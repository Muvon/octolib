// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! MockHTTP-driven coverage for the Replicate adapter's submit/poll/cancel
//! HTTP paths, complementing the pure-parser tests in `replicate.rs`.

use super::*;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

fn schema_reply() -> String {
	r#"{"latest_version":{"openapi_schema":{"components":{}}}}"#.to_string()
}

fn prediction_succeeded(output: &str) -> String {
	format!(r#"{{"id":"pred-42","status":"succeeded","output":[{output}]}}"#)
}

/// Start the mock and point the adapter at it. The guard restores the
/// environment when the test ends, pass or fail.
fn start_mock(replies: Vec<(u16, String)>) -> (MockHttp, EnvGuard) {
	let mock = MockHttp::start(replies);
	let env = EnvGuard::new()
		.set(API_BASE_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");
	(mock, env)
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_posts_prediction_after_schema_discovery() {
	let (mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(200, prediction_succeeded(r#""https://cdn.example.test/a.png""#)),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("owner/model");
	let operation = provider.submit_image(request).await.expect("prediction submitted");

	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.expect("result present");
	assert_eq!(result.artifacts.len(), 1);
	assert!(matches!(result.artifacts[0].source, ArtifactSource::Url(_)));
	let handle = operation.handle.expect("handle present");
	assert_eq!(handle.remote_id, "pred-42");
	assert_eq!(handle.task, MediaTask::TextToImage);

	let requests = mock.requests();
	assert_eq!(requests.len(), 2);
	assert_eq!(requests[0].method, "GET");
	assert_eq!(requests[0].path, "/models/owner/model");
	assert_eq!(requests[1].method, "POST");
	assert_eq!(requests[1].path, "/models/owner/model/predictions");
	let body: Value = serde_json::from_str(&requests[1].body).expect("request json");
	assert_eq!(body["input"]["prompt"], "a castle");
}

#[tokio::test]
#[serial_test::serial]
async fn poll_image_resumes_a_succeeded_prediction_from_the_handle() {
	let (_submit_mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(200, r#"{"id":"pred-42","status":"processing"}"#.to_string()),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("owner/model");
	let operation = provider.submit_image(request).await.expect("prediction submitted");
	assert_eq!(operation.status, OperationStatus::Running);
	let handle = operation.handle.expect("handle present");

	let (poll_mock, _env) = start_mock(vec![(200, prediction_succeeded(r#""https://cdn.example.test/a.png""#))]);
	let operation = provider.poll_image(&handle).await.expect("prediction polled");
	assert_eq!(operation.status, OperationStatus::Succeeded);
	assert_eq!(operation.result.expect("result present").artifacts.len(), 1);

	let requests = poll_mock.requests();
	assert_eq!(requests[0].method, "GET");
	assert_eq!(requests[0].path, "/predictions/pred-42");
}

/// A prediction Replicate reports succeeded whose output is not the requested
/// kind (audio for an image) is the job's outcome: Failed, never an error that
/// leaves it live for re-polling.
#[tokio::test]
#[serial_test::serial]
async fn poll_image_fails_a_succeeded_prediction_with_no_image_output() {
	let (_submit_mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(200, r#"{"id":"pred-42","status":"processing"}"#.to_string()),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = ImageGenerationRequest::new("a calm track").with_model("owner/model");
	let handle = provider
		.submit_image(request)
		.await
		.expect("prediction submitted")
		.handle
		.expect("handle present");

	let (_poll_mock, _env) = start_mock(vec![(
		200,
		prediction_succeeded(r#""https://cdn.example.test/track.mp3""#),
	)]);
	let operation = provider.poll_image(&handle).await.expect("outcome, not an error");
	assert_eq!(operation.status, OperationStatus::Failed);
	let error = operation.error.expect("failure present");
	assert_eq!(error.category, FailureCategory::InvalidResponse);
	assert_eq!(error.request_id.as_deref(), Some("pred-42"));
}

#[tokio::test]
#[serial_test::serial]
async fn cancel_image_posts_the_cancel_route() {
	let (mock, _env) = start_mock(vec![(200, "{}".to_string())]);

	let provider = ReplicateMediaProvider::new();
	let handle = JobHandle {
		provider: PROVIDER.to_string(),
		model: "owner/model".to_string(),
		task: MediaTask::TextToImage,
		remote_id: "pred-42".to_string(),
		cost_estimate: None,
		warnings: Vec::new(),
		expected_outputs: None,
	};
	provider.cancel_image(&handle).await.expect("prediction cancelled");

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, "/predictions/pred-42/cancel");
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_surfaces_authentication_errors() {
	let (_mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(401, r#"{"detail":"invalid token"}"#.to_string()),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("owner/model");
	let error = provider.submit_image(request).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn submit_transcription_normalizes_inline_text_output() {
	let (mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(
			200,
			r#"{"id":"pred-7","status":"succeeded","output":{"text":"hello world","language":"en"}}"#.to_string(),
		),
	]);

	let provider = ReplicateMediaProvider::new();
	let audio = MediaSource::Url {
		url: "https://cdn.example.test/audio.mp3".to_string(),
		media_type: None,
	};
	let mut request = TranscriptionRequest::new(audio).with_model("owner/transcriber");
	request.language = Some("en".to_string());
	let operation = provider
		.submit_transcription(request)
		.await
		.expect("transcription submitted");

	let result = operation.result.expect("result present");
	assert_eq!(result.text, "hello world");
	assert_eq!(result.language.as_deref(), Some("en"));

	let requests = mock.requests();
	let body: Value = serde_json::from_str(&requests[1].body).expect("request json");
	assert_eq!(body["input"]["audio"], "https://cdn.example.test/audio.mp3");
	assert_eq!(body["input"]["language"], "en");
}

#[tokio::test]
#[serial_test::serial]
async fn submit_speech_selects_the_first_audio_artifact() {
	let (_mock, _env) = start_mock(vec![
		(200, schema_reply()),
		(200, prediction_succeeded(r#""https://cdn.example.test/line.mp3""#)),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = SpeechSynthesisRequest::new("hello there", "narrator").with_model("owner/speaker");
	let operation = provider.submit_speech(request).await.expect("speech submitted");

	let result = operation.result.expect("result present");
	assert_eq!(result.artifact.media_type, "audio/mpeg");
}

#[tokio::test]
#[serial_test::serial]
async fn webhooks_must_be_https_and_are_validated_before_submission() {
	let (_mock, _env) = start_mock(vec![(200, schema_reply())]);

	let provider = ReplicateMediaProvider::new();
	// Build provider options directly: `into_provider_options` would reject the
	// plain-HTTP webhook before submission, which is a separate unit check.
	let raw = ReplicateMediaOptions {
		webhook: Some("http://hooks.example.test/replicate".to_string()),
		..ReplicateMediaOptions::default()
	};
	let mut options = ProviderOptions::new();
	options.insert(PROVIDER.to_string(), serde_json::to_value(raw).unwrap());
	let mut request = ImageGenerationRequest::new("a castle").with_model("owner/model");
	request.provider_options = options;
	let error = provider.submit_image(request).await.expect_err("http webhook rejected");
	assert!(
		matches!(error, MediaError::InvalidRequest(_)),
		"unexpected error: {error}"
	);

	// The webhook is rejected while parsing options, before any HTTP traffic.
	assert_eq!(_mock.requests().len(), 0);
}

#[tokio::test]
#[serial_test::serial]
async fn schema_discovery_failure_defers_validation_with_a_warning() {
	// Schema discovery is sent with max_retries: 0, so exactly one failed GET
	// defers validation with a warning before the prediction POST.
	let (_mock, _env) = start_mock(vec![
		(500, r#"{"detail":"boom"}"#.to_string()),
		(200, prediction_succeeded(r#""https://cdn.example.test/a.png""#)),
	]);

	let provider = ReplicateMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("owner/model");
	let operation = provider.submit_image(request).await.expect("prediction submitted");

	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.expect("result present");
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::SchemaUnavailable));
}
