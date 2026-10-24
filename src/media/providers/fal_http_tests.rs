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

//! MockHTTP-driven coverage for the fal queue adapter's submit/poll/cancel
//! HTTP paths, complementing the pure-parser tests in `fal.rs`.

use super::*;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

const MODEL: &str = "fal-ai/flux/dev";
/// Queue request URLs address the owner/app pair, dropping deeper segments.
const APP: &str = "fal-ai/flux";

/// What a test keeps alive besides the queue mock it inspects: the environment
/// (restored when the test ends, pass or fail) and fal's model catalog.
struct Scope {
	_env: EnvGuard,
	catalog: MockHttp,
}

/// fal's catalog listing for one endpoint.
fn listing(model: &str, category: &str) -> (u16, String) {
	(
		200,
		json!({"models": [{"endpoint_id": model, "metadata": {"category": category}}]}).to_string(),
	)
}

/// Start the queue mock and point the adapter at it. The catalog gets its own
/// mock: whether an endpoint's category is already cached must not shift the
/// queue's replies.
fn start_fal(catalog_reply: (u16, String), replies: Vec<(u16, String)>) -> (MockHttp, Scope) {
	let mock = MockHttp::start(replies);
	let catalog = MockHttp::start(vec![catalog_reply]);
	let env = EnvGuard::new()
		.set(API_BASE_ENV, mock.base_url())
		.set(OPENAPI_URL_ENV, format!("{}/openapi.json", mock.base_url()))
		.set(MODELS_URL_ENV, catalog.base_url())
		.set(API_KEY_ENV, "mock-key");
	(mock, Scope { _env: env, catalog })
}

fn start_mock(replies: Vec<(u16, String)>) -> (MockHttp, Scope) {
	start_fal(listing(MODEL, "text-to-image"), replies)
}

async fn submit_handle(mock: &MockHttp) -> JobHandle {
	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let operation = provider.submit_image(request).await.expect("queue submitted");
	assert_eq!(operation.status, OperationStatus::Queued);

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, format!("/{MODEL}"));
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["prompt"], "a castle");

	operation.handle.expect("handle present")
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_enqueues_and_polls_queue_position() {
	let (mock, _env) = start_mock(vec![
		(200, r#"{"request_id":"req-1","status":"IN_QUEUE"}"#.to_string()),
		(200, r#"{"status":"IN_QUEUE","queue_position":3}"#.to_string()),
	]);

	let provider = FalMediaProvider::new();
	let handle = submit_handle(&mock).await;
	assert_eq!(handle.remote_id, "req-1");
	assert_eq!(handle.task, MediaTask::TextToImage);

	let operation = provider.poll_image(&handle).await.expect("status polled");
	assert_eq!(operation.status, OperationStatus::Queued);
	assert_eq!(operation.handle.expect("handle present").remote_id, "req-1");

	let requests = mock.requests();
	assert_eq!(requests[1].method, "GET");
	assert_eq!(requests[1].path, format!("/{APP}/requests/req-1/status"));
}

#[tokio::test]
#[serial_test::serial]
async fn poll_image_fetches_the_result_once_completed() {
	let (_submit_mock, _env) = start_mock(vec![(200, r#"{"request_id":"req-1"}"#.to_string())]);
	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("queue submitted")
		.handle
		.expect("handle present");

	let (mock, _env) = start_mock(vec![
		(200, r#"{"status":"COMPLETED"}"#.to_string()),
		(
			200,
			r#"{"images":[{"url":"https://v3.fal.media/files/a.png"}]}"#.to_string(),
		),
	]);
	let operation = provider.poll_image(&handle).await.expect("result fetched");
	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.expect("result present");
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.artifacts[0].media_type, "image/png");

	let requests = mock.requests();
	assert_eq!(requests[0].path, format!("/{APP}/requests/req-1/status"));
	assert_eq!(requests[1].path, format!("/{APP}/requests/req-1"));
}

/// The prod case (2026-10-23): a music model submitted as an image finished with
/// audio only. That is the job's final outcome, so it must come back Failed —
/// as an error it left the job live and every poll re-read the same output.
#[tokio::test]
#[serial_test::serial]
async fn poll_image_fails_a_completed_request_with_no_image_output() {
	let (_mock, _env) = start_mock(vec![
		(200, r#"{"request_id":"req-1"}"#.to_string()),
		(200, r#"{"status":"COMPLETED"}"#.to_string()),
		(
			200,
			r#"{"audio":{"url":"https://v3.fal.media/files/track.mp3"}}"#.to_string(),
		),
	]);

	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a calm track").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("queue submitted")
		.handle
		.expect("handle present");

	let operation = provider.poll_image(&handle).await.expect("outcome, not an error");
	assert_eq!(operation.status, OperationStatus::Failed);
	let error = operation.error.expect("failure present");
	assert_eq!(error.category, FailureCategory::InvalidResponse);
	assert!(error.message.contains("image"), "unexpected message: {error:?}");
	assert_eq!(error.request_id.as_deref(), Some("req-1"));
}

#[tokio::test]
#[serial_test::serial]
async fn poll_image_maps_a_422_result_to_invalid_input_failure() {
	let (_mock, _env) = start_mock(vec![
		(200, r#"{"request_id":"req-1"}"#.to_string()),
		(200, r#"{"status":"COMPLETED"}"#.to_string()),
		(422, r#"{"detail":"prompt violates policy"}"#.to_string()),
	]);

	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("queue submitted")
		.handle
		.expect("handle present");

	let operation = provider.poll_image(&handle).await.expect("failure fetched");
	assert_eq!(operation.status, OperationStatus::Failed);
	let error = operation.error.expect("failure present");
	assert_eq!(error.category, FailureCategory::InvalidInput);
	assert_eq!(error.http_status, Some(422));
	assert!(error.message.contains("policy"), "unexpected message: {error:?}");
}

#[tokio::test]
#[serial_test::serial]
async fn cancel_image_puts_the_cancel_route() {
	let (mock, _env) = start_mock(vec![
		(200, r#"{"request_id":"req-1"}"#.to_string()),
		(200, "{}".to_string()),
	]);

	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("queue submitted")
		.handle
		.expect("handle present");

	provider.cancel_image(&handle).await.expect("request cancelled");

	let requests = mock.requests();
	assert_eq!(requests[1].method, "PUT");
	assert_eq!(requests[1].path, format!("/{APP}/requests/req-1/cancel"));
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_surfaces_authentication_errors() {
	let (_mock, _env) = start_mock(vec![(401, r#"{"detail":"invalid key"}"#.to_string())]);

	let provider = FalMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let error = provider.submit_image(request).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);
}

/// Minimal fal OpenAPI document: one path, one input schema with `properties`.
fn openapi_reply(model: &str, properties: Value) -> (u16, String) {
	let document = json!({
		"paths": {format!("/{model}"): {"post": {"requestBody": {"content": {
			"application/json": {"schema": {"$ref": "#/components/schemas/VideoInput"}}
		}}}}},
		"components": {"schemas": {"VideoInput": {"properties": properties}}}
	});
	(200, document.to_string())
}

fn start_video_mock(model: &str, replies: Vec<(u16, String)>) -> (MockHttp, Scope) {
	start_fal(listing(model, "text-to-video"), replies)
}

fn video_request(model: &str, duration: f64) -> VideoGenerationRequest {
	let mut request = VideoGenerationRequest::new("a wave").with_model(model);
	request.duration_secs = Some(duration);
	request
}

const QUEUED: &str = r#"{"request_id":"req-1"}"#;

/// The prod case (2026-10-23): a music endpoint submitted as an image. fal's
/// catalog says it makes audio, so nothing is queued — while its own route
/// still goes through.
#[tokio::test]
#[serial_test::serial]
async fn submit_refuses_a_route_the_endpoint_does_not_make() {
	let model = "fal-ai/test-music";
	let (mock, _scope) = start_fal(listing(model, "text-to-audio"), vec![(200, QUEUED.to_string())]);

	let error = FalMediaProvider::new()
		.submit_image(ImageGenerationRequest::new("a calm track").with_model(model))
		.await
		.expect_err("an audio endpoint is not an image route");
	assert!(
		matches!(
			error,
			MediaError::UnsupportedTask {
				task: MediaTask::TextToImage,
				..
			}
		),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "nothing may be queued");

	FalMediaProvider::new()
		.submit_speech(SpeechSynthesisRequest::new("a calm track", "default").with_model(model))
		.await
		.expect("the audio route is accepted");
	assert_eq!(mock.requests().len(), 1);
}

/// A private app is absent from fal's public catalog (404). It cannot be
/// checked, so it is submitted as before.
#[tokio::test]
#[serial_test::serial]
async fn an_endpoint_the_catalog_does_not_list_is_submitted_unchecked() {
	let (mock, _scope) = start_fal(
		(404, r#"{"error":{"type":"not_found"}}"#.to_string()),
		vec![(200, QUEUED.to_string())],
	);
	FalMediaProvider::new()
		.submit_image(ImageGenerationRequest::new("a castle").with_model("someone/private-app"))
		.await
		.expect("submitted unchecked");
	assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
#[serial_test::serial]
async fn the_catalog_is_read_once_per_endpoint() {
	let model = "fal-ai/test-catalog-cache";
	let (mock, scope) = start_fal(listing(model, "text-to-image"), vec![(200, QUEUED.to_string())]);
	let provider = FalMediaProvider::new();
	for _ in 0..2 {
		provider
			.submit_image(ImageGenerationRequest::new("a castle").with_model(model))
			.await
			.expect("submitted");
	}
	assert_eq!(scope.catalog.requests().len(), 1);
	assert_eq!(mock.requests().len(), 2);
}

/// Submit with the given `duration` property schema and return the queue POST
/// body; also checks the schema GET came first.
async fn submitted_duration_body(model: &str, property: Value, duration: f64) -> Value {
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"duration": property})),
			(200, QUEUED.to_string()),
		],
	);
	FalMediaProvider::new()
		.submit_video(video_request(model, duration))
		.await
		.expect("queue submitted");
	let requests = mock.requests();
	assert_eq!(requests[0].method, "GET");
	assert!(
		requests[0]
			.path
			.ends_with(&format!("endpoint_id={}", model.replace('/', "%2F")))
			|| requests[0].path.ends_with(&format!("endpoint_id={model}")),
		"unexpected schema path: {}",
		requests[0].path
	);
	assert_eq!(requests[1].method, "POST");
	serde_json::from_str(&requests[1].body).expect("request json")
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_uses_enum_string_when_schema_declares_it() {
	let body = submitted_duration_body(
		"bytedance/test-enum/text-to-video",
		json!({"type": "string", "enum": ["auto", "4", "5"]}),
		5.0,
	)
	.await;
	assert_eq!(body["duration"], "5");
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_uses_suffixed_enum_string() {
	let body = submitted_duration_body(
		"fal-ai/test-veo/text-to-video",
		json!({"type": "string", "enum": ["4s", "8s"]}),
		8.0,
	)
	.await;
	assert_eq!(body["duration"], "8s");
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_enum_without_match_is_invalid_request_naming_values() {
	let model = "fal-ai/test-enum-miss/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"duration": {"type": "string", "enum": ["5", "10"]}})),
			(200, QUEUED.to_string()),
		],
	);
	let error = FalMediaProvider::new()
		.submit_video(video_request(model, 7.0))
		.await
		.expect_err("7s is not allowed");
	assert!(
		matches!(&error, MediaError::InvalidRequest(message) if message.contains(model) && message.contains("5, 10")),
		"unexpected error: {error}"
	);
	assert_eq!(mock.requests().len(), 1, "no queue POST after a rejected duration");
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_integer_schema_gets_whole_number() {
	let body = submitted_duration_body("minimax/test-int/text-to-video", json!({"type": "integer"}), 5.0).await;
	assert_eq!(body["duration"], json!(5));
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_integer_schema_rejects_fractional_seconds() {
	let model = "minimax/test-int-frac/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"duration": {"type": "integer"}})),
			(200, QUEUED.to_string()),
		],
	);
	let error = FalMediaProvider::new()
		.submit_video(video_request(model, 5.5))
		.await
		.expect_err("fractional seconds rejected");
	assert!(
		matches!(error, MediaError::InvalidRequest(_)),
		"unexpected error: {error}"
	);
	assert_eq!(mock.requests().len(), 1, "no queue POST after a rejected duration");
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_number_schema_keeps_fraction() {
	let body = submitted_duration_body(
		"minimax/test-number/text-to-video",
		json!({"type": "number", "minimum": 0.92, "maximum": 15}),
		5.5,
	)
	.await;
	assert_eq!(body["duration"], json!(5.5));
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_optional_integer_unwraps_any_of() {
	let body = submitted_duration_body(
		"fal-ai/test-anyof/text-to-video",
		json!({"anyOf": [{"type": "integer"}, {"type": "null"}]}),
		6.0,
	)
	.await;
	assert_eq!(body["duration"], json!(6));
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_absent_field_errors_by_default() {
	let model = "fal-ai/test-absent-error/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"prompt": {"type": "string"}})),
			(200, QUEUED.to_string()),
		],
	);
	let error = FalMediaProvider::new()
		.submit_video(video_request(model, 5.0))
		.await
		.expect_err("duration unsupported");
	assert!(
		matches!(&error, MediaError::UnsupportedParameter { parameter, .. } if parameter == "duration_secs"),
		"unexpected error: {error}"
	);
	assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_absent_field_is_dropped_with_warning_under_warn_and_drop() {
	let model = "fal-ai/test-absent-warn/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"prompt": {"type": "string"}})),
			(200, QUEUED.to_string()),
		],
	);
	let mut request = video_request(model, 5.0);
	request.request_options.unsupported_parameter_policy = UnsupportedParameterPolicy::WarnAndDrop;
	let operation = FalMediaProvider::new()
		.submit_video(request)
		.await
		.expect("submit succeeds when duration is dropped");
	let body: Value = serde_json::from_str(&mock.requests()[1].body).expect("request json");
	assert!(body.get("duration").is_none());
	let warnings = operation.handle.expect("handle present").warnings;
	assert!(warnings
		.iter()
		.any(|warning| warning.parameter.as_deref() == Some("duration_secs")));
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_schema_404_fails_without_queue_post() {
	let model = "fal-ai/test-404/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			(404, r#"{"detail":"not found"}"#.to_string()),
			(200, QUEUED.to_string()),
		],
	);
	FalMediaProvider::new()
		.submit_video(video_request(model, 5.0))
		.await
		.expect_err("schema fetch failure surfaces");
	assert!(mock.requests().iter().all(|request| request.method == "GET"));
}

#[tokio::test]
#[serial_test::serial]
async fn video_duration_schema_is_fetched_once_per_model() {
	let model = "fal-ai/test-cache/text-to-video";
	let (mock, _env) = start_video_mock(
		model,
		vec![
			openapi_reply(model, json!({"duration": {"type": "integer"}})),
			(200, QUEUED.to_string()),
			(200, QUEUED.to_string()),
		],
	);
	let provider = FalMediaProvider::new();
	provider
		.submit_video(video_request(model, 5.0))
		.await
		.expect("first submit");
	provider
		.submit_video(video_request(model, 5.0))
		.await
		.expect("second submit");
	let methods: Vec<String> = mock.requests().into_iter().map(|request| request.method).collect();
	assert_eq!(methods, ["GET", "POST", "POST"]);
}
