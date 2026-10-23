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

//! MockHTTP-driven coverage for the OpenRouter media adapter's immediate
//! image and speech endpoints, complementing `openrouter.rs` parser tests.
#![cfg(feature = "llm")]

use super::*;

fn start_mock(replies: Vec<(u16, String)>) -> crate::llm::providers::mock_http::MockHttp {
	let mock = crate::llm::providers::mock_http::MockHttp::start(replies);
	std::env::set_var(API_BASE_ENV, mock.base_url());
	std::env::set_var(API_KEY_ENV, "mock-key");
	mock
}

fn finish_mock() {
	std::env::remove_var(API_BASE_ENV);
	std::env::remove_var(API_KEY_ENV);
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_decodes_inline_b64_artifacts() {
	let encoded = base64::engine::general_purpose::STANDARD.encode(b"image-bytes");
	let body = format!(r#"{{"data":[{{"b64_json":"{encoded}"}}]}}"#);
	let mock = start_mock(vec![(200, body)]);

	let provider = OpenRouterMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("openai/gpt-image-1");
	let operation = provider.submit_image(request).await.expect("image completed");
	assert_eq!(operation.status, OperationStatus::Succeeded);

	let result = operation.result.expect("result present");
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.artifacts[0].media_type, "image/png");
	assert!(matches!(result.artifacts[0].source, ArtifactSource::Inline(_)));

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, "/images");
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["model"], "openai/gpt-image-1");
	assert_eq!(body["prompt"], "a castle");

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_rejects_out_of_range_counts() {
	let _mock = start_mock(vec![(200, "{}".to_string())]);

	let provider = OpenRouterMediaProvider::new();
	let mut request = ImageGenerationRequest::new("a castle").with_model("openai/gpt-image-1");
	request.count = Some(11);
	let error = provider.submit_image(request).await.expect_err("count rejected");
	assert!(
		matches!(error, MediaError::InvalidRequest(_)),
		"unexpected error: {error}"
	);

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_surfaces_authentication_errors() {
	let _mock = start_mock(vec![(401, r#"{"error":{"message":"bad key"}}"#.to_string())]);

	let provider = OpenRouterMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model("openai/gpt-image-1");
	let error = provider.submit_image(request).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_speech_posts_text_and_voice() {
	let encoded = base64::engine::general_purpose::STANDARD.encode(b"audio-bytes");
	let body = format!(r#"{{"data":[{{"b64_json":"{encoded}","media_type":"audio/mpeg"}}]}}"#);
	let mock = start_mock(vec![(200, body)]);

	let provider = OpenRouterMediaProvider::new();
	let request = SpeechSynthesisRequest::new("hello there", "alloy").with_model("openai/tts-1");
	let operation = provider.submit_speech(request).await.expect("speech completed");
	assert_eq!(operation.status, OperationStatus::Succeeded);

	let result = operation.result.expect("result present");
	assert_eq!(result.artifact.media_type, "application/json");

	let requests = mock.requests();
	assert_eq!(requests[0].path, "/audio/speech");
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["input"], "hello there");
	assert_eq!(body["voice"], "alloy");

	finish_mock();
}
