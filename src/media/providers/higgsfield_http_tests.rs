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

//! MockHTTP-driven coverage for the Higgsfield adapter's submit/poll/cancel
//! HTTP paths, complementing the pure-parser tests in `higgsfield_tests.rs`.
#![cfg(feature = "llm")]

use super::*;

const MODEL: &str = "higgsfield-ai/soul";

fn start_mock(replies: Vec<(u16, String)>) -> crate::llm::providers::mock_http::MockHttp {
	let mock = crate::llm::providers::mock_http::MockHttp::start(replies);
	std::env::set_var(API_BASE_ENV, mock.base_url());
	std::env::set_var(API_KEY_ENV, "key-id:key-secret");
	mock
}

fn finish_mock() {
	std::env::remove_var(API_BASE_ENV);
	std::env::remove_var(API_KEY_ENV);
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_posts_the_endpoint_and_polls_status() {
	let mock = start_mock(vec![
		(200, r#"{"request_id":"req-9","status":"queued"}"#.to_string()),
		(200, r#"{"status":"in_progress"}"#.to_string()),
	]);

	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let operation = provider.submit_image(request).await.expect("request submitted");
	assert_eq!(operation.status, OperationStatus::Queued);
	let handle = operation.handle.expect("handle present");
	assert_eq!(handle.remote_id, "req-9");
	assert_eq!(handle.task, MediaTask::TextToImage);

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, format!("/{MODEL}"));
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["prompt"], "a castle");

	let operation = provider.poll_image(&handle).await.expect("status polled");
	assert_eq!(operation.status, OperationStatus::Running);
	assert_eq!(mock.requests()[1].path, "/requests/req-9/status");

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn poll_image_returns_completed_artifacts() {
	let _mock = start_mock(vec![
		(200, r#"{"request_id":"req-9"}"#.to_string()),
		(
			200,
			r#"{"status":"completed","images":[{"url":"https://cdn.example.test/a.jpg"}]}"#.to_string(),
		),
	]);

	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("request submitted")
		.handle
		.expect("handle present");

	let operation = provider.poll_image(&handle).await.expect("status polled");
	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.expect("result present");
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.artifacts[0].media_type, "image/jpeg");

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn cancel_image_posts_the_cancel_route() {
	let mock = start_mock(vec![
		(200, r#"{"request_id":"req-9"}"#.to_string()),
		(200, "{}".to_string()),
	]);

	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let handle = provider
		.submit_image(request)
		.await
		.expect("request submitted")
		.handle
		.expect("handle present");

	provider.cancel_image(&handle).await.expect("request cancelled");

	let requests = mock.requests();
	assert_eq!(requests[1].method, "POST");
	assert_eq!(requests[1].path, "/requests/req-9/cancel");

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_surfaces_authentication_errors() {
	let _mock = start_mock(vec![(401, r#"{"error":{"message":"invalid key"}}"#.to_string())]);

	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let error = provider.submit_image(request).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_rejects_malformed_credentials_before_any_http_traffic() {
	let mock = start_mock(vec![(200, r#"{"request_id":"req-9"}"#.to_string())]);
	std::env::set_var(API_KEY_ENV, "not-a-pair");

	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest::new("a castle").with_model(MODEL);
	let error = provider.submit_image(request).await.expect_err("bad key rejected");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);
	assert_eq!(mock.requests().len(), 0);

	finish_mock();
}
