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

/// Start the mock and point the adapter at it. The guard restores the
/// environment when the test ends, pass or fail.
fn start_mock(replies: Vec<(u16, String)>) -> (MockHttp, EnvGuard) {
	let mock = MockHttp::start(replies);
	let env = EnvGuard::new()
		.set(API_BASE_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");
	(mock, env)
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
