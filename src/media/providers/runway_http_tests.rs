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

//! MockHTTP-driven coverage for the Runway task adapter's submit/poll/cancel
//! HTTP paths, complementing the pure-parser tests in `runway.rs`.

use super::*;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

/// Start the mock and point the adapter at it. The guard restores the
/// environment when the test ends, pass or fail.
fn start_mock(replies: Vec<(u16, String)>) -> (MockHttp, EnvGuard) {
	let mock = MockHttp::start(replies);
	let env = EnvGuard::new()
		.set(API_BASE_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");
	(mock, env)
}

fn image_request() -> ImageGenerationRequest {
	let mut request = ImageGenerationRequest::new("a castle").with_model("gen4_turbo");
	request.geometry = Some(OutputGeometry::AspectRatio {
		width: 1280,
		height: 720,
	});
	request
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_posts_the_dated_contract_body() {
	let (mock, _env) = start_mock(vec![(200, r#"{"id":"task-1"}"#.to_string())]);

	let provider = RunwayMediaProvider::new();
	let operation = provider.submit_image(image_request()).await.expect("task submitted");
	assert_eq!(operation.status, OperationStatus::Queued);
	let handle = operation.handle.expect("handle present");
	assert_eq!(handle.remote_id, "task-1");
	assert_eq!(handle.task, MediaTask::TextToImage);

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, "/text_to_image");
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["model"], "gen4_turbo");
	assert_eq!(body["promptText"], "a castle");
	assert_eq!(body["ratio"], "1280:720");
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_requires_a_ratio() {
	let (_mock, _env) = start_mock(vec![(200, r#"{"id":"task-1"}"#.to_string())]);

	let provider = RunwayMediaProvider::new();
	let error = provider
		.submit_image(ImageGenerationRequest::new("a castle").with_model("gen4_turbo"))
		.await
		.expect_err("missing geometry rejected");
	assert!(
		matches!(error, MediaError::InvalidRequest(_)),
		"unexpected error: {error}"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn poll_image_returns_succeeded_artifacts() {
	let (mock, _env) = start_mock(vec![
		(200, r#"{"id":"task-1"}"#.to_string()),
		(
			200,
			r#"{"status":"SUCCEEDED","output":["https://cdn.example.test/a.png"]}"#.to_string(),
		),
	]);

	let provider = RunwayMediaProvider::new();
	let handle = provider
		.submit_image(image_request())
		.await
		.expect("task submitted")
		.handle
		.expect("handle present");

	let operation = provider.poll_image(&handle).await.expect("task polled");
	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.expect("result present");
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.artifacts[0].media_type, "image/png");

	let requests = mock.requests();
	assert_eq!(requests[1].method, "GET");
	assert_eq!(requests[1].path, "/tasks/task-1");
}

#[tokio::test]
#[serial_test::serial]
async fn cancel_image_deletes_the_task() {
	let (mock, _env) = start_mock(vec![(200, r#"{"id":"task-1"}"#.to_string()), (200, "{}".to_string())]);

	let provider = RunwayMediaProvider::new();
	let handle = provider
		.submit_image(image_request())
		.await
		.expect("task submitted")
		.handle
		.expect("handle present");

	provider.cancel_image(&handle).await.expect("task cancelled");

	let requests = mock.requests();
	assert_eq!(requests[1].method, "DELETE");
	assert_eq!(requests[1].path, "/tasks/task-1");
}

#[tokio::test]
#[serial_test::serial]
async fn submit_image_surfaces_authentication_errors() {
	let (_mock, _env) = start_mock(vec![(401, r#"{"detail":"invalid key"}"#.to_string())]);

	let provider = RunwayMediaProvider::new();
	let error = provider.submit_image(image_request()).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);
}
