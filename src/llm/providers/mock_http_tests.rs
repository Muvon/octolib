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

use super::*;

#[tokio::test]
async fn test_mock_http_serves_canned_response_and_records_request() {
	let mock = MockHttp::start(vec![(200, chat_completion_body("hello from mock"))]);

	let response = reqwest::Client::new()
		.post(&mock.url)
		.header("Content-Type", "application/json")
		.body(r#"{"model":"test"}"#)
		.send()
		.await
		.expect("request to mock");

	assert_eq!(response.status(), 200);
	let body: serde_json::Value = response.json().await.expect("json body");
	assert_eq!(body["choices"][0]["message"]["content"], "hello from mock");

	let requests = mock.requests();
	assert_eq!(requests.len(), 1);
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].body, r#"{"model":"test"}"#);
}

#[tokio::test]
async fn test_mock_http_serves_replies_in_order_then_repeats_last() {
	let mock = MockHttp::start(vec![
		(500, "{}".to_string()),
		(200, chat_completion_body("second")),
	]);

	let client = reqwest::Client::new();
	let first = client.post(&mock.url).send().await.expect("first");
	assert_eq!(first.status(), 500);
	let second = client.post(&mock.url).send().await.expect("second");
	assert_eq!(second.status(), 200);
	let third = client.post(&mock.url).send().await.expect("third repeat");
	assert_eq!(third.status(), 200);

	assert_eq!(mock.requests().len(), 3);
}
