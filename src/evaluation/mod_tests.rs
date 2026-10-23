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
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

/// Point the typesafe provider at `mock`. It posts to
/// `{TYPESAFE_BASE_URL}/v1/systemone`, and the mock answers any path.
fn route_typesafe_to(mock: &MockHttp) -> EnvGuard {
	EnvGuard::new()
		.set("TYPESAFE_BASE_URL", mock.base_url())
		.set("TYPESAFE_API_KEY", "mock-key")
}
#[tokio::test]
#[serial_test::serial]
async fn evaluate_rejects_a_request_model_that_conflicts_with_the_route() {
	let request = EvaluationRequest::new("state")
		.with_model("jev-preview")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));

	let err = evaluate("typesafe:jev-latest", request)
		.await
		.expect_err("model mismatch");
	assert!(
		err.to_string().contains("does not match 'typesafe:jev-latest'"),
		"unexpected error: {err}"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_routes_fills_the_model_and_returns_parsed_answers() {
	let body = r#"{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.92}},"usage":{"input_tokens":312,"output_tokens":48}}"#;
	let mock = MockHttp::start(vec![(200, body.to_string())]);
	let _env = route_typesafe_to(&mock);

	let request = EvaluationRequest::new("Payouts failing for 3 days.")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));
	let response = evaluate("typesafe:jev-latest", request).await.expect("mock evaluation");

	assert_eq!(response.model, "jev-1.13.0");
	assert_eq!(response.answers.get("is_urgent"), Some(&Answer::Noul { noul: 0.92 }));
	assert_eq!(response.usage.input_tokens, 312);
	assert!(
		response.usage.cost.unwrap_or(0.0) > 0.0,
		"cost priced from the jev table"
	);

	// The routed model filled the request before it hit the wire.
	let requests = mock.requests();
	let sent: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "jev-latest");
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_keeps_an_explicit_matching_request_model() {
	let body = r#"{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":10,"output_tokens":1}}"#;
	let mock = MockHttp::start(vec![(200, body.to_string())]);
	let _env = route_typesafe_to(&mock);

	let request = EvaluationRequest::new("state")
		.with_model("jev-latest")
		.with_question("is_urgent", Question::noul("urgent?"));
	let response = evaluate("typesafe:jev-latest", request).await.expect("mock evaluation");
	assert_eq!(response.answers.get("is_urgent"), Some(&Answer::Noul { noul: 0.5 }));
}
