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
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	// The typesafe provider posts to {TYPESAFE_BASE_URL}/v1/systemone; any path
	// on the mock listener is answered with the canned body.
	let base = mock.url.trim_end_matches("/v1/chat/completions").to_string();
	std::env::set_var("TYPESAFE_BASE_URL", base);
	std::env::set_var("TYPESAFE_API_KEY", "mock-key");

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

	std::env::remove_var("TYPESAFE_BASE_URL");
	std::env::remove_var("TYPESAFE_API_KEY");
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_keeps_an_explicit_matching_request_model() {
	let body = r#"{"model":"jev-1.13.0","answers":{"is_urgent":{"type":"noul","noul":0.5}},"usage":{"input_tokens":10,"output_tokens":1}}"#;
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let base = mock.url.trim_end_matches("/v1/chat/completions").to_string();
	std::env::set_var("TYPESAFE_BASE_URL", base);
	std::env::set_var("TYPESAFE_API_KEY", "mock-key");

	let request = EvaluationRequest::new("state")
		.with_model("jev-latest")
		.with_question("is_urgent", Question::noul("urgent?"));
	let response = evaluate("typesafe:jev-latest", request).await.expect("mock evaluation");
	assert_eq!(response.answers.get("is_urgent"), Some(&Answer::Noul { noul: 0.5 }));

	std::env::remove_var("TYPESAFE_BASE_URL");
	std::env::remove_var("TYPESAFE_API_KEY");
}
