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
use reqwest::header::HeaderValue;
use serde_json::json;

fn captured(status: u16, body: &str) -> CapturedResponse {
	CapturedResponse {
		status: reqwest::StatusCode::from_u16(status).unwrap(),
		headers: HeaderMap::new(),
		body: body.as_bytes().to_vec(),
	}
}

#[test]
fn blank_api_key_is_missing() {
	const VAR: &str = "OCTOLIB_TEST_EVALUATION_BLANK_KEY";
	std::env::set_var(VAR, "  ");
	assert!(matches!(api_key(VAR), Err(EvaluationError::MissingApiKey(_))));
	std::env::set_var(VAR, "k");
	assert_eq!(api_key(VAR).unwrap(), "k");
	std::env::remove_var(VAR);
	assert!(matches!(api_key(VAR), Err(EvaluationError::MissingApiKey(_))));
}

#[test]
fn retry_after_prefers_milliseconds_and_is_capped() {
	let mut headers = HeaderMap::new();
	assert!(retry_after(&headers).is_none());
	headers.insert("retry-after", HeaderValue::from_static("2"));
	assert_eq!(retry_after(&headers), Some(Duration::from_secs(2)));
	headers.insert("retry-after-ms", HeaderValue::from_static("250"));
	assert_eq!(retry_after(&headers), Some(Duration::from_millis(250)));
	headers.insert("retry-after-ms", HeaderValue::from_static("999999999"));
	assert_eq!(retry_after(&headers), Some(MAX_RETRY_AFTER));
}

#[test]
fn statuses_map_to_typed_errors_with_the_body_message() {
	assert!(matches!(
		require_success("typesafe", &captured(401, r#"{"error":{"message":"bad key"}}"#)),
		Err(EvaluationError::Authentication { message, .. }) if message == "bad key"
	));
	// TypeSafe's live shape nests the message under detail.
	assert!(matches!(
		require_success(
			"typesafe",
			&captured(
				401,
				r#"{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server."}}"#
			)
		),
		Err(EvaluationError::Authentication { message, .. }) if message == "Cannot authenticate with the server."
	));
	assert!(matches!(
		require_success("typesafe", &captured(422, r#"{"detail":"questions.q.type invalid"}"#)),
		Err(EvaluationError::InvalidRequest(message)) if message.contains("questions.q.type")
	));
	let mut limited = captured(429, r#"{"error":"slow down"}"#);
	limited.headers.insert("retry-after", HeaderValue::from_static("7"));
	assert!(matches!(
		require_success("typesafe", &limited),
		Err(EvaluationError::RateLimit { retry_after_secs: Some(7), message, .. }) if message == "slow down"
	));
	assert!(matches!(
		require_success("cloudflare", &captured(529, "overloaded")),
		Err(EvaluationError::Api { status: 529, message, .. }) if message == "overloaded"
	));
	assert!(matches!(
		require_success(
			"cloudflare",
			&captured(403, r#"{"errors":[{"message":"not enabled","code":5035}],"success":false}"#)
		),
		Err(EvaluationError::Permission { message, .. }) if message == "not enabled"
	));
	assert!(require_success("typesafe", &captured(200, "{}")).is_ok());
}

#[test]
fn payload_parses_answers_usage_and_prices_input_only() {
	let payload = json!({
		"model": "jev-1.13.0",
		"answers": {"is_urgent": {"type": "noul", "noul": 0.92}},
		"usage": {"input_tokens": 312, "output_tokens": 48}
	});
	let pricing = EvaluationPricing {
		input_price_per_1m: 0.042,
		output_price_per_1m: 0.0,
	};
	let response = parse_payload("typesafe", &payload, Some(pricing)).unwrap();
	assert_eq!(response.model, "jev-1.13.0");
	assert_eq!(response.answers["is_urgent"], Answer::Noul { noul: 0.92 });
	assert_eq!(response.usage.input_tokens, 312);
	assert!((response.usage.cost.unwrap() - 312.0 * 0.042 / 1_000_000.0).abs() < 1e-15);

	let unpriced = parse_payload("typesafe", &payload, None).unwrap();
	assert!(unpriced.usage.cost.is_none());

	assert!(parse_payload("typesafe", &json!({"model": "jev"}), None).is_err());
	assert!(parse_payload(
		"typesafe",
		&json!({"model": "jev", "answers": {"q": {"type": "maybe"}}, "usage": {"input_tokens": 1, "output_tokens": 0}}),
		None
	)
	.is_err());
}
