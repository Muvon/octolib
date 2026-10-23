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

//! HTTP-path coverage for the NVIDIA proxy: request shaping through the
//! MockHttp loopback seam, reference-cost fill, and error surfacing.

use super::*;
use crate::llm::types::Message;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_fills_reference_cost() {
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("nvidia answer"),
	)]);
	let _env = EnvGuard::new().set("NVIDIA_API_URL", mock.url.clone());

	let provider = NvidiaProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "nvidia answer");
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);
	let expected = (12.0 / 1_000_000.0) * 0.10 + (7.0 / 1_000_000.0) * 0.10;
	assert!(
		(usage.cost.expect("reference cost filled") - expected).abs() < 1e-12,
		"unexpected cost: {:?}",
		usage.cost
	);

	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["model"], "llama-3.1-8b");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_bills_reasoning_at_the_output_rate() {
	let arguments = serde_json::json!({"city": "Oslo"});
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::tool_call_body("call_1", "get_weather", &arguments, 8),
	)]);
	let _env = EnvGuard::new().set("NVIDIA_API_URL", mock.url.clone());

	let provider = NvidiaProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("weather?")], "llama-3.1-8b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	let calls = response.tool_calls.expect("tool calls present");
	assert_eq!(calls[0].id, "call_1");
	assert_eq!(calls[0].name, "get_weather");
	assert_eq!(calls[0].arguments, arguments);
	assert_eq!(response.thinking.expect("thinking present").content, "pondering");

	// 20 completion tokens split into 12 visible + 8 reasoning, and all 20 bill
	// at the output rate. Pricing only the visible 12 would under-charge.
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.output_tokens, 12);
	assert_eq!(usage.reasoning_tokens, 8);
	let expected = (5.0 / 1_000_000.0) * 0.10 + (20.0 / 1_000_000.0) * 0.10;
	assert!(
		(usage.cost.expect("reference cost filled") - expected).abs() < 1e-12,
		"unexpected cost: {:?}",
		usage.cost
	);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = MockHttp::start(vec![(401, r#"{"error":{"message":"unauthorized"}}"#.to_string())]);
	let _env = EnvGuard::new().set("NVIDIA_API_URL", mock.url.clone());

	let provider = NvidiaProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("nvidia API error"), "unexpected error: {err}");
}
