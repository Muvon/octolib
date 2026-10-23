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

//! HTTP-path coverage for the xAI Responses API client: request shaping,
//! response parsing (text, tool calls, reasoning), usage normalization and
//! error surfacing, all through the MockHttp loopback seam.

use super::*;
use crate::llm::providers::mock_http::MockHttp;

fn responses_body(output: serde_json::Value, usage: serde_json::Value) -> String {
	serde_json::json!({
		"id": "resp_mock",
		"object": "response",
		"output": output,
		"usage": usage,
	})
	.to_string()
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_parses_text_reasoning_and_usage() {
	let body = responses_body(
		json!([
			{"type": "reasoning", "summary": [{"type": "summary_text", "text": "pondering"}]},
			{"type": "message", "content": [{"type": "output_text", "text": "mocked grok answer"}]}
		]),
		json!({
			"input_tokens": 100,
			"output_tokens": 40,
			"total_tokens": 140,
			"input_tokens_details": {"cached_tokens": 30},
			"output_tokens_details": {"reasoning_tokens": 10},
			"cost_in_usd_ticks": 37_756_000
		}),
	);
	let mock = MockHttp::start(vec![(200, body)]);
	std::env::set_var("XAI_API_URL", mock.url.clone());

	let provider = XaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "grok-4.5", 0.2, 0.9, 40, 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "mocked grok answer");
	let thinking = response.thinking.expect("thinking block present");
	assert_eq!(thinking.content, "pondering");
	assert_eq!(thinking.tokens, 10);

	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 70, "clean input excludes cache reads");
	assert_eq!(usage.cache_read_tokens, 30);
	assert_eq!(usage.output_tokens, 30);
	assert_eq!(usage.reasoning_tokens, 10);
	assert_eq!(usage.cost, Some(0.0037756));
	// The response id is stored with the xAI prefix so continuation can find it.
	assert_eq!(response.id.as_deref(), Some("xai_response:resp_mock"));

	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["model"], "grok-4.5");
	assert_eq!(request["include"][0], "reasoning.encrypted_content");
	assert_eq!(request["input"][0]["role"], "user");
	assert_eq!(request["max_output_tokens"], 100);

	std::env::remove_var("XAI_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_parses_tool_calls_and_structured_output() {
	let body = responses_body(
		json!([
			{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"rust\"}"},
			{"type": "message", "content": [{"type": "output_text", "text": "{\"answer\": 42}"}]}
		]),
		json!({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
	);
	let mock = MockHttp::start(vec![(200, body)]);
	std::env::set_var("XAI_API_URL", mock.url.clone());

	let provider = XaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("find it")], "grok-4.5", 0.2, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	let calls = response.tool_calls.expect("tool calls present");
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].id, "call_1");
	assert_eq!(calls[0].name, "lookup");
	assert_eq!(calls[0].arguments, json!({"q": "rust"}));
	assert_eq!(response.structured_output, Some(json!({"answer": 42})));

	std::env::remove_var("XAI_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = MockHttp::start(vec![(400, r#"{"error":{"message":"bad request"}}"#.to_string())]);
	std::env::set_var("XAI_API_URL", mock.url.clone());

	let provider = XaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "grok-4.5", 0.2, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("xAI API error"), "unexpected error: {err}");

	std::env::remove_var("XAI_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_rejects_unsupported_model_before_any_request() {
	let provider = XaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "grok-unknown", 0.2, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned());
	let err = provider
		.chat_completion(params)
		.await
		.expect_err("unsupported model rejected");
	assert!(err.to_string().contains("not supported"), "unexpected error: {err}");
}
