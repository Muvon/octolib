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

//! HTTP-path coverage for the OctoHub Responses-API client: request shaping
//! (instructions, tools, structured output, reasoning effort), response
//! parsing (text, tool calls, reasoning, usage) and auth-aware errors, all
//! through the MockHttp loopback seam.

use super::*;
use crate::llm::types::{FunctionDefinition, OutputFormat, ReasoningEffort, ResponseMode, StructuredOutputRequest};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;
use serde_json::json;

fn octohub_body(output: serde_json::Value, usage: serde_json::Value) -> String {
	serde_json::json!({
		"id": "resp_mock",
		"output": output,
		"usage": usage,
	})
	.to_string()
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_parses_text_reasoning_and_usage() {
	let body = octohub_body(
		json!([
			{"type": "reasoning", "content": [{"type": "output_text", "text": "pondering"}]},
			{"type": "message", "content": [{"type": "output_text", "text": "mocked octohub answer"}]}
		]),
		json!({
			"input_tokens": 100,
			"output_tokens": 40,
			"total_tokens": 140,
			"cache_read_tokens": 30,
			"cache_write_tokens": 20,
			"reasoning_tokens": 10,
			"cost": 0.0125,
			"request_time_ms": 500
		}),
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let _env = EnvGuard::new().set("OCTOHUB_API_URL", mock.base_url());

	let provider = OctoHubProvider::new();
	// new(messages, model, temperature, top_p, top_k, max_tokens)
	let params = ChatCompletionParams::new(
		&[Message::system("You are helpful."), Message::user("hello")],
		"gpt-4o",
		0.5,
		0.9,
		50,
		100,
	)
	.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "mocked octohub answer");
	let thinking = response.thinking.expect("thinking block present");
	assert_eq!(thinking.content, "pondering");
	assert_eq!(thinking.tokens, 10);

	// input_tokens is forwarded clean from upstream — cache tokens stay separate.
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 100);
	assert_eq!(usage.cache_read_tokens, 30);
	assert_eq!(usage.cache_write_tokens, 20);
	assert_eq!(usage.output_tokens, 40);
	assert_eq!(usage.reasoning_tokens, 10);
	assert_eq!(usage.cost, Some(0.0125));
	assert_eq!(response.id.as_deref(), Some("resp_mock"));

	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["model"], "gpt-4o");
	assert_eq!(request["instructions"], "You are helpful.");
	assert_eq!(request["input"][0]["role"], "user");
	assert_eq!(request["temperature"], 0.5);
	// top_p is an f32 on the way in, so compare with a tolerance.
	let top_p = request["top_p"].as_f64().expect("top_p sent");
	assert!((top_p - 0.9).abs() < 1e-6, "top_p: {top_p}");
	// OctoHub has no top_k, so it is dropped rather than sent.
	assert!(request.get("top_k").is_none());
	assert_eq!(request["max_output_tokens"], 100);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_parses_tool_calls_and_structured_output() {
	let body = octohub_body(
		json!([
			{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"rust\"}"},
			{"type": "message", "content": [{"type": "output_text", "text": "{\"answer\": 42}"}]}
		]),
		json!({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}),
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let _env = EnvGuard::new().set("OCTOHUB_API_URL", mock.base_url());

	let provider = OctoHubProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("find it")], "gpt-4o", 0.5, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	let calls = response.tool_calls.expect("tool calls present");
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].id, "call_1");
	assert_eq!(calls[0].name, "lookup");
	assert_eq!(calls[0].arguments, json!({"q": "rust"}));
	// Text-parse fallback recovers structured output from the message text.
	assert_eq!(response.structured_output, Some(json!({"answer": 42})));
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_sends_tools_reasoning_and_schema() {
	let body = octohub_body(
		json!([]),
		json!({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let _env = EnvGuard::new().set("OCTOHUB_API_URL", mock.base_url());

	let provider = OctoHubProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-4o", 0.5, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::High)
		.with_tools(vec![FunctionDefinition {
			name: "lookup".to_string(),
			description: "Look things up".to_string(),
			parameters: json!({"type": "object", "properties": {}}),
			cache_control: None,
		}])
		.with_structured_output(StructuredOutputRequest {
			format: OutputFormat::JsonSchema,
			mode: ResponseMode::Strict,
			schema: Some(json!({"type": "object", "properties": {"answer": {"type": "number"}}})),
		});
	let response = provider.chat_completion(params).await.expect("mock completion");
	assert_eq!(response.content, "");

	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["reasoning_effort"], "high");
	assert_eq!(request["tools"][0]["name"], "lookup");
	assert_eq!(request["text"]["format"]["type"], "json_schema");
	assert_eq!(request["text"]["format"]["strict"], true);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_auth_aware_error() {
	let mock = MockHttp::start(vec![(401, r#"{"error":{"message":"invalid key"}}"#.to_string())]);
	let _env = EnvGuard::new().set("OCTOHUB_API_URL", mock.base_url());

	let provider = OctoHubProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-4o", 0.5, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("auth error surfaces");
	assert!(
		err.to_string().contains("passed for this request"),
		"unexpected error: {err}"
	);
	assert!(err.to_string().contains("invalid key"));
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_retries_server_errors_then_surfaces_them() {
	let mock = MockHttp::start(vec![(500, r#"{"error":{"message":"upstream exploded"}}"#.to_string())]);
	let _env = EnvGuard::new().set("OCTOHUB_API_URL", mock.base_url());

	let provider = OctoHubProvider::new();
	// 5xx is retryable; a 1ms backoff keeps the retry from sleeping for seconds.
	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-4o", 0.5, 0.9, 0, 10)
		.with_api_key("mock-key".to_owned())
		.with_max_retries(1)
		.with_retry_timeout(std::time::Duration::from_millis(1));
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(
		err.to_string().contains("OctoHub API error 500"),
		"unexpected error: {err}"
	);
	assert_eq!(mock.requests().len(), 2, "one attempt plus one retry");
}
