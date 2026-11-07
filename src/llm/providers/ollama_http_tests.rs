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

//! HTTP-path coverage for the Ollama proxy: request shaping through the
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
		crate::test_support::mock_http::chat_completion_body("ollama answer"),
	)]);
	let _env = EnvGuard::new().set("OLLAMA_API_URL", mock.url.clone());

	let provider = OllamaProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "ollama answer");
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
	assert_eq!(request["keep_alive"], "5m");
	assert_eq!(request["max_tokens"], 100);
}

#[tokio::test]
#[serial_test::serial]
async fn long_cache_keeps_the_model_loaded_for_an_hour() {
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("ollama answer"),
	)]);
	let _env = EnvGuard::new().set("OLLAMA_API_URL", mock.url.clone());

	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-oss:120b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned())
		.with_long_cache(true);
	OllamaProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");

	let request: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(request["keep_alive"], "1h");
}

#[tokio::test]
#[serial_test::serial]
async fn cache_prewarm_asks_for_one_token_and_drops_the_output_format() {
	let mock = MockHttp::start(vec![(200, crate::test_support::mock_http::chat_completion_body("o"))]);
	let _env = EnvGuard::new().set("OLLAMA_API_URL", mock.url.clone());

	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-oss:120b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned())
		.with_structured_output(crate::llm::types::StructuredOutputRequest::json())
		.with_cache_prewarm(true);
	let response = OllamaProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");

	assert_eq!(response.content, "o");
	let request: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(request["max_tokens"], 1);
	assert!(request.get("format").is_none());
	assert_eq!(request["keep_alive"], "5m");
}

#[tokio::test]
#[serial_test::serial]
async fn cached_tokens_are_billed_at_the_ollama_cached_rate() {
	let body = r#"{"id":"mock-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":10,"total_tokens":1010,"prompt_tokens_details":{"cached_tokens":800}}}"#;
	let mock = MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("OLLAMA_API_URL", mock.url.clone());

	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-oss:120b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned());
	let response = OllamaProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");

	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 200);
	assert_eq!(usage.cache_read_tokens, 800);
	let expected = (200.0 * 0.15 + 800.0 * 0.014 + 10.0 * 0.60) / 1_000_000.0;
	assert!(
		(usage.cost.expect("cost filled") - expected).abs() < 1e-12,
		"unexpected cost: {:?}",
		usage.cost
	);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_retries_server_errors_then_surfaces_them() {
	let mock = MockHttp::start(vec![(503, r#"{"error":{"message":"overloaded"}}"#.to_string())]);
	let _env = EnvGuard::new().set("OLLAMA_API_URL", mock.url.clone());

	let provider = OllamaProvider::new();
	// 5xx is retryable; a 1ms backoff keeps the retry from sleeping for seconds.
	let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 0.2, 0.9, 0, 100)
		.with_api_key("mock-key".to_owned())
		.with_max_retries(1)
		.with_retry_timeout(std::time::Duration::from_millis(1));
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("ollama API error"), "unexpected error: {err}");
	assert_eq!(mock.requests().len(), 2, "one attempt plus one retry");
}
