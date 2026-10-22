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

#[test]
fn test_supports_model() {
	let provider = NvidiaProvider::new();
	assert!(provider.supports_model("nvidia/llama-3.1-nemotron-ultra-253b-v1"));
	assert!(provider.supports_model("deepseek-ai/deepseek-v3.2"));
	assert!(provider.supports_model("minimaxai/minimax-m2.1"));
	assert!(provider.supports_model("meta/llama-3.1-405b-instruct"));
	assert!(provider.supports_model("any-model"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
	let provider = NvidiaProvider::new();
	assert_eq!(provider.name(), "nvidia");
	assert!(!provider.supports_caching("any-model"));
	assert!(provider.supports_structured_output("any-model"));
}

#[tokio::test]
#[serial_test::serial]
async fn test_chat_completion_via_mock_http() {
	std::env::set_var("NVIDIA_API_URL", "");
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(
		200,
		crate::llm::providers::mock_http::chat_completion_body("mocked answer"),
	)]);
	std::env::set_var("NVIDIA_API_URL", mock.url.clone());

	let provider = NvidiaProvider::new();
	let params = ChatCompletionParams::new(&[], "meta/llama-3.1-405b-instruct", 1.0, 1.0, 0, 10)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "mocked answer");
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);
	assert!(usage.cost.is_some(), "cost filled from reference pricing");

	let requests = mock.requests();
	assert_eq!(requests.len(), 1);
	let body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["model"], "meta/llama-3.1-405b-instruct");

	std::env::remove_var("NVIDIA_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn test_chat_completion_http_error_via_mock() {
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	std::env::set_var("NVIDIA_API_URL", mock.url.clone());

	let provider = NvidiaProvider::new();
	let params = ChatCompletionParams::new(&[], "meta/llama-3.1-405b-instruct", 1.0, 1.0, 0, 10)
		.with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("nvidia API error"), "unexpected error: {err}");

	std::env::remove_var("NVIDIA_API_URL");
}
