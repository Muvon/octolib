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
use crate::llm::utils::is_model_in_pricing_table;
use crate::test_support::env::EnvGuard;

#[test]
fn test_supports_model() {
	let provider = GroqProvider::new();
	assert!(provider.supports_model("openai/gpt-oss-120b"));
	assert!(provider.supports_model("qwen/qwen3.8-27b"));
	assert!(provider.supports_model("any-future-model"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
	let provider = GroqProvider::new();
	assert_eq!(provider.name(), "groq");
	assert!(provider.supports_structured_output("any-model"));
	// Cached-input models
	assert!(provider.supports_caching("openai/gpt-oss-120b"));
	assert!(provider.supports_caching("openai/gpt-oss-20b"));
	assert!(provider.supports_caching("openai/gpt-oss-safeguard-20b"));
	// Non-cached models
	assert!(!provider.supports_caching("qwen/qwen3.8-27b"));
}

#[test]
fn test_pricing_gpt_oss() {
	let provider = GroqProvider::new();

	let p = provider.get_model_pricing("openai/gpt-oss-120b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.15);
	assert_eq!(p.output_price_per_1m, 0.60);
	assert_eq!(p.cache_read_price_per_1m, 0.075);

	let p = provider.get_model_pricing("openai/gpt-oss-20b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.075);
	assert_eq!(p.output_price_per_1m, 0.30);
	assert_eq!(p.cache_read_price_per_1m, 0.0375);

	let p = provider.get_model_pricing("openai/gpt-oss-safeguard-20b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.075);
	assert_eq!(p.output_price_per_1m, 0.30);
	assert_eq!(p.cache_read_price_per_1m, 0.0375);
}

#[test]
fn test_pricing_qwen() {
	let provider = GroqProvider::new();

	let p = provider.get_model_pricing("qwen/qwen3.8-27b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.80);
	assert_eq!(p.output_price_per_1m, 4.00);
	assert!(!provider.supports_caching("qwen/qwen3.8-27b"));
	assert!(provider.supports_vision("qwen/qwen3.8-27b"));
}

#[test]
fn test_retired_models_removed() {
	let provider = GroqProvider::new();
	for model in [
		"qwen/qwen3.6-27b",
		"qwen/qwen3-32b",
		"meta-llama/llama-4-scout-17b-16e-instruct",
		"moonshotai/kimi-k2-instruct-0905",
		"llama-3.3-70b-versatile",
		"llama-3.1-8b-instant",
	] {
		assert!(!is_model_in_pricing_table(model, PRICING), "{model}");
		assert!(!provider.supports_caching(model), "{model}");
	}
}

#[test]
fn test_pricing_falls_back_to_reference() {
	let provider = GroqProvider::new();
	assert!(!is_model_in_pricing_table("gpt-oss-120b", PRICING));
	let p = provider.get_model_pricing("gpt-oss-120b").unwrap();
	assert!(p.input_price_per_1m > 0.0);
}

#[test]
fn test_cost_calculation_cached_input() {
	let provider = GroqProvider::new();
	let pricing = provider.get_model_pricing("openai/gpt-oss-120b").unwrap();

	// 1M regular input + 500K output, no cache
	let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
	let expected = 0.15 + 0.5 * 0.60; // $0.45
	assert!((cost - expected).abs() < 0.001);

	// 500K regular + 500K cache_read + 500K output → cached saves money
	let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
	let expected_cached = 0.5 * 0.15 + 0.5 * 0.075 + 0.5 * 0.60; // $0.4125
	assert!((cost_cached - expected_cached).abs() < 0.001);
	assert!(cost_cached < cost);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_returns_content_and_usage() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("groq mocked answer"),
	)]);
	let _env = EnvGuard::new().set("GROQ_API_URL", mock.url.clone());

	let provider = GroqProvider::new();
	let params =
		ChatCompletionParams::new(&[], "llama-3.3-70b-versatile", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "groq mocked answer");
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);

	let requests = mock.requests();
	let req_body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(req_body["model"], "llama-3.3-70b-versatile");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	let _env = EnvGuard::new().set("GROQ_API_URL", mock.url.clone());

	let provider = GroqProvider::new();
	let params =
		ChatCompletionParams::new(&[], "llama-3.3-70b-versatile", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(
		err.to_string().to_lowercase().contains("groq"),
		"unexpected error: {err}"
	);
}
