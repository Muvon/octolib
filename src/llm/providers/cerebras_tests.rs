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
#[serial_test::serial]
fn test_per_call_key_authenticates_without_environment_key() {
	let provider = CerebrasProvider::new();
	let params =
		ChatCompletionParams::new(&[], "gpt-oss-120b", 1.0, 1.0, 0, 10).with_api_key("private-request-key".to_owned());
	let original = env::var_os(CEREBRAS_API_KEY_ENV);
	env::remove_var(CEREBRAS_API_KEY_ENV);
	assert!(provider.get_api_key().is_err());
	let api_key = super::super::shared::resolve_api_key(&provider, &params).unwrap();
	let request = super::super::openai_compat::build_http_request(
		&api_key,
		CEREBRAS_API_URL,
		&serde_json::json!({"model": params.model}),
	)
	.build()
	.unwrap();
	assert_eq!(request.headers()["Authorization"], "Bearer private-request-key");
	assert_eq!(request.url().as_str(), CEREBRAS_API_URL);
	assert!(!format!("{request:?}").contains("private-request-key"));
	assert!(!std::str::from_utf8(request.body().unwrap().as_bytes().unwrap())
		.unwrap()
		.contains("private-request-key"));
	match original {
		Some(value) => env::set_var(CEREBRAS_API_KEY_ENV, value),
		None => env::remove_var(CEREBRAS_API_KEY_ENV),
	}
}

#[test]
fn test_supports_model() {
	let provider = CerebrasProvider::new();
	assert!(provider.supports_model("gpt-oss-120b"));
	assert!(provider.supports_model("qwen-3.8-27b"));
	assert!(provider.supports_model("QWEN-3.8-27B"));
	assert!(!provider.supports_model(""));
	assert!(!provider.supports_model("random-model"));
	// Retired from the Cerebras catalogue (Sep 2026 docs).
	assert!(!provider.supports_model("llama-3.1-8b"));
	assert!(!provider.supports_model("zai-glm-4.7"));
}

#[test]
fn test_default_capabilities() {
	let provider = CerebrasProvider::new();
	assert_eq!(provider.name(), "cerebras");
	assert!(!provider.supports_caching("any-model"));
	assert!(!provider.supports_vision("gpt-oss-120b"));
	assert!(provider.supports_vision("qwen-3.8-27b"));
	// The Cerebras endpoint takes text and PNG/JPEG images, not video.
	assert!(!provider.supports_video("qwen-3.8-27b"));
	assert!(provider.supports_structured_output("any-model"));
	assert_eq!(provider.get_max_input_tokens("gpt-oss-120b"), 131_072);
	// Served at 128K on paid tiers, below the 262K native context.
	assert_eq!(provider.get_max_input_tokens("qwen-3.8-27b"), 131_072);
}

#[test]
fn test_pricing_support_partial() {
	let provider = CerebrasProvider::new();
	assert!(provider.get_model_pricing("gpt-oss-120b").is_some());
	let pricing = provider
		.get_model_pricing("qwen-3.8-27b")
		.expect("qwen-3.8-27b must resolve to pricing");
	assert_eq!(pricing.input_price_per_1m, 0.99);
	assert_eq!(pricing.output_price_per_1m, 1.49);
	assert!(crate::llm::utils::is_model_in_pricing_table("QWEN-3.8-27B", PRICING));
}
