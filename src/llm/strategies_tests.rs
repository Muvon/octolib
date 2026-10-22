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
fn test_strategy_factory() {
	let anthropic_strategy = StrategyFactory::get_strategy("anthropic");
	assert_eq!(anthropic_strategy.provider_name(), "anthropic");

	let openai_strategy = StrategyFactory::get_strategy("openai");
	assert_eq!(openai_strategy.provider_name(), "openai");

	let generic_strategy = StrategyFactory::get_strategy("unknown");
	assert_eq!(generic_strategy.provider_name(), "generic");
}

#[test]
fn test_anthropic_model_validation() {
	let strategy = AnthropicStrategy;

	assert!(strategy.validate_model("claude-3-sonnet").is_ok());
	assert!(strategy.validate_model("claude-opus-4").is_ok());
	assert!(strategy.validate_model("gpt-4").is_err());
}

#[test]
fn test_openai_model_validation() {
	let strategy = OpenAIStrategy;

	assert!(strategy.validate_model("gpt-4o").is_ok());
	assert!(strategy.validate_model("gpt-3.5-turbo").is_ok());
	assert!(strategy.validate_model("claude-3-sonnet").is_err());
}

#[test]
fn test_model_limits() {
	let anthropic_strategy = AnthropicStrategy;
	let limits = anthropic_strategy.get_model_limits("claude-3-5-sonnet");

	assert_eq!(limits.max_input_tokens, 200_000);
	assert!(limits.supports_vision);
	assert!(limits.supports_caching);
	assert!(limits.supports_tools);
}

#[test]
fn test_tool_result_formatting() {
	let anthropic_strategy = AnthropicStrategy;
	let results = vec![ToolResult {
		tool_call_id: "toolu_123".to_string(),
		tool_name: "test_tool".to_string(),
		content: "result content".to_string(),
		is_error: false,
	}];

	let formatted = anthropic_strategy.format_tool_results(&results).unwrap();
	assert_eq!(formatted["role"], "user");
	assert!(formatted["content"].is_array());
}

#[test]
fn test_anthropic_model_limits_branches() {
	let strategy = AnthropicStrategy;

	// claude-3 (non-4, non-3-5): vision only for opus/sonnet
	let haiku = strategy.get_model_limits("claude-3-haiku");
	assert_eq!(haiku.max_input_tokens, 200_000);
	assert_eq!(haiku.max_output_tokens, 4_096);
	assert!(!haiku.supports_vision);
	assert!(haiku.supports_caching);

	let opus3 = strategy.get_model_limits("claude-3-opus");
	assert!(opus3.supports_vision);

	// unknown claude family
	let unknown = strategy.get_model_limits("claude-instant");
	assert_eq!(unknown.max_input_tokens, 100_000);
	assert!(!unknown.supports_caching);
	assert!(unknown.supports_tools);

	// sonnet-4 gets the modern limits
	let sonnet4 = strategy.get_model_limits("claude-sonnet-4");
	assert_eq!(sonnet4.max_output_tokens, 8_192);
	assert!(sonnet4.supports_vision);
}

#[test]
fn test_openai_model_limits_branches() {
	let strategy = OpenAIStrategy;

	let gpt4o = strategy.get_model_limits("gpt-4o-mini");
	assert_eq!(gpt4o.max_input_tokens, 128_000);
	assert_eq!(gpt4o.max_output_tokens, 16_384);
	assert!(gpt4o.supports_vision);

	let gpt4 = strategy.get_model_limits("gpt-4-turbo");
	assert_eq!(gpt4.max_output_tokens, 4_096);
	assert!(!gpt4.supports_vision);

	let gpt4_vision = strategy.get_model_limits("gpt-4-vision-preview");
	assert!(gpt4_vision.supports_vision);

	let gpt35 = strategy.get_model_limits("gpt-3.5-turbo");
	assert_eq!(gpt35.max_input_tokens, 16_385);

	let unknown = strategy.get_model_limits("llama-3");
	assert_eq!(unknown.max_input_tokens, 4_096);
	assert!(!unknown.supports_tools);
}

#[test]
fn test_validate_model_accepts_substring_families() {
	assert!(AnthropicStrategy.validate_model("my-claude-finetune").is_ok());
	assert!(OpenAIStrategy.validate_model("davinci-002").is_ok());
	assert!(OpenAIStrategy.validate_model("curie").is_ok());
	assert!(GenericStrategy::new("anything".to_string())
		.validate_model("whatever")
		.is_ok());
}

#[test]
fn test_handle_api_error_mapping() {
	let anthropic = AnthropicStrategy;
	assert!(matches!(
		anthropic.handle_api_error(400, "bad"),
		ProviderError::ApiError { .. }
	));
	assert!(matches!(
		anthropic.handle_api_error(401, ""),
		ProviderError::InvalidApiKey { .. }
	));
	assert!(matches!(
		anthropic.handle_api_error(429, ""),
		ProviderError::RateLimitExceeded { .. }
	));
	assert!(matches!(
		anthropic.handle_api_error(500, "boom"),
		ProviderError::ApiError { .. }
	));

	let openai = OpenAIStrategy;
	assert!(matches!(
		openai.handle_api_error(400, "bad"),
		ProviderError::ApiError { .. }
	));
	assert!(matches!(
		openai.handle_api_error(401, ""),
		ProviderError::InvalidApiKey { .. }
	));
	assert!(matches!(
		openai.handle_api_error(429, ""),
		ProviderError::RateLimitExceeded { .. }
	));
	assert!(matches!(
		openai.handle_api_error(503, "down"),
		ProviderError::ApiError { .. }
	));

	let generic = GenericStrategy::new("proxy".to_string());
	match generic.handle_api_error(500, "oops") {
		ProviderError::ApiError {
			provider,
			status,
			message,
		} => {
			assert_eq!(provider, "proxy");
			assert_eq!(status, 500);
			assert_eq!(message, "oops");
		}
		other => panic!("expected ApiError, got {:?}", other),
	}
}

#[test]
fn test_format_tool_results_per_strategy() {
	let results = vec![
		ToolResult {
			tool_call_id: "call_1".to_string(),
			tool_name: "tool_a".to_string(),
			content: "ok".to_string(),
			is_error: false,
		},
		ToolResult {
			tool_call_id: "call_2".to_string(),
			tool_name: "tool_b".to_string(),
			content: "failed".to_string(),
			is_error: true,
		},
	];

	let anthropic = AnthropicStrategy.format_tool_results(&results).unwrap();
	assert_eq!(anthropic["role"], "user");
	let content = anthropic["content"].as_array().unwrap();
	assert_eq!(content.len(), 2);
	assert_eq!(content[0]["tool_use_id"], "call_1");
	assert_eq!(content[0]["is_error"], false);
	assert_eq!(content[1]["is_error"], true);

	let openai = OpenAIStrategy.format_tool_results(&results).unwrap();
	let messages = openai.as_array().unwrap();
	assert_eq!(messages.len(), 2);
	assert_eq!(messages[0]["role"], "tool");
	assert_eq!(messages[0]["tool_call_id"], "call_1");
	assert_eq!(messages[0]["name"], "tool_a");

	let generic = GenericStrategy::new("x".to_string())
		.format_tool_results(&results)
		.unwrap();
	let arr = generic.as_array().unwrap();
	assert_eq!(arr.len(), 2);
	assert_eq!(arr[0]["tool_name"], "tool_a");
	assert_eq!(arr[1]["is_error"], true);
}

#[test]
fn test_extract_tool_calls_via_strategy() {
	let response = serde_json::json!({
		"tool_calls": [{ "id": "call_1", "name": "tool", "arguments": {"a": 1} }]
	});
	let exchange = ProviderExchange::new(serde_json::json!({}), response, None, "openai");
	let extracted = OpenAIStrategy.extract_tool_calls(&exchange).unwrap().unwrap();
	assert_eq!(extracted.len(), 1);
	assert_eq!(extracted[0].name, "tool");

	let empty_exchange = ProviderExchange::new(serde_json::json!({}), serde_json::json!({}), None, "openai");
	assert!(OpenAIStrategy.extract_tool_calls(&empty_exchange).unwrap().is_none());
	assert!(AnthropicStrategy.extract_tool_calls(&empty_exchange).unwrap().is_none());
	assert!(GenericStrategy::new("g".to_string())
		.extract_tool_calls(&empty_exchange)
		.unwrap()
		.is_none());
}

#[test]
fn test_apply_cache_control_is_noop_for_all_strategies() {
	let mut message = Message::user("hello");
	let config = CacheConfig::default();
	AnthropicStrategy.apply_cache_control(&mut message, &config);
	OpenAIStrategy.apply_cache_control(&mut message, &config);
	GenericStrategy::new("g".to_string()).apply_cache_control(&mut message, &config);
	assert_eq!(message.content, "hello");
}

#[test]
fn test_generic_strategy_defaults() {
	let strategy = GenericStrategy::new("proxy".to_string());
	assert_eq!(strategy.provider_name(), "generic");
	let limits = strategy.get_model_limits("any-model");
	assert_eq!(limits.max_input_tokens, 4_096);
	assert_eq!(limits.max_output_tokens, 2_048);
	assert!(!limits.supports_vision);
	assert!(!limits.supports_caching);
	assert!(!limits.supports_tools);
}

#[test]
fn test_strategy_factory_aliases_and_all_strategies() {
	assert_eq!(StrategyFactory::get_strategy("openrouter").provider_name(), "openai");
	assert_eq!(StrategyFactory::get_strategy("deepseek").provider_name(), "openai");

	let all = StrategyFactory::get_all_strategies();
	assert_eq!(all.len(), 2);
	assert!(all.contains_key("anthropic"));
	assert!(all.contains_key("openai"));
}
