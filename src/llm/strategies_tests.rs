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
