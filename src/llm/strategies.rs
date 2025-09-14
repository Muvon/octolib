// Copyright 2025 Muvon Un Limited
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

//! Provider strategy pattern for handling provider-specific logic

use crate::errors::{ProviderError, ProviderResult};
use crate::llm::config::CacheConfig;
use crate::llm::tool_calls::ProviderToolCalls;
use crate::llm::types::{Message, ProviderExchange, ToolCall};
use std::collections::HashMap;

/// Strategy trait for provider-specific operations
pub trait ProviderStrategy: Send + Sync {
    /// Get the provider name
    fn provider_name(&self) -> &'static str;

    /// Extract tool calls from provider response
    fn extract_tool_calls(
        &self,
        exchange: &ProviderExchange,
    ) -> ProviderResult<Option<Vec<ToolCall>>>;

    /// Format tool results for sending back to provider
    fn format_tool_results(&self, results: &[ToolResult]) -> ProviderResult<serde_json::Value>;

    /// Apply cache control to message based on provider requirements
    fn apply_cache_control(&self, message: &mut Message, config: &CacheConfig);

    /// Get provider-specific model limits
    fn get_model_limits(&self, model: &str) -> ModelLimits;

    /// Validate model name for this provider
    fn validate_model(&self, model: &str) -> ProviderResult<()>;

    /// Get provider-specific error handling
    fn handle_api_error(&self, status: u16, body: &str) -> ProviderError;
}

/// Tool result for formatting
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: String,
    pub is_error: bool,
}

/// Model limits for a provider
#[derive(Debug, Clone)]
pub struct ModelLimits {
    pub max_input_tokens: usize,
    pub max_output_tokens: usize,
    pub supports_vision: bool,
    pub supports_caching: bool,
    pub supports_tools: bool,
}

/// Anthropic provider strategy
pub struct AnthropicStrategy;

impl ProviderStrategy for AnthropicStrategy {
    fn provider_name(&self) -> &'static str {
        "anthropic"
    }

    fn extract_tool_calls(
        &self,
        exchange: &ProviderExchange,
    ) -> ProviderResult<Option<Vec<ToolCall>>> {
        let provider_calls = ProviderToolCalls::extract_from_exchange(exchange)
            .map_err(ProviderError::ToolCallError)?;

        match provider_calls {
            Some(calls) => {
                let tool_calls = calls
                    .to_tool_calls()
                    .map_err(ProviderError::ToolCallError)?;
                Ok(Some(tool_calls))
            }
            None => Ok(None),
        }
    }

    fn format_tool_results(&self, results: &[ToolResult]) -> ProviderResult<serde_json::Value> {
        let content: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": result.tool_call_id,
                    "content": result.content,
                    "is_error": result.is_error
                })
            })
            .collect();

        Ok(serde_json::json!({
            "role": "user",
            "content": content
        }))
    }

    fn apply_cache_control(&self, _message: &mut Message, _config: &CacheConfig) {
        // Anthropic uses cache control in the message content
        // This is handled at the provider level during message conversion
    }

    fn get_model_limits(&self, model: &str) -> ModelLimits {
        if model.contains("claude-opus-4")
            || model.contains("claude-sonnet-4")
            || model.contains("claude-3-5")
        {
            ModelLimits {
                max_input_tokens: 200_000,
                max_output_tokens: 8_192,
                supports_vision: true,
                supports_caching: true,
                supports_tools: true,
            }
        } else if model.contains("claude-3") {
            ModelLimits {
                max_input_tokens: 200_000,
                max_output_tokens: 4_096,
                supports_vision: model.contains("claude-3-opus")
                    || model.contains("claude-3-sonnet"),
                supports_caching: true,
                supports_tools: true,
            }
        } else {
            ModelLimits {
                max_input_tokens: 100_000,
                max_output_tokens: 4_096,
                supports_vision: false,
                supports_caching: false,
                supports_tools: true,
            }
        }
    }

    fn validate_model(&self, model: &str) -> ProviderResult<()> {
        if model.starts_with("claude-") || model.contains("claude") {
            Ok(())
        } else {
            Err(ProviderError::ModelNotSupported {
                provider: "anthropic".to_string(),
                model: model.to_string(),
            })
        }
    }

    fn handle_api_error(&self, status: u16, body: &str) -> ProviderError {
        match status {
            400 => ProviderError::ApiError {
                provider: "anthropic".to_string(),
                status,
                message: format!("Bad Request: {}", body),
            },
            401 => ProviderError::InvalidApiKey {
                provider: "anthropic".to_string(),
            },
            429 => ProviderError::RateLimitExceeded {
                provider: "anthropic".to_string(),
            },
            _ => ProviderError::ApiError {
                provider: "anthropic".to_string(),
                status,
                message: body.to_string(),
            },
        }
    }
}

/// OpenAI provider strategy
pub struct OpenAIStrategy;

impl ProviderStrategy for OpenAIStrategy {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn extract_tool_calls(
        &self,
        exchange: &ProviderExchange,
    ) -> ProviderResult<Option<Vec<ToolCall>>> {
        let provider_calls = ProviderToolCalls::extract_from_exchange(exchange)
            .map_err(ProviderError::ToolCallError)?;

        match provider_calls {
            Some(calls) => {
                let tool_calls = calls
                    .to_tool_calls()
                    .map_err(ProviderError::ToolCallError)?;
                Ok(Some(tool_calls))
            }
            None => Ok(None),
        }
    }

    fn format_tool_results(&self, results: &[ToolResult]) -> ProviderResult<serde_json::Value> {
        let messages: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "name": result.tool_name,
                    "content": result.content
                })
            })
            .collect();

        Ok(serde_json::json!(messages))
    }

    fn apply_cache_control(&self, _message: &mut Message, _config: &CacheConfig) {
        // OpenAI doesn't support cache control at the message level
    }

    fn get_model_limits(&self, model: &str) -> ModelLimits {
        if model.contains("gpt-4o") {
            ModelLimits {
                max_input_tokens: 128_000,
                max_output_tokens: 16_384,
                supports_vision: true,
                supports_caching: false,
                supports_tools: true,
            }
        } else if model.contains("gpt-4") {
            ModelLimits {
                max_input_tokens: 128_000,
                max_output_tokens: 4_096,
                supports_vision: model.contains("vision"),
                supports_caching: false,
                supports_tools: true,
            }
        } else if model.contains("gpt-3.5") {
            ModelLimits {
                max_input_tokens: 16_385,
                max_output_tokens: 4_096,
                supports_vision: false,
                supports_caching: false,
                supports_tools: true,
            }
        } else {
            ModelLimits {
                max_input_tokens: 4_096,
                max_output_tokens: 2_048,
                supports_vision: false,
                supports_caching: false,
                supports_tools: false,
            }
        }
    }

    fn validate_model(&self, model: &str) -> ProviderResult<()> {
        if model.starts_with("gpt-") || model.contains("davinci") || model.contains("curie") {
            Ok(())
        } else {
            Err(ProviderError::ModelNotSupported {
                provider: "openai".to_string(),
                model: model.to_string(),
            })
        }
    }

    fn handle_api_error(&self, status: u16, body: &str) -> ProviderError {
        match status {
            400 => ProviderError::ApiError {
                provider: "openai".to_string(),
                status,
                message: format!("Bad Request: {}", body),
            },
            401 => ProviderError::InvalidApiKey {
                provider: "openai".to_string(),
            },
            429 => ProviderError::RateLimitExceeded {
                provider: "openai".to_string(),
            },
            _ => ProviderError::ApiError {
                provider: "openai".to_string(),
                status,
                message: body.to_string(),
            },
        }
    }
}

/// Generic strategy for unknown providers
pub struct GenericStrategy {
    provider_name: String,
}

impl GenericStrategy {
    pub fn new(provider_name: String) -> Self {
        Self { provider_name }
    }
}

impl ProviderStrategy for GenericStrategy {
    fn provider_name(&self) -> &'static str {
        // This is a limitation - we can't return a dynamic string from a static method
        // In practice, this would need to be redesigned or we'd use an enum
        "generic"
    }

    fn extract_tool_calls(
        &self,
        exchange: &ProviderExchange,
    ) -> ProviderResult<Option<Vec<ToolCall>>> {
        let provider_calls = ProviderToolCalls::extract_from_exchange(exchange)
            .map_err(ProviderError::ToolCallError)?;

        match provider_calls {
            Some(calls) => {
                let tool_calls = calls
                    .to_tool_calls()
                    .map_err(ProviderError::ToolCallError)?;
                Ok(Some(tool_calls))
            }
            None => Ok(None),
        }
    }

    fn format_tool_results(&self, results: &[ToolResult]) -> ProviderResult<serde_json::Value> {
        // Generic format - array of tool result objects
        let tool_results: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "tool_call_id": result.tool_call_id,
                    "tool_name": result.tool_name,
                    "content": result.content,
                    "is_error": result.is_error
                })
            })
            .collect();

        Ok(serde_json::json!(tool_results))
    }

    fn apply_cache_control(&self, _message: &mut Message, _config: &CacheConfig) {
        // Generic providers don't support cache control
    }

    fn get_model_limits(&self, _model: &str) -> ModelLimits {
        // Conservative defaults for unknown providers
        ModelLimits {
            max_input_tokens: 4_096,
            max_output_tokens: 2_048,
            supports_vision: false,
            supports_caching: false,
            supports_tools: false,
        }
    }

    fn validate_model(&self, _model: &str) -> ProviderResult<()> {
        // Generic strategy accepts any model
        Ok(())
    }

    fn handle_api_error(&self, status: u16, body: &str) -> ProviderError {
        ProviderError::ApiError {
            provider: self.provider_name.clone(),
            status,
            message: body.to_string(),
        }
    }
}

/// Strategy factory for creating appropriate strategies
pub struct StrategyFactory;

impl StrategyFactory {
    /// Get strategy for a provider
    pub fn get_strategy(provider: &str) -> Box<dyn ProviderStrategy> {
        match provider {
            "anthropic" => Box::new(AnthropicStrategy),
            "openai" => Box::new(OpenAIStrategy),
            "openrouter" => Box::new(OpenAIStrategy), // OpenRouter uses OpenAI format
            "deepseek" => Box::new(OpenAIStrategy),   // DeepSeek uses OpenAI format
            _ => Box::new(GenericStrategy::new(provider.to_string())),
        }
    }

    /// Get all available strategies
    pub fn get_all_strategies() -> HashMap<&'static str, Box<dyn ProviderStrategy>> {
        let mut strategies = HashMap::new();
        strategies.insert(
            "anthropic",
            Box::new(AnthropicStrategy) as Box<dyn ProviderStrategy>,
        );
        strategies.insert(
            "openai",
            Box::new(OpenAIStrategy) as Box<dyn ProviderStrategy>,
        );
        strategies
    }
}

#[cfg(test)]
mod tests {
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
}
