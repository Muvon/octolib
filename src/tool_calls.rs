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

//! Type-safe tool call handling system for octolib

use crate::errors::{ToolCallError, ToolCallResult};
use crate::types::{ProviderExchange, ToolCall};
use serde::{Deserialize, Serialize};

/// Anthropic-specific tool use block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolUse {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// OpenAI/OpenRouter-specific tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIToolCall {
    pub id: String,
    pub function: OpenAIFunction,
    #[serde(rename = "type")]
    pub call_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIFunction {
    pub name: String,
    pub arguments: String, // JSON string that needs parsing
}

/// Generic tool call format for other providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Provider-specific tool call formats with type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "provider")]
pub enum ProviderToolCalls {
    #[serde(rename = "anthropic")]
    Anthropic { content: Vec<AnthropicToolUse> },

    #[serde(rename = "openai")]
    OpenAI { tool_calls: Vec<OpenAIToolCall> },

    #[serde(rename = "openrouter")]
    OpenRouter { tool_calls: Vec<OpenAIToolCall> },

    #[serde(rename = "deepseek")]
    DeepSeek { tool_calls: Vec<OpenAIToolCall> },

    #[serde(rename = "generic")]
    Generic { calls: Vec<GenericToolCall> },
}

impl ProviderToolCalls {
    /// Extract tool calls from provider exchange with type safety
    pub fn extract_from_exchange(exchange: &ProviderExchange) -> ToolCallResult<Option<Self>> {
        let provider = &exchange.provider;

        match provider.as_str() {
            "anthropic" => Self::extract_anthropic_calls(exchange),
            "openai" => Self::extract_openai_calls(exchange, "openai"),
            "openrouter" => Self::extract_openai_calls(exchange, "openrouter"),
            "deepseek" => Self::extract_openai_calls(exchange, "deepseek"),
            _ => Self::extract_generic_calls(exchange),
        }
    }

    /// Extract Anthropic tool calls from unified format
    fn extract_anthropic_calls(exchange: &ProviderExchange) -> ToolCallResult<Option<Self>> {
        // Check for unified format first
        if let Some(unified_calls) = exchange.response.get("tool_calls_unified") {
            if let Ok(generic_calls) =
                serde_json::from_value::<Vec<GenericToolCall>>(unified_calls.clone())
            {
                let tool_uses: Vec<AnthropicToolUse> = generic_calls
                    .into_iter()
                    .map(|call| AnthropicToolUse {
                        id: call.id,
                        name: call.name,
                        input: call.arguments,
                    })
                    .collect();

                if !tool_uses.is_empty() {
                    return Ok(Some(ProviderToolCalls::Anthropic { content: tool_uses }));
                }
            }
        }
        Ok(None)
    }

    /// Extract OpenAI-compatible tool calls from unified format (OpenAI, OpenRouter, DeepSeek)
    fn extract_openai_calls(
        exchange: &ProviderExchange,
        provider: &str,
    ) -> ToolCallResult<Option<Self>> {
        // Check for unified format first
        if let Some(unified_calls) = exchange.response.get("tool_calls_unified") {
            if let Ok(generic_calls) =
                serde_json::from_value::<Vec<GenericToolCall>>(unified_calls.clone())
            {
                let openai_calls: Vec<OpenAIToolCall> = generic_calls
                    .into_iter()
                    .map(|call| OpenAIToolCall {
                        id: call.id,
                        call_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: call.name,
                            arguments: serde_json::to_string(&call.arguments).unwrap_or_default(),
                        },
                    })
                    .collect();

                if !openai_calls.is_empty() {
                    return Ok(Some(match provider {
                        "openai" => ProviderToolCalls::OpenAI {
                            tool_calls: openai_calls,
                        },
                        "openrouter" => ProviderToolCalls::OpenRouter {
                            tool_calls: openai_calls,
                        },
                        "deepseek" => ProviderToolCalls::DeepSeek {
                            tool_calls: openai_calls,
                        },
                        _ => unreachable!(),
                    }));
                }
            }
        }
        Ok(None)
    }

    /// Extract generic tool calls from unified format (fallback for unknown providers)
    fn extract_generic_calls(exchange: &ProviderExchange) -> ToolCallResult<Option<Self>> {
        // Check for unified format
        if let Some(unified_calls) = exchange.response.get("tool_calls_unified") {
            if let Ok(generic_calls) =
                serde_json::from_value::<Vec<GenericToolCall>>(unified_calls.clone())
            {
                if !generic_calls.is_empty() {
                    return Ok(Some(ProviderToolCalls::Generic {
                        calls: generic_calls,
                    }));
                }
            }
        }
        Ok(None)
    }

    /// Convert to generic ToolCall format for processing
    pub fn to_tool_calls(&self) -> ToolCallResult<Vec<ToolCall>> {
        match self {
            ProviderToolCalls::Anthropic { content } => content
                .iter()
                .map(|tool_use| {
                    Ok(ToolCall {
                        id: tool_use.id.clone(),
                        name: tool_use.name.clone(),
                        arguments: tool_use.input.clone(),
                    })
                })
                .collect(),

            ProviderToolCalls::OpenAI { tool_calls }
            | ProviderToolCalls::OpenRouter { tool_calls }
            | ProviderToolCalls::DeepSeek { tool_calls } => {
                tool_calls
                    .iter()
                    .map(|call| {
                        // Parse the JSON string arguments
                        let arguments: serde_json::Value = if call.function.arguments.is_empty() {
                            serde_json::Value::Object(serde_json::Map::new())
                        } else {
                            serde_json::from_str(&call.function.arguments)
                                .map_err(ToolCallError::InvalidArguments)?
                        };

                        Ok(ToolCall {
                            id: call.id.clone(),
                            name: call.function.name.clone(),
                            arguments,
                        })
                    })
                    .collect()
            }

            ProviderToolCalls::Generic { calls } => calls
                .iter()
                .map(|call| {
                    Ok(ToolCall {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: call.arguments.clone(),
                    })
                })
                .collect(),
        }
    }

    /// Get the provider name
    pub fn provider(&self) -> &'static str {
        match self {
            ProviderToolCalls::Anthropic { .. } => "anthropic",
            ProviderToolCalls::OpenAI { .. } => "openai",
            ProviderToolCalls::OpenRouter { .. } => "openrouter",
            ProviderToolCalls::DeepSeek { .. } => "deepseek",
            ProviderToolCalls::Generic { .. } => "generic",
        }
    }

    /// Get the number of tool calls
    pub fn len(&self) -> usize {
        match self {
            ProviderToolCalls::Anthropic { content } => content.len(),
            ProviderToolCalls::OpenAI { tool_calls }
            | ProviderToolCalls::OpenRouter { tool_calls }
            | ProviderToolCalls::DeepSeek { tool_calls } => tool_calls.len(),
            ProviderToolCalls::Generic { calls } => calls.len(),
        }
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Validate tool calls structure
    pub fn validate(&self) -> ToolCallResult<()> {
        match self {
            ProviderToolCalls::Anthropic { content } => {
                for tool_use in content {
                    if tool_use.id.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "id".to_string(),
                        });
                    }
                    if tool_use.name.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "name".to_string(),
                        });
                    }
                }
            }

            ProviderToolCalls::OpenAI { tool_calls }
            | ProviderToolCalls::OpenRouter { tool_calls }
            | ProviderToolCalls::DeepSeek { tool_calls } => {
                for call in tool_calls {
                    if call.id.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "id".to_string(),
                        });
                    }
                    if call.function.name.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "function.name".to_string(),
                        });
                    }
                    // Validate that arguments is valid JSON
                    if !call.function.arguments.is_empty() {
                        serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                            .map_err(ToolCallError::InvalidArguments)?;
                    }
                }
            }

            ProviderToolCalls::Generic { calls } => {
                for call in calls {
                    if call.id.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "id".to_string(),
                        });
                    }
                    if call.name.is_empty() {
                        return Err(ToolCallError::MissingField {
                            field: "name".to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TokenUsage;
    use serde_json::json;

    #[test]
    fn test_anthropic_tool_call_extraction() {
        let exchange = ProviderExchange::new(
            json!({}),
            json!({
                "tool_calls_unified": [
                    {
                        "id": "toolu_123",
                        "name": "test_tool",
                        "arguments": {"param": "value"}
                    }
                ]
            }),
            Some(TokenUsage {
                prompt_tokens: 100,
                output_tokens: 50,
                total_tokens: 150,
                cached_tokens: 0,
                cost: Some(0.01),
                request_time_ms: Some(1000),
            }),
            "anthropic",
        );

        let result = ProviderToolCalls::extract_from_exchange(&exchange).unwrap();
        assert!(result.is_some());

        let tool_calls = result.unwrap();
        assert_eq!(tool_calls.provider(), "anthropic");
        assert_eq!(tool_calls.len(), 1);

        // Validate
        tool_calls.validate().unwrap();

        // Convert to generic format
        let generic_calls = tool_calls.to_tool_calls().unwrap();
        assert_eq!(generic_calls.len(), 1);
        assert_eq!(generic_calls[0].name, "test_tool");
        assert_eq!(generic_calls[0].id, "toolu_123");
    }

    #[test]
    fn test_openai_tool_call_extraction() {
        let exchange = ProviderExchange::new(
            json!({}),
            json!({
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{\"param\": \"value\"}"
                            }
                        }]
                    }
                }]
            }),
            None,
            "openai",
        );

        let result = ProviderToolCalls::extract_from_exchange(&exchange).unwrap();
        assert!(result.is_some());

        let tool_calls = result.unwrap();
        assert_eq!(tool_calls.provider(), "openai");
        assert_eq!(tool_calls.len(), 1);

        // Validate
        tool_calls.validate().unwrap();

        // Convert to generic format
        let generic_calls = tool_calls.to_tool_calls().unwrap();
        assert_eq!(generic_calls.len(), 1);
        assert_eq!(generic_calls[0].name, "test_tool");
        assert_eq!(generic_calls[0].id, "call_123");
    }

    #[test]
    fn test_invalid_tool_call_format() {
        let exchange = ProviderExchange::new(
            json!({}),
            json!({
                "tool_calls_unified": [
                    {
                        // Missing required fields
                        "name": "test_tool"
                    }
                ]
            }),
            None,
            "anthropic",
        );

        let result = ProviderToolCalls::extract_from_exchange(&exchange);
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_errors() {
        let tool_calls = ProviderToolCalls::Anthropic {
            content: vec![AnthropicToolUse {
                id: "".to_string(), // Empty ID should fail validation
                name: "test".to_string(),
                input: json!({}),
            }],
        };

        let result = tool_calls.validate();
        assert!(result.is_err());

        if let Err(ToolCallError::MissingField { field }) = result {
            assert_eq!(field, "id");
        } else {
            panic!("Expected MissingField error");
        }
    }

    #[test]
    fn test_invalid_json_arguments() {
        let tool_calls = ProviderToolCalls::OpenAI {
            tool_calls: vec![OpenAIToolCall {
                id: "call_123".to_string(),
                call_type: "function".to_string(),
                function: OpenAIFunction {
                    name: "test".to_string(),
                    arguments: "invalid json".to_string(),
                },
            }],
        };

        let result = tool_calls.to_tool_calls();
        assert!(result.is_err());
    }
}
