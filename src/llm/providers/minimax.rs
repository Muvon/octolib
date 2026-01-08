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

//! MiniMax provider implementation (Anthropic-compatible API)

use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// MiniMax pricing constants (per 1M tokens in USD)
/// Source: https://platform.minimax.io/docs/guides/pricing (as of Jan 2026)
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    ("MiniMax-M2.1-lightning", 0.30, 2.40),
    ("MiniMax-M2.1", 0.30, 1.20),
    ("MiniMax-M2", 0.30, 1.20),
];

/// Token usage breakdown for cache-aware pricing
struct CacheTokenUsage {
    regular_input_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
}

/// Calculate cost for MiniMax models with cache-aware pricing
/// - cache_creation_tokens: charged at 1.25x normal price ($0.375 per 1M for all models)
/// - cache_read_tokens: charged at 0.1x normal price ($0.03 per 1M for all models)
/// - regular_input_tokens: charged at normal price
/// - output_tokens: charged at normal price
fn calculate_cost_with_cache(model: &str, usage: CacheTokenUsage) -> Option<f64> {
    for (pricing_model, input_price, output_price) in PRICING {
        if model.contains(pricing_model) {
            // Regular input tokens at normal price
            let regular_input_cost =
                (usage.regular_input_tokens as f64 / 1_000_000.0) * input_price;

            // Cache creation tokens at fixed $0.375 per 1M tokens
            let cache_creation_cost = (usage.cache_creation_tokens as f64 / 1_000_000.0) * 0.375;

            // Cache read tokens at fixed $0.03 per 1M tokens
            let cache_read_cost = (usage.cache_read_tokens as f64 / 1_000_000.0) * 0.03;

            // Output tokens at normal price
            let output_cost = (usage.output_tokens as f64 / 1_000_000.0) * output_price;

            return Some(regular_input_cost + cache_creation_cost + cache_read_cost + output_cost);
        }
    }
    None
}

/// Helper function to calculate cost for MiniMax models
/// This is used by the helper function for individual token counts
fn calculate_minimax_cost(
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
) -> Option<f64> {
    let regular_input_tokens =
        input_tokens.saturating_sub(cache_creation_tokens + cache_read_tokens);

    let usage = CacheTokenUsage {
        regular_input_tokens: regular_input_tokens as u64,
        cache_creation_tokens: cache_creation_tokens as u64,
        cache_read_tokens: cache_read_tokens as u64,
        output_tokens: output_tokens as u64,
    };

    calculate_cost_with_cache(model, usage)
}

#[derive(Debug, Clone, Default)]
pub struct MinimaxProvider;

impl MinimaxProvider {
    pub fn new() -> Self {
        Self
    }
}

// Constants
const MINIMAX_API_KEY_ENV: &str = "MINIMAX_API_KEY";
const MINIMAX_API_URL_ENV: &str = "MINIMAX_API_URL";
const MINIMAX_API_URL: &str = "https://api.minimax.io/anthropic/v1/messages";

#[async_trait]
impl AiProvider for MinimaxProvider {
    fn name(&self) -> &str {
        "minimax"
    }

    fn supports_model(&self, model: &str) -> bool {
        // MiniMax supported models
        model.starts_with("MiniMax-M2")
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(MINIMAX_API_KEY_ENV)
            .map_err(|_| anyhow::anyhow!("MINIMAX_API_KEY not found in environment"))
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true // MiniMax supports prompt caching
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false // MiniMax doesn't support vision yet according to docs
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // MiniMax model context window limits
        if model.contains("MiniMax-M2.1") || model.contains("MiniMax-M2") {
            1_000_000 // 1M context window
        } else {
            128_000 // Default fallback
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to Anthropic format (MiniMax uses same format)
        let minimax_messages = convert_messages(&params.messages);

        // Extract system message if present
        let system_message = params
            .messages
            .iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());

        let system_cached = params
            .messages
            .iter()
            .any(|m| m.role == "system" && m.cached);

        // Validate temperature range (MiniMax requires 0.0 < temperature <= 1.0)
        if params.temperature <= 0.0 || params.temperature > 1.0 {
            return Err(anyhow::anyhow!(
                "MiniMax requires temperature in range (0.0, 1.0], got {}",
                params.temperature
            ));
        }

        // Create the request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": minimax_messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
        });

        // Add max_tokens if specified (0 means don't include it in request)
        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add system message with cache control if needed
        if system_cached {
            request_body["system"] = serde_json::json!([{
                "type": "text",
                "text": system_message,
                "cache_control": {
                    "type": "ephemeral"
                }
            }]);
        } else {
            request_body["system"] = serde_json::json!(system_message);
        }

        // Add tools if available (Anthropic format)
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                // Sort tools by name for consistent ordering
                let mut sorted_tools = tools.clone();
                sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

                let minimax_tools = sorted_tools
                    .iter()
                    .map(|f| {
                        let mut tool = serde_json::json!({
                            "name": f.name,
                            "description": f.description,
                            "input_schema": f.parameters
                        });

                        // Add cache control if present
                        if let Some(ref cache_control) = f.cache_control {
                            tool["cache_control"] = cache_control.clone();
                        }

                        tool
                    })
                    .collect::<Vec<_>>();

                request_body["tools"] = serde_json::json!(minimax_tools);
            }
        }

        // Execute the request with retry logic
        let api_url = env::var(MINIMAX_API_URL_ENV).unwrap_or_else(|_| MINIMAX_API_URL.to_string());

        let response = execute_minimax_request(
            api_key,
            api_url,
            request_body,
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        Ok(response)
    }
}

// MiniMax API structures (same as Anthropic)
#[derive(Serialize, Deserialize, Debug)]
struct MinimaxMessage {
    role: String,
    content: Vec<MinimaxContent>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum MinimaxContent {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
struct MinimaxResponse {
    content: Vec<MinimaxResponseContent>,
    usage: MinimaxUsage,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum MinimaxResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
struct MinimaxUsage {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

// Convert our session messages to MiniMax format (same as Anthropic)
fn convert_messages(messages: &[Message]) -> Vec<MinimaxMessage> {
    let mut result = Vec::new();

    for message in messages {
        // Skip system messages - they're handled separately
        if message.role == "system" {
            continue;
        }

        match message.role.as_str() {
            "tool" => {
                // Tool messages must be converted to user role with tool_result content
                let tool_call_id = message.tool_call_id.as_deref().unwrap_or("");

                let content = vec![MinimaxContent::ToolResult {
                    tool_use_id: tool_call_id.to_string(),
                    content: message.content.clone(),
                    cache_control: if message.cached {
                        Some(serde_json::json!({"type": "ephemeral"}))
                    } else {
                        None
                    },
                }];

                result.push(MinimaxMessage {
                    role: "user".to_string(), // Tool messages become user messages
                    content,
                });
            }
            _ => {
                // Handle user and assistant messages
                if message.role == "assistant" && message.tool_calls.is_some() {
                    // Assistant message with tool calls - reconstruct tool_use blocks
                    let mut content = Vec::new();

                    // Add text content if not empty
                    if !message.content.trim().is_empty() {
                        content.push(MinimaxContent::Text {
                            text: message.content.clone(),
                            cache_control: if message.cached {
                                Some(serde_json::json!({"type": "ephemeral"}))
                            } else {
                                None
                            },
                        });
                    }

                    // Add tool_use blocks from stored tool_calls in unified GenericToolCall format
                    if let Some(ref tool_calls_data) = message.tool_calls {
                        // Parse as unified GenericToolCall format
                        if let Ok(generic_calls) = serde_json::from_value::<
                            Vec<crate::llm::tool_calls::GenericToolCall>,
                        >(tool_calls_data.clone())
                        {
                            // Convert GenericToolCall to MiniMax format
                            for call in generic_calls {
                                content.push(MinimaxContent::ToolUse {
                                    id: call.id,
                                    name: call.name,
                                    input: call.arguments,
                                });
                            }
                        }
                    }

                    result.push(MinimaxMessage {
                        role: message.role.clone(),
                        content,
                    });
                } else {
                    // Handle regular user and assistant messages
                    let content = vec![MinimaxContent::Text {
                        text: message.content.clone(),
                        cache_control: if message.cached {
                            Some(serde_json::json!({"type": "ephemeral"}))
                        } else {
                            None
                        },
                    }];

                    result.push(MinimaxMessage {
                        role: message.role.clone(),
                        content,
                    });
                }
            }
        }
    }

    result
}

// Execute a single MiniMax HTTP request with smart retry delay calculation
async fn execute_minimax_request(
    api_key: String,
    api_url: String,
    request_body: serde_json::Value,
    max_retries: u32,
    base_timeout: std::time::Duration,
    cancellation_token: Option<&tokio::sync::watch::Receiver<bool>>,
) -> Result<ProviderResponse> {
    let client = Client::new();
    let start_time = std::time::Instant::now();

    let response = retry::retry_with_exponential_backoff(
        || {
            let client = client.clone();
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let request_body = request_body.clone();
            Box::pin(async move {
                client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("anthropic-version", "2023-06-01")
                    .json(&request_body)
                    .send()
                    .await
            })
        },
        max_retries,
        base_timeout,
        cancellation_token,
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "MiniMax API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = response.text().await?;
    let minimax_response: MinimaxResponse = serde_json::from_str(&response_text)?;

    // Extract content, thinking blocks, and tool calls
    let mut content_parts = Vec::new();
    let mut thinking_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for content in minimax_response.content {
        match content {
            MinimaxResponseContent::Text { text } => {
                content_parts.push(text);
            }
            MinimaxResponseContent::Thinking { thinking } => {
                thinking_parts.push(thinking);
            }
            MinimaxResponseContent::ToolUse { id, name, input } => {
                // Create generic ToolCall for processing
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: input,
                });
            }
        }
    }

    // Final content is only the text parts (thinking is separate)
    let final_content = content_parts.join("\n");

    // Extract thinking as a separate ThinkingBlock
    let (thinking, reasoning_tokens) = if thinking_parts.is_empty() {
        (None, 0)
    } else {
        let thinking_content = thinking_parts.join("\n\n");
        // Estimate reasoning tokens from content length (4 chars per token)
        let estimated = (thinking_content.len() / 4) as u64;
        (
            Some(ThinkingBlock {
                content: thinking_content,
                tokens: estimated,
            }),
            estimated,
        )
    };

    // Calculate cost with proper cache pricing
    let cached_tokens = minimax_response.usage.cache_read_input_tokens.unwrap_or(0);

    let cache_creation_tokens = minimax_response
        .usage
        .cache_creation_input_tokens
        .unwrap_or(0);

    let cost = calculate_minimax_cost(
        request_body["model"].as_str().unwrap_or(""),
        minimax_response.usage.input_tokens as u32,
        minimax_response.usage.output_tokens as u32,
        cache_creation_tokens as u32,
        cached_tokens as u32,
    );

    let usage = TokenUsage {
        prompt_tokens: minimax_response.usage.input_tokens,
        output_tokens: minimax_response.usage.output_tokens,
        reasoning_tokens, // Estimated from thinking content
        total_tokens: minimax_response.usage.input_tokens + minimax_response.usage.output_tokens,
        cached_tokens,
        cost,
        request_time_ms: Some(request_time_ms),
    };

    // Create response JSON that stores tool_calls in unified GenericToolCall format
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    if !tool_calls.is_empty() {
        let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = tool_calls
            .iter()
            .map(|tc| crate::llm::tool_calls::GenericToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
                meta: None, // MiniMax doesn't use meta fields
            })
            .collect();

        response_json["tool_calls"] = serde_json::to_value(&generic_calls).unwrap_or_default();
    }

    let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "minimax");

    Ok(ProviderResponse {
        content: final_content,
        thinking, // Extract thinking separately
        exchange,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        finish_reason: minimax_response.stop_reason,
        structured_output: None, // MiniMax doesn't support structured output
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_support() {
        let provider = MinimaxProvider::new();
        assert!(provider.supports_model("MiniMax-M2.1"));
        assert!(provider.supports_model("MiniMax-M2.1-lightning"));
        assert!(provider.supports_model("MiniMax-M2"));
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("claude-3"));
    }

    #[test]
    fn test_cost_calculation() {
        // Test MiniMax-M2.1: $0.3 input, $1.2 output
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 0, 0);
        assert_eq!(cost, Some(1.5)); // 0.3 + 1.2

        // Test with cache creation: $0.375 per 1M
        // Input: 1M total, 500K cache creation, 500K regular
        // Regular: 500K / 1M * $0.3 = $0.15
        // Cache creation: 500K / 1M * $0.375 = $0.1875
        // Output: 1M / 1M * $1.2 = $1.2
        // Total: $0.15 + $0.1875 + $1.2 = $1.5375
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 500_000, 0);
        assert_eq!(cost, Some(1.5375));

        // Test with cache read: $0.03 per 1M
        // Input: 1M total, 500K cache read, 500K regular
        // Regular: 500K / 1M * $0.3 = $0.15
        // Cache read: 500K / 1M * $0.03 = $0.015
        // Output: 1M / 1M * $1.2 = $1.2
        // Total: $0.15 + $0.015 + $1.2 = $1.365
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 0, 500_000);
        assert_eq!(cost, Some(1.365));

        // Test MiniMax-M2.1-lightning: $0.3 input, $2.4 output
        let cost = calculate_minimax_cost("MiniMax-M2.1-lightning", 1_000_000, 1_000_000, 0, 0);
        // Use approximate comparison for floating point
        assert!((cost.unwrap() - 2.7).abs() < 0.0001);
    }
}
