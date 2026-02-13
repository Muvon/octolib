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

use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
use crate::llm::utils::{
    contains_ignore_ascii_case, normalize_model_name, starts_with_ignore_ascii_case,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// MiniMax pricing constants (per 1M tokens in USD)
/// Source: https://platform.minimax.io/docs/guides/pricing (as of Feb 2026)
/// Updated with latest pricing from MiniMax official docs
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    ("MiniMax-M2.1-lightning", 0.30, 2.40),
    ("MiniMax-M2.1", 0.27, 0.95),
    ("MiniMax-M2", 0.255, 1.00),
];

/// Token usage breakdown for cache-aware pricing
struct CacheTokenUsage {
    regular_input_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
}

/// Calculate cost for MiniMax models with cache-aware pricing (case-insensitive)
/// - cache_creation_tokens: charged at 1.25x normal price ($0.375 per 1M for all models)
/// - cache_read_tokens: charged at 0.1x normal price ($0.03 per 1M for all models)
/// - regular_input_tokens: charged at normal price
/// - output_tokens: charged at normal price
fn calculate_cost_with_cache(model: &str, usage: CacheTokenUsage) -> Option<f64> {
    for (pricing_model, input_price, output_price) in PRICING {
        if contains_ignore_ascii_case(model, pricing_model) {
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
        // MiniMax supported models (case-insensitive)
        starts_with_ignore_ascii_case(model, "minimax-m2")
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

    fn supports_structured_output(&self, _model: &str) -> bool {
        true // MiniMax supports structured output via response_format
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Search through pricing table for matching model
        for (pricing_model, input_price, output_price) in PRICING {
            if contains_ignore_ascii_case(model, pricing_model) {
                // MiniMax cache pricing:
                // - Cache write: $0.375 per 1M tokens (fixed for all models)
                // - Cache read: $0.03 per 1M tokens (fixed for all models)
                return Some(crate::llm::types::ModelPricing::new(
                    *input_price,
                    *output_price,
                    0.375, // Cache write price (fixed)
                    0.03,  // Cache read price (fixed)
                ));
            }
        }
        None
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // MiniMax model context window limits (case-insensitive)
        let model_lower = normalize_model_name(model);
        if model_lower.contains("minimax-m2.1") || model_lower.contains("minimax-m2") {
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

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request_body["response_format"] = serde_json::json!({
                        "type": "json_object"
                    });
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    if let Some(schema) = &response_format.schema {
                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema
                            }
                        });

                        // Add strict mode if specified
                        if matches!(
                            response_format.mode,
                            crate::llm::types::ResponseMode::Strict
                        ) {
                            format_obj["json_schema"]["strict"] = serde_json::json!(true);
                        }

                        request_body["response_format"] = format_obj;
                    }
                }
            }
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
    id: String,
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
                    .map_err(anyhow::Error::from)
            })
        },
        max_retries,
        base_timeout,
        cancellation_token,
        || ProviderError::Cancelled.into(),
        |e| {
            matches!(
                e.downcast_ref::<ProviderError>(),
                Some(ProviderError::Cancelled)
            )
        },
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = retry::cancellable(
            async { response.text().await.map_err(anyhow::Error::from) },
            cancellation_token,
            || ProviderError::Cancelled.into(),
        )
        .await?;
        return Err(anyhow::anyhow!(
            "MiniMax API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = retry::cancellable(
        async { response.text().await.map_err(anyhow::Error::from) },
        cancellation_token,
        || ProviderError::Cancelled.into(),
    )
    .await?;
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
    let cache_read_tokens = minimax_response.usage.cache_read_input_tokens.unwrap_or(0);

    let cache_creation_tokens = minimax_response
        .usage
        .cache_creation_input_tokens
        .unwrap_or(0);

    // CRITICAL: cached_tokens should ONLY be cache_read_tokens
    // cache_creation_tokens are NEW tokens being written to cache (not cached yet)
    // MiniMax's input_tokens = regular + cache_creation + cache_read
    // So: regular_input_tokens = input_tokens - cache_creation - cache_read
    // And cached_tokens (for display) = only cache_read (actually served from cache)
    let cached_tokens = cache_read_tokens;

    let cost = calculate_minimax_cost(
        request_body["model"].as_str().unwrap_or(""),
        minimax_response.usage.input_tokens as u32,
        minimax_response.usage.output_tokens as u32,
        cache_creation_tokens as u32,
        cache_read_tokens as u32,
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

    // Try to parse structured output if it was requested
    let structured_output =
        if final_content.trim().starts_with('{') || final_content.trim().starts_with('[') {
            serde_json::from_str(&final_content).ok()
        } else {
            None
        };

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
        structured_output,
        id: Some(minimax_response.id),
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
    fn test_model_support_case_insensitive() {
        let provider = MinimaxProvider::new();
        // Test lowercase
        assert!(provider.supports_model("minimax-m2.1"));
        assert!(provider.supports_model("minimax-m2.1-lightning"));
        assert!(provider.supports_model("minimax-m2"));
        // Test uppercase
        assert!(provider.supports_model("MINIMAX-M2.1"));
        assert!(provider.supports_model("MINIMAX-M2"));
        // Test mixed case
        assert!(provider.supports_model("Minimax-M2.1"));
        assert!(provider.supports_model("MINIMAX-m2.1"));
    }

    #[test]
    fn test_cost_calculation() {
        // Test MiniMax-M2.1: $0.27 input, $0.95 output (updated Feb 2026)
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 0, 0);
        assert_eq!(cost, Some(1.22)); // 0.27 + 0.95

        // Test with cache creation: $0.375 per 1M
        // Input: 1M total, 500K cache creation, 500K regular
        // Regular: 500K / 1M * $0.27 = $0.135
        // Cache creation: 500K / 1M * $0.375 = $0.1875
        // Output: 1M / 1M * $0.95 = $0.95
        // Total: $0.135 + $0.1875 + $0.95 = $1.2725
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 500_000, 0);
        assert_eq!(cost, Some(1.2725));

        // Test with cache read: $0.03 per 1M
        // Input: 1M total, 500K cache read, 500K regular
        // Regular: 500K / 1M * $0.27 = $0.135
        // Cache read: 500K / 1M * $0.03 = $0.015
        // Output: 1M / 1M * $0.95 = $0.95
        // Total: $0.135 + $0.015 + $0.95 = $1.10
        let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 0, 500_000);
        assert_eq!(cost, Some(1.10));

        // Test MiniMax-M2.1-lightning: $0.3 input, $2.4 output
        let cost = calculate_minimax_cost("MiniMax-M2.1-lightning", 1_000_000, 1_000_000, 0, 0);
        // Use approximate comparison for floating point
        assert!((cost.unwrap() - 2.7).abs() < 0.0001);

        // Test MiniMax-M2: $0.255 input, $1.00 output (updated Feb 2026)
        let cost = calculate_minimax_cost("MiniMax-M2", 1_000_000, 1_000_000, 0, 0);
        assert_eq!(cost, Some(1.255)); // 0.255 + 1.00
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = MinimaxProvider::new();
        assert!(provider.supports_caching("MiniMax-M2.1"));
        assert!(!provider.supports_vision("MiniMax-M2.1"));
        assert!(provider.supports_structured_output("MiniMax-M2.1"));
    }
}
