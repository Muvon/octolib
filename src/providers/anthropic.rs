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

//! Anthropic provider implementation

use crate::retry;
use crate::traits::AiProvider;
use crate::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, TokenUsage, ToolCall,
};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// Anthropic pricing constants (per 1M tokens in USD)
/// Source: https://docs.anthropic.com/en/docs/about-claude/pricing (as of June 2025)
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    ("claude-opus-4-0", 15.00, 75.00),
    ("claude-opus-4-1", 15.00, 75.00),
    ("claude-sonnet-4-0", 3.00, 15.00),
    ("claude-3-7-sonnet", 3.00, 15.00),
    ("claude-3-5-sonnet", 3.00, 15.00),
    ("claude-3-5-haiku", 0.80, 4.00),
    ("claude-3-opus", 15.00, 75.00),
    ("claude-3-sonnet", 3.00, 15.00),
    ("claude-3-haiku", 0.25, 1.25),
];

/// Calculate cost for Anthropic models with proper cache pricing
fn calculate_anthropic_cost(
    model: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
    cache_creation_tokens: u64,
) -> Option<f64> {
    for (pricing_model, input_price, output_price) in PRICING {
        if model.contains(pricing_model) {
            // Regular input tokens (excluding cached and cache creation)
            let regular_input_tokens =
                prompt_tokens.saturating_sub(cached_tokens + cache_creation_tokens);
            let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input_price;

            // Cache creation tokens at 1.25x price (25% more expensive)
            let cache_creation_cost =
                (cache_creation_tokens as f64 / 1_000_000.0) * input_price * 1.25;

            // Cache read tokens at 0.1x price (90% cheaper)
            let cache_read_cost = (cached_tokens as f64 / 1_000_000.0) * input_price * 0.1;

            // Output tokens at normal price (never cached)
            let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

            let total_cost =
                regular_input_cost + cache_creation_cost + cache_read_cost + output_cost;
            return Some(total_cost);
        }
    }
    None
}

/// Check if a model supports temperature parameter
/// All Claude models support temperature
fn supports_temperature_and_top_p(model: &str) -> bool {
    !model.contains("opus-4-1")
}

/// Anthropic provider
#[derive(Debug, Clone)]
pub struct AnthropicProvider;

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicProvider {
    pub fn new() -> Self {
        Self
    }
}

// Constants
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

#[async_trait::async_trait]
impl AiProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Anthropic Claude models
        model.starts_with("claude-") || model.contains("claude")
    }

    fn get_api_key(&self) -> Result<String> {
        // API keys now only from environment variables for security
        match env::var(ANTHROPIC_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "Anthropic API key not found in environment variable: {}",
                ANTHROPIC_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Claude 3+ models support vision
        model.contains("claude-3")
            || model.contains("claude-4")
            || model.contains("claude-3.7")
            || model.contains("sonnet")
            || model.contains("opus")
            || model.contains("haiku")
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Anthropic model context window limits (what we can send as input)
        if model.contains("claude-opus-4") || model.contains("claude-sonnet-4") {
            // Claude 4 models have 200k context
            200_000
        } else if model.contains("claude-3-7") {
            // Claude 3.7 has 200k context
            200_000
        } else if model.contains("claude-3-5") {
            // Claude 3.5 models have 200k context
            200_000
        } else if model.contains("claude-3") {
            // Claude 3 models have 200k context
            200_000
        } else {
            // Default fallback for older models
            100_000
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to Anthropic format
        let anthropic_messages = convert_messages(&params.messages);

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

        // Create the request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": anthropic_messages,
        });
        request_body["temperature"] = serde_json::json!(params.temperature);

        // Opus 4.1 doesn't support using temperature and top_p together, so we do this instead
        if supports_temperature_and_top_p(&params.model) {
            request_body["top_p"] = serde_json::json!(params.top_p);
        }

        request_body["top_k"] = serde_json::json!(params.top_k);

        // Add max_tokens if specified (0 means don't include it in request)
        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add system message with cache control if needed
        if system_cached {
            let cache_ttl = crate::config::CacheTTL::short();
            request_body["system"] = serde_json::json!([{
                "type": "text",
                "text": system_message,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": cache_ttl.to_string()
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

                let anthropic_tools = sorted_tools
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

                request_body["tools"] = serde_json::json!(anthropic_tools);
            }
        }

        // Execute the request with retry logic
        let response = execute_anthropic_request(
            api_key,
            request_body,
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        Ok(response)
    }
}

// Anthropic API structures
#[derive(Serialize, Deserialize, Debug)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },
    #[serde(rename = "image")]
    Image {
        source: ImageSource,
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

#[derive(Serialize, Deserialize, Debug)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    content: Vec<AnthropicResponseContent>,
    usage: AnthropicUsage,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum AnthropicResponseContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
}

// Convert our session messages to Anthropic format
fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
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

                let content = vec![AnthropicContent::ToolResult {
                    tool_use_id: tool_call_id.to_string(),
                    content: message.content.clone(),
                    cache_control: if message.cached {
                        Some(serde_json::json!({"type": "ephemeral"}))
                    } else {
                        None
                    },
                }];

                result.push(AnthropicMessage {
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
                        content.push(AnthropicContent::Text {
                            text: message.content.clone(),
                            cache_control: if message.cached {
                                Some(serde_json::json!({"type": "ephemeral"}))
                            } else {
                                None
                            },
                        });
                    }

                    // Add tool_use blocks from stored tool_calls
                    if let Some(ref tool_calls_data) = message.tool_calls {
                        if let Some(content_blocks) =
                            tool_calls_data.get("content").and_then(|c| c.as_array())
                        {
                            // Extract tool_use blocks from stored data
                            for block in content_blocks {
                                if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                    if let (Some(id), Some(name), Some(input)) = (
                                        block.get("id").and_then(|v| v.as_str()),
                                        block.get("name").and_then(|v| v.as_str()),
                                        block.get("input"),
                                    ) {
                                        content.push(AnthropicContent::ToolUse {
                                            id: id.to_string(),
                                            name: name.to_string(),
                                            input: input.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }

                    result.push(AnthropicMessage {
                        role: message.role.clone(),
                        content,
                    });
                } else {
                    // Handle regular user and assistant messages
                    let mut content = vec![AnthropicContent::Text {
                        text: message.content.clone(),
                        cache_control: if message.cached {
                            Some(serde_json::json!({"type": "ephemeral"}))
                        } else {
                            None
                        },
                    }];

                    // Add images if present
                    if let Some(images) = &message.images {
                        for image in images {
                            if let crate::types::ImageData::Base64(data) = &image.data {
                                content.push(AnthropicContent::Image {
                                    source: ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: image.media_type.clone(),
                                        data: data.clone(),
                                    },
                                    cache_control: None,
                                });
                            }
                        }
                    }

                    result.push(AnthropicMessage {
                        role: message.role.clone(),
                        content,
                    });
                }
            }
        }
    }

    result
}

// Execute a single Anthropic HTTP request with smart retry delay calculation
async fn execute_anthropic_request(
    api_key: String,
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
            let request_body = request_body.clone();
            Box::pin(async move {
                client
                    .post(ANTHROPIC_API_URL)
                    .header("Content-Type", "application/json")
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("anthropic-beta", "prompt-caching-2024-07-31")
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

    // Extract rate limit headers before consuming response
    let mut rate_limit_headers = std::collections::HashMap::new();
    let headers = response.headers();

    // Anthropic rate limit headers
    if let Some(tokens_limit) = headers
        .get("anthropic-ratelimit-tokens-limit")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("tokens_limit".to_string(), tokens_limit.to_string());
    }
    if let Some(tokens_remaining) = headers
        .get("anthropic-ratelimit-tokens-remaining")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("tokens_remaining".to_string(), tokens_remaining.to_string());
    }
    if let Some(input_tokens_limit) = headers
        .get("anthropic-ratelimit-input-tokens-limit")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert(
            "input_tokens_limit".to_string(),
            input_tokens_limit.to_string(),
        );
    }
    if let Some(input_tokens_remaining) = headers
        .get("anthropic-ratelimit-input-tokens-remaining")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert(
            "input_tokens_remaining".to_string(),
            input_tokens_remaining.to_string(),
        );
    }
    if let Some(output_tokens_limit) = headers
        .get("anthropic-ratelimit-output-tokens-limit")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert(
            "output_tokens_limit".to_string(),
            output_tokens_limit.to_string(),
        );
    }
    if let Some(output_tokens_remaining) = headers
        .get("anthropic-ratelimit-output-tokens-remaining")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert(
            "output_tokens_remaining".to_string(),
            output_tokens_remaining.to_string(),
        );
    }

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "Anthropic API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = response.text().await?;
    let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)?;

    // Extract content and tool calls
    let mut content_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_use_blocks = Vec::new(); // Store original tool_use blocks for Anthropic

    for content in anthropic_response.content {
        match content {
            AnthropicResponseContent::Text { text } => {
                content_parts.push(text);
            }
            AnthropicResponseContent::ToolUse { id, name, input } => {
                // Store the original tool_use block for conversation history
                tool_use_blocks.push(serde_json::json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));

                // Create generic ToolCall for processing
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments: input,
                });
            }
        }
    }

    let content = content_parts.join("\n");

    // Calculate cost with proper cache pricing
    let cached_tokens = anthropic_response
        .usage
        .cache_read_input_tokens
        .unwrap_or(0);

    let cache_creation_tokens = anthropic_response
        .usage
        .cache_creation_input_tokens
        .unwrap_or(0);

    let cost = calculate_anthropic_cost(
        request_body["model"].as_str().unwrap_or(""),
        anthropic_response.usage.input_tokens,
        anthropic_response.usage.output_tokens,
        cached_tokens,
        cache_creation_tokens,
    );

    let usage = TokenUsage {
        prompt_tokens: anthropic_response.usage.input_tokens,
        output_tokens: anthropic_response.usage.output_tokens,
        total_tokens: anthropic_response.usage.input_tokens
            + anthropic_response.usage.output_tokens,
        cached_tokens,
        cost,
        request_time_ms: Some(request_time_ms),
    };

    // Create response JSON that includes original tool_use blocks for conversation history
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Add tool_calls_content for conversation history preservation
    // Wrap in content field to match the expected format in octolib
    if !tool_use_blocks.is_empty() {
        response_json["tool_calls_content"] = serde_json::json!({
            "content": tool_use_blocks
        });
    }

    let exchange = if rate_limit_headers.is_empty() {
        ProviderExchange::new(request_body, response_json, Some(usage), "anthropic")
    } else {
        ProviderExchange::with_rate_limit_headers(
            request_body,
            response_json,
            Some(usage),
            "anthropic",
            rate_limit_headers,
        )
    };

    Ok(ProviderResponse {
        content,
        exchange,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        finish_reason: anthropic_response.stop_reason,
    })
}
