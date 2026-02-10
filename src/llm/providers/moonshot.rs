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

//! Moonshot AI (Kimi) provider implementation
//!
//! PRICING UPDATE: February 2026
//! Model-specific pricing (per 1M tokens in USD):
//!
//! kimi-k2 series:
//! - Cache Hit: $0.15
//! - Cache Miss (Input): $0.60
//! - Output: $2.50
//!
//! kimi-k2.5:
//! - Cache Hit: $0.30
//! - Cache Miss (Input): $0.60
//! - Output: $2.50

use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, TokenUsage, ToolCall,
};
use crate::llm::utils::contains_ignore_ascii_case;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

// Model pricing (per 1M tokens in USD) - Updated Feb 2026
// Source: https://platform.moonshot.ai/docs/pricing/chat
const PRICING: &[(&str, f64, f64, f64)] = &[
    // Model, Cache Hit price, Cache Miss (Input) price, Output price per 1M tokens
    ("kimi-k2", 0.15, 0.60, 2.50),
    ("kimi-k2-thinking", 0.15, 0.60, 2.50),
    ("kimi-k2-thinking-turbo", 0.15, 0.60, 2.50),
    ("kimi-k2.5", 0.30, 0.60, 2.50),
    ("kimi-k2-0905", 0.15, 0.60, 2.50),
    ("kimi-k2-0915", 0.15, 0.60, 2.50),
];

/// Get pricing for a specific model (case-insensitive)
fn get_model_pricing(model: &str) -> (f64, f64, f64) {
    for (pricing_model, cache_hit, cache_miss, output) in PRICING {
        if contains_ignore_ascii_case(model, pricing_model) {
            return (*cache_hit, *cache_miss, *output);
        }
    }
    // Default to kimi-k2 pricing if model not found
    (0.15, 0.60, 2.50)
}

/// Calculate cost for Moonshot models with cache-aware pricing
fn calculate_cost_with_cache(
    model: &str,
    regular_input_tokens: u64,
    cache_hit_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    let (cache_hit_price, cache_miss_price, output_price) = get_model_pricing(model);

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * cache_miss_price;
    let cache_hit_cost = (cache_hit_tokens as f64 / 1_000_000.0) * cache_hit_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

    Some(regular_input_cost + cache_hit_cost + output_cost)
}

/// Calculate cost for Moonshot models without cache
fn calculate_cost(model: &str, prompt_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_with_cache(model, prompt_tokens, 0, completion_tokens)
}

/// Moonshot AI (Kimi) provider
#[derive(Debug, Clone)]
pub struct MoonshotProvider {
    client: Client,
}

impl Default for MoonshotProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MoonshotProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

const MOONSHOT_API_KEY_ENV: &str = "MOONSHOT_API_KEY";

// Moonshot API request/response structures (OpenAI-compatible)
#[derive(Serialize, Debug)]
struct MoonshotRequest {
    model: String,
    messages: Vec<MoonshotMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MoonshotTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MoonshotMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MoonshotToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    /// Required for kimi-k2.5 and thinking models when tool_calls are present
    /// - Some("") = empty reasoning (required for tool calls)
    /// - Some(content) = actual reasoning content
    /// - None = omit field (backward compatible for non-thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct MoonshotResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<MoonshotChoice>,
    usage: Option<MoonshotUsage>,
}

#[derive(Serialize, Deserialize, Debug)]
struct MoonshotChoice {
    index: u32,
    message: MoonshotMessage,
    finish_reason: Option<String>,
    logprobs: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug)]
struct MoonshotUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<MoonshotPromptTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct MoonshotPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MoonshotToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: MoonshotFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct MoonshotFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone)]
struct MoonshotTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: MoonshotToolFunction,
}

#[derive(Serialize, Debug, Clone)]
struct MoonshotToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

// Convert messages to Moonshot (OpenAI-compatible) format
fn convert_messages(messages: &[crate::llm::types::Message], _model: &str) -> Vec<MoonshotMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                result.push(MoonshotMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: message.tool_call_id.clone(),
                    name: message.name.clone(),
                    // Tool response messages don't need reasoning_content
                    reasoning_content: None,
                });
            }
            "assistant" if message.tool_calls.is_some() => {
                let mut content_parts = Vec::new();

                if !message.content.trim().is_empty() {
                    content_parts.push(serde_json::json!({
                        "type": "text",
                        "text": message.content
                    }));
                }

                if let Some(images) = &message.images {
                    for image in images {
                        if let crate::llm::types::ImageData::Base64(data) = &image.data {
                            content_parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", image.media_type, data)
                                }
                            }));
                        }
                    }
                }

                let content = if content_parts.is_empty() {
                    None
                } else {
                    let only_text = content_parts.len() == 1
                        && content_parts[0].get("type").and_then(|t| t.as_str()) == Some("text");

                    if only_text {
                        Some(content_parts[0]["text"].clone())
                    } else {
                        Some(serde_json::json!(content_parts))
                    }
                };

                let tool_calls = if let Ok(generic_calls) =
                    serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(
                        message.tool_calls.clone().unwrap(),
                    ) {
                    Some(
                        generic_calls
                            .into_iter()
                            .map(|tc| MoonshotToolCall {
                                id: tc.id,
                                tool_type: "function".to_string(),
                                function: MoonshotFunction {
                                    name: tc.name,
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect(),
                    )
                } else {
                    // If parsing as GenericToolCall fails, try parsing as MoonshotToolCall directly
                    // This handles cases where tool_calls are stored in provider-specific format
                    serde_json::from_value::<Vec<MoonshotToolCall>>(
                        message.tool_calls.clone().unwrap(),
                    )
                    .ok()
                };


                // Extract reasoning_content from thinking block if present.
                // Moonshot requires reasoning_content for assistant messages with tool_calls.
                // Always include the field (even if empty) for tool calls.
                let reasoning_content = Some(
                    message
                        .thinking
                        .as_ref()
                        .map(|t| t.content.clone())
                        .unwrap_or_else(String::new),
                );

                result.push(MoonshotMessage {
                    role: message.role.clone(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                    name: None,
                    reasoning_content,
                });
            }
            "user" | "assistant" | "system" => {
                let mut content_parts = vec![serde_json::json!({
                    "type": "text",
                    "text": message.content
                })];

                if let Some(images) = &message.images {
                    for image in images {
                        if let crate::llm::types::ImageData::Base64(data) = &image.data {
                            content_parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", image.media_type, data)
                                }
                            }));
                        }
                    }
                }

                let content = if content_parts.len() == 1 {
                    Some(content_parts[0]["text"].clone())
                } else {
                    Some(serde_json::json!(content_parts))
                };

                result.push(MoonshotMessage {
                    role: message.role.clone(),
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    // Regular messages without tool calls don't need reasoning_content
                    reasoning_content: None,
                });
            }
            _ => {
                tracing::warn!("Unknown message role: {}", message.role);
            }
        }
    }

    result
}

fn extract_text_content(content: &Option<serde_json::Value>) -> String {
    match content {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

#[async_trait::async_trait]
impl AiProvider for MoonshotProvider {
    fn name(&self) -> &str {
        "moonshot"
    }

    fn supports_model(&self, model: &str) -> bool {
        contains_ignore_ascii_case(model, "kimi-k2")
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(MOONSHOT_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "Moonshot AI API key not found in environment variable: {}",
                MOONSHOT_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Kimi K2.5 supports vision/multimodal
        contains_ignore_ascii_case(model, "kimi-k2.5")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Moonshot has cache-aware pricing: (cache_hit, cache_miss, output)
        let (cache_hit, cache_miss, output) = get_model_pricing(model);
        Some(crate::llm::types::ModelPricing::new(
            cache_miss, // Regular input (cache miss)
            output,     // Output price
            cache_miss, // Cache write = same as cache miss
            cache_hit,  // Cache read (cache hit)
        ))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Kimi K2 series supports up to 256K context
        if contains_ignore_ascii_case(model, "kimi-k2") {
            256_000
        } else {
            128_000
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to Moonshot format
        let messages = convert_messages(&params.messages, &params.model);

        // Kimi K2.5 only accepts temperature=1.0. Other models use standard temperature.
        let temperature = if contains_ignore_ascii_case(&params.model, "kimi-k2.5") {
            1.0
        } else {
            params.temperature
        };

        let mut request = MoonshotRequest {
            model: params.model.clone(),
            messages,
            temperature: Some(temperature),
            max_tokens: if params.max_tokens > 0 {
                Some(params.max_tokens)
            } else {
                None
            },
            stream: Some(false),
            response_format: None,
            tools: None,
            tool_choice: None,
        };

        // Add tools if available (Moonshot is OpenAI-compatible)
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let mut sorted_tools = tools.clone();
                sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

                let moonshot_tools = sorted_tools
                    .iter()
                    .map(|f| MoonshotTool {
                        tool_type: "function".to_string(),
                        function: MoonshotToolFunction {
                            name: f.name.clone(),
                            description: f.description.clone(),
                            parameters: f.parameters.clone(),
                        },
                    })
                    .collect::<Vec<_>>();

                request.tools = Some(moonshot_tools);
                request.tool_choice = Some(serde_json::json!("auto"));
            }
        }

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    // Moonshot supports JSON mode; fall back to json_object for schema requests
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
            }
        }

        let response = retry::cancellable(
            async {
                self.client
                    .post("https://api.moonshot.ai/v1/chat/completions")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .json(&request)
                    .send()
                    .await
                    .map_err(anyhow::Error::from)
            },
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
        )
        .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = retry::cancellable(
                async { response.text().await.map_err(anyhow::Error::from) },
                params.cancellation_token.as_ref(),
                || ProviderError::Cancelled.into(),
            )
            .await?;
            return Err(anyhow::anyhow!(
                "Moonshot AI API error {}: {}",
                status,
                error_text
            ));
        }

        let moonshot_response: MoonshotResponse = retry::cancellable(
            async { response.json().await.map_err(anyhow::Error::from) },
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
        )
        .await?;

        let mut response_for_exchange = serde_json::to_value(&moonshot_response)?;

        let choice = moonshot_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in Moonshot response"))?;

        let token_usage = if let Some(usage) = moonshot_response.usage {
            let prompt_tokens = usage.prompt_tokens;
            let completion_tokens = usage.completion_tokens;
            let total_tokens = usage.total_tokens;
            let cache_hit_tokens = usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens)
                .unwrap_or(0);

            let regular_input_tokens = prompt_tokens.saturating_sub(cache_hit_tokens);

            let cost = if cache_hit_tokens > 0 {
                calculate_cost_with_cache(
                    &params.model,
                    regular_input_tokens,
                    cache_hit_tokens,
                    completion_tokens,
                )
            } else {
                calculate_cost(&params.model, prompt_tokens, completion_tokens)
            };

            Some(TokenUsage {
                prompt_tokens,
                output_tokens: completion_tokens,
                reasoning_tokens: 0,
                total_tokens,
                cached_tokens: cache_hit_tokens,
                cost,
                request_time_ms: None,
            })
        } else {
            None
        };

        let content = extract_text_content(&choice.message.content);

        // Extract reasoning_content from response and convert to ThinkingBlock
        // CRITICAL: Preserve even empty reasoning_content for thinking models
        // Empty reasoning_content is required when replaying tool call messages
        let thinking = choice.message.reasoning_content.map(|rc| {
            crate::llm::types::ThinkingBlock {
                content: rc,
                tokens: 0, // Moonshot doesn't provide separate reasoning token count
            }
        });

        let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .filter_map(|call| {
                    if call.tool_type != "function" {
                        eprintln!(
                            "Warning: Unexpected tool type '{}' from Moonshot API",
                            call.tool_type
                        );
                        return None;
                    }

                    let arguments: serde_json::Value =
                        serde_json::from_str(&call.function.arguments)
                            .unwrap_or(serde_json::json!({}));

                    Some(ToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments,
                    })
                })
                .collect()
        });

        if let Some(ref tc) = tool_calls {
            let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = tc
                .iter()
                .map(|call| crate::llm::tool_calls::GenericToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    meta: None,
                })
                .collect();

            response_for_exchange["tool_calls"] =
                serde_json::to_value(&generic_calls).unwrap_or_default();
        }

        let exchange = ProviderExchange {
            request: serde_json::to_value(&request)?,
            response: response_for_exchange,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            usage: token_usage.clone(),
            provider: self.name().to_string(),
            rate_limit_headers: None,
        };

        // Try to parse structured output if requested
        let structured_output =
            if content.trim().starts_with('{') || content.trim().starts_with('[') {
                serde_json::from_str(&content).ok()
            } else {
                None
            };

        Ok(ProviderResponse {
            content,
            thinking,
            exchange,
            tool_calls,
            finish_reason: choice.finish_reason,
            structured_output,
            id: Some(moonshot_response.id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = MoonshotProvider::new();
        assert!(provider.supports_model("kimi-k2"));
        assert!(provider.supports_model("kimi-k2-thinking"));
        assert!(provider.supports_model("kimi-k2.5"));
        assert!(provider.supports_model("KIMI-K2"));
        assert!(!provider.supports_model("gpt-4"));
    }

    #[test]
    fn test_supports_vision() {
        let provider = MoonshotProvider::new();
        assert!(provider.supports_vision("kimi-k2.5"));
        assert!(!provider.supports_vision("kimi-k2"));
    }

    #[test]
    fn test_calculate_cost() {
        // kimi-k2: Input: $0.60/1M, Output: $2.50/1M
        let cost = calculate_cost("kimi-k2", 1_000_000, 500_000);
        assert!(cost.is_some());
        let expected = 0.60 + (0.5 * 2.50);
        assert!((cost.unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cost_with_cache() {
        // kimi-k2: Cache hit: $0.15/1M, Cache miss: $0.60/1M, Output: $2.50/1M
        let cost = calculate_cost_with_cache("kimi-k2", 500_000, 500_000, 250_000);
        assert!(cost.is_some());
        let expected = (0.5 * 0.60) + (0.5 * 0.15) + (0.25 * 2.50);
        assert!((cost.unwrap() - expected).abs() < 0.01);

        // Cost with cache should be less
        let cost_no_cache = calculate_cost("kimi-k2", 1_000_000, 250_000);
        assert!(cost.unwrap() < cost_no_cache.unwrap());
    }

    #[test]
    fn test_get_max_input_tokens() {
        let provider = MoonshotProvider::new();
        assert_eq!(provider.get_max_input_tokens("kimi-k2"), 256_000);
        assert_eq!(provider.get_max_input_tokens("kimi-k2.5"), 256_000);
        assert_eq!(provider.get_max_input_tokens("unknown"), 128_000);
    }

    #[test]
    fn test_reasoning_content_serialization() {
        use crate::llm::types::ThinkingBlock;

        // Test 1: Thinking model - Assistant message with tool calls and thinking
        let msg_with_thinking = crate::llm::types::Message {
            role: "assistant".to_string(),
            content: "I'll help you with that.".to_string(),
            timestamp: 0,
            cached: false,
            tool_call_id: None,
            name: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"city": "Beijing"}
            }])),
            images: None,
            thinking: Some(ThinkingBlock {
                content: "Let me check the weather".to_string(),
                tokens: 0,
            }),
            id: None,
        };

        // For thinking models, reasoning_content should be present
        let converted =
            convert_messages(std::slice::from_ref(&msg_with_thinking), "kimi-k2-thinking");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_some());
        assert_eq!(
            converted[0].reasoning_content.as_ref().unwrap(),
            "Let me check the weather"
        );

        // For Kimi K2 models, reasoning_content should be present
        let converted = convert_messages(std::slice::from_ref(&msg_with_thinking), "kimi-k2");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_some());

        // Test 2: Thinking model - Assistant message with tool calls but no thinking (empty reasoning_content)
        let msg_no_thinking = crate::llm::types::Message {
            role: "assistant".to_string(),
            content: "I'll help you with that.".to_string(),
            timestamp: 0,
            cached: false,
            tool_call_id: None,
            name: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_456",
                "name": "search",
                "arguments": {"query": "test"}
            }])),
            images: None,
            thinking: None,
            id: None,
        };

        // For thinking models, should have Some("") for tool calls even without thinking
        let converted = convert_messages(std::slice::from_ref(&msg_no_thinking), "kimi-k2.5");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_some());
        assert_eq!(converted[0].reasoning_content.as_ref().unwrap(), "");

        // For Kimi K2 models, should be Some("") even without thinking
        let converted = convert_messages(std::slice::from_ref(&msg_no_thinking), "kimi-k2");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_some());
        assert_eq!(converted[0].reasoning_content.as_ref().unwrap(), "");

        // Test 3: Regular assistant message without tool calls (no reasoning_content)
        let regular_msg = crate::llm::types::Message {
            role: "assistant".to_string(),
            content: "Hello, how can I help?".to_string(),
            timestamp: 0,
            cached: false,
            tool_call_id: None,
            name: None,
            tool_calls: None,
            images: None,
            thinking: None,
            id: None,
        };

        // Regular assistant messages without tool calls: no reasoning_content
        let converted = convert_messages(std::slice::from_ref(&regular_msg), "kimi-k2-thinking");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_none());

        let converted = convert_messages(std::slice::from_ref(&regular_msg), "kimi-k2");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_none());

        // Test 4: Tool response message (no reasoning_content)
        let tool_msg = crate::llm::types::Message {
            role: "tool".to_string(),
            content: "Weather is sunny".to_string(),
            timestamp: 0,
            cached: false,
            tool_call_id: Some("call_123".to_string()),
            name: Some("get_weather".to_string()),
            tool_calls: None,
            images: None,
            thinking: None,
            id: None,
        };

        let converted = convert_messages(&[tool_msg], "kimi-k2-thinking");
        assert_eq!(converted.len(), 1);
        assert!(converted[0].reasoning_content.is_none());

        // Test 5: Verify JSON serialization behavior
        let msg_with_reasoning = MoonshotMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::json!("test")),
            tool_calls: Some(vec![]),
            tool_call_id: None,
            name: None,
            reasoning_content: Some("thinking".to_string()),
        };

        let json = serde_json::to_value(&msg_with_reasoning).unwrap();
        assert!(json.get("reasoning_content").is_some());
        assert_eq!(
            json.get("reasoning_content").unwrap().as_str().unwrap(),
            "thinking"
        );

        // Test 6: None reasoning_content should be omitted
        let msg_without_reasoning = MoonshotMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::json!("test")),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        };

        let json = serde_json::to_value(&msg_without_reasoning).unwrap();
        assert!(json.get("reasoning_content").is_none());

        // Test 7: Empty reasoning_content (Some("")) should be serialized
        let msg_with_empty_reasoning = MoonshotMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::json!("test")),
            tool_calls: Some(vec![]),
            tool_call_id: None,
            name: None,
            reasoning_content: Some("".to_string()),
        };

        let json = serde_json::to_value(&msg_with_empty_reasoning).unwrap();
        assert!(json.get("reasoning_content").is_some());
        assert_eq!(json.get("reasoning_content").unwrap().as_str().unwrap(), "");
    }
}
