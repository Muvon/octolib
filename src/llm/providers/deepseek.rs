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

//! DeepSeek provider implementation
//!
//! PRICING UPDATE: April 2026 (revised 2026-04-24 per change log)
//! Source: <https://api-docs.deepseek.com/quick_start/pricing>
//!
//! deepseek-v4-flash (1M context, thinking by default):
//! - Cache Hit: $0.0028
//! - Cache Miss (Input): $0.14
//! - Output: $0.28
//!
//! deepseek-v4-pro (1M context, thinking by default):
//! - Cache Hit: $0.003625
//! - Cache Miss (Input): $0.435
//! - Output: $0.87
//!
//! Legacy aliases (deprecated, routed to v4-flash non-thinking/thinking modes
//! since 2026-04-24, billed at v4-flash rates; scheduled for removal
//! 2026-07-24 15:59 UTC per <https://api-docs.deepseek.com/updates>):
//! deepseek-chat (non-thinking), deepseek-reasoner (thinking)

use crate::errors::ProviderError;
use crate::llm::providers::shared;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, SamplingSupport, TokenUsage,
};
use crate::llm::utils::{is_model_in_pricing_table, PricingTuple};
use anyhow::Result;

use serde::{Deserialize, Serialize};
use std::env;

// Model pricing (per 1M tokens in USD) - Updated Apr 2026
// Source: https://api-docs.deepseek.com/quick_start/pricing
/// Format: (model, input, output, cache_write, cache_read)
/// Note: DeepSeek uses cache_hit/cache_miss model - cache_write = cache_miss (input), cache_read = cache_hit
const PRICING: &[PricingTuple] = &[
    // V4 family (1M context)
    ("deepseek-v4-pro", 0.435, 0.87, 0.435, 0.003625),
    ("deepseek-v4-flash", 0.14, 0.28, 0.14, 0.0028),
    // Legacy aliases (deprecated, routed to v4-flash rates since 2026-04-24)
    ("deepseek-chat", 0.14, 0.28, 0.14, 0.0028),
    ("deepseek-reasoner", 0.14, 0.28, 0.14, 0.0028),
];

/// Get pricing tuple for a specific model (case-insensitive)
/// Returns None if the model is not in the pricing table (not supported)
fn get_pricing_tuple(model: &str) -> Option<(f64, f64, f64, f64)> {
    crate::llm::utils::get_model_pricing(model, PRICING)
}

/// Calculate cost for DeepSeek models with cache-aware pricing (Jan 2026)
fn calculate_cost_with_cache(
    model: &str,
    regular_input_tokens: u64,
    cache_hit_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    let (input_price, output_price, _cache_write_price, cache_read_price) =
        get_pricing_tuple(model)?;

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input_price;
    let cache_hit_cost = (cache_hit_tokens as f64 / 1_000_000.0) * cache_read_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

    Some(regular_input_cost + cache_hit_cost + output_cost)
}

/// Calculate cost for DeepSeek models without cache
fn calculate_cost(model: &str, input_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_with_cache(model, input_tokens, 0, completion_tokens)
}

/// DeepSeek provider
#[derive(Debug, Clone, Default)]
pub struct DeepSeekProvider;

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self
    }
}

const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

// DeepSeek API request/response structures
#[derive(Serialize, Debug, Clone)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<DeepSeekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<DeepSeekTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeepSeekMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<DeepSeekToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekResponse {
    id: String,
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    model: Option<String>,
    choices: Vec<DeepSeekChoice>,
    usage: Option<DeepSeekUsage>,
    #[serde(default)]
    system_fingerprint: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekChoice {
    #[serde(default)]
    index: u32,
    message: DeepSeekMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
    #[serde(default)]
    prompt_cache_miss_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<DeepSeekPromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<DeepSeekCompletionTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct DeepSeekPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct DeepSeekCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeepSeekToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: DeepSeekFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeepSeekFunction {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug, Clone)]
struct DeepSeekTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: DeepSeekToolFunction,
}

#[derive(Serialize, Debug, Clone)]
struct DeepSeekToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Convert generic Messages into DeepSeek's wire format.
///
/// DeepSeek thinking-mode rule (per /guides/thinking_mode): when an assistant turn
/// produced tool_calls, its reasoning_content MUST be replayed in subsequent
/// requests — otherwise the API returns 400. For assistant turns without
/// tool_calls (and for all other roles), reasoning_content is ignored and is
/// omitted here.
fn convert_messages(messages: &[crate::llm::types::Message]) -> Vec<DeepSeekMessage> {
    messages
        .iter()
        .map(|msg| {
            let tool_calls = msg.tool_calls.as_ref().and_then(|tc| {
                shared::parse_generic_tool_calls_strict(tc, "deepseek")
                    .ok()
                    .map(|calls| {
                        calls
                            .into_iter()
                            .map(|call| DeepSeekToolCall {
                                id: call.id,
                                tool_type: "function".to_string(),
                                function: DeepSeekFunction {
                                    name: call.name,
                                    arguments: shared::arguments_to_json_string(&call.arguments),
                                },
                            })
                            .collect::<Vec<_>>()
                    })
            });

            let reasoning_content = if msg.role == "assistant" && tool_calls.is_some() {
                // Only replay actual thinking content — omit field entirely if no thinking was present.
                // DeepSeek requires reasoning_content when replaying tool-call turns that had thinking,
                // but unlike Moonshot it does NOT require an empty string when there was none.
                msg.thinking.as_ref().map(|t| t.content.clone())
            } else {
                None
            };

            DeepSeekMessage {
                role: msg.role.clone(),
                content: if msg.content.is_empty() && tool_calls.is_some() {
                    None
                } else {
                    Some(msg.content.clone())
                },
                reasoning_content,
                tool_calls,
                tool_call_id: msg.tool_call_id.clone(),
                name: msg.name.clone(),
            }
        })
        .collect()
}

#[async_trait::async_trait]
impl AiProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn supports_model(&self, model: &str) -> bool {
        // DeepSeek models - check against pricing table (strict)
        is_model_in_pricing_table(model, PRICING)
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(DEEPSEEK_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "DeepSeek API key not found in environment variable: {}",
                DEEPSEEK_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true // DeepSeek supports caching
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false // DeepSeek doesn't support vision yet
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // DeepSeek supports JSON mode as per their API documentation
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        // DeepSeek supports only `json_object` mode — it returns valid JSON but
        // ignores the supplied JSON schema, so the response shape is NOT
        // guaranteed (the `JsonSchema` request arm above downgrades to
        // `json_object`). Report false so callers route to a tolerant parser.
        false
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            crate::llm::utils::get_model_pricing(model, PRICING)?;

        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        let model_lower = crate::llm::utils::normalize_model_name(model);
        if model_lower.contains("v4") {
            1_000_000 // DeepSeek V4: 1M context
        } else {
            64_000 // Legacy models
        }
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        let model_lower = crate::llm::utils::normalize_model_name(model);
        // DeepSeek API only supports temperature (no top_p, no top_k).
        // The reasoner model silently ignores temperature, so omit it.
        SamplingSupport {
            temperature: !model_lower.contains("reasoner"),
            top_p: false,
            top_k: false,
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        let messages = convert_messages(&params.messages);

        let mut request = DeepSeekRequest {
            model: params.model.clone(),
            messages,
            temperature: self.effective_sampling_params(&params).temperature,
            max_tokens: Some(params.max_tokens),
            stream: Some(false), // We don't support streaming in octolib yet
            response_format: None,
            tools: None,
            tool_choice: None,
        };

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    // DeepSeek supports JSON mode but not full JSON schema validation
                    // Fall back to json_object mode
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
            }
        }

        // Add tools if specified
        if let Some(tools) = &params.tools {
            request.tools = Some(
                tools
                    .iter()
                    .map(|tool| DeepSeekTool {
                        tool_type: "function".to_string(),
                        function: DeepSeekToolFunction {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters: tool.parameters.clone(),
                        },
                    })
                    .collect(),
            );
        }

        let start_time = std::time::Instant::now();
        let request_timeout = params.request_timeout;
        let response = retry::retry_with_exponential_backoff(
            || {
                let client = shared::http_client();
                let api_key = api_key.clone();
                let request = request.clone();
                Box::pin(async move {
                    let req = client
                        .post("https://api.deepseek.com/chat/completions")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("Content-Type", "application/json")
                        .json(&request);

                    let captured = shared::send_and_read(req, request_timeout).await?;

                    // Return Err for retryable HTTP errors so the retry loop catches them
                    if retry::is_retryable_status(captured.status.as_u16()) {
                        return Err(anyhow::anyhow!(
                            "DeepSeek API error {}: {}",
                            captured.status,
                            captured.body
                        ));
                    }

                    Ok(captured)
                })
            },
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
            |e| {
                matches!(
                    e.downcast_ref::<ProviderError>(),
                    Some(ProviderError::Cancelled)
                )
            },
            |e: &anyhow::Error| shared::is_connection_error(e),
        )
        .await?;
        let request_time_ms = start_time.elapsed().as_millis() as u64;

        if !response.status.is_success() {
            return Err(anyhow::anyhow!(
                "DeepSeek API error {}: {}",
                response.status,
                response.body
            ));
        }

        // Parse response as JSON Value first — gives us both the raw value for exchange logging
        // and a source for typed deserialization without parsing the body twice.
        let response_json: serde_json::Value = serde_json::from_str(&response.body)?;

        let deepseek_response: DeepSeekResponse = serde_json::from_value(response_json.clone())
            .map_err(|e| {
                anyhow::anyhow!(
                    "DeepSeek API response deserialization error: {} — response: {}",
                    e,
                    response_json
                        .to_string()
                        .chars()
                        .take(500)
                        .collect::<String>()
                )
            })?;

        let response_for_exchange = response_json;

        let choice = deepseek_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in DeepSeek response"))?;

        let content = choice.message.content.unwrap_or_default();

        // Extract tool calls from response
        let tool_calls: Option<Vec<crate::llm::types::ToolCall>> =
            choice.message.tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .filter_map(|call| {
                        if call.tool_type != "function" {
                            tracing::warn!("Unexpected tool type: {}", call.tool_type);
                            return None;
                        }

                        let arguments =
                            shared::parse_tool_call_arguments_lossy(&call.function.arguments);

                        Some(crate::llm::types::ToolCall {
                            id: call.id,
                            name: call.function.name,
                            arguments,
                        })
                    })
                    .collect()
            });

        // Create exchange record for logging
        let mut response_json = response_for_exchange;
        if let Some(ref tc) = tool_calls {
            shared::set_response_tool_calls(&mut response_json, tc, None);
        }

        let exchange = ProviderExchange::new(
            serde_json::to_value(&request)?,
            response_json,
            None, // Will be set below
            self.name(),
        );

        // Calculate cost with the provider pricing table
        let token_usage = if let Some(usage) = deepseek_response.usage {
            let prompt_tokens = usage.prompt_tokens;
            let completion_tokens = usage.completion_tokens;
            let total_tokens = usage.total_tokens;

            // DeepSeek reports:
            // - prompt_cache_hit_tokens: tokens read from cache (cache READ)
            // - prompt_cache_miss_tokens: fresh input tokens (includes regular + cache WRITE)
            // DeepSeek doesn't separate regular input from cache write in their API
            let cache_read_tokens = usage.prompt_cache_hit_tokens;
            let cache_miss_tokens = usage.prompt_cache_miss_tokens;

            // Also check prompt_tokens_details.cached_tokens as alternative
            let cache_read_tokens = if cache_read_tokens == 0 {
                usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0)
            } else {
                cache_read_tokens
            };

            // For CLEAN input_tokens, we use cache_miss_tokens
            // (DeepSeek charges these at the "cache miss" rate which includes write cost)
            let input_tokens_clean = cache_miss_tokens;

            // DeepSeek doesn't expose cache_write separately - it's included in cache_miss
            let cache_write_tokens = 0_u64;

            // Calculate cost with pricing table values (Jan 2026)
            let cost = if cache_read_tokens > 0 {
                calculate_cost_with_cache(
                    &params.model,
                    input_tokens_clean,
                    cache_read_tokens,
                    completion_tokens,
                )
            } else {
                calculate_cost(&params.model, prompt_tokens, completion_tokens)
            };

            let reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens)
                .unwrap_or(0);

            Some(TokenUsage {
                input_tokens: input_tokens_clean, // CLEAN input (cache miss tokens)
                cache_read_tokens,                // Tokens read from cache
                cache_write_tokens,               // DeepSeek doesn't expose this (0)
                output_tokens: completion_tokens,
                reasoning_tokens,
                total_tokens,
                cost,
                request_time_ms: Some(request_time_ms),
            })
        } else {
            None
        };

        // Update exchange with token usage
        let mut final_exchange = exchange;
        final_exchange.usage = token_usage.clone();

        // Extract thinking block from reasoning_content if present
        let thinking = choice
            .message
            .reasoning_content
            .as_ref()
            .and_then(|reasoning| {
                if reasoning.trim().is_empty() {
                    None
                } else {
                    // Estimate tokens from content length (4 chars per token)
                    let tokens = (reasoning.len() / 4) as u64;
                    Some(crate::llm::types::ThinkingBlock {
                        content: reasoning.clone(),
                        tokens,
                    })
                }
            });

        // Try to parse structured output if it was requested
        let structured_output = shared::parse_structured_output_from_text(&content);

        Ok(ProviderResponse {
            content,
            thinking,
            exchange: final_exchange,
            tool_calls,
            finish_reason: choice.finish_reason,
            structured_output,
            id: Some(deepseek_response.id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = DeepSeekProvider::new();
        assert!(provider.supports_model("deepseek-v4-flash"));
        assert!(provider.supports_model("deepseek-v4-pro"));
        assert!(provider.supports_model("deepseek-chat"));
        assert!(provider.supports_model("deepseek-reasoner"));
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("deepseek-coder")); // Not in current API
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = DeepSeekProvider::new();
        assert!(provider.supports_model("DEEPSEEK-V4-FLASH"));
        assert!(provider.supports_model("DEEPSEEK-V4-PRO"));
        assert!(provider.supports_model("DEEPSEEK-CHAT"));
        assert!(provider.supports_model("DEEPSEEK-REASONER"));
        assert!(provider.supports_model("DeepSeek-V4-Flash"));
    }

    #[test]
    fn test_max_input_tokens() {
        let provider = DeepSeekProvider::new();
        assert_eq!(
            provider.get_max_input_tokens("deepseek-v4-flash"),
            1_000_000
        );
        assert_eq!(provider.get_max_input_tokens("deepseek-v4-pro"), 1_000_000);
        assert_eq!(provider.get_max_input_tokens("deepseek-chat"), 64_000);
        assert_eq!(provider.get_max_input_tokens("deepseek-reasoner"), 64_000);
    }

    #[test]
    fn test_calculate_cost() {
        // Test basic cost calculation with v4-flash-routed pricing (since 2026-04-24)
        // deepseek-chat: Input: $0.14/1M, Output: $0.28/1M
        let cost = calculate_cost("deepseek-chat", 1_000_000, 500_000);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();

        // Expected: (1M * $0.14) + (0.5M * $0.28) = $0.14 + $0.14 = $0.28
        let expected = 0.14 + (0.5 * 0.28);
        assert!((cost_value - expected).abs() < 0.01);

        // Both models now share the same v4-flash pricing
        let cost2 = calculate_cost("deepseek-reasoner", 1_000_000, 500_000);
        assert!(cost2.is_some());
        assert!((cost2.unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cost_with_cache() {
        // v4-flash-routed pricing: Cache hit: $0.0028/1M, Cache miss: $0.14/1M, Output: $0.28/1M
        let cost = calculate_cost_with_cache("deepseek-chat", 500_000, 500_000, 250_000);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();

        // Expected: (0.5M * $0.14) + (0.5M * $0.0028) + (0.25M * $0.28)
        //         = $0.07 + $0.0014 + $0.07 = $0.1414
        let expected = (0.5 * 0.14) + (0.5 * 0.0028) + (0.25 * 0.28);
        assert!((cost_value - expected).abs() < 0.01);

        // Cost with cache should be less than without cache for same total input
        let cost_no_cache = calculate_cost("deepseek-chat", 1_000_000, 250_000);
        assert!(cost_no_cache.is_some());
        assert!(cost_value < cost_no_cache.unwrap());
    }

    #[test]
    fn test_thinking_block_extraction() {
        // Test with reasoning_content present
        let message_with_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: Some("The answer is 9.11".to_string()),
            reasoning_content: Some("Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        // Verify reasoning_content is properly stored
        assert!(message_with_thinking.reasoning_content.is_some());
        let reasoning = message_with_thinking.reasoning_content.as_ref().unwrap();
        assert_eq!(reasoning, "Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.");

        // Test token estimation (length / 4)
        let estimated_tokens = (reasoning.len() / 4) as u64;
        assert!(estimated_tokens > 0);

        // Test without reasoning_content
        let message_without_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: Some("Hello".to_string()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        assert!(message_without_thinking.reasoning_content.is_none());

        // Test with empty reasoning_content
        let message_empty_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: Some("Hello".to_string()),
            reasoning_content: Some("".to_string()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        assert!(message_empty_thinking.reasoning_content.is_some());
        assert!(message_empty_thinking
            .reasoning_content
            .as_ref()
            .unwrap()
            .is_empty());

        // Test with null content (tool call response)
        let message_tool_call = DeepSeekMessage {
            role: "assistant".to_string(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![DeepSeekToolCall {
                id: "call_123".to_string(),
                tool_type: "function".to_string(),
                function: DeepSeekFunction {
                    name: "get_weather".to_string(),
                    arguments: "{}".to_string(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };

        assert!(message_tool_call.content.is_none());
        assert!(message_tool_call.tool_calls.is_some());
    }

    #[test]
    fn test_convert_messages_reasoning_content_replay() {
        use crate::llm::tool_calls::GenericToolCall;
        use crate::llm::types::{Message, ThinkingBlock};

        let tool_calls_json = serde_json::to_value(vec![GenericToolCall {
            id: "call_123".to_string(),
            name: "list_files".to_string(),
            arguments: serde_json::json!({"path": "."}),
            meta: None,
        }])
        .unwrap();

        // Assistant turn with tool_calls + thinking → reasoning_content must be replayed.
        let assistant_with_tools = Message {
            role: "assistant".to_string(),
            content: String::new(),
            timestamp: 0,
            cached: false,
            cache_ttl: None,
            tool_call_id: None,
            name: None,
            tool_calls: Some(tool_calls_json.clone()),
            images: None,
            videos: None,
            thinking: Some(ThinkingBlock {
                content: "I should list the files first.".to_string(),
                tokens: 8,
            }),
            id: None,
        };
        let converted = convert_messages(std::slice::from_ref(&assistant_with_tools));
        assert_eq!(converted.len(), 1);
        assert_eq!(
            converted[0].reasoning_content.as_deref(),
            Some("I should list the files first.")
        );
        assert!(converted[0].tool_calls.is_some());
        assert!(converted[0].content.is_none());

        // Assistant turn with tool_calls but no stored thinking → field omitted entirely (None).
        // DeepSeek does not require reasoning_content when there was no thinking; unlike
        // Moonshot it does NOT require an empty string sentinel.
        let assistant_tools_no_thinking = Message {
            thinking: None,
            ..assistant_with_tools.clone()
        };
        let converted = convert_messages(std::slice::from_ref(&assistant_tools_no_thinking));
        assert!(converted[0].reasoning_content.is_none());

        // Assistant turn without tool_calls → reasoning_content omitted (DeepSeek
        // ignores it on non-tool turns; sending it is harmless but unnecessary).
        let assistant_plain = Message::assistant("Hello").with_thinking(ThinkingBlock {
            content: "trivial".to_string(),
            tokens: 1,
        });
        let converted = convert_messages(std::slice::from_ref(&assistant_plain));
        assert!(converted[0].reasoning_content.is_none());

        // User / tool / system messages → never carry reasoning_content.
        let user_msg = Message::user("hi");
        let tool_msg = Message::tool("ok", "call_123", "list_files");
        let system_msg = Message::system("be helpful");
        for msg in [user_msg, tool_msg, system_msg] {
            let converted = convert_messages(std::slice::from_ref(&msg));
            assert!(converted[0].reasoning_content.is_none());
        }

        // Verify JSON serialization: None is omitted, Some("") is preserved.
        let json =
            serde_json::to_value(&convert_messages(std::slice::from_ref(&assistant_with_tools))[0])
                .unwrap();
        assert_eq!(
            json.get("reasoning_content").and_then(|v| v.as_str()),
            Some("I should list the files first.")
        );

        let json_plain =
            serde_json::to_value(&convert_messages(std::slice::from_ref(&assistant_plain))[0])
                .unwrap();
        assert!(json_plain.get("reasoning_content").is_none());
    }
}
