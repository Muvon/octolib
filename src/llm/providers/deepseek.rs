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
//! PRICING VERIFIED: August 2026 — peak/off-peak billing (effective
//! 2026-08-16 16:00 UTC; peak windows 01:00-04:00 and 06:00-10:00 UTC,
//! off-peak rates are half of peak).
//! Source: <https://api-docs.deepseek.com/quick_start/pricing>
//! (`deepseek-v4-flash` points to the retrained 0731 snapshot since 2026-07-31)
//!
//! deepseek-v4-flash (1M context, thinking by default), peak/off-peak:
//! - Cache Hit: $0.014 / $0.007
//! - Cache Miss (Input): $0.44 / $0.22
//! - Output: $1.32 / $0.66
//!
//! deepseek-v4-pro (1M context, thinking by default), peak/off-peak:
//! - Cache Hit: $0.044 / $0.022
//! - Cache Miss (Input): $1.32 / $0.66
//! - Output: $3.96 / $1.98
//!
//! deepseek-v4-flash-vision-exp (experimental multimodal, released 2026-08-21)
//! matches v4-flash on text and bills at v4-flash rates; images are tokenized
//! at up to 384 tokens each
//! (<https://api-docs.deepseek.com/news/news260821/>).
//!
//! Legacy aliases deepseek-chat / deepseek-reasoner were removed by DeepSeek
//! on 2026-07-24 15:59 UTC per <https://api-docs.deepseek.com/updates>.
//!
//! Thinking is enabled by default (effort "high"); effort is controlled via
//! the top-level `reasoning_effort` field: "low" | "high" | "max"
//! (<https://api-docs.deepseek.com/guides/thinking_mode>).

use crate::errors::ProviderError;
use crate::llm::providers::shared;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, SamplingSupport, TokenUsage,
};
use crate::llm::utils::{contains_ignore_ascii_case, is_model_in_pricing_table, PricingTuple};
use anyhow::Result;

use serde::{Deserialize, Serialize};
use std::env;

// Model pricing (per 1M tokens in USD) - Verified Aug 2026
// Source: https://api-docs.deepseek.com/quick_start/pricing
/// Format: (model, input, output, cache_write, cache_read)
/// Note: DeepSeek uses cache_hit/cache_miss model - cache_write = cache_miss (input), cache_read = cache_hit
/// DeepSeek bills peak / off-peak: peak windows are 01:00-04:00 and 06:00-10:00
/// UTC Monday-Friday; off-peak is half of peak (effective 2026-08-16 16:00 UTC).
/// `deepseek-v4-flash-vision-exp` resolves to the `deepseek-v4-flash` row by
/// substring match and bills at those rates, as DeepSeek documents.
const PRICING_PEAK: &[PricingTuple] = &[
    // V4 family (1M context), peak-hour rates
    ("deepseek-v4-pro", 1.32, 3.96, 1.32, 0.044),
    ("deepseek-v4-flash", 0.44, 1.32, 0.44, 0.014),
];

const PRICING_OFF_PEAK: &[PricingTuple] = &[
    // V4 family (1M context), off-peak rates (half of peak)
    ("deepseek-v4-pro", 0.66, 1.98, 0.66, 0.022),
    ("deepseek-v4-flash", 0.22, 0.66, 0.22, 0.007),
];

/// Peak billing windows: Monday-Friday, 01:00-04:00 and 06:00-10:00 UTC.
fn is_peak_window(days_since_epoch: u64, utc_hour: u64) -> bool {
    // 1970-01-01 was Thursday. Map Monday..Sunday to 0..6.
    let weekday = (days_since_epoch + 3) % 7;
    weekday < 5 && ((1..4).contains(&utc_hour) || (6..10).contains(&utc_hour))
}

/// Pick the pricing table that applies at `time` (tier decided by the UTC hour)
fn pricing_table_at(time: std::time::SystemTime) -> &'static [PricingTuple] {
    let secs = time
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if is_peak_window(secs / 86_400, (secs % 86_400) / 3_600) {
        PRICING_PEAK
    } else {
        PRICING_OFF_PEAK
    }
}

/// Map generic ReasoningEffort to DeepSeek's `reasoning_effort` string.
/// DeepSeek supports only "low" / "high" / "max" (default "high" when the
/// field is omitted, thinking enabled by default), so intermediate levels
/// floor to the nearest supported lower effort.
fn map_reasoning_effort(
    effort: Option<crate::llm::types::ReasoningEffort>,
) -> Option<&'static str> {
    use crate::llm::types::ReasoningEffort;
    match effort {
        Some(ReasoningEffort::Low) | Some(ReasoningEffort::Medium) => Some("low"),
        Some(ReasoningEffort::High) | Some(ReasoningEffort::XHigh) => Some("high"),
        Some(ReasoningEffort::Max) => Some("max"),
        None => None,
    }
}

/// Calculate cost for DeepSeek models with cache-aware pricing, against the
/// pricing table active at request time (legacy / peak / off-peak)
fn calculate_cost_with_cache(
    pricing: &[PricingTuple],
    model: &str,
    regular_input_tokens: u64,
    cache_hit_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    let (input_price, output_price, _cache_write_price, cache_read_price) =
        crate::llm::utils::get_model_pricing(model, pricing)?;

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input_price;
    let cache_hit_cost = (cache_hit_tokens as f64 / 1_000_000.0) * cache_read_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

    Some(regular_input_cost + cache_hit_cost + output_cost)
}

/// Calculate cost for DeepSeek models without cache
#[cfg(test)]
fn calculate_cost(
    pricing: &[PricingTuple],
    model: &str,
    input_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    calculate_cost_with_cache(pricing, model, input_tokens, 0, completion_tokens)
}

/// Split a usage report into (cache-miss, cache-hit) prompt tokens.
///
/// DeepSeek reports the split in TWO shapes and does not always send both:
///   * native: `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`
///   * OpenAI-compatible: `prompt_tokens_details.cached_tokens`, which has NO
///     miss counterpart at all
///
/// Every one of those fields is `#[serde(default)]`, so reading the miss count
/// directly yields 0 whenever only the OpenAI-compatible shape arrives — which
/// billed the entire uncached prompt as FREE and reported `input_tokens` as 0.
/// `prompt_tokens` is always present and authoritative, so the miss count is
/// derived from it whenever the explicit field is absent. The subtraction
/// saturates: a provider inconsistency must never underflow u64 into a
/// catastrophic charge.
fn split_prompt_tokens(usage: &DeepSeekUsage) -> (u64, u64) {
    let hit = if usage.prompt_cache_hit_tokens > 0 {
        usage.prompt_cache_hit_tokens
    } else {
        usage
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            .unwrap_or(0)
    };
    let miss = if usage.prompt_cache_miss_tokens > 0 {
        usage.prompt_cache_miss_tokens
    } else {
        usage.prompt_tokens.saturating_sub(hit)
    };
    (miss, hit)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeepSeekMessage {
    role: String,
    /// Plain string for text turns, OpenAI-style content parts when the turn
    /// carries images (vision route only).
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
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

            let images = msg.images.as_deref().unwrap_or_default();
            let content = if !images.is_empty() {
                let mut parts = vec![serde_json::json!({"type": "text", "text": msg.content})];
                parts.extend(images.iter().map(|image| {
                    let url = match &image.data {
                        crate::llm::types::ImageData::Base64(data) => {
                            format!("data:{};base64,{}", image.media_type, data)
                        }
                        crate::llm::types::ImageData::Url(u) => u.clone(),
                    };
                    serde_json::json!({"type": "image_url", "image_url": {"url": url}})
                }));
                Some(serde_json::json!(parts))
            } else if msg.content.is_empty() && tool_calls.is_some() {
                None
            } else {
                Some(serde_json::json!(msg.content))
            };

            DeepSeekMessage {
                role: msg.role.clone(),
                content,
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
        // DeepSeek models - check against the active pricing table (strict);
        // the model set is identical across peak/off-peak tables
        is_model_in_pricing_table(model, pricing_table_at(std::time::SystemTime::now()))
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

    fn supports_vision(&self, model: &str) -> bool {
        // Only the experimental V4-Flash-Vision route accepts image input
        contains_ignore_ascii_case(model, "vision")
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
            crate::llm::utils::get_model_pricing(
                model,
                pricing_table_at(std::time::SystemTime::now()),
            )?;

        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        1_000_000 // All supported models are V4: 1M context
    }

    fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
        // DeepSeek API only supports temperature (no top_p, no top_k).
        SamplingSupport {
            temperature: true,
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
            reasoning_effort: map_reasoning_effort(params.reasoning_effort),
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
        let extra_headers = params.extra_headers.clone();
        let response = retry::retry_with_exponential_backoff(
            || {
                let client = shared::http_client();
                let api_key = api_key.clone();
                let request = request.clone();
                let extra_headers = extra_headers.clone();
                Box::pin(async move {
                    let req = client
                        .post("https://api.deepseek.com/chat/completions")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("Content-Type", "application/json")
                        .json(&request);

                    let captured =
                        shared::send_and_read(req, request_timeout, extra_headers.as_ref()).await?;

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

        let content = choice
            .message
            .content
            .as_ref()
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string();

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
            let completion_tokens = usage.completion_tokens;
            let total_tokens = usage.total_tokens;

            // Cache misses are billed at the input ("cache miss") rate, hits at the
            // much cheaper cache-read rate. {@see split_prompt_tokens} for why the
            // miss count is derived rather than read straight off the response.
            let (input_tokens_clean, cache_read_tokens) = split_prompt_tokens(&usage);

            // DeepSeek doesn't expose cache_write separately - it's included in cache_miss
            let cache_write_tokens = 0_u64;

            // ONE path for both the cached and uncached case: with zero hits this is
            // exactly the no-cache calculation, so the two can never drift apart.
            // Tier is picked from the pricing table active now (peak windows
            // 01:00-04:00 and 06:00-10:00 UTC, off-peak — half price — otherwise).
            let pricing = pricing_table_at(std::time::SystemTime::now());
            let cost = calculate_cost_with_cache(
                pricing,
                &params.model,
                input_tokens_clean,
                cache_read_tokens,
                completion_tokens,
            );

            let reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens)
                .unwrap_or(0);
            let (output_tokens, reasoning_tokens) =
                TokenUsage::split_output(completion_tokens, reasoning_tokens);

            Some(TokenUsage {
                input_tokens: input_tokens_clean, // CLEAN input (cache miss tokens)
                cache_read_tokens,                // Tokens read from cache
                cache_write_tokens,               // DeepSeek doesn't expose this (0)
                output_tokens,
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
        // Legacy aliases removed by DeepSeek 2026-07-24
        assert!(!provider.supports_model("deepseek-chat"));
        assert!(!provider.supports_model("deepseek-reasoner"));
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("deepseek-coder")); // Not in current API
    }

    #[test]
    fn test_vision_route() {
        use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};

        let provider = DeepSeekProvider::new();
        assert!(provider.supports_model("deepseek-v4-flash-vision-exp"));
        assert!(provider.supports_vision("deepseek-v4-flash-vision-exp"));
        assert!(!provider.supports_vision("deepseek-v4-flash"));
        assert!(!provider.supports_vision("deepseek-v4-pro"));

        // Vision route bills at v4-flash rates
        assert_eq!(
            calculate_cost(PRICING_PEAK, "deepseek-v4-flash-vision-exp", 1_000_000, 0),
            calculate_cost(PRICING_PEAK, "deepseek-v4-flash", 1_000_000, 0)
        );

        let msg = Message::user("what is this?").with_images(vec![ImageAttachment {
            data: ImageData::Base64("QUJD".to_string()),
            media_type: "image/png".to_string(),
            source_type: SourceType::Clipboard,
            dimensions: None,
            size_bytes: None,
        }]);
        let content = convert_messages(std::slice::from_ref(&msg))[0]
            .content
            .clone()
            .unwrap();
        assert_eq!(content[0]["text"], "what is this?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,QUJD");

        // Text-only turns keep the plain string shape
        let plain = convert_messages(&[Message::user("hi")])[0].content.clone();
        assert_eq!(plain, Some(serde_json::json!("hi")));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = DeepSeekProvider::new();
        assert!(provider.supports_model("DEEPSEEK-V4-FLASH"));
        assert!(provider.supports_model("DEEPSEEK-V4-PRO"));
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
    }

    #[test]
    fn test_map_reasoning_effort() {
        use crate::llm::types::ReasoningEffort;
        assert_eq!(
            map_reasoning_effort(Some(ReasoningEffort::Low)),
            Some("low")
        );
        assert_eq!(
            map_reasoning_effort(Some(ReasoningEffort::Medium)),
            Some("low")
        );
        assert_eq!(
            map_reasoning_effort(Some(ReasoningEffort::High)),
            Some("high")
        );
        assert_eq!(
            map_reasoning_effort(Some(ReasoningEffort::XHigh)),
            Some("high")
        );
        assert_eq!(
            map_reasoning_effort(Some(ReasoningEffort::Max)),
            Some("max")
        );
        // None = provider default (thinking on, effort "high"); field omitted.
        assert_eq!(map_reasoning_effort(None), None);
    }

    #[test]
    fn test_tiered_pricing_peak_and_off_peak() {
        // Peak: flash $0.44 in / $1.32 out per 1M
        let peak = calculate_cost(PRICING_PEAK, "deepseek-v4-flash", 1_000_000, 500_000).unwrap();
        assert!((peak - (0.44 + 0.5 * 1.32)).abs() < 0.01);

        // Off-peak is exactly half of peak
        let off_peak =
            calculate_cost(PRICING_OFF_PEAK, "deepseek-v4-flash", 1_000_000, 500_000).unwrap();
        assert!((off_peak - peak / 2.0).abs() < 0.01);

        // Peak: pro $1.32 in / $3.96 out per 1M
        let pro = calculate_cost(PRICING_PEAK, "deepseek-v4-pro", 1_000_000, 500_000).unwrap();
        assert!((pro - (1.32 + 0.5 * 3.96)).abs() < 0.01);

        // Peak cache-hit rate: flash $0.014/1M
        let cached =
            calculate_cost_with_cache(PRICING_PEAK, "deepseek-v4-flash", 0, 1_000_000, 0).unwrap();
        assert!((cached - 0.014).abs() < 0.0001);
    }

    #[test]
    fn test_pricing_table_at_selects_tier() {
        use std::time::{Duration, SystemTime};

        let at = |secs: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(secs);

        // Monday 2026-08-17 00:00 UTC — walk a full weekday hour by hour.
        let monday_midnight = 1_786_924_800_u64;
        for hour in 0..24u64 {
            let table = pricing_table_at(at(monday_midnight + hour * 3_600));
            let expected = if is_peak_window(monday_midnight / 86_400, hour) {
                PRICING_PEAK
            } else {
                PRICING_OFF_PEAK
            };
            assert_eq!(table, expected, "hour {} misclassified", hour);
        }

        // Saturday 2026-08-22 is off-peak for the entire day, including hours
        // that would be peak on weekdays.
        let saturday_midnight = monday_midnight + 5 * 86_400;
        for hour in 0..24u64 {
            assert_eq!(
                pricing_table_at(at(saturday_midnight + hour * 3_600)),
                PRICING_OFF_PEAK,
                "Saturday hour {} must be off-peak",
                hour
            );
        }
    }

    /// Deserializes real payload shapes on purpose: the bug this guards lived in
    /// the `#[serde(default)]` fields, so constructing the struct by hand would
    /// step right over it.
    #[test]
    fn test_split_prompt_tokens_handles_both_usage_shapes() {
        let parse = |v: serde_json::Value| -> DeepSeekUsage { serde_json::from_value(v).unwrap() };

        // Native shape — hit and miss both reported.
        let native = parse(serde_json::json!({
            "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
            "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600
        }));
        assert_eq!(split_prompt_tokens(&native), (600, 400));

        // OpenAI-compatible shape — cached_tokens only, NO miss field anywhere.
        // Reading the miss field directly yielded 0 here, so the whole uncached
        // prompt was billed free and reported as 0 input tokens.
        let compat = parse(serde_json::json!({
            "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
            "prompt_tokens_details": {"cached_tokens": 400}
        }));
        assert_eq!(
            split_prompt_tokens(&compat),
            (600, 400),
            "uncached prompt tokens must not bill as free"
        );

        // No cache information at all — every prompt token is a miss.
        let plain = parse(serde_json::json!({
            "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010
        }));
        assert_eq!(split_prompt_tokens(&plain), (1000, 0));

        // Inconsistent provider data must saturate, never underflow into a
        // near-u64::MAX token count and a catastrophic charge.
        let bogus = parse(serde_json::json!({
            "prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 500}
        }));
        assert_eq!(split_prompt_tokens(&bogus), (0, 500));
    }

    #[test]
    fn test_thinking_block_extraction() {
        // Test with reasoning_content present
        let message_with_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::json!("The answer is 9.11")),
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
            content: Some(serde_json::json!("Hello")),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        assert!(message_without_thinking.reasoning_content.is_none());

        // Test with empty reasoning_content
        let message_empty_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: Some(serde_json::json!("Hello")),
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
