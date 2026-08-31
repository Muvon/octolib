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
//! 2026-08-16 16:00 UTC; peak windows 01:00-04:00 and 06:00-10:00 UTC
//! Monday-Friday, off-peak rates are half of peak).
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
//! on 2026-07-24 per <https://api-docs.deepseek.com/updates>.
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

/// Pick the pricing table that applies at `time` (tier decided by UTC weekday and hour)
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
/// DeepSeek's canonical values are "low" / "high" / "max" (default "high"
/// when omitted, thinking enabled by default); the API also accepts "medium"
/// and "xhigh" but maps both UP to "high". Intermediate internal levels
/// instead floor to the nearest lower tier so the adapter never increases
/// effort (Medium → "low", deviating from the API's own medium → "high").
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
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<DeepSeekTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    thinking: DeepSeekThinking,
}

#[derive(Serialize, Debug, Clone)]
struct DeepSeekThinking {
    #[serde(rename = "type")]
    thinking_type: &'static str,
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
/// DeepSeek thinking-mode rule (per /guides/thinking_mode): whenever the current
/// request carries tools, every prior assistant turn's complete reasoning_content
/// MUST be replayed, including turns that did not call a tool. Without tools the
/// API ignores historical reasoning, so it is omitted.
fn convert_messages(
    messages: &[crate::llm::types::Message],
    has_tools: bool,
) -> Vec<DeepSeekMessage> {
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

            let reasoning_content = if has_tools && msg.role == "assistant" {
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

fn build_request(params: &ChatCompletionParams) -> DeepSeekRequest {
    let has_tools = params.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    let messages = convert_messages(&params.messages, has_tools);
    let response_format = params.response_format.as_ref().map(|_| {
        // Preserve the existing native-wire behavior: DeepSeek exposes JSON
        // Object mode here, not JSON Schema.
        serde_json::json!({"type": "json_object"})
    });
    let tools = params.tools.as_ref().and_then(|tools| {
        (!tools.is_empty()).then(|| {
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
                .collect()
        })
    });

    DeepSeekRequest {
        model: params.model.clone(),
        messages,
        max_tokens: Some(params.max_tokens),
        stream: Some(false),
        response_format,
        tools,
        reasoning_effort: map_reasoning_effort(params.reasoning_effort),
        thinking: DeepSeekThinking {
            thinking_type: "enabled",
        },
    }
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
        // All active native routes are V4 thinking models. DeepSeek documents
        // sampling controls as unsupported in thinking mode (accepted but ignored).
        SamplingSupport::NONE
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let request = build_request(&params);

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
            // 01:00-04:00 and 06:00-10:00 UTC Monday-Friday, off-peak — half price — otherwise).
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
#[path = "deepseek_tests.rs"]
mod tests;
