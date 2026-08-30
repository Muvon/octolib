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

//! Moonshot AI (Kimi) provider implementation
//!
//! PRICING UPDATE: July 2026 (verified July 17, 2026)
//! Source: <https://platform.kimi.ai/docs/pricing/chat-k3.md>,
//!         <https://platform.kimi.ai/docs/pricing/chat-k27-code.md>,
//!         <https://platform.kimi.ai/docs/pricing/chat-k26.md>,
//!         <https://platform.kimi.ai/docs/pricing/chat-k25.md>,
//!         <https://platform.kimi.ai/docs/pricing/chat-k2.md>,
//!         <https://platform.kimi.ai/docs/pricing/chat-v1.md>
//!
//! Per 1M tokens (USD): (cache_hit, cache_miss_input, output)
//! - kimi-k3:                    $0.30 / $3.00 / $15.00
//! - kimi-k2.7-code:             $0.19 / $0.95 / $4.00
//! - kimi-k2.7-code-highspeed:   $0.38 / $1.90 / $8.00
//! - kimi-k2.6:                  $0.16 / $0.95 / $4.00
//! - kimi-k2.5:                  $0.10 / $0.60 / $3.00
//! - kimi-k2-0905-preview:       $0.15 / $0.60 / $2.50
//! - kimi-k2-0711-preview:       $0.15 / $0.60 / $2.50
//! - kimi-k2-turbo-preview:      $0.15 / $1.15 / $8.00
//! - kimi-k2-thinking:           $0.15 / $0.60 / $2.50
//! - kimi-k2-thinking-turbo:     $0.15 / $1.15 / $8.00
//! - moonshot-v1-{8k,32k,128k}:  no caching; (input, output) only
//!
//! Caching: Kimi uses AUTOMATIC context caching (no `cache_control` markers).
//! Response usage carries `cached_tokens` (top-level) and/or
//! `prompt_tokens_details.cached_tokens`. `prompt_tokens` is the TOTAL prompt
//! including cached portion, so clean input = prompt_tokens - cached_tokens.
//! Cache writes are NOT separately billed by Moonshot — only hits are discounted.
//!
//! Kimi K3 specifics (<https://platform.kimi.ai/docs/api/chat>):
//! - K3 always reasons; effort is set via top-level `reasoning_effort`
//!   ("low" / "high" / "max", default "max"). High effort = more output tokens
//!   billed at the output price, so the caller's ReasoningEffort is forwarded.
//! - Output is capped via `max_completion_tokens`; `max_tokens` is deprecated for K3.
//! - `usage.completion_tokens_details.reasoning_tokens` reports reasoning tokens.
use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, ReasoningEffort, SamplingSupport,
    TokenUsage, ToolCall,
};
use crate::llm::utils::{contains_ignore_ascii_case, is_model_in_pricing_table, PricingTuple};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

// Model pricing (per 1M tokens in USD) - Updated June 2026 (verified against official docs)
/// Format: (model, input, output, cache_write, cache_read)
///
/// Notes:
/// - `input` = cache miss price; `cache_read` = cache hit price.
/// - Moonshot does NOT bill cache writes separately, so `cache_write` = `input`.
/// - V1 legacy models have no cache support → `cache_write` = `cache_read` = `input`.
/// - Order matters: more specific patterns first (substring matching).
const PRICING: &[PricingTuple] = &[
    // Kimi K3 (1M-context multimodal reasoning flagship)
    ("kimi-k3", 3.00, 15.00, 3.00, 0.30),
    // Kimi K2.7 Code HighSpeed (high-throughput variant; must precede the
    // general kimi-k2.7 entry since that pattern is a substring of this name)
    ("kimi-k2.7-code-highspeed", 1.90, 8.00, 1.90, 0.38),
    // Kimi K2.7 Code (multimodal coding flagship, covers kimi-k2.7-code)
    ("kimi-k2.7", 0.95, 4.00, 0.95, 0.19),
    // Kimi K2.6 (multimodal)
    ("kimi-k2.6", 0.95, 4.00, 0.95, 0.16),
    // Kimi K2.5 (multimodal)
    ("kimi-k2.5", 0.60, 3.00, 0.60, 0.10),
    // Kimi K2 turbo variants (high-speed)
    ("kimi-k2-thinking-turbo", 1.15, 8.00, 1.15, 0.15),
    ("kimi-k2-turbo", 1.15, 8.00, 1.15, 0.15),
    // Kimi K2 standard variants
    ("kimi-k2-thinking", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2-0905", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2-0711", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2", 0.60, 2.50, 0.60, 0.15),
    // Moonshot V1 series (legacy, no cache support)
    ("moonshot-v1-128k", 2.00, 5.00, 2.00, 2.00),
    ("moonshot-v1-32k", 1.00, 3.00, 1.00, 1.00),
    ("moonshot-v1-8k", 0.20, 2.00, 0.20, 0.20),
];

/// Get pricing tuple for a specific model (case-insensitive)
fn get_pricing_tuple(model: &str) -> Option<(f64, f64, f64, f64)> {
    crate::llm::utils::get_model_pricing(model, PRICING)
}

/// Calculate cost for Moonshot models with cache-aware pricing
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

/// Calculate cost for Moonshot models without cache
/// Returns None if the model is not supported (not in pricing table)
fn calculate_cost(model: &str, input_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_with_cache(model, input_tokens, 0, completion_tokens)
}

/// Moonshot AI (Kimi) provider
#[derive(Debug, Clone, Default)]
pub struct MoonshotProvider;

impl MoonshotProvider {
    pub fn new() -> Self {
        Self
    }
}

const MOONSHOT_API_KEY_ENV: &str = "MOONSHOT_API_KEY";

// Moonshot API request/response structures (OpenAI-compatible)
#[derive(Serialize, Debug, Clone)]
struct MoonshotRequest {
    model: String,
    messages: Vec<MoonshotMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Kimi K3 output cap (`max_tokens` is deprecated for K3 per API docs)
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    /// Kimi K3 reasoning effort: "low" / "high" / "max" (K3 default: "max")
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MoonshotTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
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
    /// Kimi reasoning state. K2.7 requires it to be preserved across all
    /// assistant turns; K2.5/K2.6 also require it on tool-call turns.
    /// - Some("") = empty reasoning (required for tool calls)
    /// - Some(content) = actual reasoning content
    /// - None = omit field (backward compatible for non-thinking models)
    #[serde(default, alias = "reasoning", skip_serializing_if = "Option::is_none")]
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

#[derive(Serialize, Deserialize, Debug, Default)]
struct MoonshotUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    cached_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: Option<MoonshotPromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<MoonshotCompletionTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct MoonshotPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct MoonshotCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
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

fn build_moonshot_tools(tools: &[crate::llm::types::FunctionDefinition]) -> Vec<MoonshotTool> {
    let mut sorted_tools = tools.to_vec();
    sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

    sorted_tools
        .iter()
        .map(|f| MoonshotTool {
            tool_type: "function".to_string(),
            function: MoonshotToolFunction {
                name: f.name.clone(),
                description: f.description.clone(),
                parameters: sanitize_schema_for_moonshot(&f.parameters),
            },
        })
        .collect()
}

/// Sanitize a JSON Schema for Moonshot's strict validator.
///
/// Moonshot rejects schemas where a property has a `$ref` AND sibling keys
/// (`description`, `default`, etc.) — error:
///   "conflicting keywords found after $ref expansion: description"
///
/// schemars 1.x emits exactly this pattern when a typed field has a doc comment:
///   { "$ref": "#/$defs/Foo", "description": "..." }
///
/// Strategy: inline `$defs` references in-place, merging the referenced schema
/// with the sibling keys (siblings win on conflict — they carry the field-specific
/// doc). After inlining, drop `$defs` from the root since nothing references it.
/// This is lossless: descriptions, enums, and types are all preserved.
fn sanitize_schema_for_moonshot(schema: &serde_json::Value) -> serde_json::Value {
    let mut cloned = schema.clone();
    if let serde_json::Value::Object(root) = &cloned {
        // Extract $defs (if present) for inlining
        let defs = root.get("$defs").cloned();
        if let Some(serde_json::Value::Object(defs_map)) = defs {
            inline_refs(&mut cloned, &defs_map);
            // Drop $defs from root after inlining — nothing references it now
            if let serde_json::Value::Object(root_mut) = &mut cloned {
                root_mut.remove("$defs");
            }
        }
    }
    cloned
}

/// Recursively walk `value` and inline any `{"$ref": "#/$defs/Name", ...siblings}`
/// objects with the referenced definition's content. Siblings (description, etc.)
/// override fields from the inlined definition.
fn inline_refs(value: &mut serde_json::Value, defs: &serde_json::Map<String, serde_json::Value>) {
    match value {
        serde_json::Value::Object(map) => {
            // First, recurse into children so nested $refs are resolved bottom-up
            for v in map.values_mut() {
                inline_refs(v, defs);
            }

            // If this object has a $ref pointing into $defs, inline it
            if let Some(serde_json::Value::String(ref_str)) = map.get("$ref") {
                if let Some(def_name) = ref_str.strip_prefix("#/$defs/") {
                    if let Some(serde_json::Value::Object(def_obj)) = defs.get(def_name) {
                        // Build merged object: start with def content, overlay siblings
                        let mut merged = def_obj.clone();
                        for (k, v) in map.iter() {
                            if k != "$ref" {
                                merged.insert(k.clone(), v.clone());
                            }
                        }
                        *value = serde_json::Value::Object(merged);
                        // Recurse again in case the inlined def itself contains $refs
                        inline_refs(value, defs);
                    }
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for v in arr.iter_mut() {
                inline_refs(v, defs);
            }
        }
        _ => {}
    }
}

// Convert messages to Moonshot (OpenAI-compatible) format
pub(crate) fn preserves_historical_thinking(model: &str) -> bool {
    contains_ignore_ascii_case(model, "kimi-k2.6")
        || contains_ignore_ascii_case(model, "kimi-k2.7")
        || contains_ignore_ascii_case(model, "kimi-k3")
}

fn thinking_config(model: &str) -> Option<serde_json::Value> {
    contains_ignore_ascii_case(model, "kimi-k2.6").then(|| {
        serde_json::json!({
            "type": "enabled",
            "keep": "all"
        })
    })
}

/// Map generic ReasoningEffort to Kimi K3's `reasoning_effort` string.
/// K3 supports only "low" / "high" / "max" (default "max" when the field is
/// omitted), so intermediate levels floor to the nearest supported lower
/// effort. Returns None for non-K3 models and when the caller leaves the
/// effort unset (K3 then applies its own default).
fn k3_reasoning_effort(model: &str, effort: Option<ReasoningEffort>) -> Option<&'static str> {
    if !contains_ignore_ascii_case(model, "kimi-k3") {
        return None;
    }
    match effort {
        Some(ReasoningEffort::Low) | Some(ReasoningEffort::Medium) => Some("low"),
        Some(ReasoningEffort::High) | Some(ReasoningEffort::XHigh) => Some("high"),
        Some(ReasoningEffort::Max) => Some("max"),
        None => None,
    }
}

fn convert_messages(messages: &[crate::llm::types::Message], model: &str) -> Vec<MoonshotMessage> {
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

                if let Some(videos) = &message.videos {
                    for video in videos {
                        if let crate::llm::types::VideoData::Base64(data) = &video.data {
                            content_parts.push(serde_json::json!({
                                "type": "video_url",
                                "video_url": {
                                    "url": format!("data:{};base64,{}", video.media_type, data)
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

                let tool_calls = if let Some(tool_calls_data) = message.tool_calls.as_ref() {
                    let generic_calls =
                        shared::parse_generic_tool_calls_lossy(Some(tool_calls_data), "moonshot");

                    if !generic_calls.is_empty() {
                        Some(
                            generic_calls
                                .into_iter()
                                .map(|tc| MoonshotToolCall {
                                    id: tc.id,
                                    tool_type: "function".to_string(),
                                    function: MoonshotFunction {
                                        name: tc.name,
                                        arguments: shared::arguments_to_json_string(&tc.arguments),
                                    },
                                })
                                .collect(),
                        )
                    } else {
                        // If parsing as GenericToolCall fails, try parsing provider-specific format
                        serde_json::from_value::<Vec<MoonshotToolCall>>(tool_calls_data.clone())
                            .ok()
                    }
                } else {
                    None
                };

                // Extract reasoning_content from thinking block if present.
                // Moonshot requires reasoning_content for assistant messages with tool_calls.
                // Always include the field (even if empty) for tool calls.
                // CRITICAL: Handle both fresh responses and replayed messages from cache
                let reasoning_content = Some(
                    message
                        .thinking
                        .as_ref()
                        .map(|t| t.content.clone())
                        .unwrap_or_default(), // Use unwrap_or_default() instead of unwrap_or_else for consistency
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

                if let Some(videos) = &message.videos {
                    for video in videos {
                        if let crate::llm::types::VideoData::Base64(data) = &video.data {
                            content_parts.push(serde_json::json!({
                                "type": "video_url",
                                "video_url": {
                                    "url": format!("data:{};base64,{}", video.media_type, data)
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
                    // Kimi K2.7 preserve_thinking retains reasoning across every
                    // assistant turn, including turns without tool calls.
                    reasoning_content: if message.role == "assistant"
                        && preserves_historical_thinking(model)
                    {
                        message.thinking.as_ref().map(|t| t.content.clone())
                    } else {
                        None
                    },
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
        // Moonshot (Kimi) models - check against pricing table (strict)
        is_model_in_pricing_table(model, PRICING)
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

    fn supports_caching(&self, model: &str) -> bool {
        // Kimi K2.x and K3 series support automatic context caching.
        // Moonshot V1 legacy models have no caching (no cache-hit pricing).
        contains_ignore_ascii_case(model, "kimi-k2") || contains_ignore_ascii_case(model, "kimi-k3")
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Kimi K2.5, K2.6, K2.7 and K3 support vision/multimodal
        contains_ignore_ascii_case(model, "kimi-k2.5")
            || contains_ignore_ascii_case(model, "kimi-k2.6")
            || contains_ignore_ascii_case(model, "kimi-k2.7")
            || contains_ignore_ascii_case(model, "kimi-k3")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            get_pricing_tuple(model)?;
        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Kimi K3 — 1M context window per official model spec.
        if contains_ignore_ascii_case(model, "kimi-k3") {
            return 1_048_576;
        }
        // Kimi K2 family — context windows per official model spec
        // (kimi-k2.6/k2.5/k2-0905/turbo/thinking → 256K = 262_144;
        //  kimi-k2-0711 → 128K = 131_072).
        if contains_ignore_ascii_case(model, "kimi-k2-0711") {
            return 131_072;
        }
        if contains_ignore_ascii_case(model, "kimi-k2") {
            return 262_144;
        }
        // Moonshot V1 series — context window matches the variant name.
        if contains_ignore_ascii_case(model, "moonshot-v1-128k") {
            return 131_072;
        }
        if contains_ignore_ascii_case(model, "moonshot-v1-32k") {
            return 32_768;
        }
        if contains_ignore_ascii_case(model, "moonshot-v1-8k") {
            return 8_192;
        }
        // Default fallback for unknown variants
        128_000
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        // Kimi K2.5, K2.6 and K2.7 do not expose a modifiable temperature.
        // Kimi K3 has no temperature parameter at all (always-reasoning model).
        // Other Moonshot models support temperature. None support top_p or top_k.
        let fixed_temp = contains_ignore_ascii_case(model, "kimi-k2.5")
            || contains_ignore_ascii_case(model, "kimi-k2.6")
            || contains_ignore_ascii_case(model, "kimi-k2.7")
            || contains_ignore_ascii_case(model, "kimi-k3");
        SamplingSupport {
            temperature: !fixed_temp,
            top_p: false,
            top_k: false,
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to Moonshot format
        // NOTE: Moonshot uses AUTOMATIC context caching (like OpenAI)
        // Manual caching via /v1/caching endpoint is deprecated (returns "model family is invalid")
        let messages = convert_messages(&params.messages, &params.model);

        // Kimi K2.5, K2.6, K2.7 and K3 do not expose a modifiable temperature,
        // so effective sampling support omits it from the request.
        let sampling = self.effective_sampling_params(&params);
        let temperature = sampling.temperature;

        // Kimi K3 caps output via `max_completion_tokens`; `max_tokens` is
        // deprecated for K3 per the API docs. Older Kimi models keep max_tokens.
        let is_k3 = contains_ignore_ascii_case(&params.model, "kimi-k3");
        let (max_tokens, max_completion_tokens) = if params.max_tokens > 0 {
            if is_k3 {
                (None, Some(params.max_tokens))
            } else {
                (Some(params.max_tokens), None)
            }
        } else {
            (None, None)
        };

        let mut request = MoonshotRequest {
            model: params.model.clone(),
            messages,
            temperature,
            max_tokens,
            max_completion_tokens,
            reasoning_effort: k3_reasoning_effort(&params.model, params.reasoning_effort),
            stream: Some(false),
            response_format: None,
            tools: None,
            tool_choice: None,
            thinking: thinking_config(&params.model),
        };

        // Add tools if available (Moonshot is OpenAI-compatible)
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let moonshot_tools = build_moonshot_tools(tools);
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
                    // Moonshot's OpenAI-compat endpoint supports json_schema just like OpenAI
                    // but requires the "name" field in json_schema
                    if let Some(schema) = &response_format.schema {
                        // Strict structured outputs need additionalProperties:false on
                        // every nested object (no-op unless mode is Strict).
                        let schema = crate::llm::utils::normalize_strict_schema(
                            schema,
                            response_format.mode,
                        );

                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "name": "schema",
                                "schema": schema
                            }
                        });
                        if matches!(
                            response_format.mode,
                            crate::llm::types::ResponseMode::Strict
                        ) {
                            format_obj["json_schema"]["strict"] = serde_json::json!(true);
                        }
                        request.response_format = Some(format_obj);
                    } else {
                        request.response_format = Some(serde_json::json!({
                            "type": "json_object"
                        }));
                    }
                }
            }
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
                        .post("https://api.moonshot.ai/v1/chat/completions")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("Content-Type", "application/json")
                        .json(&request);

                    let captured =
                        shared::send_and_read(req, request_timeout, extra_headers.as_ref()).await?;

                    // Return Err for retryable HTTP errors so the retry loop catches them
                    if retry::is_retryable_status(captured.status.as_u16()) {
                        return Err(anyhow::anyhow!(
                            "Moonshot AI API error {}: {}",
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
                "Moonshot AI API error {}: {}",
                response.status,
                response.body
            ));
        }

        let moonshot_response: MoonshotResponse = serde_json::from_str(&response.body)?;

        let mut response_for_exchange = serde_json::to_value(&moonshot_response)?;

        let choice = moonshot_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in Moonshot response"))?;

        let token_usage = if let Some(usage) = moonshot_response.usage {
            let cache_read_tokens = std::cmp::max(
                usage.cached_tokens,
                usage
                    .prompt_tokens_details
                    .as_ref()
                    .map(|d| d.cached_tokens)
                    .unwrap_or(0),
            );

            let reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens)
                .unwrap_or(0);

            let (output_tokens, reasoning_tokens) =
                TokenUsage::split_output(usage.completion_tokens, reasoning_tokens);

            let input_tokens_clean = usage.prompt_tokens.saturating_sub(cache_read_tokens);

            let cost = if cache_read_tokens > 0 {
                calculate_cost_with_cache(
                    &params.model,
                    input_tokens_clean,
                    cache_read_tokens,
                    usage.completion_tokens,
                )
            } else {
                calculate_cost(&params.model, usage.prompt_tokens, usage.completion_tokens)
            };

            Some(TokenUsage {
                input_tokens: input_tokens_clean,
                cache_read_tokens,
                cache_write_tokens: 0,
                output_tokens,
                reasoning_tokens,
                total_tokens: usage.total_tokens,
                cost,
                request_time_ms: Some(request_time_ms),
            })
        } else {
            None
        };

        let content = extract_text_content(&choice.message.content);

        // Extract reasoning_content from response and convert to ThinkingBlock
        // CRITICAL: Preserve even empty reasoning_content for thinking models
        // Empty reasoning_content is required when replaying tool call messages
        let reasoning_token_count = token_usage
            .as_ref()
            .map(|u| u.reasoning_tokens)
            .unwrap_or(0);
        let thinking =
            choice
                .message
                .reasoning_content
                .map(|rc| crate::llm::types::ThinkingBlock {
                    content: rc,
                    tokens: reasoning_token_count,
                });

        let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
            calls
                .into_iter()
                .filter_map(|call| {
                    if call.tool_type != "function" {
                        tracing::warn!(
                            "Unexpected tool type '{}' from Moonshot API",
                            call.tool_type
                        );
                        return None;
                    }

                    let arguments =
                        shared::parse_tool_call_arguments_lossy(&call.function.arguments);

                    Some(ToolCall {
                        id: call.id,
                        name: call.function.name,
                        arguments,
                    })
                })
                .collect()
        });

        if let Some(ref tc) = tool_calls {
            shared::set_response_tool_calls(&mut response_for_exchange, tc, None);
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
        let structured_output = shared::parse_structured_output_from_text(&content);

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
#[path = "moonshot_tests.rs"]
mod tests;
