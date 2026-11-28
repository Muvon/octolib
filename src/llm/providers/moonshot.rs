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
//! PRICING VERIFIED: September 22, 2026
//! Source: <https://platform.kimi.ai/docs/pricing/chat>,
//!         <https://platform.kimi.ai/docs/models>
//!
//! Per 1M tokens (USD): (cache_hit, cache_miss_input, output)
//! - kimi-k3.1-pro:              $0.20 / $2.00 / $10.00
//! - kimi-k3.1-pro-highspeed:    $0.20 / $4.00 / $20.00
//! - kimi-k3:                    $0.30 / $3.00 / $15.00
//! - kimi-k2.7-code:             $0.19 / $0.95 / $4.00
//! - kimi-k2.7-code-highspeed:   $0.38 / $1.90 / $8.00
//! - kimi-k2.6:                  $0.16 / $0.95 / $4.00
//!
//! Discontinued per the models page (no longer served): kimi-k2.5 and
//! moonshot-v1-{8k,32k,128k} (2026-08-31); kimi-k2-0905-preview,
//! kimi-k2-0711-preview, kimi-k2-turbo-preview, kimi-k2-thinking and
//! kimi-k2-thinking-turbo (2026-05-25).
//!
//! Caching: Kimi uses AUTOMATIC context caching (no `cache_control` markers).
//! Response usage carries `cached_tokens` (top-level) and/or
//! `prompt_tokens_details.cached_tokens`. `prompt_tokens` is the TOTAL prompt
//! including cached portion, so clean input = prompt_tokens - cached_tokens.
//! K2.x bill no separate cache write. K3 bills cache write at $3.00 (5-minute
//! TTL, the default) and $6.00 (1-hour TTL); K3.1 Pro bills it at $2.00 and
//! $4.00, and K3.1 Pro Highspeed at $4.00 and $8.00. The API exposes no TTL
//! knob and no cache-write token count, so writes are billed at the input
//! rate here.
//!
//! Kimi K3 specifics (<https://platform.kimi.ai/docs/api/chat>):
//! - K3 always reasons; effort is set via top-level `reasoning_effort`
//!   ("low" / "high" / "max", default "max"). High effort = more output tokens
//!   billed at the output price, so the caller's ReasoningEffort is forwarded.
//! - Output is capped via `max_completion_tokens`; `max_tokens` is deprecated for K3.
//! - `usage.completion_tokens_details.reasoning_tokens` reports reasoning tokens.
//!
//! Requests go to the Anthropic Messages endpoint Kimi documents for
//! Claude-compatible clients — `https://api.moonshot.ai/anthropic/v1/messages`
//! (the base URL the Claude SDK is pointed at is
//! `https://api.moonshot.ai/anthropic`) — through the shared `anthropic_compat`
//! layer. A `MOONSHOT_API_URL` override is sent exactly as it is: one whose path
//! ends in `/messages` speaks Messages, any other is served by the Chat
//! Completions code below, as every override was before. `MOONSHOT_API_MODE`
//! (`anthropic` / `openai`) states the protocol for any URL and, with no URL,
//! selects that protocol's default endpoint.
//!
//! The Messages contract is the vendor's own
//! (<https://platform.kimi.ai/docs/api/messages>, checked 2026-11-26): bearer
//! auth, base64 and URL image blocks on the visual models, Messages SSE and
//! cache usage counters. The route is not live-probed from this checkout (no
//! key is configured), so what it promises is what the contract states plus the
//! requests the shared layer already sends to five other Messages endpoints.
//! What differs per protocol is stated on [`ANTHROPIC_COMPAT`].
use super::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images, MessagesUsage};
use super::shared::{self, CacheMarker};
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
	ChatCompletionParams, ImageData, ProviderExchange, ProviderResponse, ReasoningEffort, TokenUsage, ToolCall,
	VideoData,
};
use crate::llm::utils::{contains_ignore_ascii_case, is_model_in_pricing_table, PricingTuple};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

// Model pricing (per 1M tokens in USD) - verified Sep 22, 2026 against official docs
/// Format: (model, input, output, cache_write, cache_read)
///
/// Notes:
/// - `input` = cache miss price; `cache_read` = cache hit price.
/// - `cache_write` = `input`: K2.x bill no separate cache write and K3's
///   default 5-minute tier equals its input price (1-hour tier not modeled).
/// - Order matters: more specific patterns first (substring matching).
const PRICING: &[PricingTuple] = &[
	// Kimi K3.1 Pro HighSpeed (high-throughput variant; must precede the
	// general kimi-k3.1-pro entry since that pattern is a substring of this name)
	("kimi-k3.1-pro-highspeed", 4.00, 20.00, 4.00, 0.20),
	// Kimi K3.1 Pro (1M-context multimodal reasoning flagship)
	("kimi-k3.1-pro", 2.00, 10.00, 2.00, 0.20),
	// Kimi K3 (1M-context multimodal reasoning flagship)
	("kimi-k3", 3.00, 15.00, 3.00, 0.30),
	// Kimi K2.7 Code HighSpeed (high-throughput variant; must precede the
	// general kimi-k2.7 entry since that pattern is a substring of this name)
	("kimi-k2.7-code-highspeed", 1.90, 8.00, 1.90, 0.38),
	// Kimi K2.7 Code (multimodal coding flagship, covers kimi-k2.7-code)
	(KIMI_K2_7_PATTERN, 0.95, 4.00, 0.95, 0.19),
	// Kimi K2.6 (multimodal)
	("kimi-k2.6", 0.95, 4.00, 0.95, 0.16),
];

const KIMI_K2_7_PATTERN: &str = "kimi-k2.7";

pub(crate) fn setup_models() -> impl Iterator<Item = &'static str> {
	// The K2.7 row prices the full -code ID; the prefix itself is not callable.
	PRICING
		.iter()
		.map(|row| row.0)
		.filter(|model| *model != KIMI_K2_7_PATTERN)
}

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
	let (input_price, output_price, _cache_write_price, cache_read_price) = get_pricing_tuple(model)?;

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

pub(crate) const MOONSHOT_API_KEY_ENV: &str = "MOONSHOT_API_KEY";
const MOONSHOT_API_URL_ENV: &str = "MOONSHOT_API_URL";
/// Messages endpoint, the default.
const MOONSHOT_API_URL: &str = "https://api.moonshot.ai/anthropic/v1/messages";
/// Chat Completions endpoint, for `MOONSHOT_API_MODE=openai`.
const MOONSHOT_CHAT_API_URL: &str = "https://api.moonshot.ai/v1/chat/completions";
const MOONSHOT_API_MODE_ENV: &str = "MOONSHOT_API_MODE";

/// The protocol and URL for one request. A supplied URL is sent as it is,
/// whatever the protocol: it is never rewritten. An explicit mode comes first,
/// then the URL's own path; a URL naming neither endpoint was always served by
/// Chat Completions, and with no URL the default is Messages.
fn select_endpoint(mode: Option<&str>, api_url: Option<&str>) -> Result<(ApiMode, String)> {
	// An empty variable is an unset one.
	let mode = mode
		.filter(|raw| !raw.is_empty())
		.map(|raw| anthropic_compat::parse_api_mode(MOONSHOT_API_MODE_ENV, raw))
		.transpose()?;
	let api_url = api_url.filter(|api_url| !api_url.is_empty());
	let mode = mode
		.or_else(|| api_url.and_then(anthropic_compat::url_api_mode))
		.unwrap_or(match api_url {
			Some(_) => ApiMode::OpenAi,
			None => ApiMode::Anthropic,
		});
	let api_url = api_url.map_or_else(
		|| {
			match mode {
				ApiMode::Anthropic => MOONSHOT_API_URL,
				ApiMode::OpenAi => MOONSHOT_CHAT_API_URL,
			}
			.to_owned()
		},
		str::to_owned,
	);
	Ok((mode, api_url))
}

/// The route a request would take right now, for the capability methods that
/// differ per protocol. A configuration that cannot be resolved is reported as
/// Messages: the request itself then fails with the configuration error.
fn configured_mode() -> ApiMode {
	select_endpoint(
		env::var(MOONSHOT_API_MODE_ENV).ok().as_deref(),
		env::var(MOONSHOT_API_URL_ENV).ok().as_deref(),
	)
	.map(|(mode, _)| mode)
	.unwrap_or(ApiMode::Anthropic)
}

/// Cost of one Messages response, in USD. `input_tokens` is already cache-free;
/// cache writes bill at the input rate (K3's 5-minute tier is its input price)
/// and `output_tokens` includes thinking, which bills at the output rate.
fn messages_cost(model: &str, usage: &MessagesUsage) -> Option<f64> {
	let (input_price, output_price, _cache_write_price, cache_read_price) = get_pricing_tuple(model)?;
	let per_million = |tokens: u64| tokens as f64 / 1_000_000.0;
	Some(
		per_million(usage.input_tokens) * input_price
			+ per_million(usage.cache_write_tokens()) * input_price
			+ per_million(usage.cache_read_tokens) * cache_read_price
			+ per_million(usage.output_tokens) * output_price,
	)
}

/// Moonshot on the shared Messages layer.
///
/// Images: every callable Kimi model takes inline image data. The contract
/// lists a URL source too, but the live Messages endpoint rejects it —
/// `kimi-k3` with a 1x1 PNG URL answered `400 invalid_request_error
/// "unsupported image url"` on 2026-11-26 — so a URL image is refused here
/// instead of being dropped or sent to fail.
/// Video: Messages has no video block, so a video turn is refused here instead
/// of being dropped, and `MOONSHOT_API_MODE=openai` keeps the Chat route that
/// carries `video_url`.
/// Schema: only kimi-k3.1-pro was probed to decode against a schema, and only
/// on the Chat endpoint, so a schema takes the forced-tool path and is
/// validated locally here.
/// Thinking: the Messages contract documents `output_config.effort` with
/// `low` / `high` / `max` (default `max`)
/// (<https://platform.kimi.ai/docs/api/messages>), the same three values the
/// Chat route maps onto, so a requested effort is forwarded with that map
/// instead of being dropped. Models outside the Kimi K3 family keep their
/// documented default (K2.6 thinks unless disabled).
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "moonshot",
	images: |_| Images::Base64,
	videos: |_| false,
	enforces_response_schema: false,
	thinking: |model, effort, request| {
		let Some(effort) = effort.and_then(|effort| k3_reasoning_effort(model, Some(effort))) else {
			return;
		};
		// The schema, when one was requested, is already in `output_config`.
		request["output_config"]["effort"] = serde_json::json!(effort);
	},
	cost: messages_cost,
	default_system: "",
	requires_max_tokens: true,
	cache_marker: Some(CacheMarker::Ephemeral),
};
/// See `cache_lifetime` for the measurement.
const MEASURED_CACHE_LIFETIME: std::time::Duration = std::time::Duration::from_secs(3 * 60);

// Moonshot API request/response structures (OpenAI-compatible)
#[derive(Serialize, Debug, Clone)]
struct MoonshotRequest {
	model: String,
	messages: Vec<MoonshotMessage>,
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
	stream_options: serde_json::Value,
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
	/// assistant turns; K2.6 also requires it on tool-call turns.
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

/// Kimi K2.6 `thinking` block. Thinking is on by default; an explicit
/// `ReasoningEffort::None` sends `{"type": "disabled"}` (platform.kimi.ai
/// docs/guide/use-kimi-k2-thinking-model, verified Oct 2026).
fn thinking_config(model: &str, effort: Option<ReasoningEffort>) -> Option<serde_json::Value> {
	contains_ignore_ascii_case(model, "kimi-k2.6").then(|| match effort {
		Some(ReasoningEffort::None) => serde_json::json!({ "type": "disabled" }),
		_ => serde_json::json!({
			"type": "enabled",
			"keep": "all"
		}),
	})
}

/// Map generic ReasoningEffort to Kimi K3's `reasoning_effort` string.
/// K3 supports only "low" / "high" / "max" (default "max" when the field is
/// omitted), so intermediate levels floor to the nearest supported lower
/// effort. K3 documents no way to disable reasoning, so an explicit `None`
/// runs at "low" (platform.kimi.ai/docs/guide/use-kimi-k3, verified Oct 2026).
/// Returns None for non-K3 models and when the caller leaves the effort unset
/// (K3 then applies its own default).
fn k3_reasoning_effort(model: &str, effort: Option<ReasoningEffort>) -> Option<&'static str> {
	if !contains_ignore_ascii_case(model, "kimi-k3") {
		return None;
	}
	match effort {
		Some(ReasoningEffort::None) | Some(ReasoningEffort::Low) | Some(ReasoningEffort::Medium) => Some("low"),
		Some(ReasoningEffort::High) | Some(ReasoningEffort::XHigh) => Some("high"),
		Some(ReasoningEffort::Max) => Some("max"),
		None => None,
	}
}

fn convert_messages(messages: &[crate::llm::types::Message], model: &str) -> Result<Vec<MoonshotMessage>> {
	let messages = shared::chat_completion_messages(messages);
	let messages = messages.as_ref();
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

				content_parts.extend(media_parts(message, model)?);

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
					let generic_calls = shared::parse_generic_tool_calls_lossy(Some(tool_calls_data), "moonshot");

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
						serde_json::from_value::<Vec<MoonshotToolCall>>(tool_calls_data.clone()).ok()
					}
				} else {
					None
				};

				// Extract reasoning_content from thinking block if present.
				// Moonshot requires reasoning_content for assistant messages with tool_calls.
				// Always include the field (even if empty) for tool calls.
				// CRITICAL: Handle both fresh responses and replayed messages from cache
				let reasoning_content = Some(
					message.thinking.as_ref().map(|t| t.content.clone()).unwrap_or_default(), // Use unwrap_or_default() instead of unwrap_or_else for consistency
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

				content_parts.extend(media_parts(message, model)?);

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
					reasoning_content: if message.role == "assistant" && preserves_historical_thinking(model) {
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

	Ok(result)
}

/// Image and video blocks for a turn. A URL attachment is refused, never
/// skipped: the Chat route cannot hand a URL to the model, and dropping it
/// answered as though the caller's media had never been sent.
fn media_parts(message: &crate::llm::types::Message, model: &str) -> Result<Vec<serde_json::Value>> {
	let mut parts = Vec::new();
	for image in message.images.as_deref().unwrap_or_default() {
		match &image.data {
			ImageData::Base64(data) => parts.push(serde_json::json!({
				"type": "image_url",
				"image_url": { "url": format!("data:{};base64,{}", image.media_type, data) }
			})),
			ImageData::Url(_) => {
				return Err(anyhow::anyhow!(
					"moonshot model {model} takes image data inline only: a URL image is refused rather than dropped"
				))
			}
		}
	}
	for video in message.videos.as_deref().unwrap_or_default() {
		match &video.data {
			VideoData::Base64(data) => parts.push(serde_json::json!({
				"type": "video_url",
				"video_url": { "url": format!("data:{};base64,{}", video.media_type, data) }
			})),
			VideoData::Url(_) => {
				return Err(anyhow::anyhow!(
					"moonshot model {model} takes video data inline only: a URL video is refused rather than dropped"
				))
			}
		}
	}
	Ok(parts)
}

fn extract_text_content(content: &Option<serde_json::Value>) -> String {
	match content {
		Some(serde_json::Value::String(s)) => s.clone(),
		Some(serde_json::Value::Array(parts)) => parts
			.iter()
			.filter_map(|part| part.get("text").and_then(|t| t.as_str()).map(|s| s.to_string()))
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
		env::var(MOONSHOT_API_KEY_ENV)
			.map_err(|_| crate::errors::missing_api_key("moonshot", MOONSHOT_API_KEY_ENV).into())
	}

	/// Kimi K2.6+/K3 preserve reasoning across every assistant turn; older
	/// Kimi models get only the trailing message's reasoning back.
	fn replays_thinking(&self, model: &str) -> bool {
		preserves_historical_thinking(model)
	}

	fn supports_caching(&self, model: &str) -> bool {
		// Kimi K2.x and K3 series support automatic context caching.
		contains_ignore_ascii_case(model, "kimi-k2") || contains_ignore_ascii_case(model, "kimi-k3")
	}

	/// The docs say "valid for 5 minutes, and each hit resets the validity to 5 minutes"
	/// (<https://platform.kimi.ai/docs/guide/context-caching.md>), but the cache dies
	/// earlier. Measured 2026-11-07 on api.moonshot.ai: unique ~3.5k-token prompts re-sent
	/// once after an idle delay, two samples per delay. kimi-k3.1-pro, kimi-k3, kimi-k2.7-code
	/// and kimi-k2.6 hit on every sample at 1m, 3m and 4m. At 5m only kimi-k2.7-code still
	/// hit, and from 6m every sample missed. Messages, measured 2026-11-28 with ~8k-token
	/// prompts: kimi-k3.1-pro and kimi-k3 hit on every sample at 1m and 3m and missed both
	/// at 5m; 4m was not sampled. The 1h option needs `prompt_cache_options.ttl`, which
	/// octolib does not send.
	fn cache_lifetime(&self, model: &str, _use_long_cache: bool) -> Option<std::time::Duration> {
		self.supports_caching(model).then_some(MEASURED_CACHE_LIFETIME)
	}

	fn supports_vision(&self, model: &str) -> bool {
		// Kimi K2.6, K2.7 and K3 support vision/multimodal
		contains_ignore_ascii_case(model, "kimi-k2.6")
			|| contains_ignore_ascii_case(model, "kimi-k2.7")
			|| contains_ignore_ascii_case(model, "kimi-k3")
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		true
	}

	/// Only Kimi K3.1 Pro decodes against the schema: with `strict: true` and a
	/// two-field object schema, a prompt demanding a JSON array got the object
	/// on 5 of 5 K3.1 Pro calls and the array on 4 of 4 Kimi K3 calls (live,
	/// 2026-10-26, Chat endpoint). The Messages endpoint was not probed, so the
	/// claim stays on the route that has the evidence and a Messages schema
	/// takes the forced-tool path with local validation.
	fn enforces_response_schema(&self, model: &str) -> bool {
		configured_mode() == ApiMode::OpenAi && contains_ignore_ascii_case(model, "kimi-k3.1")
	}

	/// Video is a Chat Completions field (`video_url`) and the reference table
	/// lists the Kimi models that take it; the Messages protocol has no video
	/// block, so the capability follows the route instead of advertising a turn
	/// that would be refused.
	fn supports_video(&self, model: &str) -> bool {
		configured_mode() == ApiMode::OpenAi
			&& crate::llm::reference_models::get_reference_capabilities(model)
				.map(|capabilities| capabilities.video)
				.unwrap_or(false)
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		let (input_price, output_price, cache_write_price, cache_read_price) = get_pricing_tuple(model)?;
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
		// Kimi K2 family (k2.6 / k2.7-code / k2.7-code-highspeed) — 256K = 262_144
		// per official model spec.
		if contains_ignore_ascii_case(model, "kimi-k2") {
			return 262_144;
		}
		// Default fallback for unknown variants
		128_000
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let (mode, api_url) = select_endpoint(
			env::var(MOONSHOT_API_MODE_ENV).ok().as_deref(),
			env::var(MOONSHOT_API_URL_ENV).ok().as_deref(),
		)?;
		if mode == ApiMode::Anthropic {
			return anthropic_compat::chat_completion(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}

		// Convert messages to Moonshot format
		// NOTE: Moonshot uses AUTOMATIC context caching (like OpenAI)
		// Manual caching via /v1/caching endpoint is deprecated (returns "model family is invalid")
		let messages = convert_messages(&params.messages, &params.model)?;

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
			max_tokens,
			max_completion_tokens,
			reasoning_effort: k3_reasoning_effort(&params.model, params.reasoning_effort),
			stream: Some(true),
			stream_options: serde_json::json!({"include_usage": true}),
			response_format: None,
			tools: None,
			tool_choice: None,
			thinking: thinking_config(&params.model, params.reasoning_effort),
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
						let schema = crate::llm::utils::normalize_strict_schema(schema, response_format.mode);

						let mut format_obj = serde_json::json!({
							"type": "json_schema",
							"json_schema": {
								"name": "schema",
								"schema": schema
							}
						});
						if matches!(response_format.mode, crate::llm::types::ResponseMode::Strict) {
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
		let stream_timeout = params.stream_timeout;
		let extra_headers = params.extra_headers.clone();
		let response = retry::retry_provider_request(
			|| {
				let client = shared::http_client();
				let api_key = api_key.clone();
				let request = request.clone();
				let extra_headers = extra_headers.clone();
				let api_url = api_url.clone();
				Box::pin(async move {
					let req = client
						.post(&api_url)
						.bearer_auth(&api_key)
						.header("Content-Type", "application/json")
						.json(&request);

					let captured = super::streaming::send_and_read(
						req,
						request_timeout,
						stream_timeout,
						extra_headers.as_ref(),
						super::streaming::Format::Chat,
					)
					.await?;

					// Return Err for retryable HTTP errors so the retry loop catches them
					if retry::is_retryable_status(captured.status.as_u16()) {
						return Err(retry::retryable_status_error(
							&captured,
							format!("Moonshot AI API error {}: {}", captured.status, captured.body),
						));
					}

					Ok(captured)
				})
			},
			params.max_retries,
			params.retry_timeout,
			params.cancellation_token.as_ref(),
			params.api_key.as_ref(),
			params.retry_observer.as_ref(),
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

		let moonshot_response: MoonshotResponse = shared::parse_response(&response.body, params.api_key.as_ref())?;

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

			let (output_tokens, reasoning_tokens) = TokenUsage::split_output(usage.completion_tokens, reasoning_tokens);

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
		let reasoning_token_count = token_usage.as_ref().map(|u| u.reasoning_tokens).unwrap_or(0);
		let thinking = choice
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
							shared::redact_text(&call.tool_type, params.api_key.as_ref())
						);
						return None;
					}

					let arguments = shared::parse_tool_call_arguments_lossy(&call.function.arguments);

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
