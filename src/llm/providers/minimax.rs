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

//! MiniMax provider implementation. Requests go to the Anthropic-compatible
//! Messages endpoint (<https://platform.minimax.io/docs/api-reference/text-anthropic-api>)
//! through the shared `anthropic_compat` layer.
//!
//! `MINIMAX_API_URL` names another endpoint and is sent as it is: one whose
//! path ends in `/messages` speaks Messages, any other did so before and
//! still does. `MINIMAX_API_MODE=anthropic` or `openai` states the protocol
//! for any URL and, with no URL, selects that protocol's default endpoint.

use super::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images, MessagesUsage};
use super::openai_compat::{chat_completion as openai_compat_chat_completion, OpenAiCompatConfig};
use super::shared::{self, CacheMarker};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse, ReasoningEffort};
use crate::llm::utils::{get_model_pricing, is_model_in_pricing_table, normalize_model_name, PricingTuple};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::env;

/// MiniMax pricing constants (per 1M tokens in USD)
/// Source: https://platform.minimax.io/docs/guides/pricing-paygo (verified Sep 22, 2026)
/// Format: (model, input, output, cache_write, cache_read)
pub(crate) const PRICING: &[PricingTuple] = &[
	// MiniMax M3.1 Flash Preview: Token Plan / MiniMax Code only, no pay-as-you-go
	// rate published (verified Oct 5, 2026), so requests carry no per-token cost.
	// Superseded by the GA ID below; must precede it (substring match, first wins).
	("MiniMax-M3.1-Flash-Preview", 0.0, 0.0, 0.0, 0.0),
	// MiniMax M3.1 Flash (GA Oct 19, 2026; single tier, no cache-write column)
	// Source: https://platform.minimax.io/docs/guides/pricing (verified Oct 20, 2026)
	// Must precede MiniMax-M3 (substring match, first wins).
	("MiniMax-M3.1-Flash", 0.15, 0.60, 0.0, 0.015),
	// MiniMax M3 (latest generation, natively multimodal — image + video input)
	// ≤512k input tokens: 0.30 / 1.20 / cache read 0.06; >512k input tokens:
	// 0.60 / 2.40 / cache read 0.12 (tier applied in `calculate_cost_with_cache`).
	// Cache writes are free (no cache-write column in official pricing).
	("MiniMax-M3", 0.30, 1.20, 0.0, 0.06),
	// MiniMax M2.7
	("MiniMax-M2.7-highspeed", 0.60, 2.40, 0.375, 0.06),
	("MiniMax-M2.7", 0.30, 1.20, 0.375, 0.06),
	// Legacy models (still listed and priced on the official page)
	("MiniMax-M2.5-highspeed", 0.60, 2.40, 0.375, 0.03),
	("MiniMax-M2.5", 0.30, 1.20, 0.375, 0.03),
	// highspeed must precede the base M2.1 entry (substring match, first wins)
	("MiniMax-M2.1-highspeed", 0.60, 2.40, 0.375, 0.03),
	("MiniMax-M2.1", 0.30, 1.20, 0.375, 0.03),
	("MiniMax-M2", 0.30, 1.20, 0.375, 0.03),
];

/// MiniMax M3: the permanent 50% discount applies only to ≤512k input tokens.
/// Above 512k, the standard (2x) rate applies to input, output, and cache read.
const M3_DISCOUNT_MAX_INPUT: u64 = 512_000;

fn is_m3_model(model: &str) -> bool {
	normalize_model_name(model) == "minimax-m3"
}

/// MiniMax-M3.1-Flash (and its Preview ID) takes `output_config.effort` (defaults to
/// `max` when omitted). Thinking cannot be disabled, so an explicit `None` runs at `low`.
fn effort_value(model: &str, effort: ReasoningEffort) -> Option<&'static str> {
	if !matches!(
		normalize_model_name(model).as_str(),
		"minimax-m3.1-flash" | "minimax-m3.1-flash-preview"
	) {
		return None;
	}
	Some(match effort {
		ReasoningEffort::None | ReasoningEffort::Low => "low",
		ReasoningEffort::Medium => "medium",
		ReasoningEffort::High => "high",
		ReasoningEffort::XHigh => "xhigh",
		ReasoningEffort::Max => "max",
	})
}

/// MiniMax-M3 thinks only when `thinking: {"type": "adaptive"}` is sent (off when
/// omitted) and documents no `output_config.effort`, so any level except `None`
/// enables it (platform.minimax.io/docs/api-reference/text-anthropic-api, verified
/// Oct 2026).
fn thinking_value(model: &str, effort: ReasoningEffort) -> Option<serde_json::Value> {
	(is_m3_model(model) && effort != ReasoningEffort::None).then(|| serde_json::json!({ "type": "adaptive" }))
}

/// Thinking knobs for a requested effort on the Chat Completions endpoint
/// (platform.minimax.io/docs/api-reference/text-chat-openai, read 2026-11-26).
/// With no effort requested nothing is sent: the API thinks by default. M3.1
/// Flash takes `reasoning_effort` (`none` is a 400 there, so it runs at
/// `low`); M3 takes `thinking: {"type": "disabled"}` to skip thinking and its
/// `adaptive` default otherwise; M2.x takes neither knob.
pub(crate) fn openai_thinking(model: &str, effort: Option<ReasoningEffort>, request: &mut Value) {
	let Some(effort) = effort else {
		return;
	};
	if let Some(effort) = effort_value(model, effort) {
		request["reasoning_effort"] = serde_json::json!(effort);
	}
	if is_m3_model(model) && effort == ReasoningEffort::None {
		request["thinking"] = serde_json::json!({ "type": "disabled" });
	}
}

/// Thinking knobs for a requested effort on the shared Messages layer.
fn configure_thinking(model: &str, effort: Option<ReasoningEffort>, request: &mut Value) {
	let Some(effort) = effort else {
		return;
	};
	if let Some(effort) = effort_value(model, effort) {
		request["output_config"] = serde_json::json!({ "effort": effort });
	}
	if let Some(thinking) = thinking_value(model, effort) {
		request["thinking"] = thinking;
	}
}

/// Cost of one Messages response, in USD.
fn messages_cost(model: &str, usage: &MessagesUsage) -> Option<f64> {
	calculate_minimax_cost(
		model,
		usage.input_tokens,
		usage.output_tokens,
		usage.cache_write_tokens(),
		usage.cache_read_tokens,
	)
}

/// Token usage breakdown for cache-aware pricing
struct CacheTokenUsage {
	regular_input_tokens: u64,
	cache_creation_tokens: u64,
	cache_read_tokens: u64,
	output_tokens: u64,
}

/// Calculate cost for MiniMax models with cache-aware pricing (case-insensitive)
/// - cache_creation_tokens: charged at 1.25x the model's input price
/// - cache_read_tokens: charged at 0.1x the model's input price
/// - regular_input_tokens: charged at normal price
/// - output_tokens: charged at normal price
fn calculate_cost_with_cache(model: &str, usage: CacheTokenUsage) -> Option<f64> {
	let (mut input_price, mut output_price, cache_write_price, mut cache_read_price) =
		get_model_pricing(model, PRICING)?;

	// MiniMax M3: the 50% discount applies only to ≤512k input tokens.
	// Above that, the standard (2x) rate applies to input, output, and cache read.
	if is_m3_model(model) {
		let total_input = usage
			.regular_input_tokens
			.saturating_add(usage.cache_creation_tokens)
			.saturating_add(usage.cache_read_tokens);
		if total_input > M3_DISCOUNT_MAX_INPUT {
			input_price *= 2.0;
			output_price *= 2.0;
			cache_read_price *= 2.0;
		}
	}

	// Regular input tokens at normal price
	let regular_input_cost = (usage.regular_input_tokens as f64 / 1_000_000.0) * input_price;

	// Cache creation tokens at cache_write_price
	let cache_creation_cost = (usage.cache_creation_tokens as f64 / 1_000_000.0) * cache_write_price;

	// Cache read tokens at cache_read_price
	let cache_read_cost = (usage.cache_read_tokens as f64 / 1_000_000.0) * cache_read_price;

	// Output tokens at normal price
	let output_cost = (usage.output_tokens as f64 / 1_000_000.0) * output_price;

	Some(regular_input_cost + cache_creation_cost + cache_read_cost + output_cost)
}

/// Cost from the API's token counters. `input_tokens` is already clean: it
/// holds neither cache reads nor cache writes.
fn calculate_minimax_cost(
	model: &str,
	input_tokens: u64,
	output_tokens: u64,
	cache_creation_tokens: u64,
	cache_read_tokens: u64,
) -> Option<f64> {
	calculate_cost_with_cache(
		model,
		CacheTokenUsage {
			regular_input_tokens: input_tokens,
			cache_creation_tokens,
			cache_read_tokens,
			output_tokens,
		},
	)
}

#[derive(Debug, Clone, Default)]
pub struct MinimaxProvider;

impl MinimaxProvider {
	pub fn new() -> Self {
		Self
	}
}

// Constants
pub(crate) const MINIMAX_API_KEY_ENV: &str = "MINIMAX_API_KEY";
const MINIMAX_API_URL_ENV: &str = "MINIMAX_API_URL";
const MINIMAX_API_URL: &str = "https://api.minimax.io/anthropic/v1/messages";

/// MiniMax on the shared Messages layer. The endpoint ignores `response_format`,
/// so a schema takes the forced-tool path and is validated locally.
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "minimax",
	// MiniMax-M3 (including M3.1-Flash) is natively multimodal; M2.x takes text only.
	images: |model| {
		if normalize_model_name(model).contains("minimax-m3") {
			Images::Base64AndUrl
		} else {
			Images::None
		}
	},
	videos: |model| normalize_model_name(model).contains("minimax-m3"),
	enforces_response_schema: false,
	thinking: configure_thinking,
	cost: messages_cost,
	// MiniMax's own client sends this when a conversation has no system turn.
	default_system: "You are a helpful assistant.",
	requires_max_tokens: false,
	cache_marker: Some(CacheMarker::Ephemeral),
};

/// Chat Completions endpoint for `MINIMAX_API_MODE=openai`
/// (platform.minimax.io/docs/api-reference/text-chat-openai, read 2026-11-26).
const MINIMAX_CHAT_API_URL: &str = "https://api.minimax.io/v1/chat/completions";
const MINIMAX_API_MODE_ENV: &str = "MINIMAX_API_MODE";

/// MiniMax on the shared Chat Completions layer. The endpoint documents no
/// `response_format`, so a schema takes the forced-tool path, as on Messages.
const OPENAI_COMPAT: OpenAiCompatConfig = OpenAiCompatConfig {
	provider_name: "minimax",
	usage_fallback_cost: None,
	// The API returns no cost: the PRICING table bills every response.
	use_response_cost: false,
	enforces_response_schema: false,
	supports_required_tool_choice: false,
	// The endpoint documents neither `tool_choice` nor `parallel_tool_calls`.
	supports_parallel_tool_calls: false,
	cache_marker: None,
};

/// The protocol and URL for one request. A supplied URL is sent as it is,
/// whatever the protocol: it is never rewritten. An explicit mode comes
/// first, then the URL's own path; a URL naming neither endpoint was always
/// served by Messages, which is also the default.
fn select_endpoint(mode: Option<&str>, api_url: Option<&str>) -> Result<(ApiMode, String)> {
	// An empty variable is an unset one.
	let mode = mode
		.filter(|raw| !raw.is_empty())
		.map(|raw| anthropic_compat::parse_api_mode(MINIMAX_API_MODE_ENV, raw))
		.transpose()?;
	let api_url = api_url.filter(|api_url| !api_url.is_empty());
	let mode = mode
		.or_else(|| api_url.and_then(anthropic_compat::url_api_mode))
		.unwrap_or(ApiMode::Anthropic);
	let api_url = api_url.map_or_else(
		|| {
			match mode {
				ApiMode::Anthropic => MINIMAX_API_URL,
				ApiMode::OpenAi => MINIMAX_CHAT_API_URL,
			}
			.to_owned()
		},
		str::to_owned,
	);
	Ok((mode, api_url))
}

#[async_trait]
impl AiProvider for MinimaxProvider {
	fn name(&self) -> &str {
		"minimax"
	}
	fn supports_model(&self, model: &str) -> bool {
		// MiniMax models - check against pricing table (strict)
		is_model_in_pricing_table(model, PRICING)
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(MINIMAX_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("minimax", MINIMAX_API_KEY_ENV).into())
	}

	/// Automatic prefix cache for prompts of 512+ tokens; the docs state no TTL.
	/// Measured 2026-11-07 on MiniMax-M3 (default Anthropic endpoint): unique
	/// ~6.9k-token prompts re-sent once after an idle delay hit on only 8 of 14
	/// calls, misses from 5s on and hits up to 6m, so hits depend on routing and
	/// no expiry time can be read off. Do not count on a hit after any gap.
	fn supports_caching(&self, _model: &str) -> bool {
		true
	}

	fn supports_vision(&self, model: &str) -> bool {
		// Only MiniMax-M3 is natively multimodal; earlier M2.x models are text-only
		normalize_model_name(model).contains("minimax-m3")
	}

	fn supports_video(&self, model: &str) -> bool {
		// MiniMax-M3 accepts native video input on the Anthropic-compatible endpoint
		normalize_model_name(model).contains("minimax-m3")
	}

	/// Thinking blocks the model returned go back unchanged on the next turn:
	/// MiniMax requires the complete assistant content in multi-turn tool
	/// conversations, and it is context the model reads and bills.
	fn replays_thinking(&self, _model: &str) -> bool {
		true
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		false // MiniMax uses Anthropic-compat endpoint which ignores response_format
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		let (input_price, output_price, cache_write_price, cache_read_price) = get_model_pricing(model, PRICING)?;

		Some(crate::llm::types::ModelPricing::new(
			input_price,
			output_price,
			cache_write_price,
			cache_read_price,
		))
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		// MiniMax model context window limits (case-insensitive), per the
		// Anthropic API model table (verified Oct 2, 2026)
		let model_lower = normalize_model_name(model);
		if model_lower.contains("minimax-m3") {
			1_000_000 // 1M context window
		} else if model_lower.contains("minimax-m2") {
			204_800 // M2, M2.1, M2.5, M2.7 and their highspeed variants
		} else {
			128_000 // Default fallback
		}
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = shared::resolve_api_key(self, &params)?;
		let (mode, api_url) = select_endpoint(
			env::var(MINIMAX_API_MODE_ENV).ok().as_deref(),
			env::var(MINIMAX_API_URL_ENV).ok().as_deref(),
		)?;
		if mode == ApiMode::Anthropic {
			return anthropic_compat::chat_completion(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}
		let model = params.model.clone();
		let mut response = openai_compat_chat_completion(OPENAI_COMPAT, api_key, api_url, params).await?;
		let usage = response.exchange.usage.as_mut().filter(|usage| usage.cost.is_none());
		if let Some(usage) = usage {
			usage.cost = calculate_minimax_cost(
				&model,
				usage.input_tokens,
				usage.billable_output_tokens(),
				usage.cache_write_tokens,
				usage.cache_read_tokens,
			);
		}
		Ok(response)
	}
}
#[cfg(test)]
#[path = "minimax_tests.rs"]
mod tests;
