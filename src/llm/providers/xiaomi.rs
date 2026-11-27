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

//! Xiaomi MiMo provider implementation.
//!
//! The same models are served over two wire protocols:
//!
//! - OpenAI-compatible Chat Completions, the default:
//!   `https://api.xiaomimimo.com/v1/chat/completions`
//! - Anthropic Messages: `https://api.xiaomimimo.com/anthropic/v1/messages`
//!   (<https://mimo.mi.com/static/docs/api/chat/anthropic-api-post.md>)
//!
//! `XIAOMI_API_MODE` (`anthropic` or `openai`) states the protocol for any
//! URL and overrides what the URL's path says; any other value is refused
//! before a request is sent. `XIAOMI_API_URL` is sent as written when it names
//! an endpoint (`…/messages`, `…/chat/completions`) or nothing specific. A
//! documented base names a host instead: `…/v1` and `…/anthropic` are how
//! Xiaomi's guides point at the pay-as-you-go host and at a Token Plan cluster
//! (`token-plan-{cn,sgp,ams}.xiaomimimo.com`), and the resolved protocol's
//! endpoint is built on it. With no URL, the pay-as-you-go endpoint of the
//! resolved protocol. The plan is never guessed from the key: a Token Plan
//! account sets its cluster's base in `XIAOMI_API_URL`.
//!
//! The chat catalogue is fixed and known (see `MODELS`), so unknown model IDs
//! are rejected. mimo-v2.5-pro and mimo-v2.5 were retired on 2026-10-21 and
//! their IDs no longer work upstream.
//!
//! Reasoning: no effort ladder — `thinking: {"type": "enabled" | "disabled"}`
//! is the only knob (on by default) on both protocols; any effort maps to
//! `enabled` and `ReasoningEffort::None` to `disabled`. A Chat Completions
//! tool-call turn must replay its `reasoning_content` or the API returns 400;
//! Messages recommends keeping the thinking blocks, which are replayed as
//! captured.
//!
//! Structured output: native `json_schema` with strict decoding (also on the
//! Token Plan endpoint), plus `json_object`, on Chat Completions. Messages
//! documents no response format, so a schema takes the forced-tool path there
//! and is validated locally.
//!
//! PRICING UPDATE: October 2026 (overseas USD rates)
//! Source: <https://mimo.mi.com/docs/en-US/price/pay-as-you-go>
//!
//! Configuration:
//! - `XIAOMI_API_KEY`: Required API key (`sk-` pay-as-you-go, `tp-`/`ttp-`
//!   Token Plan)
//! - `XIAOMI_API_URL`: Optional endpoint or base override (e.g. a Token Plan
//!   base URL with its `tp-` key)
//! - `XIAOMI_API_MODE`: Optional protocol override (`anthropic` or `openai`)

use crate::llm::providers::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images};
use crate::llm::providers::openai_compat::{chat_completion as openai_compat_chat_completion, OpenAiCompatConfig};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, ReasoningEffort};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use serde_json::json;
use std::env;

/// Xiaomi MiMo provider
#[derive(Debug, Clone, Default)]
pub struct XiaomiProvider;

impl XiaomiProvider {
	pub fn new() -> Self {
		Self
	}
}

pub(crate) const XIAOMI_API_KEY_ENV: &str = "XIAOMI_API_KEY";
const XIAOMI_API_MODE_ENV: &str = "XIAOMI_API_MODE";
const XIAOMI_API_URL_ENV: &str = "XIAOMI_API_URL";
/// The pay-as-you-go host. Each Token Plan cluster has its own
/// (`token-plan-{cn,sgp,ams}.xiaomimimo.com`) and is named in `XIAOMI_API_URL`.
const STANDARD_ORIGIN: &str = "https://api.xiaomimimo.com";

/// (model id, max input tokens) — every text model has a 1M context window
/// and 128K max output (models page, Oct 2026).
pub(crate) const MODELS: &[(&str, usize)] = &[
	("mimo-v2.6-pro-ultraspeed", 1_048_576),
	("mimo-v2.6-pro", 1_048_576),
	("mimo-v2.6-flash", 1_048_576),
];

// Xiaomi MiMo pricing (per 1M tokens in USD, overseas real-time API) - Oct 2026
// Source: https://mimo.mi.com/docs/en-US/price/pay-as-you-go
// Format: (model, input, output, cache_write, cache_read)
// cache_write priced at input rate (cache writes are free for a limited time
// and the API reports no write tokens)
const PRICING: &[PricingTuple] = &[
	// Ultraspeed bills 10x Pro and contains "mimo-v2.6-pro", so it must come first
	("mimo-v2.6-pro-ultraspeed", 4.35, 8.70, 4.35, 0.036),
	("mimo-v2.6-pro", 0.435, 0.87, 0.435, 0.0036),
	("mimo-v2.6-flash", 0.14, 0.28, 0.14, 0.0028),
];

/// Case-insensitive lookup; requests are canonicalized to the table's exact ID.
fn find_model(model: &str) -> Option<&'static (&'static str, usize)> {
	MODELS.iter().find(|(id, _)| id.eq_ignore_ascii_case(model))
}

/// The catalogue's price for a model.
fn pricing(model: &str) -> Option<ModelPricing> {
	let (input, output, cache_write, cache_read) = crate::llm::utils::get_model_pricing(model, PRICING)?;
	Some(ModelPricing::new(input, output, cache_write, cache_read))
}

/// A documented base URL: `…/v1` names a host's Chat Completions API and
/// `…/anthropic` its Messages API, on the pay-as-you-go host and on every
/// Token Plan cluster. Returns what precedes that last path segment.
fn base_prefix(api_url: &str) -> Option<(String, ApiMode)> {
	let (scheme, rest) = api_url.split_once("://")?;
	let authority_len = rest.find('/')?;
	let path = rest[authority_len..].trim_end_matches('/');
	let (prefix, last) = path.rsplit_once('/')?;
	let mode = match last {
		"v1" => ApiMode::OpenAi,
		"anthropic" => ApiMode::Anthropic,
		_ => return None,
	};
	Some((format!("{scheme}://{}{prefix}", &rest[..authority_len]), mode))
}

/// The endpoint a protocol is served on, below a host's base.
fn endpoint(base: &str, mode: ApiMode) -> String {
	match mode {
		ApiMode::Anthropic => format!("{base}/anthropic/v1/messages"),
		ApiMode::OpenAi => format!("{base}/v1/chat/completions"),
	}
}

/// The protocol and URL for one request: `XIAOMI_API_MODE`, then what the
/// URL names, then the Chat Completions default. An endpoint or an opaque URL
/// is sent as written; a documented base has the resolved protocol's endpoint
/// built on it.
fn select_endpoint(mode_env: Option<ApiMode>, api_url_env: Option<&str>) -> (ApiMode, String) {
	// An empty variable is an unset one.
	let api_url_env = api_url_env.filter(|api_url| !api_url.is_empty());
	let mode = mode_env
		.or_else(|| {
			api_url_env.and_then(|api_url| {
				anthropic_compat::url_api_mode(api_url).or_else(|| base_prefix(api_url).map(|(_, mode)| mode))
			})
		})
		.unwrap_or(ApiMode::OpenAi);
	let api_url = match api_url_env {
		Some(api_url) => base_prefix(api_url).map_or_else(|| api_url.to_owned(), |(base, _)| endpoint(&base, mode)),
		None => endpoint(STANDARD_ORIGIN, mode),
	};
	(mode, api_url)
}

/// The mode variable, parsed. An invalid value is refused before any request.
fn configured_mode_env() -> Result<Option<ApiMode>> {
	env::var(XIAOMI_API_MODE_ENV)
		.ok()
		.filter(|raw| !raw.is_empty())
		.map(|raw| anthropic_compat::parse_api_mode(XIAOMI_API_MODE_ENV, &raw))
		.transpose()
}

/// The protocol a request resolves to right now; capability methods follow it.
fn configured_mode() -> ApiMode {
	select_endpoint(
		configured_mode_env().ok().flatten(),
		env::var(XIAOMI_API_URL_ENV).ok().as_deref(),
	)
	.0
}

/// Xiaomi's prompt cache is automatic and its Messages contract has no
/// `cache_control` field, so a caller's breakpoints stay out of the request, as
/// they already do on Chat Completions. Read hits are still reported and billed.
fn drop_cache_breakpoints(params: &mut ChatCompletionParams) {
	for message in &mut params.messages {
		message.cached = false;
	}
	for tool in params.tools.iter_mut().flatten() {
		tool.cache_control = None;
	}
}

/// Xiaomi on the shared Messages layer. Its contract
/// (<https://mimo.mi.com/static/docs/api/chat/anthropic-api-post.md>) reads the
/// key from `Authorization: Bearer` and takes an image as a URL or inline data
/// and a video as a URL or inline data for every model in `MODELS`. It documents
/// no response format, so `enforces_response_schema` stays off: the flag is
/// declared from a live probe only, and none was possible here — Token Plan may
/// not be probed automatically. A schema takes the forced-tool path and is
/// validated locally.
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "xiaomi",
	images: |model| {
		if find_model(model).is_some() {
			Images::Base64AndUrl
		} else {
			Images::None
		}
	},
	videos: |model| find_model(model).is_some(),
	enforces_response_schema: false,
	// `thinking.type` is the same switch as on Chat Completions; on by default.
	thinking: |_, effort, request| {
		if let Some(effort) = effort {
			let kind = if matches!(effort, ReasoningEffort::None) {
				"disabled"
			} else {
				"enabled"
			};
			request["thinking"] = json!({ "type": kind });
		}
	},
	// Thinking is billed inside `output_tokens`, so the billable counter is used.
	cost: |model, usage| {
		pricing(model).map(|pricing| {
			pricing.calculate_cost(
				usage.input_tokens,
				usage.cache_write_tokens(),
				usage.cache_read_tokens,
				usage.output_tokens,
			)
		})
	},
	default_system: "",
};

#[async_trait::async_trait]
impl AiProvider for XiaomiProvider {
	fn name(&self) -> &str {
		"xiaomi"
	}

	fn supports_model(&self, model: &str) -> bool {
		find_model(model).is_some()
	}

	fn supports_vision(&self, _model: &str) -> bool {
		// Full-modal understanding: text, image, audio and video input
		true
	}

	fn supports_video(&self, _model: &str) -> bool {
		true
	}

	fn supports_caching(&self, _model: &str) -> bool {
		// Automatic prompt cache on both protocols; hits are billed at the
		// cache-hit rate and reported via `prompt_tokens_details.cached_tokens`
		// on Chat Completions, `cache_read_input_tokens` on Messages. No fixed TTL:
		// entries are evicted by LRU when storage fills
		// (https://mimo.mi.com/static/docs/news/latest/updqate.md). Measured
		// 2026-11-07 on mimo-v2.6-pro (Token Plan, Singapore): unique ~11k-token
		// prompts re-sent once after an idle delay hit in full on 30 of 32 calls,
		// with delays from 5s to 60m; one miss at 3m and one partial hit at 8m
		// whose twins hit. The cache outlives 60m of silence; the upper bound is
		// not measured.
		true
	}

	/// The idle gap measured to still hit on Chat Completions; see
	/// `supports_caching`. Messages has not been measured, so it claims nothing.
	fn cache_lifetime(&self, _model: &str, _use_long_cache: bool) -> Option<std::time::Duration> {
		(configured_mode() == ApiMode::OpenAi).then(|| crate::llm::config::CacheTTL::long().to_duration())
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		// Native on Chat Completions, the forced-tool path on Messages.
		true
	}

	/// `json_schema` decodes against the schema: with `strict: true` and a
	/// two-field object schema, a prompt demanding a JSON array got the object
	/// on 4 of 4 mimo-v2.6-flash/pro calls (live, 2026-10-26); the Token Plan
	/// endpoint behaves the same (2 of 2, live, 2026-10-27). Messages documents
	/// no response format and was not probed, so it reports false there.
	fn enforces_response_schema(&self, _model: &str) -> bool {
		configured_mode() == ApiMode::OpenAi
	}

	fn supports_required_tool_choice(&self, _model: &str) -> bool {
		// Only tool_choice "auto" is documented, on both protocols
		false
	}

	fn replays_thinking(&self, _model: &str) -> bool {
		// Assistant turns carry their thinking back on every request:
		// `reasoning_content` on Chat Completions, the captured blocks on Messages
		true
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		find_model(model).map(|(_, max)| *max).unwrap_or(1_048_576)
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(XIAOMI_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("xiaomi", XIAOMI_API_KEY_ENV).into())
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		pricing(model)
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		// The mode is validated here: an invalid value makes no request.
		let mode_env = configured_mode_env()?;
		let (mode, api_url) = select_endpoint(mode_env, env::var(XIAOMI_API_URL_ENV).ok().as_deref());

		if let Some((canonical, _)) = find_model(&params.model) {
			params.model = canonical.to_string();
		}
		let model = params.model.clone();

		if mode == ApiMode::Anthropic {
			drop_cache_breakpoints(&mut params);
			return anthropic_compat::chat_completion(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "xiaomi",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: true,
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		// MiMo does not return cost; derive it from the pricing table
		if let Some(ref mut usage) = response.exchange.usage {
			if usage.cost.is_none() {
				if let Some(pricing) = self.get_model_pricing(&model) {
					usage.cost = Some(pricing.calculate_cost(
						usage.input_tokens,
						usage.cache_write_tokens,
						usage.cache_read_tokens,
						usage.billable_output_tokens(),
					));
				}
			}
		}

		Ok(response)
	}
}

#[cfg(test)]
#[path = "xiaomi_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "xiaomi_anthropic_tests.rs"]
mod anthropic_tests;
