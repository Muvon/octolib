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

//! Tinker provider implementation (Thinking Machines).
//!
//! Requests go to Tinker's Anthropic Messages endpoint (beta) by default:
//! `https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1/messages`
//! (<https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/>).
//! It takes images on Inkling models, also inside tool results, and streams;
//! the OpenAI-compatible endpoint documents neither. `TINKER_API_MODE=openai`
//! keeps that endpoint:
//! `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions`
//!
//! Serves the Inkling family plus open-weight checkpoints (Nemotron, GLM,
//! Kimi, Qwen, GPT-OSS, DeepSeek) and any sampler checkpoint saved during
//! training (`tinker://...` paths). Model IDs contain colons
//! (`thinkingmachines/Inkling:peft:262144`); the factory splits only on the
//! first colon, so they pass through intact.
//! Short names `inkling` and `inkling-small` resolve to the SERVERLESS
//! INFERENCE ids (`...:peft:262144:sampling-nvfp4`), which is the tier this
//! endpoint serves — the bare `thinkingmachines/Inkling` spelling is the same
//! model in the TRAINING table, at a different rate and window.
//! Reasoning on Chat Completions: the server separates chain-of-thought into
//! `reasoning_content` by default (`separate_reasoning` defaults to true since
//! June 2026), which the shared compat layer already parses. `reasoning_effort`
//! accepts the full OpenAI ladder including "xhigh" (0.99).
//!
//! The endpoints are a beta meant for testing and low-traffic use, not
//! production. Structured output is documented on neither, so it is reported
//! as unsupported. Prompt caching is automatic; see `supports_caching`.
//!
//! Sources: <https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/anthropic/>,
//! <https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/>
//! Pricing: <https://tinker-docs.thinkingmachines.ai/tinker/models/>
//!
//! Configuration:
//! - `TINKER_API_KEY`: Required API key
//! - `TINKER_API_URL`: Optional endpoint override, sent as written
//! - `TINKER_API_MODE`: `anthropic` or `openai`, the protocol `TINKER_API_URL`
//!   speaks; without it the URL's path decides

use super::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images, MessagesUsage};
use super::shared::CacheMarker;
use crate::llm::providers::openai_compat::{chat_completion as openai_compat_chat_completion, OpenAiCompatConfig};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, ReasoningEffort};
use crate::llm::utils::{contains_ignore_ascii_case, get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Tinker provider
#[derive(Debug, Clone, Default)]
pub struct TinkerProvider;

impl TinkerProvider {
	pub fn new() -> Self {
		Self
	}
}

pub(crate) const TINKER_API_KEY_ENV: &str = "TINKER_API_KEY";
const TINKER_API_URL_ENV: &str = "TINKER_API_URL";
const TINKER_API_MODE_ENV: &str = "TINKER_API_MODE";
/// Messages endpoint, the default.
const TINKER_MESSAGES_API_URL: &str =
	"https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1/messages";
/// Chat Completions endpoint, for `TINKER_API_MODE=openai`.
const TINKER_CHAT_API_URL: &str =
	"https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions";

/// The protocol and URL for one request. A supplied URL is sent as it is,
/// whatever the protocol: it is never rewritten. An explicit mode comes first,
/// then the URL's own path; a URL naming neither endpoint was always served by
/// Chat Completions, and with no URL the default is Messages.
fn select_endpoint(mode: Option<&str>, api_url: Option<&str>) -> Result<(ApiMode, String)> {
	// An empty variable is an unset one.
	let mode = mode
		.filter(|raw| !raw.is_empty())
		.map(|raw| anthropic_compat::parse_api_mode(TINKER_API_MODE_ENV, raw))
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
				ApiMode::Anthropic => TINKER_MESSAGES_API_URL,
				ApiMode::OpenAi => TINKER_CHAT_API_URL,
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
		env::var(TINKER_API_MODE_ENV).ok().as_deref(),
		env::var(TINKER_API_URL_ENV).ok().as_deref(),
	)
	.map(|(mode, _)| mode)
	.unwrap_or(ApiMode::Anthropic)
}

/// The Inkling family, the only models the Messages contract takes images for
/// ("on Inkling models only"). Live on 2026-11-28, Inkling and Inkling-Small read
/// a base64 PNG in a user turn and inside a `tool_result`; gpt-oss-120b answered
/// 400 "Image input is not supported for this model", and a URL source is not
/// supported.
fn is_inkling(model: &str) -> bool {
	contains_ignore_ascii_case(model, "inkling")
}

/// Thinking knobs on Messages, probed live on 2026-11-28. Inkling and
/// Inkling-Small took every `output_config.effort` level and
/// `thinking: {"type": "disabled"}`; `max` goes as `xhigh`, which the contract
/// says it behaves as. gpt-oss-120b took `low|medium|high` and answered 400 to
/// `xhigh`, `max` and disabled thinking ("This model accepts one of: low, medium,
/// high"), so it clamps to that ladder and an off request sends nothing. Every
/// other model sends nothing: the contract says "Not all models support effort"
/// and answers 400 to those, Nemotron takes a boolean on Chat Completions, and
/// this account could not sample the rest on Messages to probe them.
fn messages_thinking(model: &str, effort: Option<ReasoningEffort>, request: &mut serde_json::Value) {
	let Some(effort) = effort else {
		return;
	};
	let value = if is_inkling(model) {
		match effort {
			ReasoningEffort::None => {
				request["thinking"] = serde_json::json!({ "type": "disabled" });
				return;
			}
			ReasoningEffort::Low => "low",
			ReasoningEffort::Medium => "medium",
			ReasoningEffort::High => "high",
			ReasoningEffort::XHigh | ReasoningEffort::Max => "xhigh",
		}
	} else if contains_ignore_ascii_case(model, "gpt-oss") {
		match effort {
			ReasoningEffort::None => return,
			ReasoningEffort::Low => "low",
			ReasoningEffort::Medium => "medium",
			ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max => "high",
		}
	} else {
		return;
	};
	// The schema, when one was requested, is already in `output_config`.
	request["output_config"]["effort"] = serde_json::json!(value);
}

/// Cost of one Messages response, in USD, from the same table as the Chat
/// route. Live on 2026-11-28 the endpoint reported every uncached prompt token
/// as `cache_creation_input_tokens` and `input_tokens` as 0; Tinker has no
/// cache-write fee (cache_write = input in `PRICING`), so either counter bills
/// at the input rate. `output_tokens` includes thinking.
fn messages_cost(model: &str, usage: &MessagesUsage) -> Option<f64> {
	tinker_model_pricing(model).map(|pricing| {
		pricing.calculate_cost(
			usage.input_tokens,
			usage.cache_write_tokens(),
			usage.cache_read_tokens,
			usage.output_tokens,
		)
	})
}

/// Tinker on the shared Messages layer. Live on 2026-11-28 (Inkling,
/// Inkling-Small, gpt-oss-120b): text, tool calls, a replayed tool turn with
/// its thinking block (signature empty), SSE streaming and prefix-cache reads
/// all worked. Video has no block here. Schema: undocumented, so it takes the
/// forced-tool path and is validated locally.
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "tinker",
	images: |model| {
		if is_inkling(model) {
			Images::Base64
		} else {
			Images::None
		}
	},
	videos: |_| false,
	enforces_response_schema: false,
	thinking: messages_thinking,
	cost: messages_cost,
	default_system: "",
	requires_max_tokens: true,
	cache_marker: Some(CacheMarker::Ephemeral),
};

/// (model, input, output, cache_write, cache_read) per 1M tokens, from
/// <https://tinker-docs.thinkingmachines.ai/tinker/models/> (Sep 2026,
/// limited-time 50% discount prices — re-verify when the promotion ends).
/// Cached input is Tinker's prefill-cache read price; there is no separate
/// cache-write fee, so cache_write = input.
///
/// First match wins on substring matching, so longer IDs come first:
/// serverless (`:sampling-nvfp4`) before `:peft:` variants before base IDs,
/// and Inkling-2-Small and Inkling-Small before Inkling (the base Inkling ID is
/// a substring of every other Inkling ID).
const PRICING: &[PricingTuple] = &[
	// Serverless inference (beta)
	// Inkling 2 Small lists $0.30 / $0.03 cached / $1.00 on all three of its ids (2026-11-07).
	(
		"thinkingmachines/inkling-2-small:peft:262144:sampling-nvfp4",
		0.30,
		1.00,
		0.30,
		0.03,
	),
	(
		"thinkingmachines/inkling-small:peft:262144:sampling-nvfp4",
		0.30,
		1.20,
		0.30,
		0.06,
	),
	(
		"thinkingmachines/inkling:peft:262144:sampling-nvfp4",
		1.00,
		4.05,
		1.00,
		0.17,
	),
	// Inkling family — 256K extended-context variants
	// Inkling 2 Mini lists $0.15 / $0.015 cached / $0.50 on both its ids (2026-11-24).
	// Neither Inkling 2 model is on Tinker's serverless tier (Inkling and Inkling-Small
	// only): their `:sampling-nvfp4` ids answer 400 "Sampling is not supported".
	("thinkingmachines/inkling-2-mini:peft:262144", 0.15, 0.50, 0.15, 0.015),
	("thinkingmachines/inkling-2-small:peft:262144", 0.30, 1.00, 0.30, 0.03),
	("thinkingmachines/inkling-small:peft:262144", 1.16, 2.89, 1.16, 0.232),
	("thinkingmachines/inkling:peft:262144", 3.74, 9.36, 3.74, 0.748),
	("thinkingmachines/inkling-2-mini", 0.15, 0.50, 0.15, 0.015),
	("thinkingmachines/inkling-2-small", 0.30, 1.00, 0.30, 0.03),
	("thinkingmachines/inkling-small", 0.58, 1.44, 0.58, 0.116),
	("thinkingmachines/inkling", 1.87, 4.68, 1.87, 0.374),
	// NVIDIA Nemotron
	(
		"nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16:peft:262144",
		0.26,
		0.66,
		0.26,
		0.052,
	),
	(
		"nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16",
		0.195,
		0.495,
		0.195,
		0.039,
	),
	(
		"nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16:peft:262144",
		3.32,
		8.30,
		3.32,
		0.664,
	),
	(
		"nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16",
		2.49,
		6.225,
		2.49,
		0.498,
	),
	// Super-VL: the `:peft:` variant widens the context, not the price.
	(
		"nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-bf16:peft:262144",
		0.57,
		1.44,
		0.57,
		0.114,
	),
	(
		"nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-bf16",
		0.57,
		1.44,
		0.57,
		0.114,
	),
	(
		"nvidia/nvidia-nemotron-3-super-120b-a12b-bf16:peft:262144",
		0.76,
		1.92,
		0.76,
		0.152,
	),
	("nvidia/nvidia-nemotron-3-super-120b-a12b-bf16", 0.57, 1.44, 0.57, 0.114),
	("nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16", 0.195, 0.495, 0.195, 0.039),
	// GLM / Kimi
	("zai-org/glm-5.3-flash:peft:262144", 0.84, 2.11, 0.84, 0.168),
	("zai-org/glm-5.3:peft:262144", 4.86, 12.15, 4.86, 0.972),
	("moonshotai/kimi-k2.6:peft:131072", 5.15, 12.81, 5.15, 1.03),
	("moonshotai/kimi-k2.6", 2.205, 5.49, 2.205, 0.441),
	// Qwen
	("qwen/qwen3.8-27b:peft:262144", 2.48, 7.46, 2.48, 0.496),
	("qwen/qwen3.8-27b", 1.86, 5.595, 1.86, 0.372),
	("qwen/qwen3.6-35b-a3b", 0.54, 1.335, 0.54, 0.108),
	("qwen/qwen3.5-397b-a17b:peft:262144", 4.00, 10.00, 4.00, 0.80),
	("qwen/qwen3.5-397b-a17b", 3.00, 7.50, 3.00, 0.60),
	("qwen/qwen3.5-35b-a3b-base", 0.54, 1.335, 0.54, 0.108),
	("qwen/qwen3.5-9b-base", 0.66, 1.995, 0.66, 0.132),
	("qwen/qwen3.5-9b", 0.66, 1.995, 0.66, 0.132),
	("qwen/qwen3.5-4b", 0.33, 1.005, 0.33, 0.066),
	("qwen/qwen3-8b", 0.195, 0.60, 0.195, 0.039),
	// OpenAI GPT-OSS
	("openai/gpt-oss-120b:peft:131072", 0.78, 1.94, 0.78, 0.156),
	("openai/gpt-oss-120b", 0.33, 0.84, 0.33, 0.066),
	("openai/gpt-oss-20b", 0.18, 0.45, 0.18, 0.036),
	// DeepSeek
	("deepseek-ai/deepseek-v4.1-flash:peft:262144", 0.60, 3.60, 0.60, 0.12),
	(
		"deepseek-ai/deepseek-v4.1-flash-1017:peft:262144",
		0.60,
		3.60,
		0.60,
		0.12,
	),
	("deepseek-ai/deepseek-v3.1", 1.695, 4.215, 1.695, 0.339),
];

/// (model id, context window) from the same pricing page. Same
/// specific-first ordering as `PRICING`.
const CONTEXTS: &[(&str, usize)] = &[
	("thinkingmachines/inkling-2-small:peft:262144:sampling-nvfp4", 262_144),
	("thinkingmachines/inkling-small:peft:262144:sampling-nvfp4", 262_144),
	("thinkingmachines/inkling:peft:262144:sampling-nvfp4", 262_144),
	("thinkingmachines/inkling-2-mini:peft:262144", 262_144),
	("thinkingmachines/inkling-2-small:peft:262144", 262_144),
	("thinkingmachines/inkling-small:peft:262144", 262_144),
	("thinkingmachines/inkling:peft:262144", 262_144),
	("thinkingmachines/inkling-2-mini", 65_536),
	("thinkingmachines/inkling-2-small", 65_536),
	("thinkingmachines/inkling-small", 65_536),
	("thinkingmachines/inkling", 65_536),
	("nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16:peft:262144", 262_144),
	("nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16", 65_536),
	("nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16:peft:262144", 262_144),
	("nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16", 65_536),
	("nvidia/nvidia-nemotron-3-super-120b-a12b-bf16:peft:262144", 262_144),
	("nvidia/nvidia-nemotron-3-super-120b-a12b-bf16", 65_536),
	("nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16", 65_536),
	(
		"nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-bf16:peft:262144",
		262_144,
	),
	("nvidia/nvidia-nemotron-3.5-super-vl-120b-a12b-bf16", 65_536),
	("zai-org/glm-5.3-flash:peft:262144", 262_144),
	("zai-org/glm-5.3:peft:262144", 262_144),
	("moonshotai/kimi-k2.6:peft:131072", 131_072),
	("moonshotai/kimi-k2.6", 32_768),
	("qwen/qwen3.8-27b:peft:262144", 262_144),
	("qwen/qwen3.8-27b", 65_536),
	("qwen/qwen3.6-35b-a3b", 65_536),
	("qwen/qwen3.5-397b-a17b:peft:262144", 262_144),
	("qwen/qwen3.5-397b-a17b", 65_536),
	("qwen/qwen3.5-35b-a3b-base", 65_536),
	("qwen/qwen3.5-9b-base", 65_536),
	("qwen/qwen3.5-9b", 65_536),
	("qwen/qwen3.5-4b", 65_536),
	("qwen/qwen3-8b", 32_768),
	("openai/gpt-oss-120b:peft:131072", 131_072),
	("openai/gpt-oss-120b", 32_768),
	("openai/gpt-oss-20b", 32_768),
	("deepseek-ai/deepseek-v4.1-flash:peft:262144", 262_144),
	("deepseek-ai/deepseek-v4.1-flash-1017:peft:262144", 262_144),
	("deepseek-ai/deepseek-v3.1", 32_768),
];

/// The short names resolve to the SERVERLESS INFERENCE ids, not the bare base
/// ids. This provider only ever does inference — both of its endpoints are the
/// serverless offering —
/// and the models page lists a different id per table for the same model name:
/// "Inkling" is `thinkingmachines/Inkling:peft:262144:sampling-nvfp4` under
/// Serverless Inference and `thinkingmachines/Inkling` under Training. The bare
/// id therefore carried the TRAINING tier's rate and window into every inference
/// call made by short name — 1.87/4.68 and 64K instead of 1.00/4.05 and 256K,
/// i.e. an 87% over-bill and a context window quoted at a quarter of the truth.
/// Both bare ids stay priced in the tables above: a caller may still name one
/// explicitly, and that spelling means the training tier.
fn resolve_model(model: &str) -> &str {
	if model.eq_ignore_ascii_case("inkling") {
		"thinkingmachines/Inkling:peft:262144:sampling-nvfp4"
	} else if model.eq_ignore_ascii_case("inkling-small") {
		"thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4"
	} else {
		// Preserve fully qualified IDs and case-sensitive checkpoint paths.
		model
	}
}

fn tinker_model_pricing(model: &str) -> Option<ModelPricing> {
	let (input, output, cache_write, cache_read) = get_model_pricing(resolve_model(model), PRICING)?;
	Some(ModelPricing::new(input, output, cache_write, cache_read))
}

fn tinker_model_context(model: &str) -> Option<usize> {
	let normalized = normalize_model_name(resolve_model(model));
	CONTEXTS
		.iter()
		.find(|(name, _)| normalized.contains(&normalize_model_name(name)))
		.map(|(_, context)| *context)
}

#[async_trait::async_trait]
impl AiProvider for TinkerProvider {
	fn name(&self) -> &str {
		"tinker"
	}

	fn supports_model(&self, model: &str) -> bool {
		// Any non-empty ID: a known Tinker model or a sampler checkpoint
		// path (`tinker://...`) saved during training.
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(TINKER_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("tinker", TINKER_API_KEY_ENV).into())
	}

	/// The docs only say "We provide a discount on cached prefill tokens". Measured
	/// 2026-11-07: every model this account can sample reported cached tokens on a
	/// repeat call (Inkling, Inkling-Small, Inkling-2-Small, the Inkling serverless id,
	/// Nemotron-3-Super, gpt-oss-120b and gpt-oss-20b), so the cache is the server's,
	/// not a per-model feature.
	fn supports_caching(&self, _model: &str) -> bool {
		true
	}

	/// Measured on both routes: unique 3.5k-8k-token prompts re-sent once after an
	/// idle delay, two samples per delay, hit on every sample from 1m to 60m. Chat
	/// Completions on 2026-11-07 (gpt-oss-120b, gpt-oss-20b, `inkling`, the serverless
	/// id, and Inkling-2-Small); Messages on 2026-11-28 (gpt-oss-120b and `inkling`).
	/// The upper bound is not measured.
	fn cache_lifetime(&self, _model: &str, _use_long_cache: bool) -> Option<std::time::Duration> {
		Some(crate::llm::config::CacheTTL::long().to_duration())
	}

	/// Measured on Messages on 2026-11-28: sending a turn back with its thinking
	/// block raised the prompt on Inkling (472 tokens against 189 without it),
	/// while gpt-oss-120b counted 359 either way. Chat Completions was not
	/// measured.
	fn replays_thinking(&self, model: &str) -> bool {
		configured_mode() == ApiMode::Anthropic && is_inkling(model)
	}

	/// Images reach Inkling models on Messages only: the OpenAI-compatible
	/// endpoint documents no image input (only the SDK renderers handle images).
	fn supports_vision(&self, model: &str) -> bool {
		configured_mode() == ApiMode::Anthropic && is_inkling(model)
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		// response_format is not documented for this endpoint.
		false
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		tinker_model_pricing(model)
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		tinker_model_context(model).unwrap_or(262_144)
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let (mode, api_url) = select_endpoint(
			env::var(TINKER_API_MODE_ENV).ok().as_deref(),
			env::var(TINKER_API_URL_ENV).ok().as_deref(),
		)?;
		params.model = resolve_model(&params.model).to_string();
		if mode == ApiMode::Anthropic {
			// The Messages contract names `x-api-key`.
			return anthropic_compat::chat_completion_with_x_api_key(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "tinker",
				usage_fallback_cost: None,
				use_response_cost: true,
				enforces_response_schema: false,
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
				cache_marker: None,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		// Tinker does not report cost in the response — derive it from the
		// Tinker pricing table. Unknown checkpoint paths have no entry, so
		// their cost stays None rather than guessing a model family.
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
#[path = "tinker_tests.rs"]
mod tests;
