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

//! Alibaba Cloud Model Studio (DashScope) provider implementation.
//!
//! With no configuration, requests go to the OpenAI-compatible endpoint at
//! `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`.
//!
//! Messages is available through the shared `anthropic_compat` layer. Model
//! Studio has no single public Messages host: the documented pay-as-you-go base
//! is per workspace
//! (`https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages`,
//! <https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages>),
//! and Token Plan and Coding Plan have their own `/apps/anthropic/v1/messages`
//! bases. Two configurations are supported:
//!
//! - `ALIBABA_WORKSPACE_ID` (one DNS label, validated before it enters the URL)
//!   selects the official regional pay-as-you-go Messages base, and Messages
//!   then becomes the default for that account.
//! - `ALIBABA_API_URL` names the full endpoint — the form a Token Plan, Coding
//!   Plan, mainland-China or dedicated deployment needs. It is sent exactly as
//!   configured and is never redirected, and its path decides the protocol
//!   unless `ALIBABA_API_MODE` overrides it.
//!
//! With neither set, the safe Chat Completions default stands and
//! `ALIBABA_API_MODE=anthropic` is refused before a request is sent. Nothing is
//! derived from the key, the account or the region, and the legacy generic
//! `dashscope-intl` `/api/v2/apps/claude-code-proxy` path is not substituted.
//!
//! Hosts the Qwen family plus third-party models (DeepSeek, GLM, Kimi) at
//! Alibaba's own rates.
//!
//! PRICING UPDATE: September 2026
//! Source: <https://www.alibabacloud.com/help/en/model-studio/model-pricing>
//!
//! Configuration:
//! - `ALIBABA_API_KEY`: Required API key
//! - `ALIBABA_WORKSPACE_ID`: Optional workspace identity for the pay-as-you-go
//!   Messages default
//! - `ALIBABA_API_URL`: Optional full endpoint override (Token Plan, China, workspace)
//! - `ALIBABA_API_MODE`: `openai` or `anthropic`; without a URL, `anthropic`
//!   needs `ALIBABA_WORKSPACE_ID`

use super::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images, MessagesUsage};
use super::shared::CacheMarker;
use crate::llm::config::CacheTTL;
use crate::llm::providers::openai_compat::{chat_completion as openai_compat_chat_completion, OpenAiCompatConfig};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse, ReasoningEffort};
use crate::llm::utils::{contains_ignore_ascii_case, normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Alibaba Cloud Model Studio provider
#[derive(Debug, Clone)]
pub struct AlibabaProvider;

impl Default for AlibabaProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl AlibabaProvider {
	pub fn new() -> Self {
		Self
	}
}

pub(crate) const ALIBABA_API_KEY_ENV: &str = "ALIBABA_API_KEY";
const ALIBABA_API_URL_ENV: &str = "ALIBABA_API_URL";
const ALIBABA_API_MODE_ENV: &str = "ALIBABA_API_MODE";
/// Optional workspace identity: with it and no URL, the documented regional
/// pay-as-you-go Messages base is selected instead of Chat Completions.
pub(crate) const ALIBABA_WORKSPACE_ID_ENV: &str = "ALIBABA_WORKSPACE_ID";
/// Chat Completions endpoint, the default when no workspace is configured.
const ALIBABA_API_URL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions";

/// The documented regional pay-as-you-go Messages base
/// (<https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages>).
/// The workspace identity is one DNS label, so it is validated as one before it
/// enters the URL: no dot, slash, colon, query, fragment, userinfo or space can
/// reach the host, and nothing here accepts or re-parses a full URL.
const PAYG_MESSAGES_HOST_SUFFIX: &str = "ap-southeast-1.maas.aliyuncs.com";
const PAYG_MESSAGES_PATH: &str = "/apps/anthropic/v1/messages";

fn validate_workspace_id(workspace: &str) -> Result<()> {
	let label_ok = !workspace.is_empty()
		&& workspace.len() <= 63
		&& !workspace.starts_with('-')
		&& !workspace.ends_with('-')
		&& workspace
			.bytes()
			.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-');
	if !label_ok {
		return Err(crate::errors::ProviderError::ConfigurationError {
			message: format!(
				"{ALIBABA_WORKSPACE_ID_ENV} must be a single DNS label (lowercase letters, digits and hyphens, 1-63 characters, no dot or slash), got {workspace:?}"
			),
		}
		.into());
	}
	Ok(())
}

fn payg_messages_url(workspace: &str) -> Result<String> {
	validate_workspace_id(workspace)?;
	Ok(format!(
		"https://{workspace}.{PAYG_MESSAGES_HOST_SUFFIX}{PAYG_MESSAGES_PATH}"
	))
}

/// The protocol and URL for one request. A supplied URL is sent as it is,
/// whatever the protocol: it is never rewritten and never redirected elsewhere.
/// Precedence: `ALIBABA_API_URL`, then `ALIBABA_API_MODE`, then
/// `ALIBABA_WORKSPACE_ID`, then the Chat Completions default.
///
/// A workspace identity selects the official regional pay-as-you-go Messages
/// base, so Messages is the default for an account that states its workspace.
/// Token Plan and Coding Plan bases are per plan and are not derived from it:
/// those accounts set the full Messages URL in `ALIBABA_API_URL`. With neither a
/// workspace nor a URL the safe existing Chat Completions default stands, and
/// `ALIBABA_API_MODE=anthropic` alone is refused rather than pointed at a
/// guessed or legacy host.
fn select_endpoint(mode: Option<&str>, api_url: Option<&str>, workspace: Option<&str>) -> Result<(ApiMode, String)> {
	// An empty variable is an unset one.
	let mode = mode
		.filter(|raw| !raw.is_empty())
		.map(|raw| anthropic_compat::parse_api_mode(ALIBABA_API_MODE_ENV, raw))
		.transpose()?;
	let api_url = api_url.filter(|api_url| !api_url.is_empty());
	let workspace = workspace.filter(|workspace| !workspace.is_empty());
	if let Some(api_url) = api_url {
		// An explicit URL is used unchanged; its path decides the protocol
		// unless an explicit mode overrides it.
		let mode = mode
			.or_else(|| anthropic_compat::url_api_mode(api_url))
			.unwrap_or(ApiMode::OpenAi);
		return Ok((mode, api_url.to_owned()));
	}
	match (mode, workspace) {
		(Some(ApiMode::OpenAi), _) => Ok((ApiMode::OpenAi, ALIBABA_API_URL.to_owned())),
		(_, Some(workspace)) => Ok((ApiMode::Anthropic, payg_messages_url(workspace)?)),
		(Some(ApiMode::Anthropic), None) => Err(crate::errors::ProviderError::ConfigurationError {
			message: format!(
				"ALIBABA_API_MODE=anthropic needs {ALIBABA_API_URL_ENV} or {ALIBABA_WORKSPACE_ID_ENV}: Model Studio's Messages endpoint is per workspace or per plan, so no default host can be selected without one"
			),
		}
		.into()),
		(None, None) => Ok((ApiMode::OpenAi, ALIBABA_API_URL.to_owned())),
	}
}

/// The route a request would take right now, for the capability methods that
/// differ per protocol. A configuration that cannot be resolved is reported as
/// Chat Completions: the request itself then fails with the configuration error.
fn configured_mode() -> ApiMode {
	select_endpoint(
		env::var(ALIBABA_API_MODE_ENV).ok().as_deref(),
		env::var(ALIBABA_API_URL_ENV).ok().as_deref(),
		env::var(ALIBABA_WORKSPACE_ID_ENV).ok().as_deref(),
	)
	.map(|(mode, _)| mode)
	.unwrap_or(ApiMode::OpenAi)
}

/// Cost of one Messages response, in USD: the same table and long-context tier
/// as the Chat route, with `output_tokens` covering thinking.
fn messages_cost(model: &str, usage: &MessagesUsage) -> Option<f64> {
	let cache_write_tokens = usage.cache_write_tokens();
	calculate_local_usage_cost(
		model,
		usage.input_tokens,
		cache_write_tokens,
		usage.cache_read_tokens,
		usage.output_tokens,
		None,
	)
	.or_else(|| {
		model_pricing(model).map(|pricing| {
			pricing.calculate_cost(
				usage.input_tokens,
				cache_write_tokens,
				usage.cache_read_tokens,
				usage.output_tokens,
			)
		})
	})
}

/// The provider's own table first, then the shared reference table for models
/// it does not list.
fn model_pricing(model: &str) -> Option<crate::llm::types::ModelPricing> {
	if let Some((input, output, cache_write, cache_read)) = crate::llm::utils::get_model_pricing(model, PRICING) {
		return Some(crate::llm::types::ModelPricing::new(
			input,
			output,
			cache_write,
			cache_read,
		));
	}
	crate::llm::reference_models::get_reference_pricing(model)
}

/// Alibaba on the shared Messages layer, used for a configured Messages URL or
/// a configured workspace identity.
///
/// Images: inline data for every model the reference table does not call
/// text-only, refused for the ones it does. A live Token Plan Messages call on
/// 2026-11-26 with `qwen3.8-max` took the base64 block and rejected only the
/// 1x1 test image on size (`height or width must be larger than 10`), so the
/// block itself is accepted there; a URL image is unproven on that route and
/// is refused rather than dropped. Video: Messages has no video block, so a
/// video turn is refused instead of being dropped — the Chat route, and its
/// `video_url`, stay available. Schema: `output_config.format` is unprobed on
/// this endpoint, so a schema takes the forced-tool path and is validated
/// locally. Thinking: `output_config.effort` is the documented knob here
/// (`thinking.budget_tokens` is deprecated by the contract), mapped per model
/// family and clamped to the values the contract lists.
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "alibaba",
	images: |model| match crate::llm::reference_models::get_reference_capabilities(model) {
		Some(capabilities) if !capabilities.vision => Images::None,
		_ => Images::Base64,
	},
	videos: |_| false,
	enforces_response_schema: false,
	thinking: |model, effort, request| {
		let Some(effort) = effort.and_then(|effort| messages_effort(model, effort)) else {
			return;
		};
		// The schema, when one was requested, is already in `output_config`.
		request["output_config"]["effort"] = serde_json::json!(effort);
	},
	cost: messages_cost,
	default_system: "",
	requires_max_tokens: true,
	// Set per request; see `cache_marker`.
	cache_marker: None,
};

/// The marker every cached block carries, on both routes. A marker switches a
/// request from the implicit to the explicit cache ("Explicit cache and implicit
/// cache are mutually exclusive",
/// <https://www.alibabacloud.com/help/en/model-studio/context-cache>): hits bill
/// about half the implicit rate, and the cache lives 5 minutes from its last hit.
/// Live on the Token Plan on 2026-11-28, qwen4-max, qwen3.8-max, qwen3.8-flash
/// and qwen3.7-plus wrote an explicit cache on Messages, and qwen4-max and
/// qwen3.8-flash on Chat Completions; glm-5.3 and deepseek-v4.1-flash ignored
/// the marker on both and kept the implicit cache. The one-hour TTL goes only
/// where the caller asks for the long cache and the model takes it.
fn cache_marker(model: &str, use_long_cache: bool) -> CacheMarker {
	if use_long_cache && takes_hour_cache(model) {
		CacheMarker::EphemeralHour
	} else {
		CacheMarker::Ephemeral
	}
}

/// Model families with the explicit cache in the international deployment, as
/// the context-cache page lists them (read 2026-11-28); a dated snapshot such as
/// `qwen3.7-plus-2026-05-26` belongs to its family. GLM, DeepSeek and Kimi are
/// not listed. Macaron is listed but left out: its one published cache rate
/// covers both caches, so the table already prices it.
const EXPLICIT_CACHE_MODELS: &[&str] = &[
	"qwen4-max",
	"qwen3.8-max",
	"qwen3.7-max",
	"qwen3.8-2.4t-a95b",
	"qwen3.8-27b",
	"qwen3.8-flash",
	"qwen3.7-plus",
	"qwen3.7-flash",
	"qwen3.6-plus",
	"qwen3.6-flash",
	"qwen3.5-plus",
	"qwen3.5-flash",
	"qwen3-coder-flash",
	"qwen3-vl-plus",
	"qwen-plus",
	"qwen-flash",
];

fn uses_explicit_cache(model: &str) -> bool {
	let model = normalize_model_name(model);
	EXPLICIT_CACHE_MODELS.iter().any(|family| {
		model.strip_prefix(family).is_some_and(|rest| {
			rest.is_empty()
				|| rest
					.strip_prefix('-')
					.is_some_and(|snapshot| snapshot.starts_with(|c: char| c.is_ascii_digit()))
		})
	})
}

/// "Only Qwen4 series models are supported" for the one-hour TTL
/// (<https://www.alibabacloud.com/help/en/model-studio/explicit-cache-qwen4>).
/// Live on 2026-11-28, qwen4-max filled `ephemeral_1h_input_tokens` on both
/// routes; qwen3.8-flash took the same marker as a 5-minute write.
fn takes_hour_cache(model: &str) -> bool {
	normalize_model_name(model).starts_with("qwen4")
}

/// The marker whose explicit cache prices this request: the model has the
/// explicit cache, and a message is a breakpoint for it.
fn explicit_cache(params: &ChatCompletionParams) -> Option<CacheMarker> {
	(uses_explicit_cache(&params.model) && params.messages.iter().any(|message| message.cached))
		.then(|| cache_marker(&params.model, params.use_long_cache))
}

// Explicit-cache rates as a share of the input rate: writes "Typically 125%" and
// hits "Typically 10%" (<https://www.alibabacloud.com/help/en/model-studio/context-cache>),
// a one-hour write "Approximately 200%"
// (<https://www.alibabacloud.com/help/en/model-studio/explicit-cache-qwen4>). The page
// says qwen4-max, qwen3.8-max and qwen3.8-flash differ and shows their prices only in
// the console, so the typical shares stand in for them.
const EXPLICIT_CACHE_WRITE_SHARE: f64 = 1.25;
const EXPLICIT_CACHE_HOUR_WRITE_SHARE: f64 = 2.0;
const EXPLICIT_CACHE_HIT_SHARE: f64 = 0.10;

/// `output_config.effort` values the Model Studio Messages contract lists for
/// each model family
/// (<https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages>),
/// from lowest to highest. A family the contract does not list sends no knob,
/// and an effort floors to the nearest documented value at or below it — the
/// same rule the Chat route applies. `ReasoningEffort::None` floors too: this
/// route documents no way to switch reasoning off.
fn messages_effort(model: &str, effort: ReasoningEffort) -> Option<&'static str> {
	let model = normalize_model_name(model);
	// qwen4-max: low, high (default), xhigh; "medium maps to high, max maps to
	// xhigh" — applied here rather than sent, so no undocumented value travels.
	// qwen3.8 / qwen3.7 series: low, medium, xhigh; high and max map to xhigh.
	// glm-5.x: low, high, max; any other value is refused by the endpoint.
	// deepseek-v4: high, max; low and medium map to high, xhigh maps to max.
	// kimi-k3.1-pro: low, medium, high, max.
	let tiers: [&'static str; 5] = if model.starts_with("qwen4-max") {
		["low", "high", "high", "xhigh", "xhigh"]
	} else if model.starts_with("qwen3.8") || model.starts_with("qwen3.7") {
		["low", "medium", "xhigh", "xhigh", "xhigh"]
	} else if model.starts_with("glm-5") {
		["low", "high", "high", "max", "max"]
	} else if model.starts_with("deepseek-v4") {
		["high", "high", "high", "max", "max"]
	} else if model.starts_with("kimi-k3.1") {
		["low", "medium", "high", "high", "max"]
	} else {
		return None;
	};
	Some(match effort {
		ReasoningEffort::None | ReasoningEffort::Low => tiers[0],
		ReasoningEffort::Medium => tiers[1],
		ReasoningEffort::High => tiers[2],
		ReasoningEffort::XHigh => tiers[3],
		ReasoningEffort::Max => tiers[4],
	})
}

// Model Studio international pricing (per 1M tokens in USD) - Oct 2026; rows added
// since carry their own date.
// Source: https://www.alibabacloud.com/help/en/model-studio/model-pricing plus the
// per-model pages linked from https://www.alibabacloud.com/help/en/model-studio/models
// Format: (model, input, output, cache_write, cache_read)
// List prices only; limited-time promotions are not tracked.
// Context caching is implicit: writes bill at the input rate, hits at cache_read
// (the published implicit cache-hit rate, or the explicit cache-read rate when
// that is the only one published). Models without context caching bill hits at
// the input rate.
// Tiered models are priced at their lowest tier; longer prompts bill up to 4x more.
// Busy/idle models are priced at the busy rate (idle is half). Hybrid-thinking
// open-weight models are priced at the non-thinking output rate.
pub(crate) const PRICING: &[PricingTuple] = &[
	// Model page, Nov 2026.
	("qwen4-max", 1.50, 4.50, 1.50, 0.075),
	// The 0902 snapshot is listed at the moving alias's rates.
	("qwen3.8-max-0902", 2.00, 6.00, 2.00, 0.25),
	("qwen3.8-max", 2.00, 6.00, 2.00, 0.25),
	("qwen3.8-flash", 0.15, 0.47, 0.15, 0.016),
	("qwen3.8-omni-flash", 0.15, 0.47, 0.15, 0.016),
	("qwen3.7-max", 2.50, 7.50, 2.50, 0.50),
	("qwen3.7-plus", 0.40, 1.60, 0.40, 0.08),
	("qwen3.7-flash", 0.03, 0.13, 0.03, 0.006),
	("qwen3.6-max-preview", 1.30, 7.80, 1.30, 0.13),
	// Explicit cache only (10% hit rate); not on the international implicit list.
	("qwen3.6-plus", 0.50, 3.00, 0.50, 0.05),
	("qwen3.6-flash", 0.25, 1.50, 0.25, 0.025),
	("qwen3.5-plus", 0.40, 2.40, 0.40, 0.04),
	// Qwen3.5-Omni-Plus: text/image/video input at $1.40 and text output at
	// $8.30 (international list price). Audio input ($11.00) and text-and-audio
	// output ($44.00) are not representable in a single-rate row, so the text
	// rates are used. Context caching is unsupported, so cache hits bill at
	// the input rate.
	("qwen3.5-omni-plus", 1.40, 8.30, 1.40, 1.40),
	// Qwen3.5-Omni-Flash: text/image/video input at $0.40 and text output at
	// $2.20 (international list price). Audio input ($3.00) and text-and-audio
	// output ($11.90) are not representable in a single-rate row, so the text
	// rates are used. Context caching is unsupported, so cache hits bill at
	// the input rate.
	("qwen3.5-omni-flash", 0.40, 2.20, 0.40, 0.40),
	// Qwen3-Omni-Flash: text input at $0.43 and text output after text-only
	// input at $1.66 (international list price). Image/video input ($0.78),
	// audio input ($3.81) and text output after multimodal input ($3.06) are
	// not representable in a single-rate row, so the text rates are used. Not
	// on the implicit-cache list, so cache hits bill at the input rate.
	("qwen3-omni-flash", 0.43, 1.66, 0.43, 0.43),
	("qwen3.5-flash", 0.10, 0.40, 0.10, 0.01),
	("qwen3-coder-plus", 1.00, 5.00, 1.00, 0.20),
	("qwen3-coder-flash", 0.30, 1.50, 0.30, 0.06),
	("qwen3-coder-next", 0.30, 1.50, 0.30, 0.30),
	("qwen3-coder-480b-a35b-instruct", 1.50, 7.50, 1.50, 1.50),
	("qwen3-coder-30b-a3b-instruct", 0.45, 2.25, 0.45, 0.45),
	("qwen3-vl-plus", 0.20, 1.60, 0.20, 0.04),
	("qwen3-vl-flash", 0.05, 0.40, 0.05, 0.01),
	("qwen3-vl-235b-a22b-thinking", 0.40, 4.00, 0.40, 0.40),
	("qwen3-vl-235b-a22b-instruct", 0.40, 1.60, 0.40, 0.40),
	("qwen3-max", 1.20, 6.00, 1.20, 0.24),
	("qwen-vl-max", 0.80, 3.20, 0.80, 0.16),
	("qwen-max", 1.60, 6.40, 1.60, 0.32),
	// Must precede qwen-plus; covers qwen-plus-character-ja too.
	("qwen-plus-character", 0.50, 1.40, 0.50, 0.10),
	("qwen-plus", 0.40, 1.20, 0.40, 0.08),
	("qwen-flash", 0.05, 0.40, 0.05, 0.01),
	("qwen-turbo", 0.05, 0.20, 0.05, 0.01),
	// Open-weight Qwen checkpoints; the 3.5/3.6 ones have no context caching.
	("qwen3.8-2.4t-a95b", 2.00, 6.00, 2.00, 0.25),
	("qwen3.8-27b", 0.50, 3.00, 0.50, 0.10),
	("qwen3.6-35b-a3b", 0.375, 2.25, 0.375, 0.375),
	("qwen3.6-27b", 0.60, 3.60, 0.60, 0.60),
	("qwen3.5-397b-a17b", 0.60, 3.60, 0.60, 0.60),
	("qwen3.5-122b-a10b", 0.40, 3.20, 0.40, 0.40),
	("qwen3.5-35b-a3b", 0.25, 2.00, 0.25, 0.25),
	("qwen3.5-27b", 0.30, 2.40, 0.30, 0.30),
	("qwen3-235b-a22b-thinking-2507", 0.23, 2.30, 0.23, 0.23),
	("qwen3-235b-a22b-instruct-2507", 0.23, 0.92, 0.23, 0.23),
	("qwen3-235b-a22b", 0.70, 2.80, 0.70, 0.70),
	("qwen3-32b", 0.16, 0.64, 0.16, 0.16),
	("qwen3-8b", 0.18, 0.70, 0.18, 0.18),
	// Third-party models resold by Model Studio at Alibaba's own rates.
	// V4.1 Flash and the dated V4 snapshots bill busy/idle; the moving V4
	// aliases have flat rates. V4.1 Flash cache hits bill a fixed $0.006 at
	// busy hours ($0.003 idle), the same in every region.
	("deepseek-v4.1-flash", 0.30, 1.20, 0.30, 0.006),
	("deepseek-v4-pro-0813", 1.32, 3.96, 1.32, 0.132),
	("deepseek-v4-pro", 2.40, 4.80, 2.40, 0.20),
	("deepseek-v4-flash-0731", 0.44, 1.32, 0.44, 0.044),
	("deepseek-v4-flash", 0.20, 0.40, 0.20, 0.04),
	("deepseek-v3.2", 0.57, 1.71, 0.57, 0.114),
	// GLM cache hits bill 20% of input in Singapore (glm-5.1: 18.57%).
	// Prime bills 2x glm-5.3 and must precede it.
	("glm-5.3-prime", 2.80, 8.80, 2.80, 0.56),
	("glm-5.3", 1.40, 4.40, 1.40, 0.28),
	// Must precede glm-5.2.
	("glm-5.2-fast-preview", 2.80, 8.80, 2.80, 0.56),
	("glm-5.2", 1.40, 4.40, 1.40, 0.28),
	("glm-5.1", 1.40, 4.40, 1.40, 0.26),
	// Published cached-input rate, not the 20% default; must precede kimi-k3.
	("kimi-k3.1-pro", 2.00, 10.00, 2.00, 0.20),
	("kimi-k3", 3.00, 15.00, 3.00, 0.60),
	("kimi-k2.7-code", 0.95, 4.00, 0.95, 0.19),
	// Macaron models from `mindlab` (model pages, Nov 2026): pay-as-you-go in
	// Singapore only. One published cache rate covers implicit and explicit hits.
	("macaron-v1.1-coding-venti", 1.19, 3.74, 1.19, 0.22),
	("macaron-v1.1-venti", 1.19, 3.74, 1.19, 0.22),
	("macaron-v1-tall", 0.45, 2.60, 0.45, 0.08),
];

const QWEN_PLUS_LONG_CONTEXT_THRESHOLD: u64 = 256_000;

// Implicit cache lifetimes, for models without the explicit cache. Model Studio gives
// the implicit cache "no fixed validity period"
// (<https://www.alibabacloud.com/help/en/model-studio/context-cache>). Measured
// 2026-11-07 on the Token Plan endpoint (token-plan.ap-southeast-1): unique
// 3.5k-12k-token prompts re-sent once after an idle delay, two samples per delay.
// - deepseek-v4.1-flash: every sample hit up to 30m, both missed at 45m and 60m.
// - glm-5.3: every sample hit up to 10m, one of two missed at 15m.
const MEASURED_CACHE_LIFETIMES: &[(&str, std::time::Duration)] = &[
	("deepseek-v4.1-flash", std::time::Duration::from_secs(30 * 60)),
	("glm-5.3", std::time::Duration::from_secs(10 * 60)),
];
// Assumed for models not measured yet.
const UNMEASURED_CACHE_LIFETIME: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// `explicit` is the marker of an explicit-cache request; it replaces the
/// table's implicit cache rates.
fn calculate_local_usage_cost(
	model: &str,
	input_tokens: u64,
	cache_write_tokens: u64,
	cache_read_tokens: u64,
	output_tokens: u64,
	explicit: Option<CacheMarker>,
) -> Option<f64> {
	let (mut input, mut output, mut cache_write, mut cache_read) =
		crate::llm::utils::get_model_pricing(model, PRICING)?;
	let total_input_tokens = input_tokens
		.saturating_add(cache_write_tokens)
		.saturating_add(cache_read_tokens);

	if normalize_model_name(model).contains("qwen3.7-plus") && total_input_tokens > QWEN_PLUS_LONG_CONTEXT_THRESHOLD {
		// List-price tier for prompts in (256K, 1M].
		input = 1.20;
		output = 4.80;
		cache_write = 1.20;
		cache_read = 0.24;
	}
	if let Some(marker) = explicit {
		cache_write = input
			* match marker {
				CacheMarker::Ephemeral => EXPLICIT_CACHE_WRITE_SHARE,
				CacheMarker::EphemeralHour => EXPLICIT_CACHE_HOUR_WRITE_SHARE,
			};
		cache_read = input * EXPLICIT_CACHE_HIT_SHARE;
	}

	Some(
		(input_tokens as f64 / 1_000_000.0) * input
			+ (cache_write_tokens as f64 / 1_000_000.0) * cache_write
			+ (cache_read_tokens as f64 / 1_000_000.0) * cache_read
			+ (output_tokens as f64 / 1_000_000.0) * output,
	)
}

#[async_trait::async_trait]
impl AiProvider for AlibabaProvider {
	fn name(&self) -> &str {
		"alibaba"
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(ALIBABA_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("alibaba", ALIBABA_API_KEY_ENV).into())
	}

	fn supports_caching(&self, _model: &str) -> bool {
		// Both routes report cache writes and hits: Messages as
		// `cache_creation_input_tokens` / `cache_read_input_tokens`, Chat
		// Completions under `prompt_tokens_details`.
		true
	}

	/// A model with the explicit cache keeps it 5 minutes from its last hit, an
	/// hour with the one-hour TTL. Live on the Token Plan on 2026-11-28, on both
	/// routes: qwen4-max and qwen3.8-flash hit 2 of 2 after 4 idle minutes and
	/// missed 2 of 2 after 6; qwen4-max with the one-hour TTL hit 2 of 2 after 55
	/// and missed 2 of 2 after 65. Other models keep the implicit cache; see
	/// `MEASURED_CACHE_LIFETIMES`.
	fn cache_lifetime(&self, model: &str, use_long_cache: bool) -> Option<std::time::Duration> {
		if uses_explicit_cache(model) {
			let hour = cache_marker(model, use_long_cache) == CacheMarker::EphemeralHour;
			return Some(CacheTTL::from_long_cache_flag(hour).to_duration());
		}
		let measured = MEASURED_CACHE_LIFETIMES
			.iter()
			.find(|(pattern, _)| contains_ignore_ascii_case(model, pattern))
			.map(|(_, lifetime)| *lifetime);
		Some(measured.unwrap_or(UNMEASURED_CACHE_LIFETIME))
	}

	/// On Chat, GLM and qwen4-max read every prior turn's reasoning back (see
	/// `preserves_historical_thinking`), so it occupies context the model
	/// reads and bills. The shared Messages layer replays every captured
	/// thinking block, and live on the Token Plan Messages endpoint on
	/// 2026-11-28 that raised `input_tokens` for every model tried
	/// (qwen3.8-flash 428 with the block vs 241 without, glm-5.3 311 vs 292),
	/// so there every model replays.
	fn replays_thinking(&self, model: &str) -> bool {
		configured_mode() == ApiMode::Anthropic || preserves_historical_thinking(model)
	}

	// supports_vision and get_max_input_tokens come from the reference table:
	// inline images and the context window are the same on both routes.

	/// Video is a Chat Completions field (`video_url`). The Messages protocol
	/// has no video block, so a video turn is refused there and the capability
	/// says so, instead of advertising a turn that cannot be served.
	fn supports_video(&self, model: &str) -> bool {
		configured_mode() == ApiMode::OpenAi
			&& crate::llm::reference_models::get_reference_capabilities(model)
				.map(|capabilities| capabilities.video)
				.unwrap_or(false)
	}

	/// Alibaba supports native JSON Schema for selected Qwen families. Other
	/// hosted models use shared forced-tool enforcement plus local validation.
	fn supports_structured_output(&self, _model: &str) -> bool {
		true
	}

	/// The live probe behind `natively_enforces_response_schema` ran on the Chat
	/// endpoint only, and the Messages contract's `output_config.format` is
	/// unprobed here, so the claim stays on the route that has the evidence:
	/// a Messages schema takes the forced-tool path with local validation.
	fn enforces_response_schema(&self, model: &str) -> bool {
		configured_mode() == ApiMode::OpenAi && natively_enforces_response_schema(model)
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		model_pricing(model)
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let (mode, api_url) = select_endpoint(
			env::var(ALIBABA_API_MODE_ENV).ok().as_deref(),
			env::var(ALIBABA_API_URL_ENV).ok().as_deref(),
			env::var(ALIBABA_WORKSPACE_ID_ENV).ok().as_deref(),
		)?;
		let marker = cache_marker(&params.model, params.use_long_cache);
		let explicit = explicit_cache(&params);
		let model = params.model.clone();
		if mode == ApiMode::Anthropic {
			let config = AnthropicCompatConfig {
				cache_marker: Some(marker),
				..ANTHROPIC_COMPAT
			};
			let mut response = anthropic_compat::chat_completion(config, api_key, api_url, params).await?;
			// The config's cost hook prices the implicit cache.
			if let (Some(explicit), Some(usage)) = (explicit, response.exchange.usage.as_mut()) {
				usage.cost = calculate_local_usage_cost(
					&model,
					usage.input_tokens,
					usage.cache_write_tokens,
					usage.cache_read_tokens,
					usage.billable_output_tokens(),
					Some(explicit),
				);
			}
			return Ok(response);
		}
		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "alibaba",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: natively_enforces_response_schema(&model),
				// Thinking-mode tool calls accept only auto/none. Schema repair
				// therefore uses auto plus explicit prompt guidance and local
				// validation instead of an unsupported required policy.
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
				cache_marker: Some(marker),
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		if let Some(ref mut usage) = response.exchange.usage {
			if usage.cost.is_none() {
				let input_tokens = usage.input_tokens;
				let cache_write_tokens = usage.cache_write_tokens;
				let cache_read_tokens = usage.cache_read_tokens;
				let output_tokens = usage.billable_output_tokens();
				usage.cost = calculate_local_usage_cost(
					&model,
					input_tokens,
					cache_write_tokens,
					cache_read_tokens,
					output_tokens,
					explicit,
				)
				.or_else(|| {
					self.get_model_pricing(&model).map(|pricing| {
						pricing.calculate_cost(input_tokens, cache_write_tokens, cache_read_tokens, output_tokens)
					})
				});
			}
		}

		Ok(response)
	}
}

/// JSON Schema is native only for the Alibaba model families explicitly listed
/// by Model Studio. Snapshot suffixes inherit their family's capability.
/// qwen4-max was probed live on the Token Plan endpoint (2026-11-17): a strict
/// object schema against a prompt asking for a JSON array returned the object
/// on 5 of 5 calls, 3 with thinking and 2 with `enable_thinking: false`. The
/// pay-as-you-go endpoint was not probed (no key for it); its model page lists
/// JSON Schema as supported.
fn natively_enforces_response_schema(model: &str) -> bool {
	let model = normalize_model_name(model);
	[
		"qwen4-max",
		"qwen3.8-max",
		"qwen3.8-flash",
		"qwen3.7-max",
		"qwen3.7-plus",
		"qwen3.7-flash",
	]
	.iter()
	.any(|prefix| model.starts_with(prefix))
}

/// Model Studio's GLM models keep earlier turns' reasoning in context by
/// default (`clear_thinking` defaults to false), and Preserved Thinking needs
/// that `reasoning_content` sent back complete, unmodified and in its original
/// order. Without it every call starts with none of the model's earlier
/// reasoning: a glm-5.3 max-effort run re-derived the same plan (20-88k chars)
/// on every step until it timed out. qwen4-max has `preserve_thinking`
/// on by default in Chat Completions and asks for all historical thinking back
/// in `reasoning_content` (model page, Nov 2026). DeepSeek and earlier Qwen
/// models follow other rules on this host.
pub(crate) fn preserves_historical_thinking(model: &str) -> bool {
	contains_ignore_ascii_case(model, "glm") || contains_ignore_ascii_case(model, "qwen4-max")
}

#[cfg(test)]
#[path = "alibaba_tests.rs"]
mod tests;
