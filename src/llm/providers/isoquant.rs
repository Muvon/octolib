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

//! Isoquant provider implementation.
//!
//! Speaks Isoquant's Anthropic Messages endpoint,
//! `https://api.isoquant.ai/v1/messages`, and keeps its OpenAI-compatible Chat
//! Completions endpoint, `https://api.isoquant.ai/v1/chat/completions`. Each
//! request takes one protocol, chosen in this order:
//! - `ISOQUANT_API_MODE=anthropic|openai` overrides everything else.
//! - `ISOQUANT_API_URL` is sent exactly as written. A path ending in `/messages`
//!   or `/chat/completions` names the protocol; any other custom URL keeps Chat
//!   Completions, the only API this provider called before.
//! - Otherwise Messages, except for a model the catalog serves on Chat
//!   Completions only.
//!
//! A failed request is retried on the endpoint it was sent to, never on the
//! other protocol.
//!
//! Models added by Isoquant work without a release. Any model id is accepted
//! and Isoquant itself refuses ones it does not serve. The public catalog at
//! `GET /v1/models` is fetched once per process, through [`preload`] or before
//! the first chat request, and supplies context window, what each protocol
//! takes for a model, pricing and reasoning levels. `KNOWN_MODELS` below is
//! the offline fallback. A model missing from the fetched catalog (added
//! since, or visible to one account only) still runs, on reference prices and
//! with no reasoning level sent. Messages takes nothing else from a model's
//! name; Chat Completions keeps the reference tables it always fell back to.
//!
//! Every inference request sends `Isoquant-ZDR: required`, which requires zero
//! data retention. Isoquant still records usage and billing metadata. The key
//! travels in `x-api-key` on Messages, as Isoquant documents it, and in
//! `Authorization: Bearer` on Chat Completions. The public catalog
//! (`GET /v1/models`, model metadata only) is read without the key or that
//! header.
//!
//! Configuration:
//! - `ISOQUANT_API_KEY`: Required API key
//! - `ISOQUANT_API_MODE`: Optional `anthropic` or `openai`; overrides the URL
//! - `ISOQUANT_API_URL`: Optional endpoint override; the catalog is read from
//!   the `/models` path beside it

use crate::llm::providers::anthropic_compat::{self, AnthropicCompatConfig, ApiMode, Images};
use crate::llm::providers::openai_compat::{chat_completion as openai_compat_chat_completion, OpenAiCompatConfig};
use crate::llm::providers::shared::CacheMarker;
use crate::llm::reference_models::{get_reference_capabilities, get_reference_pricing};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, ReasoningEffort};
use crate::llm::utils::{contains_ignore_ascii_case, normalize_model_name};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::env;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::OnceCell;

pub(crate) const ISOQUANT_API_KEY_ENV: &str = "ISOQUANT_API_KEY";
const ISOQUANT_API_MODE_ENV: &str = "ISOQUANT_API_MODE";
const ISOQUANT_API_URL_ENV: &str = "ISOQUANT_API_URL";
const CHAT_COMPLETIONS_URL: &str = "https://api.isoquant.ai/v1/chat/completions";
const MESSAGES_URL: &str = "https://api.isoquant.ai/v1/messages";
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const MESSAGES_PATH: &str = "/messages";
const ZDR_HEADER: &str = "Isoquant-ZDR";
const ZDR_REQUIRED: &str = "required";
/// A slow catalog must not hold up the chat request waiting behind it.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(10);
/// The catalog prices tokens in billionths of a dollar: 1000 of them per token
/// is $1 per 1M tokens.
const NANO_USD_PER_TOKEN_PER_USD_PER_1M: f64 = 1_000.0;
const KNOWN_CONTEXT_WINDOW: usize = 1_048_576;
const DEFAULT_MAX_INPUT_TOKENS: usize = 8_192;
/// Messages lists `max_tokens` as a required field. 16384 is the default it
/// applies when the field is absent (catalog and docs, 2026-11-27), the same
/// one Chat Completions applies to an omitted field.
const MESSAGES_DEFAULT_MAX_TOKENS: u32 = 16_384;

/// Reasoning levels from lowest to highest, with their wire names.
const EFFORT_LADDER: &[(ReasoningEffort, &str)] = &[
	(ReasoningEffort::None, "none"),
	(ReasoningEffort::Low, "low"),
	(ReasoningEffort::Medium, "medium"),
	(ReasoningEffort::High, "high"),
	(ReasoningEffort::XHigh, "xhigh"),
	(ReasoningEffort::Max, "max"),
];

/// Used until the catalog loads, or when it cannot be reached. From
/// `GET /v1/models`, 2026-11-27. Format: (model, input, output, cached input,
/// reasoning levels), prices per 1M tokens. Isoquant publishes no cache-write
/// price, so writes bill at the input rate. GLM-5.3 has no `none` level: the
/// host answers 400, and a request for no reasoning runs at `low`.
pub(crate) const KNOWN_MODELS: &[(&str, f64, f64, f64, &[&str])] = &[
	(
		"deepseek-v4.1-flash-1017-fast",
		0.035,
		0.37,
		0.005,
		&["none", "low", "high", "max"],
	),
	("glm-5.3-flash", 0.07, 0.15, 0.025, &["low", "high", "max"]),
];

/// Models proven to decode against `json_schema` on Chat Completions. A strict
/// object schema with a prompt demanding a JSON array returned the object on 3
/// of 3 calls per model (`/v1/chat/completions`, 2026-11-07). Models added
/// later take the forced-tool path until probed. Messages was probed the same
/// way on 2026-11-27 with the same result, but that route has no per-model
/// flag (`ANTHROPIC_COMPAT`), so it claims nothing.
const SCHEMA_ENFORCING_MODELS: &[&str] = &["deepseek-v4.1-flash-1017-fast", "glm-5.3-flash"];
/// The only model with a measured cache lifetime, on Chat Completions; see
/// `cache_lifetime`.
const MEASURED_CACHE_MODEL: &str = "deepseek-v4.1-flash-1017-fast";
const MEASURED_CACHE_LIFETIME: Duration = Duration::from_secs(60 * 60);

/// What one protocol takes for a model, as the catalog's `apis` block lists it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IsoquantApi {
	vision: bool,
	structured_output: bool,
	/// The protocol's `tool_choice` lists `required`.
	required_tool_choice: bool,
}

/// Both `KNOWN_MODELS` on each protocol (`GET /v1/models`, 2026-11-27): image
/// input and `json_schema` output on both, a `required` tool choice on Chat
/// Completions only.
const KNOWN_CHAT_API: IsoquantApi = IsoquantApi {
	vision: true,
	structured_output: true,
	required_tool_choice: true,
};
const KNOWN_MESSAGES_API: IsoquantApi = IsoquantApi {
	vision: true,
	structured_output: true,
	required_tool_choice: false,
};

#[derive(Debug, Clone)]
struct IsoquantModel {
	id: String,
	context_window: usize,
	pricing: Option<ModelPricing>,
	/// Lowest first; empty when the model has no reasoning levels.
	efforts: Vec<(ReasoningEffort, &'static str)>,
	/// `None` when the catalog does not serve the model on that protocol.
	chat: Option<IsoquantApi>,
	messages: Option<IsoquantApi>,
}

static KNOWN_CATALOG: LazyLock<Vec<IsoquantModel>> = LazyLock::new(|| {
	KNOWN_MODELS
		.iter()
		.map(|(id, input, output, cached, efforts)| IsoquantModel {
			id: id.to_string(),
			context_window: KNOWN_CONTEXT_WINDOW,
			pricing: Some(ModelPricing::new(*input, *output, *input, *cached)),
			efforts: listed_efforts(efforts),
			chat: Some(KNOWN_CHAT_API),
			messages: Some(KNOWN_MESSAGES_API),
		})
		.collect()
});

/// Process-wide catalog, filled by [`preload`] or the first chat call.
static CATALOG: OnceCell<Vec<IsoquantModel>> = OnceCell::const_new();

#[derive(Deserialize)]
struct CatalogResponse {
	#[serde(default)]
	data: Vec<CatalogEntry>,
}

#[derive(Deserialize)]
struct CatalogEntry {
	id: String,
	isoquant: Option<CatalogDetails>,
}

#[derive(Deserialize)]
struct CatalogDetails {
	apis: Option<CatalogApis>,
	context_length: Option<usize>,
	pricing: Option<CatalogPricing>,
	reasoning: Option<CatalogReasoning>,
}

#[derive(Deserialize)]
struct CatalogApis {
	chat_completions: Option<CatalogApi>,
	messages: Option<CatalogApi>,
}

/// One protocol's `apis` block. Its other fields differ per protocol and are
/// not read.
#[derive(Deserialize)]
struct CatalogApi {
	#[serde(default)]
	input_modalities: Vec<String>,
	#[serde(default)]
	structured_outputs: Vec<String>,
	#[serde(default)]
	tool_choice: Vec<String>,
}

#[derive(Deserialize)]
struct CatalogPricing {
	input_nano_usd_per_token: Option<String>,
	output_nano_usd_per_token: Option<String>,
	cached_input_nano_usd_per_token: Option<String>,
}

#[derive(Deserialize)]
struct CatalogReasoning {
	#[serde(default)]
	efforts: Vec<String>,
}

fn listed_efforts<S: AsRef<str>>(listed: &[S]) -> Vec<(ReasoningEffort, &'static str)> {
	EFFORT_LADDER
		.iter()
		.filter(|(_, name)| listed.iter().any(|level| level.as_ref() == *name))
		.copied()
		.collect()
}

fn price_per_1m(nano_usd_per_token: &str) -> Option<f64> {
	let nano_usd = nano_usd_per_token.parse::<f64>().ok()?;
	Some(nano_usd / NANO_USD_PER_TOKEN_PER_USD_PER_1M)
}

fn parse_pricing(pricing: &CatalogPricing) -> Option<ModelPricing> {
	let input = price_per_1m(pricing.input_nano_usd_per_token.as_deref()?)?;
	let output = price_per_1m(pricing.output_nano_usd_per_token.as_deref()?)?;
	let cached = pricing
		.cached_input_nano_usd_per_token
		.as_deref()
		.and_then(price_per_1m)
		.unwrap_or(input);
	Some(ModelPricing::new(input, output, input, cached))
}

fn parse_api(api: CatalogApi) -> IsoquantApi {
	IsoquantApi {
		vision: api.input_modalities.iter().any(|modality| modality == "image"),
		structured_output: api.structured_outputs.iter().any(|format| format == "json_schema"),
		required_tool_choice: api.tool_choice.iter().any(|choice| choice == "required"),
	}
}

/// Keeps a model the catalog serves on at least one protocol, with what each of
/// them takes for it. Nothing is inferred from the model's name or brand.
fn parse_catalog(entries: Vec<CatalogEntry>) -> Vec<IsoquantModel> {
	entries
		.into_iter()
		.filter_map(|entry| {
			let details = entry.isoquant?;
			let apis = details.apis?;
			let chat = apis.chat_completions.map(parse_api);
			let messages = apis.messages.map(parse_api);
			if chat.is_none() && messages.is_none() {
				return None;
			}
			Some(IsoquantModel {
				id: entry.id,
				context_window: details.context_length?,
				pricing: details.pricing.as_ref().and_then(parse_pricing),
				efforts: details
					.reasoning
					.map(|reasoning| listed_efforts(&reasoning.efforts))
					.unwrap_or_default(),
				chat,
				messages,
			})
		})
		.collect()
}

/// The catalog path beside either completion URL: its path up to a terminal
/// `/messages` or `/chat/completions` segment, then `/models`. Only scheme,
/// host and path count: the completion URL's query, fragment and userinfo are
/// not forwarded, whatever credentials they carry, because the catalog read
/// is keyless.
fn models_url(api_url: &str) -> String {
	let Ok(mut url) = reqwest::Url::parse(api_url) else {
		// Nothing reqwest could request either; the string form reaches the error.
		return format!("{api_url}/models");
	};
	let path = url.path().trim_end_matches('/');
	let base = path
		.strip_suffix(CHAT_COMPLETIONS_PATH)
		.or_else(|| path.strip_suffix(MESSAGES_PATH))
		.unwrap_or(path);
	url.set_path(&format!("{base}/models"));
	url.set_query(None);
	url.set_fragment(None);
	// Both setters fail only for cannot-be-a-base URLs; for one, there is no userinfo to clear.
	let _ = url.set_username("");
	let _ = url.set_password(None);
	url.to_string()
}

async fn fetch_catalog(url: &str) -> Result<Vec<IsoquantModel>> {
	let response = super::shared::http_client()
		.get(url)
		.timeout(CATALOG_TIMEOUT)
		.send()
		.await
		.context("Failed to fetch Isoquant model catalog")?;
	if !response.status().is_success() {
		let status = response.status();
		let text = response.text().await.unwrap_or_default();
		return Err(anyhow!("Isoquant models API error {status}: {text}"));
	}
	let catalog: CatalogResponse = response
		.json()
		.await
		.context("Failed to parse Isoquant model catalog")?;
	let models = parse_catalog(catalog.data);
	// Caching an empty list would reject every model until the process restarts.
	if models.is_empty() {
		return Err(anyhow!("Isoquant model catalog lists no models"));
	}
	Ok(models)
}

async fn load_catalog(api_url: &str) -> Result<()> {
	let url = models_url(api_url);
	CATALOG
		.get_or_try_init(|| async move { fetch_catalog(&url).await })
		.await
		.map(|_| ())
}

/// Fetch the catalog once per process. Idempotent: the first chat call does the
/// same lazily, and a failure leaves `KNOWN_MODELS` in charge until the next
/// attempt.
pub async fn preload() -> Result<()> {
	configured_mode_env()?;
	load_catalog(api_url_env().as_deref().unwrap_or(CHAT_COMPLETIONS_URL)).await
}

fn find_model<'a>(models: &'a [IsoquantModel], model: &str) -> Option<&'a IsoquantModel> {
	let normalized = normalize_model_name(model);
	models
		.iter()
		.find(|entry| normalize_model_name(&entry.id) == normalized)
}

fn known_model(model: &str) -> Option<&'static IsoquantModel> {
	CATALOG
		.get()
		.and_then(|catalog| find_model(catalog, model))
		.or_else(|| find_model(&KNOWN_CATALOG, model))
}

/// The highest listed level not above the request, or the lowest listed one.
/// Isoquant answers 400 to any unlisted level, including `medium` on both
/// current models (live, 2026-11-07).
fn select_effort(efforts: &[(ReasoningEffort, &'static str)], requested: ReasoningEffort) -> Option<&'static str> {
	efforts
		.iter()
		.rev()
		.find(|(level, _)| *level as u8 <= requested as u8)
		.or(efforts.first())
		.map(|(_, name)| *name)
}

/// The reasoning level to send for a request, whichever knob the protocol
/// carries it in, or `None` to send nothing because the model's levels are
/// unknown.
pub(crate) fn reasoning_effort_value(model: &str, effort: ReasoningEffort) -> Option<&'static str> {
	select_effort(&known_model(model)?.efforts, effort)
}

fn model_pricing(model: &str) -> Option<ModelPricing> {
	known_model(model)
		.and_then(|entry| entry.pricing)
		.or_else(|| get_reference_pricing(model))
}

/// The protocol the standard endpoint serves a model on when nothing pins it.
fn preferred_mode(model: Option<&IsoquantModel>) -> ApiMode {
	match model {
		Some(entry) if entry.messages.is_none() => ApiMode::OpenAi,
		_ => ApiMode::Anthropic,
	}
}

fn standard_url(mode: ApiMode) -> &'static str {
	match mode {
		ApiMode::Anthropic => MESSAGES_URL,
		ApiMode::OpenAi => CHAT_COMPLETIONS_URL,
	}
}

/// A variable counts as set only when non-empty, like every other provider.
fn non_empty_env(name: &str) -> Option<String> {
	env::var(name).ok().filter(|raw| !raw.is_empty())
}

fn api_url_env() -> Option<String> {
	non_empty_env(ISOQUANT_API_URL_ENV)
}

/// The mode variable, parsed once. An invalid value is refused before any
/// request goes out, the catalog fetch included.
fn configured_mode_env() -> Result<Option<ApiMode>> {
	non_empty_env(ISOQUANT_API_MODE_ENV)
		.map(|raw| anthropic_compat::parse_api_mode(ISOQUANT_API_MODE_ENV, &raw))
		.transpose()
}

/// The endpoint a request goes to and the protocol it speaks there: the mode
/// variable, then the URL's own path, then (for the standard endpoint only)
/// the catalog. The URL is used exactly as configured.
fn select_endpoint(mode_env: Option<ApiMode>, api_url_env: Option<&str>, model: &str) -> (ApiMode, String) {
	let mode = mode_env
		.or_else(|| api_url_env.and_then(anthropic_compat::url_api_mode))
		.unwrap_or_else(|| match api_url_env {
			// A custom URL naming neither endpoint has always been served by Chat
			// Completions.
			Some(_) => ApiMode::OpenAi,
			None => preferred_mode(known_model(model)),
		});
	let api_url = api_url_env.map_or_else(|| standard_url(mode).to_owned(), str::to_owned);
	(mode, api_url)
}

/// The protocol the configured endpoint speaks for a model. The capability
/// methods report what is true on it. An unusable configuration is refused by
/// the request itself; the capabilities then say no more than Chat Completions.
fn configured_mode(model: &str) -> ApiMode {
	let mode_env = configured_mode_env().ok().flatten();
	select_endpoint(mode_env, api_url_env().as_deref(), model).0
}

/// Whether a route can force a tool call. The catalog's word counts on Chat
/// Completions only: the Messages transport here has no required tool choice,
/// whatever an entry lists.
fn takes_required_tool_choice(api: IsoquantApi, mode: ApiMode) -> bool {
	mode == ApiMode::OpenAi && api.required_tool_choice
}

/// What the configured protocol takes for the model, or `None` when nothing is
/// known. Messages takes nothing from a model's name: what another host serves
/// under the same id says nothing about this endpoint.
fn configured_api(model: &str) -> Option<IsoquantApi> {
	let mode = configured_mode(model);
	let known = known_model(model).and_then(|entry| match mode {
		ApiMode::Anthropic => entry.messages,
		ApiMode::OpenAi => entry.chat,
	});
	if known.is_some() || mode == ApiMode::Anthropic {
		return known;
	}
	get_reference_capabilities(model).map(|capabilities| IsoquantApi {
		vision: capabilities.vision,
		structured_output: capabilities.structured_output,
		required_tool_choice: false,
	})
}

/// Drops the headers this provider owns on a route and sets the one it
/// requires, so a caller can neither opt out of zero data retention nor put a
/// second key, or another protocol version, on the wire.
fn protect_request_headers(params: &mut ChatCompletionParams, mode: ApiMode) {
	let headers = params.extra_headers.get_or_insert_with(Default::default);
	headers.retain(|name, _| !is_protected_header(name, mode));
	headers.insert(ZDR_HEADER.to_string(), ZDR_REQUIRED.to_string());
}

fn is_protected_header(name: &str, mode: ApiMode) -> bool {
	let matches = |protected: &str| name.eq_ignore_ascii_case(protected);
	matches(ZDR_HEADER)
		|| matches("authorization")
		|| matches("x-api-key")
		|| (mode == ApiMode::Anthropic && matches("anthropic-version"))
}

const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "isoquant",
	// Messages documents an image as inline data only. A URL image is refused
	// rather than sent in a shape the endpoint does not name.
	images: |model| match known_model(model).and_then(|entry| entry.messages) {
		Some(api) if api.vision => Images::Base64,
		_ => Images::None,
	},
	videos: |_| false,
	// Probed 2026-11-27 on `/v1/messages` with a strict object schema and a
	// prompt demanding an array: both current models returned the object on 3
	// of 3 calls. This flag is per endpoint, not per model, so it stays off: a
	// model added since would claim enforcement it was never probed for, and
	// the forced-tool path plus local validation serves every model.
	enforces_response_schema: false,
	// Reasoning levels are the model's on either protocol; Messages takes the
	// selected one under `output_config.effort`.
	thinking: |model, effort, request| {
		if let Some(level) = effort.and_then(|effort| reasoning_effort_value(model, effort)) {
			request["output_config"]["effort"] = serde_json::json!(level);
		}
	},
	// Isoquant returns no cost. The catalog rates price every counter, thinking
	// included in `output_tokens`.
	cost: |model, usage| {
		model_pricing(model).map(|pricing| {
			pricing.calculate_cost(
				usage.input_tokens,
				usage.cache_write_tokens(),
				usage.cache_read_tokens,
				usage.output_tokens,
			)
		})
	},
	default_system: "",
	requires_max_tokens: true,
	cache_marker: Some(CacheMarker::Ephemeral),
};

/// Isoquant provider
#[derive(Debug, Clone, Default)]
pub struct IsoquantProvider;

impl IsoquantProvider {
	pub fn new() -> Self {
		Self
	}
}

#[async_trait::async_trait]
impl AiProvider for IsoquantProvider {
	fn name(&self) -> &str {
		"isoquant"
	}

	fn supports_model(&self, model: &str) -> bool {
		// Not checked against the catalog: it is fetched once, so a long-running
		// process would otherwise reject models added after that.
		!model.is_empty()
	}

	fn supports_vision(&self, model: &str) -> bool {
		configured_api(model).is_some_and(|api| api.vision)
	}

	fn supports_video(&self, _model: &str) -> bool {
		// input_modalities: ["text", "image"] on both protocols
		false
	}

	/// Measured on Messages on 2026-11-28: sending a turn back with its thinking
	/// block raised `input_tokens` on glm-5.3-flash (263 against 250 without it),
	/// while deepseek-v4.1-flash-1017-fast counted 204 either way. Chat
	/// Completions was not measured.
	fn replays_thinking(&self, model: &str) -> bool {
		configured_mode(model) == ApiMode::Anthropic && contains_ignore_ascii_case(model, "glm")
	}

	fn supports_caching(&self, model: &str) -> bool {
		// Automatic prefix caching on Chat Completions; hits bill at the cached
		// input rate. Messages has not been measured, so it claims nothing.
		configured_mode(model) == ApiMode::OpenAi
			&& known_model(model)
				.and_then(|entry| entry.pricing)
				.is_some_and(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
	}

	/// The docs give no lifetime. Measured 2026-11-07 on `/v1/chat/completions`: unique
	/// ~3.5k-token prompts re-sent once after an idle delay, two samples per delay.
	/// deepseek-v4.1-flash-1017-fast hit on every sample from 1m to 60m. glm-5.3-flash
	/// missed one sample at 1m and every sample from 15m, so it has no lifetime it is
	/// known to survive. Not measured on Messages.
	fn cache_lifetime(&self, model: &str, _use_long_cache: bool) -> Option<Duration> {
		(configured_mode(model) == ApiMode::OpenAi && normalize_model_name(model) == MEASURED_CACHE_MODEL)
			.then_some(MEASURED_CACHE_LIFETIME)
	}

	fn supports_structured_output(&self, model: &str) -> bool {
		configured_api(model).is_some_and(|api| api.structured_output)
	}

	fn enforces_response_schema(&self, model: &str) -> bool {
		// Proven on Chat Completions only; see `SCHEMA_ENFORCING_MODELS`.
		if configured_mode(model) != ApiMode::OpenAi {
			return false;
		}
		let normalized = normalize_model_name(model);
		SCHEMA_ENFORCING_MODELS.iter().any(|id| normalized == *id)
	}

	fn supports_required_tool_choice(&self, model: &str) -> bool {
		configured_api(model).is_some_and(|api| takes_required_tool_choice(api, configured_mode(model)))
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		match known_model(model) {
			Some(entry) => entry.context_window,
			None => get_reference_capabilities(model)
				.map(|capabilities| capabilities.max_input_tokens)
				.unwrap_or(DEFAULT_MAX_INPUT_TOKENS),
		}
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(ISOQUANT_API_KEY_ENV)
			.map_err(|_| crate::errors::missing_api_key("isoquant", ISOQUANT_API_KEY_ENV).into())
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		model_pricing(model)
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		// The mode is validated before any request goes out, the catalog fetch included.
		let mode_env = configured_mode_env()?;
		let api_url_env = api_url_env();

		// A failed load is retried on the next call; `KNOWN_MODELS` serves this one.
		if let Err(error) = load_catalog(api_url_env.as_deref().unwrap_or(CHAT_COMPLETIONS_URL)).await {
			tracing::warn!("Isoquant model catalog unavailable: {error:#}");
		}

		let model = params.model.clone();
		// The catalog decides the route only when nothing pins it, so it loads first.
		let (mode, api_url) = select_endpoint(mode_env, api_url_env.as_deref(), &model);
		protect_request_headers(&mut params, mode);

		if mode == ApiMode::Anthropic {
			// Messages lists `max_tokens` as a required field; 0 is the caller's
			// "provider default".
			if params.max_tokens == 0 {
				params.max_tokens = MESSAGES_DEFAULT_MAX_TOKENS;
			}
			return anthropic_compat::chat_completion_with_x_api_key(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "isoquant",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: self.enforces_response_schema(&model),
				supports_required_tool_choice: self.supports_required_tool_choice(&model),
				supports_parallel_tool_calls: true,
				cache_marker: None,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		// Isoquant does not return cost; derive it from the catalog rates
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
#[path = "isoquant_tests.rs"]
mod tests;
