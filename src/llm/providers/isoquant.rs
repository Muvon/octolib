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
//! Uses Isoquant's OpenAI-compatible chat endpoint at:
//! `https://api.isoquant.ai/v1/chat/completions`
//!
//! Models added by Isoquant work without a release. Any model id is accepted
//! and Isoquant itself refuses ones it does not serve. The public catalog at
//! `GET /v1/models` is fetched once per process, through [`preload`] or before
//! the first chat request, and supplies context window, vision, pricing and
//! reasoning levels. `KNOWN_MODELS` below is the offline fallback. A model
//! missing from the fetched catalog (added since, or visible to one account
//! only) still runs, on reference prices and with no reasoning level sent.
//!
//! Every chat request sends `Isoquant-ZDR: required`, which requires zero data
//! retention. Isoquant still records usage and billing metadata.
//!
//! Configuration:
//! - `ISOQUANT_API_KEY`: Required API key
//! - `ISOQUANT_API_URL`: Optional endpoint override; the catalog is read from
//!   the `/models` path beside it

use crate::llm::providers::openai_compat::{
	chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::reference_models::{get_reference_capabilities, get_reference_pricing};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, ReasoningEffort, SamplingSupport};
use crate::llm::utils::normalize_model_name;
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::env;
use std::sync::LazyLock;
use std::time::Duration;
use tokio::sync::OnceCell;

const ISOQUANT_API_KEY_ENV: &str = "ISOQUANT_API_KEY";
const ISOQUANT_API_URL_ENV: &str = "ISOQUANT_API_URL";
const ISOQUANT_API_URL: &str = "https://api.isoquant.ai/v1/chat/completions";
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const ZDR_HEADER: &str = "Isoquant-ZDR";
const ZDR_REQUIRED: &str = "required";
/// A slow catalog must not hold up the chat request waiting behind it.
const CATALOG_TIMEOUT: Duration = Duration::from_secs(10);
/// The catalog prices tokens in billionths of a dollar: 1000 of them per token
/// is $1 per 1M tokens.
const NANO_USD_PER_TOKEN_PER_USD_PER_1M: f64 = 1_000.0;
const KNOWN_CONTEXT_WINDOW: usize = 1_048_576;
const DEFAULT_MAX_INPUT_TOKENS: usize = 8_192;

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
/// `GET /v1/models`, 2026-11-07. Format: (model, input, output, cached input,
/// reasoning levels), prices per 1M tokens. Isoquant publishes no cache-write
/// price, so writes bill at the input rate. GLM-5.3 has no `none` level: the
/// host answers 400, and a request for no reasoning runs at `low`.
const KNOWN_MODELS: &[(&str, f64, f64, f64, &[&str])] = &[
	(
		"deepseek-v4.1-flash-1017-fast",
		0.035,
		0.37,
		0.005,
		&["none", "low", "high"],
	),
	("glm-5.3-flash", 0.07, 0.15, 0.025, &["low", "high", "max"]),
];

/// Models proven to decode against `json_schema`. A strict object schema with a
/// prompt demanding a JSON array returned the object on 3 of 3 calls per model
/// (`/v1/chat/completions`, 2026-11-07). Models added later take the forced-tool
/// path until probed.
const SCHEMA_ENFORCING_MODELS: &[&str] = &["deepseek-v4.1-flash-1017-fast", "glm-5.3-flash"];
/// The only model with a measured cache lifetime; see `cache_lifetime`.
const MEASURED_CACHE_MODEL: &str = "deepseek-v4.1-flash-1017-fast";
const MEASURED_CACHE_LIFETIME: std::time::Duration = std::time::Duration::from_secs(60 * 60);

#[derive(Debug, Clone)]
struct IsoquantModel {
	id: String,
	context_window: usize,
	vision: bool,
	structured_output: bool,
	required_tool_choice: bool,
	pricing: Option<ModelPricing>,
	/// Lowest first; empty when the model has no reasoning levels.
	efforts: Vec<(ReasoningEffort, &'static str)>,
}

static KNOWN_CATALOG: LazyLock<Vec<IsoquantModel>> = LazyLock::new(|| {
	KNOWN_MODELS
		.iter()
		.map(|(id, input, output, cached, efforts)| IsoquantModel {
			id: id.to_string(),
			context_window: KNOWN_CONTEXT_WINDOW,
			vision: true,
			structured_output: true,
			required_tool_choice: true,
			pricing: Some(ModelPricing::new(*input, *output, *input, *cached)),
			efforts: listed_efforts(efforts),
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
	chat_completions: Option<CatalogChatApi>,
}

#[derive(Deserialize)]
struct CatalogChatApi {
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

/// Keeps entries served on Chat Completions, the only API this provider calls.
fn parse_catalog(entries: Vec<CatalogEntry>) -> Vec<IsoquantModel> {
	entries
		.into_iter()
		.filter_map(|entry| {
			let details = entry.isoquant?;
			let chat = details.apis?.chat_completions?;
			Some(IsoquantModel {
				id: entry.id,
				context_window: details.context_length?,
				vision: chat.input_modalities.iter().any(|modality| modality == "image"),
				structured_output: chat.structured_outputs.iter().any(|format| format == "json_schema"),
				required_tool_choice: chat.tool_choice.iter().any(|choice| choice == "required"),
				pricing: details.pricing.as_ref().and_then(parse_pricing),
				efforts: details
					.reasoning
					.map(|reasoning| listed_efforts(&reasoning.efforts))
					.unwrap_or_default(),
			})
		})
		.collect()
}

fn models_url(api_url: &str) -> String {
	let base = api_url.strip_suffix(CHAT_COMPLETIONS_PATH).unwrap_or(api_url);
	format!("{base}/models")
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
		return Err(anyhow!("Isoquant model catalog lists no chat models"));
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
	load_catalog(&get_api_url(ISOQUANT_API_URL_ENV, ISOQUANT_API_URL)).await
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

/// The `reasoning_effort` value to send, or `None` to send nothing because the
/// model's levels are unknown.
pub(crate) fn reasoning_effort_value(model: &str, effort: ReasoningEffort) -> Option<&'static str> {
	select_effort(&known_model(model)?.efforts, effort)
}

/// Replaces any caller value, so a request cannot opt out of zero data retention.
fn require_zero_data_retention(params: &mut ChatCompletionParams) {
	let headers = params.extra_headers.get_or_insert_with(Default::default);
	headers.retain(|name, _| !name.eq_ignore_ascii_case(ZDR_HEADER));
	headers.insert(ZDR_HEADER.to_string(), ZDR_REQUIRED.to_string());
}

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
		match known_model(model) {
			Some(entry) => entry.vision,
			None => get_reference_capabilities(model).is_some_and(|capabilities| capabilities.vision),
		}
	}

	fn supports_video(&self, _model: &str) -> bool {
		// input_modalities: ["text", "image"]
		false
	}

	fn supports_caching(&self, model: &str) -> bool {
		// Automatic prefix caching; hits bill at the cached input rate
		known_model(model)
			.and_then(|entry| entry.pricing)
			.is_some_and(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
	}

	/// The docs give no lifetime. Measured 2026-11-07 on `/v1/chat/completions`: unique
	/// ~3.5k-token prompts re-sent once after an idle delay, two samples per delay.
	/// deepseek-v4.1-flash-1017-fast hit on every sample from 1m to 60m. glm-5.3-flash
	/// missed one sample at 1m and every sample from 15m, so it has no lifetime it is
	/// known to survive.
	fn cache_lifetime(&self, model: &str, _use_long_cache: bool) -> Option<std::time::Duration> {
		(normalize_model_name(model) == MEASURED_CACHE_MODEL).then_some(MEASURED_CACHE_LIFETIME)
	}

	fn supports_structured_output(&self, model: &str) -> bool {
		match known_model(model) {
			Some(entry) => entry.structured_output,
			None => get_reference_capabilities(model).is_some_and(|capabilities| capabilities.structured_output),
		}
	}

	fn enforces_response_schema(&self, model: &str) -> bool {
		let normalized = normalize_model_name(model);
		SCHEMA_ENFORCING_MODELS.iter().any(|id| normalized == *id)
	}

	fn supports_required_tool_choice(&self, model: &str) -> bool {
		known_model(model).is_some_and(|entry| entry.required_tool_choice)
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		match known_model(model) {
			Some(entry) => entry.context_window,
			None => get_reference_capabilities(model)
				.map(|capabilities| capabilities.max_input_tokens)
				.unwrap_or(DEFAULT_MAX_INPUT_TOKENS),
		}
	}

	fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
		// DeepSeek accepts temperature and top_p but answers 400 to top_k (live,
		// 2026-11-07), although its catalog entry lists none of them.
		SamplingSupport::TEMPERATURE_AND_TOP_P
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(ISOQUANT_API_KEY_ENV)
			.map_err(|_| crate::errors::missing_api_key("isoquant", ISOQUANT_API_KEY_ENV).into())
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		known_model(model)
			.and_then(|entry| entry.pricing)
			.or_else(|| get_reference_pricing(model))
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = get_api_url(ISOQUANT_API_URL_ENV, ISOQUANT_API_URL);

		// A failed load is retried on the next call; `KNOWN_MODELS` serves this one.
		if let Err(error) = load_catalog(&api_url).await {
			tracing::warn!("Isoquant model catalog unavailable: {error:#}");
		}

		require_zero_data_retention(&mut params);
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "isoquant",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: self.enforces_response_schema(&model),
				supports_required_tool_choice: self.supports_required_tool_choice(&model),
				supports_parallel_tool_calls: true,
			},
			self.supported_sampling_params(&model),
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
