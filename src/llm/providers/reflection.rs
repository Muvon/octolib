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

//! Reflection AI provider implementation.
//!
//! Uses Reflection's OpenAI-compatible chat endpoint at:
//! `https://api.reflection.ai/openai/v1/chat/completions`
//!
//! Serves Beam, a text-only model that always reasons. Model ids are
//! case-sensitive upstream, so requests are canonicalized to the id in `MODELS`.
//!
//! Every earlier assistant turn must carry non-empty `reasoning_content`, or the
//! API answers 400. Turns without reasoning go back with a placeholder; see
//! `openai_compat::convert_messages`.
//!
//! Data retention is not chosen per request. Each response reports it in the
//! `reflection-data-retention` header, and a test key on 2026-11-07 reported
//! `standard`, under which request content may be kept.
//!
//! Configuration:
//! - `REFLECTION_API_KEY`: Required API key
//! - `REFLECTION_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
	chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, SamplingSupport};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

pub(crate) const REFLECTION_API_KEY_ENV: &str = "REFLECTION_API_KEY";
const REFLECTION_API_URL_ENV: &str = "REFLECTION_API_URL";
const REFLECTION_API_URL: &str = "https://api.reflection.ai/openai/v1/chat/completions";

/// (model id, max input tokens), from `GET /openai/v1/models`, 2026-11-07.
pub(crate) const MODELS: &[(&str, usize)] = &[("Beam-1.0-501B-A23B", 1_048_576)];
const DEFAULT_MAX_INPUT_TOKENS: usize = 1_048_576;

// Pricing (per 1M tokens in USD) - Nov 2026
// Format: (model, input, output, cache_write, cache_read)
// Source: Ollama Cloud's listing of Beam (https://ollama.com/library/beam).
// Reflection shows its own rates only after sign-in. No write fee is listed, so
// cache writes bill at the input rate. The reference row stays on OpenRouter's
// baseline because it prices Beam on other hosts.
const PRICING: &[PricingTuple] = &[("Beam-1.0-501B-A23B", 0.125, 0.40, 0.125, 0.015)];

/// Case-insensitive lookup; requests are canonicalized to the table's exact ID.
fn find_model(model: &str) -> Option<&'static (&'static str, usize)> {
	MODELS.iter().find(|(id, _)| id.eq_ignore_ascii_case(model))
}

/// Reflection AI provider
#[derive(Debug, Clone, Default)]
pub struct ReflectionProvider;

impl ReflectionProvider {
	pub fn new() -> Self {
		Self
	}
}

#[async_trait::async_trait]
impl AiProvider for ReflectionProvider {
	fn name(&self) -> &str {
		"reflection"
	}

	fn supports_model(&self, model: &str) -> bool {
		find_model(model).is_some()
	}

	fn supports_vision(&self, _model: &str) -> bool {
		// Image input is a 400 (live, 2026-11-07)
		false
	}

	fn supports_video(&self, _model: &str) -> bool {
		false
	}

	fn supports_caching(&self, _model: &str) -> bool {
		// Automatic: 12,224 of 12,273 prompt tokens cached on a repeat call (live, 2026-11-07)
		true
	}

	/// The docs give no lifetime. Measured 2026-11-07 on `/openai/v1/chat/completions`:
	/// unique ~36k-token prompts (a ~5k one never hit) re-sent once after an idle delay,
	/// two samples per delay. Every sample hit up to 10m; at 15m and 30m one of two
	/// missed, and from 45m every sample missed.
	fn cache_lifetime(&self, _model: &str, _use_long_cache: bool) -> Option<std::time::Duration> {
		Some(std::time::Duration::from_secs(10 * 60))
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		true
	}

	/// A strict object schema with a prompt demanding a JSON array returned the
	/// object on 3 of 3 Beam calls; `json_object` returned an object on 2 of 2
	/// (`/openai/v1/chat/completions`, 2026-11-07).
	fn enforces_response_schema(&self, _model: &str) -> bool {
		true
	}

	fn supports_required_tool_choice(&self, _model: &str) -> bool {
		// `required` and a named function both forced a call on an unrelated prompt (live, 2026-11-07)
		true
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		find_model(model).map_or(DEFAULT_MAX_INPUT_TOKENS, |(_, max)| *max)
	}

	fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
		// `top_k` is a 400 `unsupported_parameter` (live, 2026-11-07)
		SamplingSupport::TEMPERATURE_AND_TOP_P
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(REFLECTION_API_KEY_ENV)
			.map_err(|_| crate::errors::missing_api_key("reflection", REFLECTION_API_KEY_ENV).into())
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		let (input, output, cache_write, cache_read) = crate::llm::utils::get_model_pricing(model, PRICING)?;
		Some(ModelPricing::new(input, output, cache_write, cache_read))
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = get_api_url(REFLECTION_API_URL_ENV, REFLECTION_API_URL);

		if let Some((canonical, _)) = find_model(&params.model) {
			params.model = canonical.to_string();
		}
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "reflection",
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

		// Reflection does not return cost; derive it from the pricing table
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
#[path = "reflection_tests.rs"]
mod tests;
