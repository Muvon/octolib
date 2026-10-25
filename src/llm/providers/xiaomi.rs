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
//! Uses the MiMo Open Platform OpenAI-compatible chat endpoint at:
//! `https://api.xiaomimimo.com/v1/chat/completions`
//!
//! The chat catalogue is fixed and known (see `MODELS`), so unknown model IDs
//! are rejected. mimo-v2.5-pro and mimo-v2.5 were retired on 2026-10-21 and
//! their IDs no longer work upstream.
//!
//! Reasoning: no effort ladder — `thinking: {"type": "enabled" | "disabled"}`
//! is the only knob (on by default); the openai_compat layer maps any effort
//! to `enabled` and `ReasoningEffort::None` to `disabled`. In thinking mode
//! temperature/top_p are pinned upstream to 1.0/0.95. Tool-call turns must
//! replay their `reasoning_content` or the API returns 400.
//!
//! Structured output is `json_object` only (no `json_schema`), so schemas are
//! sent as prompt guidance and validated locally.
//!
//! PRICING UPDATE: October 2026 (overseas USD rates)
//! Source: <https://mimo.mi.com/docs/en-US/price/pay-as-you-go>
//!
//! Configuration:
//! - `XIAOMI_API_KEY`: Required API key (`sk-` pay-as-you-go key)
//! - `XIAOMI_API_URL`: Optional endpoint override (e.g. a Token Plan base URL
//!   with its `tp-` key)

use crate::llm::providers::openai_compat::{
	chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, SamplingSupport};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

/// Xiaomi MiMo provider
#[derive(Debug, Clone, Default)]
pub struct XiaomiProvider;

impl XiaomiProvider {
	pub fn new() -> Self {
		Self
	}
}

const XIAOMI_API_KEY_ENV: &str = "XIAOMI_API_KEY";
const XIAOMI_API_URL_ENV: &str = "XIAOMI_API_URL";
const XIAOMI_API_URL: &str = "https://api.xiaomimimo.com/v1/chat/completions";

/// (model id, max input tokens) — every text model has a 1M context window
/// and 128K max output (models page, Oct 2026).
const MODELS: &[(&str, usize)] = &[
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
		// Automatic prompt cache; hits are billed at the cache-hit rate and
		// reported via `prompt_tokens_details.cached_tokens`
		true
	}

	fn supports_structured_output(&self, _model: &str) -> bool {
		true
	}

	fn enforces_response_schema(&self, _model: &str) -> bool {
		// Only `json_object` mode exists; schemas are prompt guidance
		false
	}

	fn supports_required_tool_choice(&self, _model: &str) -> bool {
		// Only tool_choice "auto" is documented
		false
	}

	fn replays_thinking(&self, _model: &str) -> bool {
		// Assistant turns carry their `reasoning_content` back on every request
		true
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		find_model(model).map(|(_, max)| *max).unwrap_or(1_048_576)
	}

	fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
		// temperature and top_p are documented; top_k is not
		SamplingSupport::TEMPERATURE_AND_TOP_P
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(XIAOMI_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("xiaomi", XIAOMI_API_KEY_ENV).into())
	}

	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		let (input, output, cache_write, cache_read) = crate::llm::utils::get_model_pricing(model, PRICING)?;
		Some(ModelPricing::new(input, output, cache_write, cache_read))
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = get_api_url(XIAOMI_API_URL_ENV, XIAOMI_API_URL);

		if let Some((canonical, _)) = find_model(&params.model) {
			params.model = canonical.to_string();
		}
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "xiaomi",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: false,
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
			},
			self.supported_sampling_params(&model),
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
