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

//! Ollama provider implementation.
//!
//! Uses Ollama's OpenAI-compatible endpoint by default:
//! `https://ollama.com/v1/chat/completions`
//!
//! Configuration:
//! - `OLLAMA_API_KEY`: Optional API key (required for cloud/private deployments)
//! - `OLLAMA_API_URL`: Override endpoint
//!
//! Prompt caching is automatic. An idle model is unloaded, and its cache lost,
//! once `keep_alive` runs out, so requests set it from `use_long_cache` and a
//! pre-warm is a one-token completion that restarts the timer.

use crate::llm::config::CacheTTL;
use crate::llm::providers::openai_compat::{
	chat_completion as openai_compat_chat_completion, get_api_url, get_optional_api_key, OpenAiCompatConfig,
};
use crate::llm::traits::{AiProvider, KeepalivePolicy};
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{calculate_cost_from_pricing_table, get_model_pricing, PricingTuple};
use anyhow::Result;

/// Ollama provider
#[derive(Debug, Clone)]
pub struct OllamaProvider;

impl Default for OllamaProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl OllamaProvider {
	pub fn new() -> Self {
		Self
	}
}

const OLLAMA_API_KEY_ENV: &str = "OLLAMA_API_KEY";
const OLLAMA_API_URL_ENV: &str = "OLLAMA_API_URL";
const OLLAMA_API_URL: &str = "https://ollama.com/v1/chat/completions";
/// Ollama has no zero-output completion, so a pre-warm asks for the cheapest one.
const CACHE_PREWARM_MAX_TOKENS: u32 = 1;

/// Ollama Cloud's own rates, which differ from the reference rows it resells:
/// (model, input, output, cache_write, cache_read) per 1M tokens, checked against
/// https://ollama.com/pricing on 2026-11-07. Peak rates: DeepSeek bills half of
/// these off-peak. Ollama lists no write rate, and `-` for cached input means
/// input rate. `clef` is omitted because it publishes no output rate. Specific
/// patterns come first.
const PRICING: &[PricingTuple] = &[
	("beam", 0.125, 0.40, 0.125, 0.015),
	("deepseek-v4.1-flash", 0.30, 1.20, 0.30, 0.006),
	("deepseek-v4-pro", 1.32, 3.96, 1.32, 0.044),
	("gemma4", 0.14, 0.40, 0.14, 0.05),
	("glm-5.3-flash", 0.15, 0.50, 0.15, 0.03),
	("glm-5.3", 1.40, 4.40, 1.40, 0.26),
	("glm-5.2", 1.40, 4.40, 1.40, 0.26),
	("gpt-oss:120b", 0.15, 0.60, 0.15, 0.014),
	("gpt-oss:20b", 0.07, 0.30, 0.07, 0.035),
	("kimi-k3.1-pro", 2.00, 10.00, 2.00, 0.20),
	("kimi-k3", 3.00, 15.00, 3.00, 0.30),
	("kimi-k2.7-code", 0.95, 4.00, 0.95, 0.19),
	("kimi-k2.6", 0.95, 4.00, 0.95, 0.16),
	("mimo-v2.6-flash", 0.14, 0.28, 0.14, 0.0028),
	("mimo-v2.6-pro", 0.435, 0.87, 0.435, 0.0036),
	("minimax-m3.1-flash", 0.15, 0.60, 0.15, 0.015),
	("minimax-m3", 0.60, 2.40, 0.60, 0.12),
	("minimax-m2.7", 0.30, 1.20, 0.30, 0.06),
	("mistral-large-4", 0.70, 2.10, 0.70, 0.07),
	("mistral-large-3", 0.50, 1.50, 0.50, 0.50),
	("muse-spark-open", 1.25, 4.25, 1.25, 0.15),
	("nemotron-3-nano", 0.06, 0.24, 0.06, 0.06),
	("nemotron-3-super", 0.015, 0.60, 0.015, 0.015),
	("nemotron-3-ultra", 0.10, 3.00, 0.10, 0.10),
];

#[async_trait::async_trait]
impl AiProvider for OllamaProvider {
	fn name(&self) -> &str {
		"ollama"
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		Ok(get_optional_api_key(OLLAMA_API_KEY_ENV))
	}

	/// Ollama's OpenAI layer replays reasoning for the trailing assistant
	/// message only, except Kimi models that preserve it across every turn.
	fn replays_thinking(&self, model: &str) -> bool {
		crate::llm::providers::moonshot::preserves_historical_thinking(model)
	}

	/// Ollama reuses a matching prompt prefix on its own and reports the hit in
	/// `prompt_tokens_details.cached_tokens`; Ollama Cloud bills it at a cached rate.
	fn supports_caching(&self, _model: &str) -> bool {
		true
	}

	fn keepalive_policy(&self, _model: &str, use_long_cache: bool) -> Option<KeepalivePolicy> {
		// 90% of the `keep_alive` sent with each request leaves a 10% margin for
		// network latency and scheduler jitter.
		Some(KeepalivePolicy {
			interval: CacheTTL::from_long_cache_flag(use_long_cache).to_duration() * 9 / 10,
		})
	}

	fn supports_video(&self, _model: &str) -> bool {
		false
	}

	// supports_vision, supports_video, supports_structured_output, get_max_input_tokens
	// are resolved via reference capabilities (trait defaults)

	/// Ollama Cloud does not constrain decoding: a schema reaches the model only
	/// as a forced tool call, which it may answer with the wrong shape. Routers
	/// read this flag to send structured requests to a lane that guarantees it.
	fn enforces_response_schema(&self, _model: &str) -> bool {
		false
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		get_model_pricing(model, PRICING)
			.map(|(input, output, cache_write, cache_read)| {
				crate::llm::types::ModelPricing::new(input, output, cache_write, cache_read)
			})
			// Try reference pricing for cloud-equivalent cost estimation
			.or_else(|| crate::llm::reference_models::get_reference_pricing(model))
	}

	async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = get_api_url(OLLAMA_API_URL_ENV, OLLAMA_API_URL);
		let model = params.model.clone();

		if params.cache_prewarm {
			params.max_tokens = CACHE_PREWARM_MAX_TOKENS;
			// A one-token reply cannot pass schema validation. The format only
			// constrains decoding, so the cached prompt prefix stays the same.
			params.response_format = None;
		}

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "ollama",
				usage_fallback_cost: None,
				use_response_cost: true,
				enforces_response_schema: false,
				supports_required_tool_choice: true,
				supports_parallel_tool_calls: true,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		// Fill cost from the pricing tables if the API didn't return one
		if let Some(ref mut usage) = response.exchange.usage {
			if usage.cost.is_none() {
				usage.cost = calculate_cost_from_pricing_table(
					&model,
					PRICING,
					usage.input_tokens,
					0,
					usage.cache_read_tokens,
					usage.billable_output_tokens(),
				)
				.or_else(|| {
					crate::llm::reference_models::calculate_reference_cost(
						&model,
						usage.input_tokens,
						usage.cache_read_tokens,
						usage.billable_output_tokens(),
					)
				});
			}
		}

		Ok(response)
	}
}

#[cfg(test)]
#[path = "ollama_http_tests.rs"]
mod http_tests;
#[cfg(test)]
#[path = "ollama_tests.rs"]
mod tests;
