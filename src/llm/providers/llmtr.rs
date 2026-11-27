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

//! LLMTR provider implementation.
//!
//! Uses LLMTR's OpenAI-compatible endpoint at:
//! `https://llmtr.com/v1/chat/completions`
//!
//! Model IDs use the `vendor/model` form (e.g., `qwen/qwen3.8-27b-free`).
//!
//! Configuration:
//! - `LLMTR_API_KEY`: Required API key
//! - `LLMTR_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
	chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// LLMTR provider
#[derive(Debug, Clone)]
pub struct LlmtrProvider;

impl Default for LlmtrProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl LlmtrProvider {
	pub fn new() -> Self {
		Self
	}
}

pub(crate) const LLMTR_API_KEY_ENV: &str = "LLMTR_API_KEY";
const LLMTR_API_URL_ENV: &str = "LLMTR_API_URL";
const LLMTR_API_URL: &str = "https://llmtr.com/v1/chat/completions";
pub(crate) const LLMTR_MODELS_URL: &str = "https://llmtr.com/v1/models";

#[async_trait::async_trait]
impl AiProvider for LlmtrProvider {
	fn name(&self) -> &str {
		"llmtr"
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(LLMTR_API_KEY_ENV).map_err(|_| crate::errors::missing_api_key("llmtr", LLMTR_API_KEY_ENV).into())
	}

	fn supports_caching(&self, _model: &str) -> bool {
		false
	}

	/// Schema enforcement has not been verified against the LLMTR endpoint.
	fn enforces_response_schema(&self, _model: &str) -> bool {
		false
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		crate::llm::reference_models::get_reference_pricing(model)
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = get_api_url(LLMTR_API_URL_ENV, LLMTR_API_URL);
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "llmtr",
				usage_fallback_cost: None,
				use_response_cost: true,
				enforces_response_schema: false,
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		if let Some(ref mut usage) = response.exchange.usage {
			if usage.cost.is_none() {
				usage.cost = crate::llm::reference_models::calculate_reference_cost(
					&model,
					usage.input_tokens,
					usage.cache_read_tokens,
					usage.billable_output_tokens(),
				);
			}
		}

		Ok(response)
	}
}

#[cfg(test)]
#[path = "llmtr_tests.rs"]
mod tests;
