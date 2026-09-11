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

//! Inception Labs provider implementation.
//!
//! Uses Inception's OpenAI-compatible chat endpoint at:
//! `https://api.inceptionlabs.ai/v1/chat/completions`
//!
//! Serves the Mercury diffusion LLM family. The chat catalogue is fixed and
//! known (see `MODELS`), so unknown model IDs are rejected — update the table
//! when Inception changes the catalogue (`GET /v1/models`). Mercury Edit 2 is
//! FIM/edit-only and intentionally absent: the chat endpoint 404s it.
//!
//! Sampling: temperature only, clamped to 0.5–1.0 upstream (out-of-range
//! values are reset to the model default with a warning). Reasoning:
//! `reasoning_effort` accepts `instant`/`low`/`medium`/`high`; XHigh and Max
//! collapse to `high` via the generic openai_compat ceiling. Prompt caching
//! is automatic; cached input is billed at a discount and reported via
//! `prompt_tokens_details.cached_tokens`.
//!
//! PRICING UPDATE: September 2026 (Mercury 2.5 at 80% launch discount)
//! Source: <https://docs.inceptionlabs.ai/get-started/models>
//!
//! Configuration:
//! - `INCEPTION_API_KEY`: Required API key
//! - `INCEPTION_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, SamplingSupport};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

/// Inception Labs provider
#[derive(Debug, Clone, Default)]
pub struct InceptionProvider;

impl InceptionProvider {
    pub fn new() -> Self {
        Self
    }
}

const INCEPTION_API_KEY_ENV: &str = "INCEPTION_API_KEY";
const INCEPTION_API_URL_ENV: &str = "INCEPTION_API_URL";
const INCEPTION_API_URL: &str = "https://api.inceptionlabs.ai/v1/chat/completions";

/// (model id, max input tokens) — from the Inception models table
/// (`GET /v1/models` reports `context_length`).
const MODELS: &[(&str, usize)] = &[("mercury-2.5", 260_000), ("mercury-2", 128_000)];

// Inception pricing (per 1M tokens in USD) - Sep 2026
// Source: https://docs.inceptionlabs.ai/get-started/models
// Format: (model, input, output, cache_write, cache_read)
// cache_write priced at input rate (Inception bills no separate write fee)
const PRICING: &[PricingTuple] = &[
    // Mercury 2.5 at 80% launch discount: $0.04 in / $0.004 cached / $0.15 out
    ("mercury-2.5", 0.04, 0.15, 0.04, 0.004),
    ("mercury-2", 0.25, 0.75, 0.25, 0.025),
];

/// Case-insensitive lookup; requests are canonicalized to the table's exact ID.
fn find_model(model: &str) -> Option<&'static (&'static str, usize)> {
    MODELS.iter().find(|(id, _)| id.eq_ignore_ascii_case(model))
}

#[async_trait::async_trait]
impl AiProvider for InceptionProvider {
    fn name(&self) -> &str {
        "inception"
    }

    fn supports_model(&self, model: &str) -> bool {
        find_model(model).is_some()
    }

    fn supports_vision(&self, _model: &str) -> bool {
        // Text-only models (input_modalities: ["text"])
        false
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Automatic prefix caching with discounted cached-input billing
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // supported_features: ["tools", "json_mode", "structured_outputs"]
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        // strict json_schema constrains output to match the schema exactly
        true
    }

    fn supports_required_tool_choice(&self, _model: &str) -> bool {
        // Documented: tool_choice "required" forces a tool call
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        find_model(model).map(|(_, max)| *max).unwrap_or(128_000)
    }

    fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
        // supported_sampling_parameters: ["temperature", "stop"]
        SamplingSupport::TEMPERATURE_ONLY
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(INCEPTION_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Inception API key not found in environment variable: {}",
                INCEPTION_API_KEY_ENV
            )
        })
    }

    fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
        let (input, output, cache_write, cache_read) =
            crate::llm::utils::get_model_pricing(model, PRICING)?;
        Some(ModelPricing::new(input, output, cache_write, cache_read))
    }

    async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(INCEPTION_API_URL_ENV, INCEPTION_API_URL);

        if let Some((canonical, _)) = find_model(&params.model) {
            params.model = canonical.to_string();
        }
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "inception",
                usage_fallback_cost: None,
                use_response_cost: false,
                enforces_response_schema: true,
                supports_required_tool_choice: true,
            },
            SamplingSupport::TEMPERATURE_ONLY,
            api_key,
            api_url,
            params,
        )
        .await?;

        // Inception does not return cost; derive it from the pricing table
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
#[path = "inception_tests.rs"]
mod tests;
