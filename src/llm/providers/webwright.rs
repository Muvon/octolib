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

//! WebWright provider implementation.
//!
//! Uses WebWright's OpenAI-compatible chat endpoint at:
//! `https://webwright.ai/v1/chat/completions`
//!
//! WebWright serves a single proprietary model, `wren-r1` (200k-token context,
//! `owned_by: "webwright"`). The catalogue is fixed and known (see `MODELS`),
//! so unknown model IDs are rejected rather than served by `wren-r1`. Because
//! `wren-r1` matches no reference model, every capability and the context
//! window are declared here instead of resolved from the reference tables.
//!
//! Reasoning: WebWright publishes low and medium reasoning modes on the
//! Standard tier and high-to-max on Premium. The client cannot see which tier
//! an account holds, so `reasoning_effort` is clamped to low/medium (the
//! `webwright` row in `openai_compat`'s hosted effort levels).
//!
//! Configuration:
//! - `WEBWRIGHT_API_KEY`: Required API key
//! - `WEBWRIGHT_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

/// WebWright provider
#[derive(Debug, Clone, Default)]
pub struct WebWrightProvider;

impl WebWrightProvider {
    pub fn new() -> Self {
        Self
    }
}

const WEBWRIGHT_API_KEY_ENV: &str = "WEBWRIGHT_API_KEY";
const WEBWRIGHT_API_URL_ENV: &str = "WEBWRIGHT_API_URL";
const WEBWRIGHT_API_URL: &str = "https://webwright.ai/v1/chat/completions";

/// (model id, max input tokens) — WebWright's `GET /v1/models` reports
/// `context_length: 200000` for the single deployed model.
const MODELS: &[(&str, usize)] = &[("wren-r1", 200_000)];

// WebWright pricing (per 1M tokens in USD) — Standard tier, Oct 2026
// Source: https://webwright.ai/pricing
// Format: (model, input, output, cache_write, cache_read)
// WebWright bills no separate cache-write fee, so cache writes price at the
// input rate; cached input reads at $0.10/1M.
const PRICING: &[PricingTuple] = &[("wren-r1", 0.40, 1.60, 0.40, 0.10)];

/// Case-insensitive lookup; requests are canonicalized to the table's exact ID.
fn find_model(model: &str) -> Option<&'static (&'static str, usize)> {
    MODELS.iter().find(|(id, _)| id.eq_ignore_ascii_case(model))
}

#[async_trait::async_trait]
impl AiProvider for WebWrightProvider {
    fn name(&self) -> &str {
        "webwright"
    }

    fn supports_model(&self, model: &str) -> bool {
        find_model(model).is_some()
    }

    fn supports_vision(&self, _model: &str) -> bool {
        // Text-only chat completions; the API documents no image input.
        false
    }

    fn supports_video(&self, _model: &str) -> bool {
        false
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // WebWright publishes a cached-input rate ($0.10/1M), so prefix
        // caching is billed and reported through the prompt token details.
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // No documented json_schema or json_object guarantee.
        false
    }

    /// WebWright is an OpenAI-compatible chat endpoint; it does not publish
    /// evidence of constrained decoding, so schema enforcement stays off.
    fn enforces_response_schema(&self, _model: &str) -> bool {
        false
    }

    fn supports_required_tool_choice(&self, _model: &str) -> bool {
        // No documented forced tool selection.
        false
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        find_model(model).map(|(_, max)| *max).unwrap_or(200_000)
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(WEBWRIGHT_API_KEY_ENV)
            .map_err(|_| crate::errors::missing_api_key("webwright", WEBWRIGHT_API_KEY_ENV).into())
    }

    fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
        let (input, output, cache_write, cache_read) =
            crate::llm::utils::get_model_pricing(model, PRICING)?;
        Some(ModelPricing::new(input, output, cache_write, cache_read))
    }

    async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(WEBWRIGHT_API_URL_ENV, WEBWRIGHT_API_URL);

        if let Some((canonical, _)) = find_model(&params.model) {
            params.model = canonical.to_string();
        }
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "webwright",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: false,
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // WebWright reports the billed cost in `usage.cost`; fall back to the
        // pricing table only when a response omits it.
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
#[path = "webwright_tests.rs"]
mod tests;
