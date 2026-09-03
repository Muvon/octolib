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

//! Meta Model API provider implementation.
//!
//! Uses Meta's OpenAI-compatible endpoint at:
//! `https://api.meta.ai/v1/chat/completions`
//!
//! Serves the Muse Spark family (`muse-spark-1.3`, `1.2`, `1.1` plus the
//! discounted `-contributor` variants) — multimodal reasoning models with a
//! 1,048,576-token context window. The old `api.llama.com` Llama API preview
//! was sunset on 2026-07-06 and is not used.
//!
//! PRICING UPDATE: September 2026
//! Source: <https://dev.meta.ai/docs/pricing-rate-limits>
//!
//! Configuration:
//! - `META_API_KEY` (or `MODEL_API_KEY`, the documented SDK variable): Required API key
//! - `META_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{is_model_in_pricing_table, PricingTuple};
use anyhow::Result;
use std::env;

/// Meta Model API provider
#[derive(Debug, Clone)]
pub struct MetaProvider;

impl Default for MetaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaProvider {
    pub fn new() -> Self {
        Self
    }
}

const META_API_KEY_ENV: &str = "META_API_KEY";
const MODEL_API_KEY_ENV: &str = "MODEL_API_KEY";
const META_API_URL_ENV: &str = "META_API_URL";
const META_API_URL: &str = "https://api.meta.ai/v1/chat/completions";

// Meta Model API pricing (per 1M tokens in USD) - Sep 2026
// Source: https://dev.meta.ai/docs/pricing-rate-limits
// Format: (model, input, output, cache_write, cache_read)
// Prompt caching is implicit: writes bill at the input rate, hits at the
// discounted cached-input rate. The -contributor variants trade lower rates
// for permission to train on prompts/completions. Contributor patterns must
// stay before their base versions because matching is substring-based.
const PRICING: &[PricingTuple] = &[
    // Contributor tier (1.3 / 1.2): $0.10 in / $0.20 out / $0.002 cached
    ("muse-spark-1.3-contributor", 0.10, 0.20, 0.10, 0.002),
    ("muse-spark-1.2-contributor", 0.10, 0.20, 0.10, 0.002),
    // Standard tier (1.1 / 1.2 / 1.3 share the same rates)
    ("muse-spark-1.3", 1.25, 4.25, 1.25, 0.15),
    ("muse-spark-1.2", 1.25, 4.25, 1.25, 0.15),
    ("muse-spark-1.1", 1.25, 4.25, 1.25, 0.15),
];

/// Context window shared by every Muse Spark version.
const MAX_INPUT_TOKENS: usize = 1_048_576;

#[async_trait::async_trait]
impl AiProvider for MetaProvider {
    fn name(&self) -> &str {
        "meta"
    }

    /// Closed first-party catalog: only priced Muse Spark models are accepted.
    fn supports_model(&self, model: &str) -> bool {
        is_model_in_pricing_table(model, PRICING)
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(META_API_KEY_ENV)
            .or_else(|_| env::var(MODEL_API_KEY_ENV))
            .map_err(|_| {
                anyhow::anyhow!(
                    "Meta Model API key not found in environment variables: {} or {}",
                    META_API_KEY_ENV,
                    MODEL_API_KEY_ENV
                )
            })
    }

    /// Prompt caching is implicit on Model API; hits report `cached_tokens`.
    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    /// Text, image, video, audio and PDF input on every Muse Spark version.
    fn supports_vision(&self, _model: &str) -> bool {
        true
    }

    fn supports_video(&self, _model: &str) -> bool {
        true
    }

    /// `response_format: json_schema` constrains decoding server-side, so
    /// responses are schema-enforced natively.
    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    /// Only `tool_choice: "auto"` is accepted; none/required/named return 400.
    fn supports_required_tool_choice(&self, _model: &str) -> bool {
        false
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        MAX_INPUT_TOKENS
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        crate::llm::utils::get_model_pricing(model, PRICING).map(
            |(input, output, cache_write, cache_read)| {
                crate::llm::types::ModelPricing::new(input, output, cache_write, cache_read)
            },
        )
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(META_API_URL_ENV, META_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "meta",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: true,
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Meta reports token counts but no cost in the response — derive it
        // from the local pricing table so downstream cost tracking sees a value.
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
#[path = "meta_tests.rs"]
mod tests;
