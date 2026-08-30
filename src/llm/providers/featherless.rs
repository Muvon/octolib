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

//! Featherless provider implementation.
//!
//! Uses Featherless's OpenAI-compatible endpoint at:
//! `https://api.featherless.ai/v1/chat/completions`
//!
//! Featherless is a serverless inference platform exposing a large catalogue of
//! open-weight models (Qwen, Llama, Mistral, DeepSeek, RWKV, QRWKV) using
//! HuggingFace-style namespaced model IDs (e.g. `Qwen/Qwen2.5-7B-Instruct`).
//!
//! Feather Chat is subscription-based. Feather Developer uses prepaid credits
//! and charges successful requests from published per-model token rates.
//!
//! Source: <https://featherless.ai/docs/api-overview-and-common-options>
//!
//! Configuration:
//! - `FEATHERLESS_API_KEY`: Required API key
//! - `FEATHERLESS_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{get_model_pricing, PricingTuple};
use anyhow::Result;
use std::env;

/// Featherless provider
#[derive(Debug, Clone)]
pub struct FeatherlessProvider;

impl Default for FeatherlessProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatherlessProvider {
    pub fn new() -> Self {
        Self
    }
}

const FEATHERLESS_API_KEY_ENV: &str = "FEATHERLESS_API_KEY";
const FEATHERLESS_API_URL_ENV: &str = "FEATHERLESS_API_URL";
const FEATHERLESS_API_URL: &str = "https://api.featherless.ai/v1/chat/completions";

/// Feather Developer request prices per 1M tokens, verified Aug 26, 2026.
/// Format: (model ID pattern, input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
    ("deepseek-ai/DeepSeek-V4-Flash-0731", 0.14, 0.28, 0.14, 0.03),
    ("deepseek-ai/DeepSeek-V4-Flash", 0.14, 0.28, 0.14, 0.03),
    ("deepseek-ai/DeepSeek-V4-Pro", 1.60, 3.20, 1.60, 0.20),
    ("deepseek-ai/DeepSeek-V3.2", 0.2995, 0.45, 0.2995, 0.06),
    ("zai-org/GLM-5.2", 0.75, 2.40, 0.75, 0.15),
    ("moonshotai/Kimi-K3", 2.00, 10.00, 2.00, 0.30),
    ("MiniMaxAI/MiniMax-M3", 0.55, 2.20, 0.55, 0.06),
    ("google/gemma-4-31B", 0.12, 0.36, 0.12, 0.10),
    ("google/gemma-4-26B", 0.07, 0.34, 0.07, 0.05),
    ("openai/gpt-oss-120b", 0.10, 0.55, 0.10, 0.02),
    ("openai/gpt-oss-20b", 0.04, 0.15, 0.04, 0.04),
];

fn featherless_model_pricing(model: &str) -> Option<crate::llm::types::ModelPricing> {
    let (input, output, cache_write, cache_read) = get_model_pricing(model, PRICING)?;
    Some(crate::llm::types::ModelPricing::new(
        input,
        output,
        cache_write,
        cache_read,
    ))
}

#[async_trait::async_trait]
impl AiProvider for FeatherlessProvider {
    fn name(&self) -> &str {
        "featherless"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(FEATHERLESS_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Featherless API key not found in environment variable: {}",
                FEATHERLESS_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, model: &str) -> bool {
        featherless_model_pricing(model)
            .map(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
            .unwrap_or(false)
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        featherless_model_pricing(model)
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(FEATHERLESS_API_URL_ENV, FEATHERLESS_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "featherless",
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

        // Derive cost from Feather Developer pricing, with reference fallback.
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
#[path = "featherless_tests.rs"]
mod tests;
