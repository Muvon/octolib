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

//! BytePlus ModelArk provider implementation.
//!
//! Uses BytePlus's OpenAI-compatible endpoint at:
//! `https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions`
//!
//! Hosts ByteDance Seed models plus third-party models (GLM, DeepSeek, Kimi, etc.).
//! Also supports the Coding Plan subscription via endpoint override.
//!
//! PRICING UPDATE: April 2026
//! Source: <https://docs.byteplus.com/en/docs/ModelArk/1544106>
//!
//! Configuration:
//! - `BYTEPLUS_API_KEY`: Required API key
//! - `BYTEPLUS_API_URL`: Optional endpoint override (e.g. coding plan URL)

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// BytePlus ModelArk provider
#[derive(Debug, Clone)]
pub struct BytePlusProvider;

impl Default for BytePlusProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl BytePlusProvider {
    pub fn new() -> Self {
        Self
    }
}

const BYTEPLUS_API_KEY_ENV: &str = "BYTEPLUS_API_KEY";
const BYTEPLUS_API_URL_ENV: &str = "BYTEPLUS_API_URL";
const BYTEPLUS_API_URL: &str = "https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions";

// BytePlus ModelArk pricing (per 1M tokens in USD) - Apr 2026
// Source: https://docs.byteplus.com/en/docs/ModelArk/1544106
// Format: (model, input, output, cache_write, cache_read)
// cache_write = input price, cache_read = cache-hit price
const PRICING: &[PricingTuple] = &[
    // Seed 2.1 family (Aug 2026) — Turbo only; Pro has no published USD rates
    ("seed-2-1-turbo", 0.50, 2.50, 0.50, 0.50),
    // Seed 2.0 family (256K context)
    ("seed-2-0-pro", 0.50, 3.00, 0.50, 0.10),
    ("seed-2-0-code-preview", 0.50, 3.00, 0.50, 0.10),
    ("seed-2-0-lite", 0.25, 2.00, 0.25, 0.05),
    ("seed-2-0-mini", 0.10, 0.40, 0.10, 0.02),
    // Coding Plan aliases
    ("dola-seed-2.0-pro", 0.50, 3.00, 0.50, 0.10),
    ("dola-seed-2.0-lite", 0.25, 2.00, 0.25, 0.05),
    ("dola-seed-2.0-code", 0.50, 3.00, 0.50, 0.10),
    ("bytedance-seed-code", 0.50, 3.00, 0.50, 0.10),
    // Seed 1.x family (128K context)
    ("seed-1-8", 0.25, 2.00, 0.25, 0.05),
    ("seed-1-6-flash", 0.075, 0.30, 0.075, 0.015),
    ("seed-1-6", 0.25, 2.00, 0.25, 0.05),
    // Third-party models hosted on BytePlus (BytePlus-specific pricing)
    ("glm-4-7-251222", 0.60, 2.20, 0.60, 0.11),
    ("gpt-oss-120b-250805", 0.10, 0.50, 0.10, 0.00),
];

const SEED_2_LONG_CONTEXT_THRESHOLD: u64 = 128_000;

/// Calculate BytePlus usage cost, including the documented 2x Seed 2.0 tier
/// when the prompt exceeds 128K tokens.
fn calculate_usage_cost(
    model: &str,
    input_tokens: u64,
    cache_write_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let (mut input, mut output, mut cache_write, mut cache_read) =
        crate::llm::utils::get_model_pricing(model, PRICING)?;
    let total_input_tokens = input_tokens
        .saturating_add(cache_write_tokens)
        .saturating_add(cache_read_tokens);
    let normalized = normalize_model_name(model);
    if (normalized.contains("seed-2-0") || normalized.contains("seed-2.0"))
        && total_input_tokens > SEED_2_LONG_CONTEXT_THRESHOLD
    {
        input *= 2.0;
        output *= 2.0;
        cache_write *= 2.0;
        cache_read *= 2.0;
    }

    Some(
        (input_tokens as f64 / 1_000_000.0) * input
            + (cache_write_tokens as f64 / 1_000_000.0) * cache_write
            + (cache_read_tokens as f64 / 1_000_000.0) * cache_read
            + (output_tokens as f64 / 1_000_000.0) * output,
    )
}

#[async_trait::async_trait]
impl AiProvider for BytePlusProvider {
    fn name(&self) -> &str {
        "byteplus"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(BYTEPLUS_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "BytePlus API key not found in environment variable: {}",
                BYTEPLUS_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try local pricing table first (BytePlus-specific prices)
        if let Some((input, output, cache_write, cache_read)) =
            crate::llm::utils::get_model_pricing(model, PRICING)
        {
            return Some(crate::llm::types::ModelPricing::new(
                input,
                output,
                cache_write,
                cache_read,
            ));
        }
        crate::llm::reference_models::get_reference_pricing(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(BYTEPLUS_API_URL_ENV, BYTEPLUS_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "byteplus",
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

        // Derive cost from pricing if not returned in the response
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                let input_tokens = usage.input_tokens;
                let cache_write_tokens = usage.cache_write_tokens;
                let cache_read_tokens = usage.cache_read_tokens;
                let output_tokens = usage.billable_output_tokens();
                usage.cost = calculate_usage_cost(
                    &model,
                    input_tokens,
                    cache_write_tokens,
                    cache_read_tokens,
                    output_tokens,
                )
                .or_else(|| {
                    self.get_model_pricing(&model).map(|pricing| {
                        pricing.calculate_cost(
                            input_tokens,
                            cache_write_tokens,
                            cache_read_tokens,
                            output_tokens,
                        )
                    })
                });
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "byteplus_tests.rs"]
mod tests;
