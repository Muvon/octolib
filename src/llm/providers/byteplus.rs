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
use crate::llm::utils::PricingTuple;
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
        // Fall back to reference pricing for third-party models
        crate::llm::reference_pricing::get_reference_pricing(model)
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
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Derive cost from pricing if not returned in the response
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                if let Some(pricing) = self.get_model_pricing(&model) {
                    usage.cost = Some(pricing.calculate_cost(
                        usage.input_tokens,
                        usage.cache_write_tokens,
                        usage.cache_read_tokens,
                        usage.output_tokens,
                    ));
                }
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = BytePlusProvider::new();
        assert!(provider.supports_model("seed-2-0-pro-260328"));
        assert!(provider.supports_model("seed-2-0-lite-260228"));
        assert!(provider.supports_model("dola-seed-2.0-pro"));
        assert!(provider.supports_model("glm-4-7-251222"));
        assert!(provider.supports_model("any-model"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = BytePlusProvider::new();
        assert_eq!(provider.name(), "byteplus");
        assert!(provider.supports_caching("any-model"));
        assert!(provider.supports_structured_output("any-model"));
    }

    #[test]
    fn test_pricing_seed_models() {
        let provider = BytePlusProvider::new();

        let p = provider.get_model_pricing("seed-2-0-pro-260328").unwrap();
        assert_eq!(p.input_price_per_1m, 0.50);
        assert_eq!(p.output_price_per_1m, 3.00);
        assert_eq!(p.cache_read_price_per_1m, 0.10);

        let p = provider.get_model_pricing("seed-2-0-mini-260215").unwrap();
        assert_eq!(p.input_price_per_1m, 0.10);
        assert_eq!(p.output_price_per_1m, 0.40);

        let p = provider.get_model_pricing("seed-1-6-flash-250715").unwrap();
        assert_eq!(p.input_price_per_1m, 0.075);
        assert_eq!(p.output_price_per_1m, 0.30);
    }

    #[test]
    fn test_pricing_coding_plan_aliases() {
        let provider = BytePlusProvider::new();

        let p = provider.get_model_pricing("dola-seed-2.0-pro").unwrap();
        assert_eq!(p.input_price_per_1m, 0.50);
        assert_eq!(p.output_price_per_1m, 3.00);

        let p = provider.get_model_pricing("dola-seed-2.0-lite").unwrap();
        assert_eq!(p.input_price_per_1m, 0.25);

        let p = provider.get_model_pricing("bytedance-seed-code").unwrap();
        assert_eq!(p.input_price_per_1m, 0.50);
    }

    #[test]
    fn test_pricing_falls_back_to_reference() {
        let provider = BytePlusProvider::new();
        // GLM-5.1 is not in local PRICING but is in reference_pricing
        let p = provider.get_model_pricing("glm-5.1").unwrap();
        assert!(p.input_price_per_1m > 0.0);
    }

    #[test]
    fn test_cost_calculation() {
        let provider = BytePlusProvider::new();
        let pricing = provider.get_model_pricing("seed-2-0-pro-260328").unwrap();

        // 1M input + 500K output, no cache
        let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
        let expected = 0.50 + 0.5 * 3.00; // $2.00
        assert!((cost - expected).abs() < 0.001);

        // With cache: 500K regular + 500K cache_read + 500K output
        let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
        let expected_cached = 0.25 + 0.5 * 0.10 + 0.5 * 3.00; // $1.80
        assert!((cost_cached - expected_cached).abs() < 0.001);
        assert!(cost_cached < cost);
    }
}
