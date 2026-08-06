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

//! Alibaba Cloud Model Studio (DashScope) provider implementation.
//!
//! Uses the OpenAI-compatible endpoint at:
//! `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`
//!
//! Hosts the Qwen family plus third-party models (DeepSeek, GLM) at Alibaba's
//! own rates. Mainland China accounts, Token Plan subscriptions and dedicated
//! workspace deployments use a different host — override `ALIBABA_API_URL`
//! with the full endpoint including `/chat/completions`.
//!
//! PRICING UPDATE: August 2026
//! Source: <https://www.alibabacloud.com/help/en/model-studio/model-pricing>
//!
//! Configuration:
//! - `ALIBABA_API_KEY`: Required API key
//! - `ALIBABA_API_URL`: Optional endpoint override (Token Plan, China, workspace)

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

/// Alibaba Cloud Model Studio provider
#[derive(Debug, Clone)]
pub struct AlibabaProvider;

impl Default for AlibabaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AlibabaProvider {
    pub fn new() -> Self {
        Self
    }
}

const ALIBABA_API_KEY_ENV: &str = "ALIBABA_API_KEY";
const ALIBABA_API_URL_ENV: &str = "ALIBABA_API_URL";
const ALIBABA_API_URL: &str =
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions";

// Model Studio international pricing (per 1M tokens in USD) - Aug 2026
// Source: https://www.alibabacloud.com/help/en/model-studio/model-pricing
// Format: (model, input, output, cache_write, cache_read)
// Context caching is implicit: writes bill at the input rate, hits at cache_read.
// Tiered models are priced at the 0-256K tier; longer contexts bill up to 4x more.
// cache_read follows Alibaba's 10%-of-input context-cache rate where the price
// page lists no explicit cache-hit column.
const PRICING: &[PricingTuple] = &[
    ("qwen3.8-max", 2.00, 6.00, 2.00, 0.20),
    ("qwen3.7-max", 2.50, 7.50, 2.50, 0.25),
    ("qwen3.7-plus", 0.40, 1.60, 0.40, 0.04),
    ("qwen3.6-plus", 0.50, 3.00, 0.50, 0.05),
    ("qwen3.6-flash", 0.25, 1.50, 0.25, 0.025),
    ("qwen3.5-flash", 0.10, 0.40, 0.10, 0.01),
    ("qwen3-coder-plus", 1.00, 5.00, 1.00, 0.10),
    ("qwen3-coder-flash", 0.30, 1.50, 0.30, 0.03),
    ("qwen3-vl-plus", 0.20, 1.60, 0.20, 0.02),
    ("qwen-max", 1.60, 6.40, 1.60, 0.16),
    ("qwen-plus", 0.40, 1.20, 0.40, 0.04),
    ("qwen-turbo", 0.05, 0.20, 0.05, 0.005),
    // Third-party models resold by Model Studio at Alibaba's own rates
    ("deepseek-v4-pro", 2.40, 4.80, 2.40, 0.24),
    ("deepseek-v4-flash", 0.20, 0.40, 0.20, 0.02),
    ("glm-5.2", 1.40, 4.40, 1.40, 0.14),
];

#[async_trait::async_trait]
impl AiProvider for AlibabaProvider {
    fn name(&self) -> &str {
        "alibaba"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(ALIBABA_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Alibaba API key not found in environment variable: {}",
                ALIBABA_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    // supports_vision, supports_video, get_max_input_tokens are resolved via
    // reference capabilities (trait defaults)

    /// Compatible mode maps `json_schema` onto `json_object`: the schema is not
    /// enforced, and the request is rejected outright unless the prompt itself
    /// contains the word "json". Never advertise structured output here.
    fn supports_structured_output(&self, _model: &str) -> bool {
        false
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
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
        let api_url = get_api_url(ALIBABA_API_URL_ENV, ALIBABA_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "alibaba",
                usage_fallback_cost: None,
                use_response_cost: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

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
        let provider = AlibabaProvider::new();
        assert!(provider.supports_model("qwen3.8-max"));
        assert!(provider.supports_model("qwen3-coder-plus"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = AlibabaProvider::new();
        assert_eq!(provider.name(), "alibaba");
        assert!(provider.supports_caching("qwen3.8-max"));
        assert!(!provider.supports_structured_output("qwen3.8-max"));
        assert!(!provider.enforces_response_schema("qwen3.8-max"));
        // qwen3.8-max carries a 1M context window via reference capabilities
        assert_eq!(provider.get_max_input_tokens("qwen3.8-max"), 1_000_000);
        // Verified live: 3.8-max/3.7-plus/3.6-flash take images and video,
        // 3.7-max rejects image parts outright
        assert!(provider.supports_vision("qwen3.8-max"));
        assert!(provider.supports_vision("qwen3.7-plus"));
        assert!(provider.supports_vision("qwen3.6-flash"));
        assert!(provider.supports_video("qwen3.8-max"));
        assert!(provider.supports_video("qwen3.6-flash"));
        assert!(!provider.supports_vision("qwen3.7-max"));
    }

    #[test]
    fn test_pricing_qwen() {
        let provider = AlibabaProvider::new();

        let p = provider.get_model_pricing("qwen3.8-max").unwrap();
        assert_eq!(p.input_price_per_1m, 2.00);
        assert_eq!(p.output_price_per_1m, 6.00);
        assert_eq!(p.cache_read_price_per_1m, 0.20);

        // Alibaba list price, not the discounted third-party reference price
        let p = provider.get_model_pricing("qwen3.7-max").unwrap();
        assert_eq!(p.input_price_per_1m, 2.50);
        assert_eq!(p.output_price_per_1m, 7.50);

        let p = provider.get_model_pricing("qwen3.6-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 0.25);
        assert_eq!(p.cache_read_price_per_1m, 0.025);

        // Dated aliases must resolve to their family
        let p = provider
            .get_model_pricing("deepseek-v4-flash-0731")
            .unwrap();
        assert_eq!(p.input_price_per_1m, 0.20);

        // Unversioned aliases must not shadow more specific entries
        let p = provider.get_model_pricing("qwen-plus-latest").unwrap();
        assert_eq!(p.input_price_per_1m, 0.40);
        assert_eq!(p.output_price_per_1m, 1.20);
    }

    #[test]
    fn test_pricing_falls_back_to_reference() {
        let provider = AlibabaProvider::new();
        // Not in the local table, resolved from reference pricing
        let p = provider.get_model_pricing("qwen3-max-2026-01-25").unwrap();
        assert!(p.input_price_per_1m > 0.0);
    }

    #[test]
    fn test_cost_calculation() {
        let provider = AlibabaProvider::new();
        let pricing = provider.get_model_pricing("qwen3.8-max").unwrap();

        // 1M input + 500K output, no cache
        let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
        let expected = 2.00 + 0.5 * 6.00; // $5.00
        assert!((cost - expected).abs() < 0.001);

        // Cache hits must be cheaper than fresh input
        let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
        assert!(cost_cached < cost);
    }
}
