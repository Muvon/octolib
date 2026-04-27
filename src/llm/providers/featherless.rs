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
//! Pricing is **subscription-based** ($10–$200/month) with concurrency limits
//! rather than per-token billing, so no provider-specific pricing table is
//! defined. Cost estimates fall back to reference pricing for well-known
//! open-weight models when available.
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

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Featherless uses subscription billing — no per-token pricing.
        // Fall back to reference pricing so cost estimates exist for popular
        // open-weight models served by the platform.
        crate::llm::reference_pricing::get_reference_pricing(model)
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
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Derive cost from reference pricing if not returned in the response
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
        let provider = FeatherlessProvider::new();
        assert!(provider.supports_model("Qwen/Qwen2.5-7B-Instruct"));
        assert!(provider.supports_model("meta-llama/Meta-Llama-3.1-8B-Instruct"));
        assert!(provider.supports_model("mistralai/Mistral-7B-Instruct-v0.3"));
        assert!(provider.supports_model("any-future-model"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = FeatherlessProvider::new();
        assert_eq!(provider.name(), "featherless");
        assert!(provider.supports_structured_output("any-model"));
        assert!(!provider.supports_caching("Qwen/Qwen2.5-7B-Instruct"));
        assert!(!provider.supports_caching("any-model"));
    }

    #[test]
    fn test_pricing_reference_fallback() {
        let provider = FeatherlessProvider::new();
        // Reference pricing should resolve for well-known open-weight models
        let pricing = provider.get_model_pricing("meta-llama/Llama-3.1-8B-Instruct");
        assert!(pricing.is_some());
        let p = pricing.unwrap();
        assert!(p.input_price_per_1m >= 0.0);
        assert!(p.output_price_per_1m >= 0.0);
    }
}
