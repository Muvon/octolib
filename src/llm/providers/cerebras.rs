// Copyright 2025 Muvon Un Limited
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

//! Cerebras provider implementation.
//!
//! Uses Cerebras OpenAI-compatible endpoint by default:
//! `https://api.cerebras.ai/v1/chat/completions`
//!
//! Configuration:
//! - `CEREBRAS_API_KEY`: Required API key
//! - `CEREBRAS_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{
    calculate_cost_from_pricing_table, get_model_pricing, normalize_model_name, PricingTuple,
};
use anyhow::Result;
use std::env;

/// Cerebras provider
#[derive(Debug, Clone)]
pub struct CerebrasProvider;

impl Default for CerebrasProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CerebrasProvider {
    pub fn new() -> Self {
        Self
    }
}

const CEREBRAS_API_KEY_ENV: &str = "CEREBRAS_API_KEY";
const CEREBRAS_API_URL_ENV: &str = "CEREBRAS_API_URL";
const CEREBRAS_API_URL: &str = "https://api.cerebras.ai/v1/chat/completions";

/// Cerebras supported model IDs (Mar 2026 docs)
/// Source: https://inference-docs.cerebras.ai/models/overview
const SUPPORTED_MODELS: &[&str] = &[
    // Production models
    "gpt-oss-120b",
    "llama-3.1-8b",
    // Preview models
    "qwen-3-235b-a22b-instruct-2507",
    "zai-glm-4.7",
];

/// Cerebras model pricing (per 1M tokens in USD)
/// Format: (model, input, output, cache_write, cache_read)
///
/// Source: https://www.cerebras.ai/pricing (checked Mar 31, 2026)
const PRICING: &[PricingTuple] = &[
    ("zai-glm-4.7", 2.25, 2.75, 2.25, 2.25),
    ("qwen-3-235b-a22b-instruct-2507", 0.60, 1.20, 0.60, 0.60),
    ("gpt-oss-120b", 0.35, 0.75, 0.35, 0.35),
    ("llama-3.1-8b", 0.10, 0.10, 0.10, 0.10),
];

fn calculate_cost(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    calculate_cost_from_pricing_table(model, PRICING, input_tokens, 0, 0, output_tokens)
}

#[async_trait::async_trait]
impl AiProvider for CerebrasProvider {
    fn name(&self) -> &str {
        "cerebras"
    }

    fn supports_model(&self, model: &str) -> bool {
        let model_norm = normalize_model_name(model);
        SUPPORTED_MODELS
            .iter()
            .any(|m| normalize_model_name(m) == model_norm)
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(CEREBRAS_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cerebras API key not found in environment variable: {}",
                CEREBRAS_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        128_000
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            get_model_pricing(model, PRICING)?;
        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(CEREBRAS_API_URL_ENV, CEREBRAS_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "cerebras",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                usage.cost = calculate_cost(&model, usage.input_tokens, usage.output_tokens);
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
        let provider = CerebrasProvider::new();
        assert!(provider.supports_model("llama-3.3-70b"));
        assert!(provider.supports_model("gpt-oss-120b"));
        assert!(provider.supports_model("QWEN-3-32B"));
        assert!(!provider.supports_model(""));
        assert!(!provider.supports_model("random-model"));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = CerebrasProvider::new();
        assert_eq!(provider.name(), "cerebras");
        assert!(!provider.supports_caching("any-model"));
        assert!(!provider.supports_vision("any-model"));
        assert!(provider.supports_structured_output("any-model"));
        assert_eq!(provider.get_max_input_tokens("any-model"), 128_000);
    }

    #[test]
    fn test_pricing_support_partial() {
        let provider = CerebrasProvider::new();
        assert!(provider.get_model_pricing("llama-3.1-8b").is_some());
        assert!(provider
            .get_model_pricing("qwen-3-235b-a22b-thinking-2507")
            .is_some());
        assert!(provider.get_model_pricing("llama-3.3-70b").is_none());
        assert!(crate::llm::utils::is_model_in_pricing_table(
            "LLAMA-3.1-8B",
            PRICING
        ));
    }
}
