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

/// Cerebras supported model IDs (Sep 2026 docs)
/// Source: https://inference-docs.cerebras.ai/models/overview
const SUPPORTED_MODELS: &[&str] = &[
    // Shared-tier production models
    "gpt-oss-120b",
    "qwen-3.8-27b",
];

/// Cerebras model pricing (per 1M tokens in USD)
/// Format: (model, input, output, cache_write, cache_read)
///
/// Source: https://www.cerebras.ai/pricing (gpt-oss-120b, checked Mar 31,
/// 2026) and https://inference-docs.cerebras.ai/models/qwen-3.8-27b
/// (checked Sep 13, 2026). No separate cache rates are published, so
/// cache fields mirror input.
const PRICING: &[PricingTuple] = &[
    ("qwen-3.8-27b", 0.99, 1.49, 0.99, 0.99),
    ("gpt-oss-120b", 0.35, 0.75, 0.35, 0.35),
];

fn calculate_cost(
    model: &str,
    input_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    calculate_cost_from_pricing_table(
        model,
        PRICING,
        input_tokens,
        0,
        cache_read_tokens,
        output_tokens,
    )
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

    // supports_vision is resolved via reference capabilities (trait default)

    fn supports_video(&self, model: &str) -> bool {
        // The Cerebras qwen-3.8-27b endpoint accepts text and base64-encoded
        // PNG/JPEG images only, unlike the native route that also takes video.
        if normalize_model_name(model) == "qwen-3.8-27b" {
            return false;
        }
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.video)
            .unwrap_or(false)
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Cerebras serves Qwen 3.8 27B at 128K (131,072) on paid tiers —
        // below the 262K native context in the reference tables.
        if normalize_model_name(model) == "qwen-3.8-27b" {
            return 131_072;
        }
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.max_input_tokens)
            .unwrap_or(262_144)
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
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
                enforces_response_schema: true,
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                usage.cost = calculate_cost(
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
#[path = "cerebras_tests.rs"]
mod tests;
