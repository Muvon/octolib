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

//! Fireworks AI provider implementation.
//!
//! Uses the OpenAI-compatible endpoint at:
//! `https://api.fireworks.ai/inference/v1/chat/completions`
//!
//! Fireworks hosts a large catalog of open-weight and frontier models
//! (Kimi K2, DeepSeek V3/V4, GLM 4.x/5.x, Qwen 3, Llama 4, gpt-oss, etc.) using
//! `accounts/fireworks/models/<name>` model IDs. As an aggregator we accept any
//! non-empty model string. Current serverless headline models use Fireworks'
//! published route prices; other models retain shared reference estimates.
//!
//! Caching: Fireworks performs automatic prompt-prefix caching. Cached input
//! tokens are surfaced via `usage.prompt_tokens_details.cached_tokens` (already
//! parsed by the shared `openai_compat` layer) and billed at the cached rate
//! at the Fireworks route's cached rate when one is published.
//!
//! Source: <https://docs.fireworks.ai/api-reference/post-chatcompletions>
//!
//! Configuration:
//! - `FIREWORKS_API_KEY`: Required API key
//! - `FIREWORKS_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Fireworks AI provider
#[derive(Debug, Clone)]
pub struct FireworksProvider;

impl Default for FireworksProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FireworksProvider {
    pub fn new() -> Self {
        Self
    }
}

const FIREWORKS_API_KEY_ENV: &str = "FIREWORKS_API_KEY";
const FIREWORKS_API_URL_ENV: &str = "FIREWORKS_API_URL";
const FIREWORKS_API_URL: &str = "https://api.fireworks.ai/inference/v1/chat/completions";

/// Fireworks Standard serverless prices per 1M tokens, verified Aug 26, 2026.
/// Format: (model path pattern, input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
    ("qwen3p8-2p4t-a95b", 2.00, 6.00, 2.00, 0.25),
    ("qwen3p7-plus", 0.40, 1.60, 0.40, 0.08),
    ("deepseek-v4-pro-0813", 1.32, 3.96, 1.32, 0.044),
    ("deepseek-v4-pro", 1.74, 3.48, 1.74, 0.145),
    ("deepseek-v4-flash", 0.22, 0.66, 0.22, 0.007),
    ("kimi-k2p7-code", 0.95, 4.00, 0.95, 0.19),
    ("kimi-k3", 3.00, 15.00, 3.00, 0.30),
    ("minimax-m3", 0.30, 1.20, 0.30, 0.06),
    ("glm-5p2", 1.40, 4.40, 1.40, 0.14),
];

fn fireworks_model_pricing(model: &str) -> Option<crate::llm::types::ModelPricing> {
    let (input, output, cache_write, cache_read) = get_model_pricing(model, PRICING)?;
    Some(crate::llm::types::ModelPricing::new(
        input,
        output,
        cache_write,
        cache_read,
    ))
}

#[async_trait::async_trait]
impl AiProvider for FireworksProvider {
    fn name(&self) -> &str {
        "fireworks"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(FIREWORKS_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Fireworks API key not found in environment variable: {}",
                FIREWORKS_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Fireworks performs automatic prompt-prefix caching across hosted
        // text/vision LLMs and reports `cached_tokens` in usage.
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // Fireworks supports `response_format` (json_object / json_schema /
        // grammar) on the chat completions endpoint for all served models.
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        let normalized = normalize_model_name(model);
        if normalized.contains("qwen3p8-2p4t-a95b")
            || normalized.contains("qwen3p7-plus")
            || normalized.contains("kimi-k2p7-code")
        {
            262_144
        } else if normalized.contains("minimax-m3") {
            512_000
        } else if normalized.contains("kimi-k3")
            || normalized.contains("glm-5p2")
            || normalized.contains("deepseek-v4")
        {
            1_040_000
        } else {
            crate::llm::reference_models::get_reference_capabilities(model)
                .map(|caps| caps.max_input_tokens)
                .unwrap_or(262_144)
        }
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        fireworks_model_pricing(model)
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(FIREWORKS_API_URL_ENV, FIREWORKS_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "fireworks",
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

        // Derive cost from Fireworks route pricing, with reference fallback for
        // unlisted models. Fireworks does not return cost in the response.
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
#[path = "fireworks_tests.rs"]
mod tests;
