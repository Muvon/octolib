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

//! Groq provider implementation.
//!
//! Uses Groq's OpenAI-compatible endpoint at:
//! `https://api.groq.com/openai/v1/chat/completions`
//!
//! Hosts open-weight models (Llama, GPT-OSS, Qwen3, Kimi K2) on custom LPU
//! hardware. Cached input pricing is offered for selected GPT-OSS and Kimi
//! models.
//!
//! PRICING UPDATE: April 2026
//! Source: <https://groq.com/pricing>
//!
//! Configuration:
//! - `GROQ_API_KEY`: Required API key
//! - `GROQ_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::PricingTuple;
use anyhow::Result;
use std::env;

/// Groq provider
#[derive(Debug, Clone)]
pub struct GroqProvider;

impl Default for GroqProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GroqProvider {
    pub fn new() -> Self {
        Self
    }
}

const GROQ_API_KEY_ENV: &str = "GROQ_API_KEY";
const GROQ_API_URL_ENV: &str = "GROQ_API_URL";
const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

// Groq pricing (per 1M tokens in USD) - Apr 2026
// Source: https://groq.com/pricing
// Format: (model, input, output, cache_write, cache_read)
// cache_write priced at input rate (Groq does not bill a separate write fee)
const PRICING: &[PricingTuple] = &[
    // GPT-OSS — cached input pricing available
    ("openai/gpt-oss-120b", 0.15, 0.60, 0.15, 0.075),
    ("openai/gpt-oss-20b", 0.075, 0.30, 0.075, 0.0375),
    ("openai/gpt-oss-safeguard-20b", 0.075, 0.30, 0.075, 0.0375),
    // Qwen3.6 (preview) — replacement for the retired qwen3-32b / llama-4-scout
    ("qwen/qwen3.6-27b", 0.60, 3.00, 0.60, 0.60),
    // Qwen3.8-27B (Aug 2026) — multimodal; no cached-input discount published
    ("qwen/qwen3.8-27b", 0.80, 4.00, 0.80, 0.80),
    // Llama
    ("llama-3.3-70b-versatile", 0.59, 0.79, 0.59, 0.59),
    ("llama-3.1-8b-instant", 0.05, 0.08, 0.05, 0.05),
    (
        "meta-llama/llama-4-scout-17b-16e-instruct",
        0.11,
        0.34,
        0.11,
        0.11,
    ),
    // Qwen3
    ("qwen/qwen3-32b", 0.29, 0.59, 0.29, 0.29),
    // Kimi K2 (Moonshot served on Groq) — cached input pricing
    ("moonshotai/kimi-k2-instruct-0905", 1.00, 3.00, 1.00, 0.50),
];

// Models with documented prompt-caching support on Groq.
const CACHED_INPUT_MODELS: &[&str] = &[
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct-0905",
];

#[async_trait::async_trait]
impl AiProvider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(GROQ_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Groq API key not found in environment variable: {}",
                GROQ_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, model: &str) -> bool {
        let lower = model.to_ascii_lowercase();
        CACHED_INPUT_MODELS
            .iter()
            .any(|m| lower.contains(&m.to_ascii_lowercase()))
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try Groq-specific pricing first
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
        let api_url = get_api_url(GROQ_API_URL_ENV, GROQ_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "groq",
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
#[path = "groq_tests.rs"]
mod tests;
