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
use crate::llm::utils::{normalize_model_name, PricingTuple};
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
// Except for qwen3.8-max's separately published $0.25 rate and DeepSeek V4 Pro,
// implicit cache hits cost 20% of uncached input.
const PRICING: &[PricingTuple] = &[
    ("qwen3.8-max", 2.00, 6.00, 2.00, 0.25),
    ("qwen3.8-flash", 0.113, 0.382, 0.113, 0.0226),
    // Dated Qwen 3.7 snapshots retain list price; moving aliases have current promos.
    ("qwen3.7-max-2026-06-08", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-2026-05-20", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-2026-05-17", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-preview", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max", 1.25, 3.75, 1.25, 0.25),
    ("qwen3.7-plus-2026-05-26", 0.40, 1.60, 0.40, 0.08),
    ("qwen3.7-plus", 0.32, 1.28, 0.32, 0.064),
    ("qwen3.6-plus", 0.50, 3.00, 0.50, 0.10),
    ("qwen3.6-flash", 0.25, 1.50, 0.25, 0.05),
    ("qwen3.5-flash", 0.10, 0.40, 0.10, 0.02),
    ("qwen3-coder-plus", 1.00, 5.00, 1.00, 0.20),
    ("qwen3-coder-flash", 0.30, 1.50, 0.30, 0.06),
    ("qwen3-vl-plus", 0.20, 1.60, 0.20, 0.04),
    ("qwen-max", 1.60, 6.40, 1.60, 0.32),
    ("qwen-plus", 0.40, 1.20, 0.40, 0.08),
    ("qwen-turbo", 0.05, 0.20, 0.05, 0.01),
    // Third-party models resold by Model Studio at Alibaba's own rates
    ("deepseek-v4-pro", 2.40, 4.80, 2.40, 0.24),
    ("deepseek-v4-flash", 0.20, 0.40, 0.20, 0.04),
    ("glm-5.2", 1.40, 4.40, 1.40, 0.28),
];

const QWEN_PLUS_LONG_CONTEXT_THRESHOLD: u64 = 256_000;

fn calculate_local_usage_cost(
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

    if normalize_model_name(model).contains("qwen3.7-plus")
        && total_input_tokens > QWEN_PLUS_LONG_CONTEXT_THRESHOLD
    {
        if normalize_model_name(model).contains("qwen3.7-plus-2026-05-26") {
            // Dated snapshot list-price tier for prompts in (256K, 1M].
            input = 1.20;
            output = 4.80;
            cache_write = 1.20;
            cache_read = 0.24;
        } else {
            // Moving alias: current 20%-off tier for prompts in (256K, 1M].
            input = 0.96;
            output = 3.84;
            cache_write = 0.96;
            cache_read = 0.192;
        }
    }

    Some(
        (input_tokens as f64 / 1_000_000.0) * input
            + (cache_write_tokens as f64 / 1_000_000.0) * cache_write
            + (cache_read_tokens as f64 / 1_000_000.0) * cache_read
            + (output_tokens as f64 / 1_000_000.0) * output,
    )
}

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

    /// Alibaba supports native JSON Schema for selected Qwen families. Other
    /// hosted models use shared forced-tool enforcement plus local validation.
    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, model: &str) -> bool {
        natively_enforces_response_schema(model)
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
                enforces_response_schema: natively_enforces_response_schema(&model),
                // Thinking-mode tool calls accept only auto/none. Schema repair
                // therefore uses auto plus explicit prompt guidance and local
                // validation instead of an unsupported required policy.
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                let input_tokens = usage.input_tokens;
                let cache_write_tokens = usage.cache_write_tokens;
                let cache_read_tokens = usage.cache_read_tokens;
                let output_tokens = usage.billable_output_tokens();
                usage.cost = calculate_local_usage_cost(
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

/// JSON Schema is native only for the Alibaba model families explicitly listed
/// by Model Studio. Snapshot suffixes inherit their family's capability.
fn natively_enforces_response_schema(model: &str) -> bool {
    let model = normalize_model_name(model);
    [
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.7-flash",
    ]
    .iter()
    .any(|prefix| model.starts_with(prefix))
}

#[cfg(test)]
#[path = "alibaba_tests.rs"]
mod tests;
