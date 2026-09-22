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
//! PRICING UPDATE: September 2026
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

// Model Studio international pricing (per 1M tokens in USD) - Sep 2026
// Source: https://www.alibabacloud.com/help/en/model-studio/model-pricing plus the
// per-model pages linked from https://www.alibabacloud.com/help/en/model-studio/models
// Format: (model, input, output, cache_write, cache_read)
// List prices only; limited-time promotions are not tracked.
// Context caching is implicit: writes bill at the input rate, hits at cache_read
// (the published implicit cache-hit rate, or the explicit cache-read rate when
// that is the only one published). Models without context caching bill hits at
// the input rate.
// Tiered models are priced at their lowest tier; longer prompts bill up to 4x more.
// Busy/idle models are priced at the busy rate (idle is half).
const PRICING: &[PricingTuple] = &[
    ("qwen3.8-max", 2.00, 6.00, 2.00, 0.25),
    ("qwen3.8-flash", 0.15, 0.47, 0.15, 0.016),
    ("qwen3.7-max", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-plus", 0.40, 1.60, 0.40, 0.08),
    ("qwen3.7-flash", 0.03, 0.13, 0.03, 0.006),
    ("qwen3.6-max-preview", 1.30, 7.80, 1.30, 0.13),
    ("qwen3.6-plus", 0.50, 3.00, 0.50, 0.10),
    ("qwen3.6-flash", 0.25, 1.50, 0.25, 0.05),
    ("qwen3.5-plus", 0.40, 2.40, 0.40, 0.04),
    ("qwen3.5-flash", 0.10, 0.40, 0.10, 0.02),
    ("qwen3-coder-plus", 1.00, 5.00, 1.00, 0.20),
    ("qwen3-coder-flash", 0.30, 1.50, 0.30, 0.06),
    ("qwen3-vl-plus", 0.20, 1.60, 0.20, 0.04),
    ("qwen3-vl-flash", 0.05, 0.40, 0.05, 0.01),
    ("qwen3-max", 1.20, 6.00, 1.20, 0.24),
    ("qwen-vl-max", 0.80, 3.20, 0.80, 0.16),
    ("qwen-max", 1.60, 6.40, 1.60, 0.32),
    ("qwen-plus", 0.40, 1.20, 0.40, 0.08),
    ("qwen-flash", 0.05, 0.40, 0.05, 0.01),
    ("qwen-turbo", 0.05, 0.20, 0.05, 0.01),
    // Open-weight Qwen checkpoints; the 3.5/3.6 ones have no context caching.
    ("qwen3.8-2.4t-a95b", 2.00, 6.00, 2.00, 0.25),
    ("qwen3.8-27b", 0.50, 3.00, 0.50, 0.10),
    ("qwen3.6-35b-a3b", 0.375, 2.25, 0.375, 0.375),
    ("qwen3.6-27b", 0.60, 3.60, 0.60, 0.60),
    ("qwen3.5-397b-a17b", 0.60, 3.60, 0.60, 0.60),
    ("qwen3.5-122b-a10b", 0.40, 3.20, 0.40, 0.40),
    ("qwen3.5-35b-a3b", 0.25, 2.00, 0.25, 0.25),
    ("qwen3.5-27b", 0.30, 2.40, 0.30, 0.30),
    // Third-party models resold by Model Studio at Alibaba's own rates.
    // V4.1 Flash and the dated V4 snapshots bill busy/idle; the moving V4
    // aliases have flat rates.
    ("deepseek-v4.1-flash", 0.30, 1.20, 0.30, 0.03),
    ("deepseek-v4-pro-0813", 1.32, 3.96, 1.32, 0.132),
    ("deepseek-v4-pro", 2.40, 4.80, 2.40, 0.20),
    ("deepseek-v4-flash-0731", 0.44, 1.32, 0.44, 0.044),
    ("deepseek-v4-flash", 0.20, 0.40, 0.20, 0.04),
    ("glm-5.3", 1.40, 4.40, 1.40, 0.28),
    ("glm-5.2", 1.40, 4.40, 1.40, 0.28),
    ("kimi-k3", 3.00, 15.00, 3.00, 0.30),
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
        // List-price tier for prompts in (256K, 1M].
        input = 1.20;
        output = 4.80;
        cache_write = 1.20;
        cache_read = 0.24;
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
