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

//! Cloudflare Workers AI provider implementation
//!
//! Uses the OpenAI-compatible endpoint at
//! `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions`.
//!
//! Authentication: Requires API token and Account ID.
//!
//! **How to get credentials:**
//! 1. Cloudflare Dashboard → My Profile → API Tokens
//! 2. Create Token → Use template "Workers AI" or create custom with Workers AI permissions
//! 3. Copy the API token
//! 4. Get Account ID from Cloudflare Dashboard → Workers & Pages (in URL or sidebar)
//! 5. Set environment variables:
//!    - export CLOUDFLARE_API_KEY="your-api-token"
//!    - export CLOUDFLARE_ACCOUNT_ID="your-account-id"
//!    - export CLOUDFLARE_API_URL="..." (optional endpoint override)
//!
//! The API token is sent as a Bearer token in the Authorization header.
//!
//! Frontier models (DeepSeek V4, GLM-5.x, Kimi K2.6/K2.7) require the Workers
//! Paid plan or prepaid AI Gateway credits; on the Free plan they return
//! HTTP 403 with Cloudflare error 5035.

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Cloudflare Workers AI provider
#[derive(Debug, Clone)]
pub struct CloudflareWorkersAiProvider;

impl Default for CloudflareWorkersAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudflareWorkersAiProvider {
    pub fn new() -> Self {
        Self
    }

    /// Get Cloudflare API token
    fn get_api_token(&self) -> Result<String> {
        env::var(CLOUDFLARE_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cloudflare API token not found. Set {} environment variable.\n\
                To create an API token:\n\
                1. Cloudflare Dashboard → My Profile → API Tokens\n\
                2. Create Token → Use 'Workers AI' template or create custom\n\
                3. Ensure token has Workers AI permissions",
                CLOUDFLARE_API_KEY_ENV
            )
        })
    }

    /// Get Cloudflare Account ID
    fn get_account_id(&self) -> Result<String> {
        env::var(CLOUDFLARE_ACCOUNT_ID_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cloudflare Account ID not found. Set {} environment variable.\n\
                Find your Account ID in Cloudflare Dashboard → Workers & Pages (in URL or sidebar)",
                CLOUDFLARE_ACCOUNT_ID_ENV
            )
        })
    }
}

const CLOUDFLARE_API_KEY_ENV: &str = "CLOUDFLARE_API_KEY";
const CLOUDFLARE_ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_API_URL_ENV: &str = "CLOUDFLARE_API_URL";

/// Cloudflare Workers AI text-generation prices per 1M tokens, verified
/// Sep 18, 2026 against the pricing page and each model page. Prompt caching
/// is implicit: cache writes bill at the input rate and hits at the cached
/// rate; models without a cached rate bill hits at the input rate.
/// Longer IDs precede the IDs they contain because lookup is a substring match.
/// Format: (model ID, input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
    ("@cf/meta/llama-3.2-1b-instruct", 0.027, 0.201, 0.027, 0.027),
    (
        "@cf/meta/llama-3.2-3b-instruct",
        0.0509,
        0.335,
        0.0509,
        0.0509,
    ),
    (
        "@cf/meta/llama-3.2-11b-vision-instruct",
        0.0485,
        0.676,
        0.0485,
        0.0485,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
        0.045,
        0.384,
        0.045,
        0.045,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8",
        0.152,
        0.287,
        0.152,
        0.152,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-awq",
        0.123,
        0.266,
        0.123,
        0.123,
    ),
    ("@cf/meta/llama-3.1-8b-instruct", 0.282, 0.827, 0.282, 0.282),
    (
        "@cf/meta/llama-3.1-70b-instruct-fp8-fast",
        0.293,
        2.253,
        0.293,
        0.293,
    ),
    (
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        0.293,
        2.253,
        0.293,
        0.293,
    ),
    (
        "@cf/meta/llama-3-8b-instruct-awq",
        0.123,
        0.266,
        0.123,
        0.123,
    ),
    ("@cf/meta/llama-3-8b-instruct", 0.282, 0.827, 0.282, 0.282),
    ("@cf/meta/llama-2-7b-chat-fp16", 0.556, 6.667, 0.556, 0.556),
    ("@cf/meta/llama-guard-3-8b", 0.484, 0.030, 0.484, 0.484),
    (
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        0.270,
        0.850,
        0.270,
        0.270,
    ),
    (
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        0.497,
        4.881,
        0.497,
        0.497,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        0.440,
        1.320,
        0.440,
        0.014,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        1.320,
        3.960,
        1.320,
        0.044,
    ),
    (
        "@cf/mistral/mistral-7b-instruct-v0.1",
        0.110,
        0.190,
        0.110,
        0.110,
    ),
    (
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        0.351,
        0.555,
        0.351,
        0.351,
    ),
    ("@cf/google/gemma-3-12b-it", 0.345, 0.556, 0.345, 0.345),
    ("@cf/google/gemma-4-26b-a4b-it", 0.100, 0.300, 0.100, 0.100),
    (
        "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
        0.351,
        0.555,
        0.351,
        0.351,
    ),
    ("@cf/qwen/qwq-32b", 0.660, 1.000, 0.660, 0.660),
    (
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        0.660,
        1.000,
        0.660,
        0.660,
    ),
    ("@cf/qwen/qwen3-30b-a3b-fp8", 0.0509, 0.335, 0.0509, 0.0509),
    // The cached rate is on the model page only; the pricing page omits it.
    ("@cf/qwen/qwen3.8-27b", 0.450, 3.200, 0.450, 0.050),
    ("@cf/openai/gpt-oss-120b", 0.350, 0.750, 0.350, 0.350),
    ("@cf/openai/gpt-oss-20b", 0.200, 0.300, 0.200, 0.200),
    (
        "@cf/ibm-granite/granite-4.0-h-micro",
        0.017,
        0.112,
        0.017,
        0.017,
    ),
    ("@cf/zai-org/glm-4.7-flash", 0.0605, 0.400, 0.0605, 0.0605),
    ("@cf/zai-org/glm-5.2", 1.400, 4.400, 1.400, 0.260),
    ("@cf/zai-org/glm-5.3-flash", 0.150, 0.500, 0.150, 0.030),
    ("@cf/zai-org/glm-5.3", 1.400, 4.400, 1.400, 0.260),
    (
        "@cf/nvidia/nemotron-3-120b-a12b",
        0.500,
        1.500,
        0.500,
        0.500,
    ),
    ("@cf/moonshotai/kimi-k2.5", 0.600, 3.000, 0.600, 0.100),
    ("@cf/moonshotai/kimi-k2.6", 0.950, 4.000, 0.950, 0.160),
    ("@cf/moonshotai/kimi-k2.7-code", 0.950, 4.000, 0.950, 0.190),
];

/// Per-model facts from the Workers AI model pages, verified Sep 18, 2026.
/// The last flag marks models whose request schema is the OpenAI chat shape
/// (`tool_choice` incl. "required", `reasoning_effort` low|medium|high,
/// `response_format` json_schema); the rest use the legacy Workers AI schema.
/// Longer IDs precede the IDs they contain because lookup is a substring match.
/// Format: (model ID, context window, vision, function calling, OpenAI-shaped schema).
type ModelFacts = (&'static str, usize, bool, bool, bool);
const MODELS: &[ModelFacts] = &[
    (
        "@cf/meta/llama-3.2-1b-instruct",
        60_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.2-3b-instruct",
        80_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.2-11b-vision-instruct",
        128_000,
        true,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8",
        32_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        24_000,
        false,
        true,
        false,
    ),
    ("@cf/meta/llama-guard-3-8b", 131_072, false, false, false),
    (
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        131_000,
        true,
        true,
        false,
    ),
    (
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        80_000,
        false,
        false,
        false,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        1_048_576,
        false,
        true,
        true,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        1_048_576,
        false,
        true,
        true,
    ),
    (
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        128_000,
        false,
        true,
        false,
    ),
    ("@cf/google/gemma-4-26b-a4b-it", 256_000, true, true, true),
    (
        "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
        128_000,
        false,
        false,
        false,
    ),
    ("@cf/qwen/qwq-32b", 24_000, false, false, false),
    (
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        32_768,
        false,
        false,
        false,
    ),
    ("@cf/qwen/qwen3-30b-a3b-fp8", 32_768, false, true, false),
    ("@cf/qwen/qwen3.8-27b", 262_144, true, true, true),
    ("@cf/openai/gpt-oss-120b", 128_000, false, true, false),
    ("@cf/openai/gpt-oss-20b", 128_000, false, true, false),
    (
        "@cf/ibm-granite/granite-4.0-h-micro",
        131_000,
        false,
        true,
        false,
    ),
    ("@cf/zai-org/glm-4.7-flash", 131_072, false, true, true),
    ("@cf/zai-org/glm-5.2", 262_144, false, true, true),
    ("@cf/zai-org/glm-5.3-flash", 1_310_720, true, true, true),
    ("@cf/zai-org/glm-5.3", 1_310_720, false, true, true),
    (
        "@cf/nvidia/nemotron-3-120b-a12b",
        256_000,
        false,
        true,
        true,
    ),
    ("@cf/moonshotai/kimi-k2.6", 262_144, true, true, true),
    ("@cf/moonshotai/kimi-k2.7-code", 262_144, true, true, true),
];

fn model_facts(model: &str) -> Option<&'static ModelFacts> {
    let normalized = normalize_model_name(model);
    MODELS
        .iter()
        .find(|(id, ..)| normalized.contains(&normalize_model_name(id)))
}

fn cloudflare_model_pricing(model: &str) -> Option<crate::llm::types::ModelPricing> {
    let (input, output, cache_write, cache_read) = get_model_pricing(model, PRICING)?;
    Some(crate::llm::types::ModelPricing::new(
        input,
        output,
        cache_write,
        cache_read,
    ))
}

fn default_cloudflare_api_url(account_id: &str) -> String {
    format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/v1/chat/completions",
        account_id
    )
}

#[async_trait::async_trait]
impl AiProvider for CloudflareWorkersAiProvider {
    fn name(&self) -> &str {
        "cloudflare"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // Cloudflare Workers AI requires both API token and account ID
        let api_token = self.get_api_token()?;
        let _account_id = self.get_account_id()?; // Validate it exists
        Ok(api_token) // Return API token as the "API key"
    }

    fn supports_required_tool_choice(&self, model: &str) -> bool {
        model_facts(model).is_some_and(|(_, _, _, _, openai_schema)| *openai_schema)
    }

    fn supports_caching(&self, model: &str) -> bool {
        cloudflare_model_pricing(model)
            .map(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
            .unwrap_or(false)
    }

    fn supports_vision(&self, model: &str) -> bool {
        if let Some((_, _, vision, _, _)) = model_facts(model) {
            return *vision;
        }
        // Check Cloudflare-specific naming patterns first
        let model_lower = normalize_model_name(model);
        if model_lower.contains("vision") || model_lower.contains("@cf/llava") {
            return true;
        }
        // Fall back to reference capabilities for the underlying model
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.vision)
            .unwrap_or(false)
    }

    /// Schemas are enforced through the forced-tool path (see
    /// `enforces_response_schema`), so structured output needs function calling.
    fn supports_structured_output(&self, model: &str) -> bool {
        match model_facts(model) {
            Some((_, _, _, function_calling, _)) => *function_calling,
            None => crate::llm::reference_models::get_reference_capabilities(model)
                .map(|c| c.structured_output)
                .unwrap_or(false),
        }
    }

    /// JSON Mode is best effort — "Workers AI can't guarantee that the model
    /// responds according to the requested JSON Schema" — and does not support
    /// streaming, so the schema is enforced locally via a forced tool call.
    fn enforces_response_schema(&self, _model: &str) -> bool {
        false
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        cloudflare_model_pricing(model)
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        if let Some((_, context_window, ..)) = model_facts(model) {
            return *context_window;
        }
        // Use reference capabilities for model-specific context windows
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.max_input_tokens)
            .unwrap_or(4_096) // Conservative default for Cloudflare's smaller models
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let account_id = self.get_account_id()?;
        let api_url = get_api_url(
            CLOUDFLARE_API_URL_ENV,
            &default_cloudflare_api_url(&account_id),
        );

        let model = params.model.clone();
        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "cloudflare",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: false,
                supports_required_tool_choice: self.supports_required_tool_choice(&model),
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
                        usage.billable_output_tokens(),
                    ));
                }
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "cloudflare_tests.rs"]
mod tests;
