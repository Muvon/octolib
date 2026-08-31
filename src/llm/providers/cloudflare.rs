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
//! Authentication: Requires API token and Account ID.
//!
//! **How to get credentials:**
//! 1. Cloudflare Dashboard → My Profile → API Tokens
//! 2. Create Token → Use template "Workers AI" or create custom with Workers AI permissions
//! 3. Copy the API token
//! 4. Get Account ID from Cloudflare Dashboard → Workers & Pages (in URL or sidebar)
//! 5. Set environment variables:
//!    - export CLOUDFLARE_API_TOKEN="your-api-token"
//!    - export CLOUDFLARE_ACCOUNT_ID="your-account-id"
//!
//! The API token is sent as a Bearer token in the Authorization header.

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
        env::var(CLOUDFLARE_API_TOKEN_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cloudflare API token not found. Set {} environment variable.\n\
                To create an API token:\n\
                1. Cloudflare Dashboard → My Profile → API Tokens\n\
                2. Create Token → Use 'Workers AI' template or create custom\n\
                3. Ensure token has Workers AI permissions",
                CLOUDFLARE_API_TOKEN_ENV
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

const CLOUDFLARE_API_TOKEN_ENV: &str = "CLOUDFLARE_API_TOKEN";
const CLOUDFLARE_ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_API_URL_ENV: &str = "CLOUDFLARE_API_URL";

/// Cloudflare Workers AI prices per 1M tokens, verified Aug 26, 2026.
/// Format: (model ID, input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
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
    ("@cf/qwen/qwen3.8-27b", 0.450, 3.200, 0.450, 0.450),
    ("@cf/zai-org/glm-5.2", 1.400, 4.400, 1.400, 0.260),
    ("@cf/zai-org/glm-5.3-flash", 0.150, 0.500, 0.150, 0.030),
    ("@cf/zai-org/glm-5.3", 1.400, 4.400, 1.400, 0.260),
    ("@cf/zai-org/glm-4.7-flash", 0.060, 0.400, 0.060, 0.060),
    (
        "@cf/nvidia/nemotron-3-120b-a12b",
        0.500,
        1.500,
        0.500,
        0.500,
    ),
    ("@cf/moonshotai/kimi-k2.5", 0.600, 3.000, 0.600, 0.100),
    ("@cf/moonshotai/kimi-k2.7-code", 0.950, 4.000, 0.950, 0.190),
    ("@cf/moonshotai/kimi-k2.6", 0.950, 4.000, 0.950, 0.160),
    ("@cf/openai/gpt-oss-120b", 0.350, 0.750, 0.350, 0.350),
    ("@cf/openai/gpt-oss-20b", 0.200, 0.300, 0.200, 0.200),
    ("@cf/google/gemma-4-26b-a4b-it", 0.100, 0.300, 0.100, 0.100),
];

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

    fn supports_caching(&self, model: &str) -> bool {
        cloudflare_model_pricing(model)
            .map(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
            .unwrap_or(false)
    }

    fn supports_vision(&self, model: &str) -> bool {
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

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        cloudflare_model_pricing(model)
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
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
