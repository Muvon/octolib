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
use crate::llm::utils::normalize_model_name;
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

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, model: &str) -> bool {
        let model_lower = normalize_model_name(model);
        model_lower.contains("vision")
            || model_lower.contains("vl")
            || model_lower.contains("llava")
            || model_lower.contains("@cf/llava")
            || model_lower.contains("qwen-vl")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, _model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Cloudflare Workers AI has usage-based pricing - return zero for now
        // so compression analysis can still work (will assume always beneficial)
        Some(crate::llm::types::ModelPricing::new(0.0, 0.0, 0.0, 0.0))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Cloudflare Workers AI model context window limits (case-insensitive)
        let model_lower = normalize_model_name(model);
        if model_lower.contains("llama-3.1") || model_lower.contains("llama-3.2") {
            32_768 // Llama 3.1 and 3.2 models have 32K context
        } else if model_lower.contains("llama-2") {
            4_096 // Llama 2 models have 4K context
        } else if model_lower.contains("mistral-7b") {
            8_192 // Mistral 7B has 8K context
        } else if model_lower.contains("qwen1.5") {
            32_768 // Qwen 1.5 models have 32K context
        } else if model_lower.contains("codellama") {
            16_384 // CodeLlama has 16K context
        } else if model_lower.contains("gemma") {
            8_192 // Gemma models have 8K context
        } else {
            4_096 // Conservative default for smaller models
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let account_id = self.get_account_id()?;
        let api_url = get_api_url(
            CLOUDFLARE_API_URL_ENV,
            &default_cloudflare_api_url(&account_id),
        );

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "cloudflare",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = CloudflareWorkersAiProvider::new();

        // Cloudflare Workers AI accepts any non-empty model identifier
        assert!(provider.supports_model("llama-3.1-70b-instruct"));
        assert!(provider.supports_model("@cf/meta/llama-3.1-70b-instruct"));
        assert!(provider.supports_model("@hf/meta/llama-3.1-8b-instruct"));
        assert!(provider.supports_model("mistral-7b-instruct-v0.1"));
        assert!(provider.supports_model("gemma-2-27b-it"));
        assert!(provider.supports_model("gpt-4"));
        assert!(provider.supports_model("claude-3"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = CloudflareWorkersAiProvider::new();

        // Test uppercase
        assert!(provider.supports_model("LLAMA-3.1-70B-INSTRUCT"));
        assert!(provider.supports_model("MISTRAL-7B-INSTRUCT-V0.1"));
        // Test mixed case
        assert!(provider.supports_model("Llama-3.1-70B-Instruct"));
        assert!(provider.supports_model("GEMMA-2-27B-IT"));
    }
}
