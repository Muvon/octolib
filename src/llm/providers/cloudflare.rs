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
        env::var(CLOUDFLARE_API_TOKEN_ENV)
            .map_err(|_| anyhow::anyhow!("CLOUDFLARE_API_TOKEN not found in environment"))
    }

    /// Get Cloudflare Account ID
    fn get_account_id(&self) -> Result<String> {
        env::var(CLOUDFLARE_ACCOUNT_ID_ENV)
            .map_err(|_| anyhow::anyhow!("CLOUDFLARE_ACCOUNT_ID not found in environment"))
    }
}

const CLOUDFLARE_API_TOKEN_ENV: &str = "CLOUDFLARE_API_TOKEN";
const CLOUDFLARE_ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";

#[async_trait::async_trait]
impl AiProvider for CloudflareWorkersAiProvider {
    fn name(&self) -> &str {
        "cloudflare"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Cloudflare Workers AI supported models (case-insensitive)
        let model_lower = normalize_model_name(model);
        model_lower.contains("llama")
            || model_lower.contains("mistral")
            || model_lower.contains("qwen")
            || model_lower.contains("phi")
            || model_lower.contains("tinyllama")
            || model_lower.contains("gemma")
            || model_lower.contains("codellama")
            || model_lower.contains("deepseek")
            || model_lower.contains("neural-chat")
            || model_lower.contains("openchat")
            || model_lower.contains("starling")
            || model_lower.contains("zephyr")
            || model.starts_with("@cf/")
            || model.starts_with("@hf/")
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

    fn supports_vision(&self, _model: &str) -> bool {
        false
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

    async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
        Err(anyhow::anyhow!(
            "Cloudflare Workers AI provider not fully implemented in octolib"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = CloudflareWorkersAiProvider::new();

        // Cloudflare Workers AI supported models
        assert!(provider.supports_model("llama-3.1-70b-instruct"));
        assert!(provider.supports_model("@cf/meta/llama-3.1-70b-instruct"));
        assert!(provider.supports_model("@hf/meta/llama-3.1-8b-instruct"));
        assert!(provider.supports_model("mistral-7b-instruct-v0.1"));
        assert!(provider.supports_model("gemma-2-27b-it"));

        // Unsupported models
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("claude-3"));
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
