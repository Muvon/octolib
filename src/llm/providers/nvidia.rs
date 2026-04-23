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

//! NVIDIA NIM provider implementation.
//!
//! Uses NVIDIA's OpenAI-compatible endpoint at:
//! `https://integrate.api.nvidia.com/v1/chat/completions`
//!
//! Hosts 100+ models from various providers (Nemotron, Llama, DeepSeek, etc.).
//! Model IDs use namespace/model format (e.g., `nvidia/llama-3.1-nemotron-ultra-253b-v1`).
//!
//! Configuration:
//! - `NVIDIA_API_KEY`: Required API key
//! - `NVIDIA_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// NVIDIA NIM provider
#[derive(Debug, Clone)]
pub struct NvidiaProvider;

impl Default for NvidiaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl NvidiaProvider {
    pub fn new() -> Self {
        Self
    }
}

const NVIDIA_API_KEY_ENV: &str = "NVIDIA_API_KEY";
const NVIDIA_API_URL_ENV: &str = "NVIDIA_API_URL";
const NVIDIA_API_URL: &str = "https://integrate.api.nvidia.com/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for NvidiaProvider {
    fn name(&self) -> &str {
        "nvidia"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(NVIDIA_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "NVIDIA API key not found in environment variable: {}",
                NVIDIA_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // NVIDIA NIM doesn't expose its own pricing — fall back to reference pricing
        // based on the underlying model (e.g., z-ai/glm-5.1 → glm-5.1 reference entry).
        crate::llm::reference_pricing::get_reference_pricing(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(NVIDIA_API_URL_ENV, NVIDIA_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "nvidia",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // NVIDIA doesn't return cost in the response — derive it from reference pricing
        // so downstream consumers (CLI token display, cost tracking) see a value.
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                usage.cost = crate::llm::reference_pricing::calculate_reference_cost(
                    &model,
                    usage.input_tokens,
                    usage.output_tokens,
                );
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
        let provider = NvidiaProvider::new();
        assert!(provider.supports_model("nvidia/llama-3.1-nemotron-ultra-253b-v1"));
        assert!(provider.supports_model("deepseek-ai/deepseek-v3.2"));
        assert!(provider.supports_model("minimaxai/minimax-m2.1"));
        assert!(provider.supports_model("meta/llama-3.1-405b-instruct"));
        assert!(provider.supports_model("any-model"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = NvidiaProvider::new();
        assert_eq!(provider.name(), "nvidia");
        assert!(!provider.supports_caching("any-model"));
        assert!(provider.supports_structured_output("any-model"));
    }
}
