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

//! Ollama provider implementation.
//!
//! Uses Ollama's OpenAI-compatible endpoint by default:
//! `https://ollama.com/v1/chat/completions`
//!
//! Configuration:
//! - `OLLAMA_API_KEY`: Optional API key (required for cloud/private deployments)
//! - `OLLAMA_API_URL`: Override endpoint

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, get_optional_api_key,
    OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;

/// Ollama provider
#[derive(Debug, Clone)]
pub struct OllamaProvider;

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self
    }
}

const OLLAMA_API_KEY_ENV: &str = "OLLAMA_API_KEY";
const OLLAMA_API_URL_ENV: &str = "OLLAMA_API_URL";
const OLLAMA_API_URL: &str = "https://ollama.com/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        Ok(get_optional_api_key(OLLAMA_API_KEY_ENV))
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    // supports_vision, supports_video, supports_structured_output, get_max_input_tokens
    // are resolved via reference capabilities (trait defaults)

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try reference pricing for cloud-equivalent cost estimation
        crate::llm::reference_pricing::get_reference_pricing(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(OLLAMA_API_URL_ENV, OLLAMA_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "ollama",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Fill cost from reference pricing if the API didn't return one
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
        let provider = OllamaProvider::new();
        assert!(provider.supports_model("llama3.2"));
        assert!(provider.supports_model("qwen2.5"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.name(), "ollama");
        assert!(!provider.supports_caching("any-model"));
    }

    #[test]
    fn test_vision_model_specific() {
        let provider = OllamaProvider::new();
        // Vision models detected via reference capabilities
        assert!(provider.supports_vision("llava:latest"));
        assert!(provider.supports_vision("qwen2.5-vl:72b"));
        assert!(provider.supports_vision("gemma3:27b"));
        // Text-only models correctly report no vision
        assert!(!provider.supports_vision("llama3.1:8b"));
        assert!(!provider.supports_vision("mistral:7b"));
        // Unknown models default to false
        assert!(!provider.supports_vision("unknown-model"));
    }

    #[test]
    fn test_video_model_specific() {
        let provider = OllamaProvider::new();
        // Only VL models support video
        assert!(provider.supports_video("qwen2.5-vl:72b"));
        assert!(!provider.supports_video("llama3.1:8b"));
        assert!(!provider.supports_video("llava:latest"));
    }

    #[test]
    fn test_structured_output_model_specific() {
        let provider = OllamaProvider::new();
        assert!(provider.supports_structured_output("llama3.1:8b"));
        assert!(provider.supports_structured_output("qwen2.5:72b"));
        assert!(!provider.supports_structured_output("mistral:7b"));
    }

    #[test]
    fn test_context_window_model_specific() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.get_max_input_tokens("llama3.1:8b"), 131_072);
        assert_eq!(provider.get_max_input_tokens("mistral:7b"), 32_768);
        // Unknown models get conservative default
        assert_eq!(provider.get_max_input_tokens("unknown-model"), 8_192);
    }
}
