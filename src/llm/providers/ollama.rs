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

    fn supports_vision(&self, _model: &str) -> bool {
        // Ollama is a local provider with many models - support all by default
        // The actual capability depends on the specific model being used
        true
    }

    fn supports_video(&self, _model: &str) -> bool {
        // Ollama is a local provider - support all by default
        // The actual capability depends on the specific model being used
        true
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        128_000
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(OLLAMA_API_URL_ENV, OLLAMA_API_URL);

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "ollama",
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
        assert!(provider.supports_structured_output("any-model"));
        assert_eq!(provider.get_max_input_tokens("any-model"), 128_000);
    }

    #[test]
    fn test_supports_vision_default() {
        let provider = OllamaProvider::new();
        // Ollama supports all models by default (aggregator behavior)
        assert!(provider.supports_vision("llava:latest"));
        assert!(provider.supports_vision("qwen2.5-vl"));
        assert!(provider.supports_vision("llama3.2"));
        assert!(provider.supports_vision("any-model"));
    }

    #[test]
    fn test_supports_video_default() {
        let provider = OllamaProvider::new();
        // Ollama supports video by default (aggregator behavior)
        assert!(provider.supports_video("llava:latest"));
        assert!(provider.supports_video("qwen2.5-vl"));
        assert!(provider.supports_video("llama3.2"));
        assert!(provider.supports_video("any-model"));
    }
}
