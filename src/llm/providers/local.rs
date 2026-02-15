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

//! Local provider for OpenAI-compatible local model servers.
//!
//! ## Supported Local Servers
//! - **Ollama**: Default `http://localhost:11434/v1/chat/completions`
//! - **LM Studio**: `http://localhost:1234/v1/chat/completions`
//! - **LocalAI**: `http://localhost:8080/v1/chat/completions`
//! - Any other OpenAI-compatible local server
//!
//! ## Configuration
//! - `LOCAL_API_URL`: API endpoint (default: Ollama local)
//! - `LOCAL_API_KEY`: Optional API key

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, get_optional_api_key,
    OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;

/// Local provider for local model servers
#[derive(Debug, Clone)]
pub struct LocalProvider;

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalProvider {
    pub fn new() -> Self {
        Self
    }
}

const LOCAL_API_KEY_ENV: &str = "LOCAL_API_KEY";
const LOCAL_API_URL_ENV: &str = "LOCAL_API_URL";
const LOCAL_API_URL: &str = "http://localhost:11434/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        Ok(get_optional_api_key(LOCAL_API_KEY_ENV))
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, _model: &str) -> bool {
        // Local provider supports many models - default to true
        // Actual capability depends on the specific model being used
        true
    }

    fn supports_video(&self, _model: &str) -> bool {
        // Local provider supports video depending on the model
        // Default to true and let the underlying model handle it
        true
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        // Local models often support large contexts (128K is common)
        128_000
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(LOCAL_API_URL_ENV, LOCAL_API_URL);

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "local",
                usage_fallback_cost: Some(0.0),
                use_response_cost: false,
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
        let provider = LocalProvider::new();

        assert!(provider.supports_model("llama3.2"));
        assert!(provider.supports_model("mistral-7b"));
        assert!(provider.supports_model("gpt4all-j"));
        assert!(provider.supports_model("any-model-name"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_get_api_key_optional() {
        let provider = LocalProvider::new();
        let result = provider.get_api_key();
        assert!(result.is_ok());
    }

    #[test]
    fn test_supports_structured_output() {
        let provider = LocalProvider::new();
        assert!(provider.supports_structured_output("any-model"));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = LocalProvider::new();

        assert_eq!(provider.name(), "local");
        assert!(!provider.supports_caching("any-model"));
        assert!(provider.supports_vision("any-model"));
        assert!(provider.supports_video("any-model"));
        assert_eq!(provider.get_max_input_tokens("any-model"), 128_000);
    }
}
