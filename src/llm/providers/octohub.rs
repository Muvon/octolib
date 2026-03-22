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

//! OctoHub provider implementation.
//!
//! Proxies requests through an OctoHub server which handles response chaining,
//! logging, and multi-provider routing. Uses OpenAI-compatible chat completions
//! endpoint under the hood.
//!
//! Configuration:
//! - `OCTOHUB_API_KEY`: Optional API key for OctoHub server authentication
//! - `OCTOHUB_API_URL`: Required OctoHub server endpoint (default: http://127.0.0.1:8080/v1/chat/completions)

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, get_optional_api_key,
    OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;

const OCTOHUB_API_KEY_ENV: &str = "OCTOHUB_API_KEY";
const OCTOHUB_API_URL_ENV: &str = "OCTOHUB_API_URL";
const OCTOHUB_API_URL: &str = "http://127.0.0.1:8080/v1/chat/completions";

/// OctoHub provider - routes through OctoHub proxy server
#[derive(Debug, Clone)]
pub struct OctoHubProvider;

impl Default for OctoHubProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OctoHubProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AiProvider for OctoHubProvider {
    fn name(&self) -> &str {
        "octohub"
    }

    /// OctoHub accepts any model - it routes to the appropriate provider
    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // OctoHub API key is optional (server may run without auth)
        Ok(get_optional_api_key(OCTOHUB_API_KEY_ENV))
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Caching depends on the underlying provider, assume supported
        true
    }

    fn supports_vision(&self, _model: &str) -> bool {
        // Vision depends on the underlying model
        true
    }

    fn supports_video(&self, _model: &str) -> bool {
        false
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        // Conservative default; actual limit depends on underlying provider
        128_000
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = get_optional_api_key(OCTOHUB_API_KEY_ENV);
        let api_url = get_api_url(OCTOHUB_API_URL_ENV, OCTOHUB_API_URL);

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "octohub",
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
    fn test_supports_any_model() {
        let provider = OctoHubProvider::new();
        assert!(provider.supports_model("gpt-4o"));
        assert!(provider.supports_model("claude-3.5-sonnet"));
        assert!(provider.supports_model("any-model-name"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_provider_name() {
        let provider = OctoHubProvider::new();
        assert_eq!(provider.name(), "octohub");
    }

    #[test]
    fn test_capabilities() {
        let provider = OctoHubProvider::new();
        assert!(provider.supports_caching("any"));
        assert!(provider.supports_vision("any"));
        assert!(!provider.supports_video("any"));
        assert!(provider.supports_structured_output("any"));
        assert_eq!(provider.get_max_input_tokens("any"), 128_000);
    }
}
