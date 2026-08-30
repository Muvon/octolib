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

    fn supports_video(&self, _model: &str) -> bool {
        false
    }

    // supports_vision, supports_video, supports_structured_output, get_max_input_tokens
    // are resolved via reference capabilities (trait defaults)

    fn enforces_response_schema(&self, model: &str) -> bool {
        // Cloud does not constrain decoding, but the shared OpenAI-compatible
        // path forces a tool call, validates locally, and fails closed.
        self.supports_structured_output(model)
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try reference pricing for cloud-equivalent cost estimation
        crate::llm::reference_models::get_reference_pricing(model)
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
                enforces_response_schema: false,
                supports_required_tool_choice: true,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Fill cost from reference pricing if the API didn't return one
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                usage.cost = crate::llm::reference_models::calculate_reference_cost(
                    &model,
                    usage.input_tokens,
                    usage.cache_read_tokens,
                    usage.billable_output_tokens(),
                );
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod tests;
