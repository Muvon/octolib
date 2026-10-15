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

//! WebWright provider implementation.
//!
//! Uses WebWright's OpenAI-compatible endpoint:
//! `https://webwright.ai/v1/chat/completions`
//!
//! Configuration:
//! - `WEBWRIGHT_API_KEY`: Required API key
//! - `WEBWRIGHT_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// WebWright provider
#[derive(Debug, Clone)]
pub struct WebWrightProvider;

impl Default for WebWrightProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl WebWrightProvider {
    pub fn new() -> Self {
        Self
    }
}

const WEBWRIGHT_API_KEY_ENV: &str = "WEBWRIGHT_API_KEY";
const WEBWRIGHT_API_URL_ENV: &str = "WEBWRIGHT_API_URL";
const WEBWRIGHT_API_URL: &str = "https://webwright.ai/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for WebWrightProvider {
    fn name(&self) -> &str {
        "webwright"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(WEBWRIGHT_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "WebWright API key not found in environment variable: {}",
                WEBWRIGHT_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_video(&self, _model: &str) -> bool {
        false
    }

    /// WebWright does not publish evidence that it enforces a supplied JSON
    /// schema, so callers should route structured requests through a tolerant
    /// parser rather than assume the shape is guaranteed.
    fn enforces_response_schema(&self, _model: &str) -> bool {
        false
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(WEBWRIGHT_API_URL_ENV, WEBWRIGHT_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "webwright",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: false,
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Fill cost from reference pricing if the API did not return one
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
#[path = "webwright_tests.rs"]
mod tests;
