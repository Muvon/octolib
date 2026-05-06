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

//! Fireworks AI provider implementation.
//!
//! Uses the OpenAI-compatible endpoint at:
//! `https://api.fireworks.ai/inference/v1/chat/completions`
//!
//! Fireworks hosts a large catalog of open-weight and frontier models
//! (Kimi K2, DeepSeek V3/V4, GLM 4.x/5.x, Qwen 3, Llama 4, gpt-oss, etc.) using
//! `accounts/fireworks/models/<name>` model IDs. As an aggregator we accept any
//! non-empty model string and resolve capabilities/pricing through the
//! reference tables (substring matching naturally strips the namespace prefix).
//!
//! Caching: Fireworks performs automatic prompt-prefix caching. Cached input
//! tokens are surfaced via `usage.prompt_tokens_details.cached_tokens` (already
//! parsed by the shared `openai_compat` layer) and billed at the cached rate
//! when reference pricing exposes it.
//!
//! Source: <https://docs.fireworks.ai/api-reference/post-chatcompletions>
//!
//! Configuration:
//! - `FIREWORKS_API_KEY`: Required API key
//! - `FIREWORKS_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// Fireworks AI provider
#[derive(Debug, Clone)]
pub struct FireworksProvider;

impl Default for FireworksProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FireworksProvider {
    pub fn new() -> Self {
        Self
    }
}

const FIREWORKS_API_KEY_ENV: &str = "FIREWORKS_API_KEY";
const FIREWORKS_API_URL_ENV: &str = "FIREWORKS_API_URL";
const FIREWORKS_API_URL: &str = "https://api.fireworks.ai/inference/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for FireworksProvider {
    fn name(&self) -> &str {
        "fireworks"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(FIREWORKS_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Fireworks API key not found in environment variable: {}",
                FIREWORKS_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Fireworks performs automatic prompt-prefix caching across hosted
        // text/vision LLMs and reports `cached_tokens` in usage.
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // Fireworks supports `response_format` (json_object / json_schema /
        // grammar) on the chat completions endpoint for all served models.
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Fireworks doesn't return cost in responses — fall back to reference
        // pricing keyed on the underlying model name. Reference matching uses
        // substring lookup so the `accounts/fireworks/models/` prefix is harmless.
        crate::llm::reference_pricing::get_reference_pricing(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(FIREWORKS_API_URL_ENV, FIREWORKS_API_URL);
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "fireworks",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Derive cost from reference pricing (Fireworks doesn't return it).
        // Includes cache_read tokens at the reference cache rate when available.
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                if let Some(pricing) = self.get_model_pricing(&model) {
                    usage.cost = Some(pricing.calculate_cost(
                        usage.input_tokens,
                        usage.cache_write_tokens,
                        usage.cache_read_tokens,
                        usage.output_tokens,
                    ));
                }
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
        let provider = FireworksProvider::new();
        assert!(provider.supports_model("accounts/fireworks/models/kimi-k2-instruct-0905"));
        assert!(provider.supports_model("accounts/fireworks/models/deepseek-v3"));
        assert!(provider.supports_model("accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"));
        assert!(provider.supports_model("any-future-model"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = FireworksProvider::new();
        assert_eq!(provider.name(), "fireworks");
        assert!(provider.supports_caching("any-model"));
        assert!(provider.supports_structured_output("any-model"));
    }

    #[test]
    fn test_pricing_reference_fallback() {
        let provider = FireworksProvider::new();
        // Reference pricing should resolve for well-known open-weight models,
        // including when wrapped in the Fireworks namespace prefix.
        let pricing = provider.get_model_pricing("accounts/fireworks/models/kimi-k2-instruct-0905");
        assert!(pricing.is_some());

        let pricing = provider.get_model_pricing("accounts/fireworks/models/deepseek-v3");
        assert!(pricing.is_some());

        let pricing = provider.get_model_pricing("accounts/fireworks/models/gpt-oss-120b");
        assert!(pricing.is_some());
    }
}
