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

//! Google AI Studio (Gemini API) provider implementation.
//!
//! The API-key sibling of the google-vertex provider: same models, same list
//! prices, but authenticated with a plain AI Studio key instead of a service
//! account — via Google's OpenAI-compatible endpoint:
//! `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`
//!
//! Configuration:
//! - `GOOGLE_STUDIO_API_KEY`: Required API key (aistudio.google.com/apikey)
//! - `GOOGLE_STUDIO_API_URL`: Optional endpoint override
//!
//! Model discovery: Available models are lazy-loaded from the Gemini API on first
//! chat_completion() call. The list is cached for the lifetime of the process.

use crate::llm::providers::google_vertex::{
    calculate_usage_cost, fetch_available_models, gemini_max_input_tokens, gemini_sampling_support,
    get_cached_input_limit, is_model_cached, CachedModel, PRICING,
};
use crate::llm::providers::openai_compat::{
    chat_completion_with_sampling, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse, SamplingSupport};
use crate::llm::utils::{get_model_pricing, normalize_model_name};
use anyhow::Result;
use std::env;
use tokio::sync::OnceCell;

/// Google AI Studio provider
#[derive(Debug, Clone, Default)]
pub struct GoogleStudioProvider;

impl GoogleStudioProvider {
    pub fn new() -> Self {
        Self
    }
}

const GOOGLE_STUDIO_API_KEY_ENV: &str = "GOOGLE_STUDIO_API_KEY";
const GOOGLE_STUDIO_API_URL_ENV: &str = "GOOGLE_STUDIO_API_URL";
const GOOGLE_STUDIO_API_URL: &str =
    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";

/// Process-wide cache of available models, populated on first chat_completion()
static MODELS_CACHE: OnceCell<Vec<CachedModel>> = OnceCell::const_new();

#[async_trait::async_trait]
impl AiProvider for GoogleStudioProvider {
    fn name(&self) -> &str {
        "google-studio"
    }

    fn supports_model(&self, model: &str) -> bool {
        if model.is_empty() {
            return false;
        }
        // Use cached model list if available (populated on first chat_completion)
        is_model_cached(&MODELS_CACHE, model).unwrap_or(true)
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(GOOGLE_STUDIO_API_KEY_ENV) {
            Ok(key) if !key.trim().is_empty() => Ok(key),
            _ => Err(anyhow::anyhow!(
                "Google AI Studio API key not found in environment variable: {}. \
                Get an API key at https://aistudio.google.com/apikey",
                GOOGLE_STUDIO_API_KEY_ENV
            )),
        }
    }

    // Gemini 2.5+ caches implicitly; cached tokens come back in usage and bill
    // at the cache-read rate — same behavior as the Vertex lane.
    fn supports_caching(&self, model: &str) -> bool {
        let normalized = normalize_model_name(model);
        normalized.contains("gemini-3") || normalized.contains("gemini-2.5")
    }

    fn supports_vision(&self, model: &str) -> bool {
        normalize_model_name(model).contains("gemini")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            get_model_pricing(model, PRICING)?;
        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Prefer cached value from API if available
        if let Some(limit) = get_cached_input_limit(&MODELS_CACHE, model) {
            return limit;
        }
        gemini_max_input_tokens(model)
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        gemini_sampling_support(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(GOOGLE_STUDIO_API_URL_ENV, GOOGLE_STUDIO_API_URL);
        let model = params.model.clone();

        // Lazy-load available models on first call (errors silently ignored; retries next call)
        let key = api_key.clone();
        let url = api_url.clone();
        let _ = MODELS_CACHE
            .get_or_try_init(|| async move { fetch_available_models(&key, &url).await })
            .await;

        let mut response = chat_completion_with_sampling(
            OpenAiCompatConfig {
                provider_name: "google-studio",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: true,
                supports_required_tool_choice: false,
            },
            self.supported_sampling_params(&model),
            api_key,
            api_url,
            params,
        )
        .await?;

        // Gemini's OpenAI-compat endpoint doesn't report cost — fill from the pricing table
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                usage.cost = calculate_usage_cost(
                    &model,
                    usage.input_tokens,
                    usage.cache_write_tokens,
                    usage.cache_read_tokens,
                    usage.billable_output_tokens(),
                );
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "google_studio_tests.rs"]
mod tests;
