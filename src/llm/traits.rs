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

//! AI Provider trait definition

use crate::llm::types::{ChatCompletionParams, ProviderResponse, SamplingParams};
use anyhow::Result;

/// Trait that all AI providers must implement
#[async_trait::async_trait]
pub trait AiProvider: Send + Sync {
    /// Get the provider name (e.g., "openrouter", "openai", "anthropic")
    fn name(&self) -> &str;

    /// Check if the provider supports the given model
    fn supports_model(&self, model: &str) -> bool;

    /// Send a chat completion request
    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse>;

    /// Get API key for this provider from environment variables
    /// Each provider should implement this to check their specific environment variable
    fn get_api_key(&self) -> Result<String>;

    /// Which sampling parameters this model supports.
    ///
    /// Returns `SamplingParams` where `Some(default)` means the parameter is supported
    /// and `None` means it must be omitted from API requests.
    ///
    /// Default: all parameters supported (temperature, top_p, top_k).
    /// Override in providers where specific models reject sampling parameters.
    fn supported_sampling_params(&self, _model: &str) -> SamplingParams {
        SamplingParams::default()
    }

    /// Compute effective sampling parameters by merging user-requested values
    /// with what the model actually supports.
    fn effective_sampling_params(&self, params: &ChatCompletionParams) -> SamplingParams {
        self.supported_sampling_params(&params.model).effective(
            params.temperature,
            params.top_p,
            params.top_k,
        )
    }

    /// Check if the provider/model supports caching
    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    /// Get maximum input tokens for a model (actual context window size)
    /// This is what we can send to the API - the provider handles output limits internally
    fn get_max_input_tokens(&self, model: &str) -> usize {
        crate::llm::reference_capabilities::get_reference_capabilities(model)
            .map(|c| c.max_input_tokens)
            .unwrap_or(8_192)
    }

    /// Check if the provider/model supports vision capabilities
    fn supports_vision(&self, model: &str) -> bool {
        crate::llm::reference_capabilities::get_reference_capabilities(model)
            .map(|c| c.vision)
            .unwrap_or(false)
    }

    /// Check if the provider/model supports video capabilities
    fn supports_video(&self, model: &str) -> bool {
        crate::llm::reference_capabilities::get_reference_capabilities(model)
            .map(|c| c.video)
            .unwrap_or(false)
    }

    /// Check if the provider supports structured output
    fn supports_structured_output(&self, model: &str) -> bool {
        crate::llm::reference_capabilities::get_reference_capabilities(model)
            .map(|c| c.structured_output)
            .unwrap_or(false)
    }

    /// Get pricing information for a model
    /// Returns None if pricing is not available or model is not recognized.
    /// Default falls back to reference pricing for well-known open models.
    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        crate::llm::reference_pricing::get_reference_pricing(model)
    }
}
