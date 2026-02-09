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

//! AI Provider trait definition

use crate::llm::types::{ChatCompletionParams, ProviderResponse};
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

    /// Check if the provider/model supports caching
    fn supports_caching(&self, _model: &str) -> bool {
        // Default implementation - providers can override
        false
    }

    /// Get maximum input tokens for a model (actual context window size)
    /// This is what we can send to the API - the provider handles output limits internally
    fn get_max_input_tokens(&self, model: &str) -> usize;

    /// Check if the provider/model supports vision capabilities
    fn supports_vision(&self, _model: &str) -> bool {
        // Default implementation - providers can override
        false
    }

    /// Check if the provider supports structured output
    fn supports_structured_output(&self, _model: &str) -> bool {
        // Default implementation - providers can override
        false
    }

    /// Get pricing information for a model
    /// Returns None if pricing is not available or model is not recognized
    fn get_model_pricing(&self, _model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Default implementation - providers should override with actual pricing
        None
    }
}
