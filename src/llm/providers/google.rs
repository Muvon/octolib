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

//! Google Vertex AI provider implementation

use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// Google Vertex AI provider
#[derive(Debug, Clone)]
pub struct GoogleVertexProvider;

impl Default for GoogleVertexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleVertexProvider {
    pub fn new() -> Self {
        Self
    }
}

const GOOGLE_API_KEY_ENV: &str = "GOOGLE_API_KEY";

#[async_trait::async_trait]
impl AiProvider for GoogleVertexProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn supports_model(&self, model: &str) -> bool {
        model.contains("gemini") || model.contains("palm") || model.starts_with("text-")
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(GOOGLE_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "Google API key not found in environment variable: {}",
                GOOGLE_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, model: &str) -> bool {
        model.contains("gemini-1.5") || model.contains("gemini-2")
    }

    fn supports_vision(&self, model: &str) -> bool {
        model.contains("gemini")
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        if model.contains("gemini-2") {
            2_000_000
        } else if model.contains("gemini-1.5") {
            1_000_000
        } else {
            32_768
        }
    }

    async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
        Err(anyhow::anyhow!(
            "Google Vertex AI provider not fully implemented in octolib"
        ))
    }
}
