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

//! Amazon Bedrock provider implementation

use crate::traits::AiProvider;
use crate::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// Amazon Bedrock provider
#[derive(Debug, Clone)]
pub struct AmazonBedrockProvider;

impl Default for AmazonBedrockProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AmazonBedrockProvider {
    pub fn new() -> Self {
        Self
    }
}

const AMAZON_ACCESS_KEY_ID_ENV: &str = "AMAZON_ACCESS_KEY_ID";

#[async_trait::async_trait]
impl AiProvider for AmazonBedrockProvider {
    fn name(&self) -> &str {
        "amazon"
    }

    fn supports_model(&self, model: &str) -> bool {
        model.contains("claude") || model.contains("titan") || model.contains("llama")
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(AMAZON_ACCESS_KEY_ID_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "Amazon access key not found in environment variable: {}",
                AMAZON_ACCESS_KEY_ID_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, model: &str) -> bool {
        model.contains("claude")
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        200_000
    }

    async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
        Err(anyhow::anyhow!(
            "Amazon Bedrock provider not fully implemented in octolib"
        ))
    }
}
