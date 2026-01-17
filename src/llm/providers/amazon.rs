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

use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::normalize_model_name;
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

    /// Get AWS access key ID
    fn get_aws_access_key_id(&self) -> Result<String> {
        env::var(AWS_ACCESS_KEY_ID_ENV)
            .map_err(|_| anyhow::anyhow!("AWS_ACCESS_KEY_ID not found in environment"))
    }

    /// Get AWS secret access key
    fn get_aws_secret_access_key(&self) -> Result<String> {
        env::var(AWS_SECRET_ACCESS_KEY_ENV)
            .map_err(|_| anyhow::anyhow!("AWS_SECRET_ACCESS_KEY not found in environment"))
    }
}

const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";
const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";

#[async_trait::async_trait]
impl AiProvider for AmazonBedrockProvider {
    fn name(&self) -> &str {
        "amazon"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Amazon Bedrock supported models (case-insensitive)
        let model_lower = normalize_model_name(model);
        model_lower.contains("claude")
            || model_lower.contains("titan")
            || model_lower.contains("llama")
            || model_lower.contains("anthropic.")
            || model_lower.contains("meta.")
            || model_lower.contains("amazon.")
            || model_lower.contains("ai21.")
            || model_lower.contains("cohere.")
            || model_lower.contains("mistral.")
    }

    fn get_api_key(&self) -> Result<String> {
        // Amazon Bedrock requires both access key ID and secret access key
        let access_key_id = self.get_aws_access_key_id()?;
        let _secret_access_key = self.get_aws_secret_access_key()?; // Validate it exists
        Ok(access_key_id) // Return access key ID as the "API key"
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Claude models on Bedrock support vision (case-insensitive)
        let model_lower = normalize_model_name(model);
        model_lower.contains("claude-3")
            || model_lower.contains("claude-4")
            || model_lower.contains("anthropic.claude")
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Amazon Bedrock model context window limits (case-insensitive)
        let model_lower = normalize_model_name(model);
        if model_lower.contains("claude") || model_lower.contains("anthropic.claude") {
            200_000 // Claude models have 200K context
        } else if model_lower.contains("llama3-2-90b") || model_lower.contains("meta.llama3-2-90b")
        {
            128_000 // Llama 3.2 90B has 128K context
        } else if model_lower.contains("llama") || model_lower.contains("meta.llama") {
            32_768 // Other Llama models typically 32K
        } else if model_lower.contains("titan") || model_lower.contains("amazon.titan") {
            32_000 // Titan models have 32K context
        } else {
            32_768 // Conservative default
        }
    }

    async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
        Err(anyhow::anyhow!(
            "Amazon Bedrock provider not fully implemented in octolib"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = AmazonBedrockProvider::new();

        // Amazon Bedrock supported models
        assert!(provider.supports_model("anthropic.claude-3-haiku-20240307-v1:0"));
        assert!(provider.supports_model("anthropic.claude-3-5-sonnet-20241022-v2:0"));
        assert!(provider.supports_model("meta.llama3-2-90b-instruct-v1:0"));
        assert!(provider.supports_model("amazon.titan-embed-text-v2:0"));

        // Unsupported models
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("deepseek-chat"));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = AmazonBedrockProvider::new();

        // Test uppercase
        assert!(provider.supports_model("ANTHROPIC.CLAUDE-3-HAIKU-20240307-V1:0"));
        assert!(provider.supports_model("META.LLAMA3-2-90B-INSTRUCT-V1:0"));
        // Test mixed case
        assert!(provider.supports_model("Anthropic.Claude-3-Haiku"));
        assert!(provider.supports_model("AMAZON.TITAN-EMBED-TEXT-V2:0"));
    }

    #[test]
    fn test_supports_vision_case_insensitive() {
        let provider = AmazonBedrockProvider::new();

        // Test lowercase
        assert!(provider.supports_vision("claude-3-haiku"));
        assert!(provider.supports_vision("claude-3-sonnet"));

        // Test uppercase
        assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
        assert!(provider.supports_vision("CLAUDE-3-SONNET"));
        // Test mixed case
        assert!(provider.supports_vision("Anthropic.Claude-3-Haiku"));
    }
}
