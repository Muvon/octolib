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
//!
//! Authentication: Uses Amazon Bedrock API keys (long-term credentials).
//!
//! **How to get a Bedrock API key:**
//! 1. AWS Console → IAM → Users → Select/Create user
//! 2. Security credentials → Create service-specific credential
//! 3. Select "Amazon Bedrock" as the service
//! 4. Copy the generated API key (format: bedrock-<region>-<account>:<secret>)
//! 5. Set environment variable: export AWS_BEARER_TOKEN_BEDROCK="your-api-key"
//!
//! **Note:** These are NOT regular AWS access keys. They are Bedrock-specific API keys
//! that work with the OpenAI-compatible endpoint without requiring AWS SigV4 signing.
//!
//! Alternative: You can implement AWS SigV4 signing, but API keys are simpler for most use cases.

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, get_optional_api_key,
    OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::normalize_model_name;
use anyhow::Result;

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

const AWS_BEARER_TOKEN_BEDROCK_ENV: &str = "AWS_BEARER_TOKEN_BEDROCK";
const AWS_BEDROCK_REGION_ENV: &str = "AWS_BEDROCK_REGION";
const AWS_BEDROCK_API_URL_ENV: &str = "AWS_BEDROCK_API_URL";

fn default_bedrock_api_url() -> String {
    let region = std::env::var(AWS_BEDROCK_REGION_ENV).unwrap_or_else(|_| "us-east-1".to_string());
    format!(
        "https://bedrock-runtime.{}.amazonaws.com/openai/v1/chat/completions",
        region
    )
}

#[async_trait::async_trait]
impl AiProvider for AmazonBedrockProvider {
    fn name(&self) -> &str {
        "amazon"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        let token = get_optional_api_key(AWS_BEARER_TOKEN_BEDROCK_ENV);
        if token.is_empty() {
            Err(anyhow::anyhow!(
                "Amazon Bedrock API key not found. Set {} environment variable.\n\
                To create a Bedrock API key:\n\
                1. AWS Console → IAM → Users → Select/Create user\n\
                2. Security credentials → Create service-specific credential\n\
                3. Select 'Amazon Bedrock' as the service\n\
                4. Copy the generated API key (format: bedrock-<region>-<account>:<secret>)\n\
                Note: These are NOT regular AWS access keys.",
                AWS_BEARER_TOKEN_BEDROCK_ENV
            ))
        } else {
            Ok(token)
        }
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
            || model_lower.contains("nova")
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

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(AWS_BEDROCK_API_URL_ENV, &default_bedrock_api_url());

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "amazon",
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
    fn test_supports_model() {
        let provider = AmazonBedrockProvider::new();

        // Amazon Bedrock accepts any non-empty model identifier
        assert!(provider.supports_model("anthropic.claude-3-haiku-20240307-v1:0"));
        assert!(provider.supports_model("anthropic.claude-3-5-sonnet-20241022-v2:0"));
        assert!(provider.supports_model("meta.llama3-2-90b-instruct-v1:0"));
        assert!(provider.supports_model("amazon.titan-embed-text-v2:0"));
        assert!(provider.supports_model("gpt-4"));
        assert!(provider.supports_model("deepseek-chat"));
        assert!(!provider.supports_model(""));
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
