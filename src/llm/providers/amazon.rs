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

//! Amazon Bedrock provider implementation
//!
//! Authentication: Uses Amazon Bedrock API keys (long-term credentials).
//!
//! **How to get a Bedrock API key:**
//! 1. AWS Console → IAM → Users → Select/Create user
//! 2. Security credentials → Create service-specific credential
//! 3. Select "Amazon Bedrock" as the service
//! 4. Copy the generated API key (format: bedrock-region-account:secret)
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
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
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

/// Bedrock on-demand Standard rates (us-east-1, per 1M tokens) for hosted
/// open-weight and partner models whose Bedrock price differs from the
/// maker's. Patterns keep Bedrock's `vendor.` prefix and match raw (no
/// sanitizing), so `deepseek.v3.2` never catches DeepSeek's own
/// `deepseek-v3.2`. Source: AWS Price List API, AmazonBedrock offer published
/// 2026-09-30. `global.` cross-region IDs bill below in-region IDs and must
/// precede them. Models without prompt caching bill cache columns at input.
/// Format: (model, input, output, cache_write, cache_read)
const PRICING: &[PricingTuple] = &[
    ("global.moonshotai.kimi-k3", 3.00, 15.00, 3.75, 0.30),
    ("moonshotai.kimi-k3", 3.30, 16.50, 4.125, 0.33),
    ("global.xai.grok-4.7", 2.00, 6.00, 2.00, 0.50),
    ("xai.grok-4.7", 2.20, 6.60, 2.20, 0.55),
    ("global.xai.grok-4.6", 2.00, 6.00, 2.00, 0.50),
    ("xai.grok-4.6", 2.20, 6.60, 2.20, 0.55),
    ("qwen.qwen3-235b-a22b-2507", 0.22, 0.88, 0.22, 0.22),
    ("qwen.qwen3-32b", 0.15, 0.60, 0.15, 0.15),
    ("qwen.qwen3-coder-480b-a35b", 0.45, 1.80, 0.45, 0.45),
    ("qwen.qwen3-coder-next", 0.50, 1.20, 0.50, 0.50),
    ("qwen.qwen3-next-80b-a3b", 0.14, 1.20, 0.14, 0.14),
    ("openai.gpt-oss-120b", 0.15, 0.60, 0.15, 0.15),
    ("openai.gpt-oss-20b", 0.07, 0.30, 0.07, 0.07),
    ("deepseek.v3.2", 0.62, 1.85, 0.62, 0.62),
    // DeepSeek V3.1: `deepseek.v3.1` on the Mantle endpoint, `deepseek.v3-v1:0`
    // on bedrock-runtime.
    ("deepseek.v3.1", 0.58, 1.68, 0.58, 0.58),
    ("deepseek.v3-v1", 0.58, 1.68, 0.58, 0.58),
    ("deepseek.r1", 1.35, 5.40, 1.35, 1.35),
    ("zai.glm-4.7-flash", 0.07, 0.40, 0.07, 0.07),
    ("minimax.minimax-m2.1", 0.30, 1.20, 0.30, 0.30),
    ("google.gemma-4-31b", 0.14, 0.40, 0.14, 0.14),
    ("google.gemma-4-26b-a4b", 0.13, 0.40, 0.13, 0.13),
    ("google.gemma-4-e2b", 0.04, 0.08, 0.04, 0.04),
];

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
        // Bedrock-specific known models
        let model_lower = normalize_model_name(model);
        if model_lower.contains("claude-3")
            || model_lower.contains("claude-4")
            || model_lower.contains("anthropic.claude")
        {
            return true;
        }
        // Fall back to reference capabilities for other models on Bedrock
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.vision)
            .unwrap_or(false)
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Bedrock-specific known models
        let model_lower = normalize_model_name(model);
        if model_lower.contains("claude") || model_lower.contains("anthropic.claude") {
            return 200_000;
        }
        if model_lower.contains("titan") || model_lower.contains("amazon.titan") {
            return 32_000;
        }
        // Fall back to reference capabilities for other models on Bedrock
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.max_input_tokens)
            .unwrap_or(32_768)
    }

    fn supports_structured_output(&self, model: &str) -> bool {
        // Bedrock structured outputs cover the Anthropic Claude routes. Other
        // hosted families resolve through reference capabilities — the Nova
        // model cards list structured outputs as not supported.
        let model_lower = normalize_model_name(model);
        if model_lower.contains("claude") {
            return true;
        }
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.structured_output)
            .unwrap_or(false)
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Bedrock's own rate where it differs from the maker's; otherwise the
        // reference table's model-level rate.
        get_model_pricing(model, PRICING)
            .map(|(input, output, cache_write, cache_read)| {
                crate::llm::types::ModelPricing::new(input, output, cache_write, cache_read)
            })
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(AWS_BEDROCK_API_URL_ENV, &default_bedrock_api_url());

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "amazon",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: self.enforces_response_schema(&params.model),
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await
    }
}

#[cfg(test)]
#[path = "amazon_tests.rs"]
mod tests;
