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

//! Provider factory for creating AI provider instances

use crate::llm::providers::{
    AmazonBedrockProvider, AnthropicProvider, CloudflareWorkersAiProvider, DeepSeekProvider,
    GoogleVertexProvider, OpenAiProvider, OpenRouterProvider,
};
use crate::llm::traits::AiProvider;
use anyhow::Result;

/// Provider factory to create the appropriate provider based on model string
pub struct ProviderFactory;

impl ProviderFactory {
    /// Parse a model string in format "provider:model" and return (provider_name, model_name)
    /// Provider prefix is now REQUIRED
    pub fn parse_model(model: &str) -> Result<(String, String)> {
        if let Some(pos) = model.find(':') {
            let provider = model[..pos].to_string();
            let model_name = model[pos + 1..].to_string();

            if provider.is_empty() || model_name.is_empty() {
                return Err(anyhow::anyhow!(
                    "Invalid model format. Use 'provider:model' (e.g., 'openai:gpt-4o')"
                ));
            }

            Ok((provider, model_name))
        } else {
            Err(anyhow::anyhow!("Invalid model format '{}'. Must specify provider like 'openai:gpt-4o' or 'openrouter:anthropic/claude-3.5-sonnet'", model))
        }
    }

    /// Create a provider instance based on the provider name
    pub fn create_provider(provider_name: &str) -> Result<Box<dyn AiProvider>> {
        match provider_name.to_lowercase().as_str() {
            "openrouter" => Ok(Box::new(OpenRouterProvider::new())),
            "openai" => Ok(Box::new(OpenAiProvider::new())),
            "anthropic" => Ok(Box::new(AnthropicProvider::new())),
            "google" => Ok(Box::new(GoogleVertexProvider::new())),
            "amazon" => Ok(Box::new(AmazonBedrockProvider::new())),
            "cloudflare" => Ok(Box::new(CloudflareWorkersAiProvider::new())),
            "deepseek" => Ok(Box::new(DeepSeekProvider::new())),
            _ => Err(anyhow::anyhow!("Unsupported provider: {}. Supported providers: openrouter, openai, anthropic, google, amazon, cloudflare, deepseek", provider_name)),
        }
    }

    /// Get the appropriate provider for a given model string
    pub fn get_provider_for_model(model: &str) -> Result<(Box<dyn AiProvider>, String)> {
        let (provider_name, model_name) = Self::parse_model(model)?;
        let provider = Self::create_provider(&provider_name)?;

        // Verify the provider supports this model
        if !provider.supports_model(&model_name) {
            return Err(anyhow::anyhow!(
                "Provider '{}' does not support model '{}'",
                provider_name,
                model_name
            ));
        }

        Ok((provider, model_name))
    }

    /// Get list of all supported providers
    pub fn supported_providers() -> Vec<&'static str> {
        vec![
            "openrouter",
            "openai",
            "anthropic",
            "google",
            "amazon",
            "cloudflare",
            "deepseek",
        ]
    }

    /// Validate model format without creating provider
    pub fn validate_model_format(model: &str) -> Result<()> {
        Self::parse_model(model)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model() {
        // Test with provider prefix
        let result = ProviderFactory::parse_model("openrouter:anthropic/claude-3.5-sonnet");
        assert!(result.is_ok());
        let (provider, model) = result.unwrap();
        assert_eq!(provider, "openrouter");
        assert_eq!(model, "anthropic/claude-3.5-sonnet");

        // Test with different provider
        let result = ProviderFactory::parse_model("openai:gpt-4o");
        assert!(result.is_ok());
        let (provider, model) = result.unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");

        // Test DeepSeek provider
        let result = ProviderFactory::parse_model("deepseek:deepseek-chat");
        assert!(result.is_ok());
        let (provider, model) = result.unwrap();
        assert_eq!(provider, "deepseek");
        assert_eq!(model, "deepseek-chat");

        // Test invalid format (no colon)
        let result = ProviderFactory::parse_model("gpt-4o");
        assert!(result.is_err());

        // Test invalid format (empty provider)
        let result = ProviderFactory::parse_model(":gpt-4o");
        assert!(result.is_err());

        // Test invalid format (empty model)
        let result = ProviderFactory::parse_model("openai:");
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_providers() {
        let providers = ProviderFactory::supported_providers();
        assert!(providers.contains(&"openai"));
        assert!(providers.contains(&"anthropic"));
        assert!(providers.contains(&"openrouter"));
        assert!(providers.contains(&"google"));
        assert!(providers.contains(&"amazon"));
        assert!(providers.contains(&"cloudflare"));
        assert!(providers.contains(&"deepseek"));
    }

    #[test]
    fn test_validate_model_format() {
        assert!(ProviderFactory::validate_model_format("openai:gpt-4o").is_ok());
        assert!(ProviderFactory::validate_model_format("anthropic:claude-3.5-sonnet").is_ok());
        assert!(ProviderFactory::validate_model_format("gpt-4o").is_err());
        assert!(ProviderFactory::validate_model_format(":model").is_err());
        assert!(ProviderFactory::validate_model_format("provider:").is_err());
    }

    #[test]
    fn test_create_provider() {
        // Test creating valid providers
        assert!(ProviderFactory::create_provider("openai").is_ok());
        assert!(ProviderFactory::create_provider("anthropic").is_ok());
        assert!(ProviderFactory::create_provider("openrouter").is_ok());
        assert!(ProviderFactory::create_provider("google").is_ok());
        assert!(ProviderFactory::create_provider("amazon").is_ok());
        assert!(ProviderFactory::create_provider("cloudflare").is_ok());
        assert!(ProviderFactory::create_provider("deepseek").is_ok());

        // Test case insensitive
        assert!(ProviderFactory::create_provider("OpenAI").is_ok());
        assert!(ProviderFactory::create_provider("ANTHROPIC").is_ok());

        // Test invalid provider
        assert!(ProviderFactory::create_provider("invalid").is_err());
    }

    #[test]
    fn test_provider_capabilities() {
        let openai = ProviderFactory::create_provider("openai").unwrap();
        assert_eq!(openai.name(), "openai");
        assert!(openai.supports_model("gpt-4o"));
        assert!(openai.supports_vision("gpt-4o"));
        assert!(openai.supports_caching("gpt-4o")); // OpenAI now supports caching

        let anthropic = ProviderFactory::create_provider("anthropic").unwrap();
        assert_eq!(anthropic.name(), "anthropic");
        assert!(anthropic.supports_model("claude-3.5-sonnet"));
        assert!(anthropic.supports_vision("claude-3.5-sonnet"));
        assert!(anthropic.supports_caching("claude-3.5-sonnet"));

        let openrouter = ProviderFactory::create_provider("openrouter").unwrap();
        assert_eq!(openrouter.name(), "openrouter");
        assert!(openrouter.supports_model("any-model")); // OpenRouter accepts any model
        assert!(openrouter.supports_vision("claude-3.5-sonnet"));
        assert!(openrouter.supports_caching("claude-3.5-sonnet"));
    }

    #[test]
    fn test_get_provider_for_model() {
        // Test valid model strings
        let result = ProviderFactory::get_provider_for_model("openai:gpt-4o");
        assert!(result.is_ok());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(model, "gpt-4o");

        let result = ProviderFactory::get_provider_for_model("anthropic:claude-3.5-sonnet");
        assert!(result.is_ok());
        let (provider, model) = result.unwrap();
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(model, "claude-3.5-sonnet");

        // Test invalid format
        let result = ProviderFactory::get_provider_for_model("gpt-4o");
        assert!(result.is_err());

        // Test unsupported provider
        let result = ProviderFactory::get_provider_for_model("invalid:model");
        assert!(result.is_err());
    }
}
