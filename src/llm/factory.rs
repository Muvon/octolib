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

//! Provider factory for creating AI provider instances

use crate::llm::providers::{
    AlibabaProvider, AmazonBedrockProvider, AnthropicProvider, BytePlusProvider, CerebrasProvider,
    CliProvider, CloudflareWorkersAiProvider, DeepSeekProvider, FeatherlessProvider,
    FireworksProvider, GoogleStudioProvider, GoogleVertexProvider, GroqProvider, HetznerProvider,
    LocalProvider, MetaProvider, MinimaxProvider, MoonshotProvider, NvidiaProvider,
    OctoHubProvider, OllamaProvider, OpenAiProvider, OpenCodeGoProvider, OpenCodeZenProvider,
    OpenRouterProvider, TogetherProvider, XaiProvider, ZaiProvider,
};
use crate::llm::traits::AiProvider;
use anyhow::Result;

/// Provider factory to create the appropriate provider based on model string
pub struct ProviderFactory;

impl ProviderFactory {
    /// Parse a model string in format "provider:model" and return (provider_name, model_name)
    /// Provider prefix is now REQUIRED
    pub fn parse_model(model: &str) -> Result<(String, String)> {
        let model = model.trim();
        if let Some(pos) = model.find(':') {
            let provider = model[..pos].trim().to_string();
            let model_name = model[pos + 1..].trim().to_string();

            if provider.is_empty() || model_name.is_empty() {
                return Err(anyhow::anyhow!(
                    "Invalid model format. Use 'provider:model' (e.g., 'openai:gpt-4o')"
                ));
            }

            Ok((provider, model_name))
        } else {
            Err(anyhow::anyhow!(
                "Invalid model format '{}'. Must specify provider like 'provider:model'",
                model
            ))
        }
    }

    /// Create a provider instance based on the provider name
    pub fn create_provider(provider_name: &str) -> Result<Box<dyn AiProvider>> {
        match provider_name.to_lowercase().as_str() {
            "openrouter" => Ok(Box::new(OpenRouterProvider::new())),
            "openai" => Ok(Box::new(OpenAiProvider::new())),
            "cerebras" => Ok(Box::new(CerebrasProvider::new())),
            "local" => Ok(Box::new(LocalProvider::new())),
            "ollama" => Ok(Box::new(OllamaProvider::new())),
            "anthropic" => Ok(Box::new(AnthropicProvider::new())),
            "byteplus" => Ok(Box::new(BytePlusProvider::new())),
            "alibaba" => Ok(Box::new(AlibabaProvider::new())),
            "google-vertex" => Ok(Box::new(GoogleVertexProvider::new())),
            "google-studio" => Ok(Box::new(GoogleStudioProvider::new())),
            "groq" => Ok(Box::new(GroqProvider::new())),
            "amazon" => Ok(Box::new(AmazonBedrockProvider::new())),
            "cloudflare" => Ok(Box::new(CloudflareWorkersAiProvider::new())),
            "deepseek" => Ok(Box::new(DeepSeekProvider::new())),
            "featherless" => Ok(Box::new(FeatherlessProvider::new())),
            "fireworks" => Ok(Box::new(FireworksProvider::new())),
            "hetzner" => Ok(Box::new(HetznerProvider::new())),
            "meta" => Ok(Box::new(MetaProvider::new())),
            "minimax" => Ok(Box::new(MinimaxProvider::new())),
            "moonshot" | "kimi" => Ok(Box::new(MoonshotProvider::new())),
            "nvidia" => Ok(Box::new(NvidiaProvider::new())),
            "octohub" => Ok(Box::new(OctoHubProvider::new())),
            "opencode-zen" => Ok(Box::new(OpenCodeZenProvider::new())),
            "opencode-go" => Ok(Box::new(OpenCodeGoProvider::new())),
            "together" => Ok(Box::new(TogetherProvider::new())),
            "xai" => Ok(Box::new(XaiProvider::new())),
            "zai" => Ok(Box::new(ZaiProvider::new())),
            "cli" => Err(anyhow::anyhow!(
                "CLI provider requires a model string like 'cli:<backend>/<model>'. Use ProviderFactory::get_provider_for_model instead."
            )),
            _ => Err(anyhow::anyhow!("Unsupported provider: {}. Supported: openai, anthropic, openrouter, cerebras, local, ollama, google-vertex, google-studio, groq, alibaba, amazon, cloudflare, deepseek, featherless, fireworks, hetzner, meta, minimax, moonshot, nvidia, octohub, opencode-zen, opencode-go, together, xai, zai, byteplus, cli", provider_name))
        }
    }

    /// Get the appropriate provider for a given model string
    pub fn get_provider_for_model(model: &str) -> Result<(Box<dyn AiProvider>, String)> {
        let (provider_name, model_name) = Self::parse_model(model)?;
        let provider: Box<dyn AiProvider> = if provider_name.eq_ignore_ascii_case("cli") {
            Box::new(CliProvider::new_for_model(&model_name)?)
        } else {
            Self::create_provider(&provider_name)?
        };

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
            "cerebras",
            "local",
            "ollama",
            "anthropic",
            "google-vertex",
            "google-studio",
            "groq",
            "alibaba",
            "amazon",
            "cloudflare",
            "deepseek",
            "featherless",
            "fireworks",
            "hetzner",
            "meta",
            "minimax",
            "moonshot",
            "nvidia",
            "octohub",
            "opencode-zen",
            "opencode-go",
            "together",
            "xai",
            "zai",
            "byteplus",
            "cli",
        ]
    }

    /// Validate model format without creating provider
    pub fn validate_model_format(model: &str) -> Result<()> {
        Self::parse_model(model)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "factory_tests.rs"]
mod tests;
