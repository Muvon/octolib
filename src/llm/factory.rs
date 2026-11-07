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
	AlibabaProvider, AmazonBedrockProvider, AnthropicProvider, BytePlusProvider, CerebrasProvider, ChatGptProvider,
	CliProvider, CloudflareWorkersAiProvider, DeepSeekProvider, FeatherlessProvider, FireworksProvider,
	GoogleStudioProvider, GoogleVertexProvider, GroqProvider, HetznerProvider, InceptionProvider, IsoquantProvider,
	LocalProvider, MetaProvider, MinimaxProvider, MoonshotProvider, NvidiaProvider, OctoHubProvider, OllamaProvider,
	OpenAiProvider, OpenCodeGoProvider, OpenCodeZenProvider, OpenRouterProvider, ReflectionProvider, TinkerProvider,
	TogetherProvider, WebWrightProvider, XaiProvider, XiaomiProvider, ZaiProvider,
};
use crate::llm::traits::AiProvider;
use anyhow::Result;

/// Offline metadata for configuring a provider and selecting a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderSetupInfo {
	/// Provider ID, as returned by [`ProviderFactory::supported_providers`].
	pub name: &'static str,
	/// Primary API key variable, including optional keys for local, Ollama and OctoHub.
	/// `None` means authentication uses login or a credentials file instead.
	pub api_key_env: Option<&'static str>,
	/// Environment variables required in addition to api_key_env;
	/// alternatives and vars with defaults are not listed.
	pub extra_env: &'static [&'static str],
	/// Callable IDs from local model/pricing tables, in table order without duplicates.
	/// Empty when only pricing patterns or a remote catalog are available.
	pub models: Vec<String>,
	/// Public model list in the OpenAI `/v1/models` shape (`data[].id`) that answers
	/// without an API key. Set where `models` is empty and the provider publishes one.
	pub models_url: Option<&'static str>,
}

impl ProviderSetupInfo {
	fn new(
		name: &'static str,
		api_key_env: Option<&'static str>,
		extra_env: &'static [&'static str],
		models: impl IntoIterator<Item = &'static str>,
	) -> Self {
		let mut seen = std::collections::HashSet::new();
		Self {
			name,
			api_key_env,
			extra_env,
			models: models
				.into_iter()
				.filter(|model| seen.insert(*model))
				.map(str::to_string)
				.collect(),
			models_url: None,
		}
	}
}

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
            "chatgpt" => Ok(Box::new(ChatGptProvider::new())),
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
            "inception" => Ok(Box::new(InceptionProvider::new())),
            "isoquant" => Ok(Box::new(IsoquantProvider::new())),
            "meta" => Ok(Box::new(MetaProvider::new())),
            "minimax" => Ok(Box::new(MinimaxProvider::new())),
            "moonshot" | "kimi" => Ok(Box::new(MoonshotProvider::new())),
            "nvidia" => Ok(Box::new(NvidiaProvider::new())),
            "octohub" => Ok(Box::new(OctoHubProvider::new())),
            "opencode-zen" => Ok(Box::new(OpenCodeZenProvider::new())),
            "opencode-go" => Ok(Box::new(OpenCodeGoProvider::new())),
            "reflection" => Ok(Box::new(ReflectionProvider::new())),
            "tinker" => Ok(Box::new(TinkerProvider::new())),
            "together" => Ok(Box::new(TogetherProvider::new())),
            "webwright" => Ok(Box::new(WebWrightProvider::new())),
            "xai" => Ok(Box::new(XaiProvider::new())),
            "xiaomi" => Ok(Box::new(XiaomiProvider::new())),
            "zai" => Ok(Box::new(ZaiProvider::new())),
            "cli" => Err(anyhow::anyhow!(
                "CLI provider requires a model string like 'cli:<backend>/<model>'. Use ProviderFactory::get_provider_for_model instead."
            )),
            _ => Err(anyhow::anyhow!("Unsupported provider: {}. Supported: openai, chatgpt, anthropic, openrouter, cerebras, local, ollama, google-vertex, google-studio, groq, alibaba, amazon, cloudflare, deepseek, featherless, fireworks, hetzner, inception, isoquant, meta, minimax, moonshot, nvidia, octohub, opencode-zen, opencode-go, reflection, tinker, together, webwright, xai, xiaomi, zai, byteplus, cli", provider_name))
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
			"chatgpt",
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
			"inception",
			"isoquant",
			"meta",
			"minimax",
			"moonshot",
			"nvidia",
			"octohub",
			"opencode-zen",
			"opencode-go",
			"reflection",
			"tinker",
			"together",
			"webwright",
			"xai",
			"xiaomi",
			"zai",
			"byteplus",
			"cli",
		]
	}

	/// Get offline setup metadata for every supported provider except CLI backends.
	/// Does not read credentials, perform login, or fetch remote model catalogs;
	/// callers fetch `models_url` themselves.
	pub fn provider_setup_info() -> Vec<ProviderSetupInfo> {
		use crate::llm::providers::{
			alibaba, amazon, anthropic, byteplus, cerebras, cloudflare, deepseek, featherless, fireworks,
			google_studio, google_vertex, groq, hetzner, inception, isoquant, local, meta, minimax, moonshot, nvidia,
			octohub, ollama, openai, opencode, openrouter, reflection, tinker, together, webwright, xai, xiaomi, zai,
		};

		vec![
			ProviderSetupInfo {
				models_url: Some(openrouter::OPENROUTER_MODELS_URL),
				..ProviderSetupInfo::new("openrouter", Some(openrouter::OPENROUTER_API_KEY_ENV), &[], [])
			},
			ProviderSetupInfo::new(
				"openai",
				Some(openai::OPENAI_API_KEY_ENV),
				&[],
				openai::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new("chatgpt", None, &[], []),
			ProviderSetupInfo::new(
				"cerebras",
				Some(cerebras::CEREBRAS_API_KEY_ENV),
				&[],
				cerebras::SUPPORTED_MODELS.iter().copied(),
			),
			ProviderSetupInfo::new("local", Some(local::LOCAL_API_KEY_ENV), &[], []),
			ProviderSetupInfo {
				models_url: Some(ollama::OLLAMA_MODELS_URL),
				..ProviderSetupInfo::new("ollama", Some(ollama::OLLAMA_API_KEY_ENV), &[], [])
			},
			ProviderSetupInfo::new(
				"anthropic",
				Some(anthropic::ANTHROPIC_API_KEY_ENV),
				&[],
				anthropic::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new("google-vertex", None, &[], google_vertex::setup_models()),
			ProviderSetupInfo::new(
				"google-studio",
				Some(google_studio::GOOGLE_STUDIO_API_KEY_ENV),
				&[],
				google_vertex::setup_models(),
			),
			ProviderSetupInfo::new(
				"groq",
				Some(groq::GROQ_API_KEY_ENV),
				&[],
				groq::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"alibaba",
				Some(alibaba::ALIBABA_API_KEY_ENV),
				&[],
				alibaba::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"amazon",
				Some(amazon::AWS_BEARER_TOKEN_BEDROCK_ENV),
				&[],
				amazon::setup_models(),
			),
			ProviderSetupInfo::new(
				"cloudflare",
				Some(cloudflare::CLOUDFLARE_API_KEY_ENV),
				&[cloudflare::CLOUDFLARE_ACCOUNT_ID_ENV],
				cloudflare::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"deepseek",
				Some(deepseek::DEEPSEEK_API_KEY_ENV),
				&[],
				deepseek::PRICING_PEAK.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"featherless",
				Some(featherless::FEATHERLESS_API_KEY_ENV),
				&[],
				featherless::setup_models(),
			),
			// Fireworks pricing rows are path suffix patterns, not callable route IDs.
			ProviderSetupInfo::new("fireworks", Some(fireworks::FIREWORKS_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new(
				"hetzner",
				Some(hetzner::HETZNER_API_KEY_ENV),
				&[],
				hetzner::MODELS.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"inception",
				Some(inception::INCEPTION_API_KEY_ENV),
				&[],
				inception::MODELS.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"isoquant",
				Some(isoquant::ISOQUANT_API_KEY_ENV),
				&[],
				isoquant::KNOWN_MODELS.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"meta",
				Some(meta::META_API_KEY_ENV),
				&[],
				meta::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"minimax",
				Some(minimax::MINIMAX_API_KEY_ENV),
				&[],
				minimax::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"moonshot",
				Some(moonshot::MOONSHOT_API_KEY_ENV),
				&[],
				moonshot::setup_models(),
			),
			ProviderSetupInfo::new("nvidia", Some(nvidia::NVIDIA_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new("octohub", Some(octohub::OCTOHUB_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new("opencode-zen", Some(opencode::OPENCODE_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new("opencode-go", Some(opencode::OPENCODE_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new(
				"reflection",
				Some(reflection::REFLECTION_API_KEY_ENV),
				&[],
				reflection::MODELS.iter().map(|row| row.0),
			),
			// Tinker pricing keys are case-normalized patterns; requests preserve model ID case.
			ProviderSetupInfo::new("tinker", Some(tinker::TINKER_API_KEY_ENV), &[], []),
			ProviderSetupInfo::new(
				"together",
				Some(together::TOGETHER_API_KEY_ENV),
				&[],
				together::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"webwright",
				Some(webwright::WEBWRIGHT_API_KEY_ENV),
				&[],
				webwright::MODELS.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"xai",
				Some(xai::XAI_API_KEY_ENV),
				&[],
				xai::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"xiaomi",
				Some(xiaomi::XIAOMI_API_KEY_ENV),
				&[],
				xiaomi::MODELS.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"zai",
				Some(zai::ZAI_API_KEY_ENV),
				&[],
				zai::PRICING.iter().map(|row| row.0),
			),
			ProviderSetupInfo::new(
				"byteplus",
				Some(byteplus::BYTEPLUS_API_KEY_ENV),
				&[],
				byteplus::setup_models(),
			),
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
