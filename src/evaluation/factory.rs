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

use super::errors::{EvaluationError, EvaluationResult};
use super::providers::{
	CloudflareEvaluationProvider, OctoHubEvaluationProvider, OllamaEvaluationProvider, OpenAiEvaluationProvider,
	OpenRouterEvaluationProvider, TypeSafeEvaluationProvider,
};
// Deprecated: Inception is removed entirely from octolib on 2026-12-28; the
// factory route stays supported until then.
#[allow(deprecated)]
use super::providers::InceptionEvaluationProvider;
use super::traits::EvaluationProvider;

/// Offline metadata for configuring an evaluation provider and selecting a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationProviderSetupInfo {
	/// Provider ID, as returned by [`EvaluationProviderFactory::supported_providers`].
	pub name: &'static str,
	/// Primary API key variable, including optional keys such as Ollama's.
	pub api_key_env: Option<&'static str>,
	/// Environment variables required by the default endpoint in addition to the API key.
	pub extra_env: &'static [&'static str],
	/// Callable model IDs in table order. Empty when the provider has no model list.
	pub models: Vec<String>,
}

impl EvaluationProviderSetupInfo {
	fn new(
		name: &'static str,
		api_key_env: &'static str,
		extra_env: &'static [&'static str],
		models: impl IntoIterator<Item = &'static str>,
	) -> Self {
		Self {
			name,
			api_key_env: Some(api_key_env),
			extra_env,
			models: models.into_iter().map(str::to_string).collect(),
		}
	}
}

pub struct EvaluationProviderFactory;

impl EvaluationProviderFactory {
	pub fn parse_model(value: &str) -> EvaluationResult<(String, String)> {
		let value = value.trim();
		let (provider, model) = value
			.split_once(':')
			.ok_or_else(|| EvaluationError::InvalidModelFormat(value.to_string()))?;
		let provider = provider.trim().to_ascii_lowercase();
		let model = model.trim().to_string();
		if provider.is_empty() || model.is_empty() {
			return Err(EvaluationError::InvalidModelFormat(value.to_string()));
		}
		Ok((provider, model))
	}

	pub fn create_provider(name: &str) -> EvaluationResult<Box<dyn EvaluationProvider>> {
		match name.to_ascii_lowercase().as_str() {
			"cloudflare" => Ok(Box::new(CloudflareEvaluationProvider::new())),
			// Deprecated: removed entirely on 2026-12-28; route stays until then.
			#[allow(deprecated)]
			"inception" => Ok(Box::new(InceptionEvaluationProvider::new())),
			"octohub" => Ok(Box::new(OctoHubEvaluationProvider::new())),
			"ollama" => Ok(Box::new(OllamaEvaluationProvider::new())),
			"openai" => Ok(Box::new(OpenAiEvaluationProvider::new())),
			"openrouter" => Ok(Box::new(OpenRouterEvaluationProvider::new())),
			"typesafe" => Ok(Box::new(TypeSafeEvaluationProvider::new())),
			other => Err(EvaluationError::UnsupportedProvider(other.to_string())),
		}
	}

	pub fn get_provider_for_model(value: &str) -> EvaluationResult<(Box<dyn EvaluationProvider>, String)> {
		let (provider_name, model) = Self::parse_model(value)?;
		let provider = Self::create_provider(&provider_name)?;
		if !provider.supports_model(&model) {
			return Err(EvaluationError::UnsupportedModel {
				provider: provider_name,
				model,
			});
		}
		Ok((provider, model))
	}

	pub fn supported_providers() -> &'static [&'static str] {
		&[
			"cloudflare",
			"inception",
			"octohub",
			"ollama",
			"openai",
			"openrouter",
			"typesafe",
		]
	}

	/// Get setup metadata without reading credentials or fetching model catalogs.
	pub fn provider_setup_info() -> Vec<EvaluationProviderSetupInfo> {
		use super::providers::{cloudflare, octohub, ollama, openai, openrouter, typesafe};

		// Inception is deprecated and will be removed on 2026-12-28, so omit it from setup.
		vec![
			EvaluationProviderSetupInfo::new(
				"cloudflare",
				cloudflare::API_KEY_ENV,
				&[cloudflare::ACCOUNT_ID_ENV],
				cloudflare::MODELS.iter().map(|row| row.0),
			),
			EvaluationProviderSetupInfo::new("octohub", octohub::API_KEY_ENV, &[], [octohub::AUTO_MODEL]),
			EvaluationProviderSetupInfo::new("ollama", ollama::API_KEY_ENV, &[], []),
			EvaluationProviderSetupInfo::new("openai", openai::API_KEY_ENV, &[], [openai::MODEL]),
			EvaluationProviderSetupInfo::new(
				"openrouter",
				openrouter::API_KEY_ENV,
				&[],
				openrouter::MODELS.iter().map(|row| row.0),
			),
			EvaluationProviderSetupInfo::new("typesafe", typesafe::API_KEY_ENV, &[], [typesafe::JEV_LATEST_MODEL]),
		]
	}
}

#[cfg(test)]
#[path = "factory_tests.rs"]
mod tests;
