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
    CloudflareEvaluationProvider, OctoHubEvaluationProvider, TypeSafeEvaluationProvider,
};
use super::traits::EvaluationProvider;

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
            "octohub" => Ok(Box::new(OctoHubEvaluationProvider::new())),
            "typesafe" => Ok(Box::new(TypeSafeEvaluationProvider::new())),
            other => Err(EvaluationError::UnsupportedProvider(other.to_string())),
        }
    }

    pub fn get_provider_for_model(
        value: &str,
    ) -> EvaluationResult<(Box<dyn EvaluationProvider>, String)> {
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
        &["cloudflare", "octohub", "typesafe"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_provider_model_pairs_to_their_native_model() {
        let (provider, model) =
            EvaluationProviderFactory::get_provider_for_model(" TypeSafe : jev-latest ").unwrap();
        assert_eq!(provider.name(), "typesafe");
        assert_eq!(model, "jev-latest");

        let (provider, model) =
            EvaluationProviderFactory::get_provider_for_model("cloudflare:typesafe/jev").unwrap();
        assert_eq!(provider.name(), "cloudflare");
        assert_eq!(model, "typesafe/jev");

        let (provider, model) =
            EvaluationProviderFactory::get_provider_for_model("octohub:jev").unwrap();
        assert_eq!(provider.name(), "octohub");
        assert_eq!(model, "jev");
    }

    #[test]
    fn rejects_unknown_providers_models_and_malformed_ids() {
        assert!(matches!(
            EvaluationProviderFactory::get_provider_for_model("openai:jev-latest"),
            Err(EvaluationError::UnsupportedProvider(_))
        ));
        assert!(matches!(
            EvaluationProviderFactory::get_provider_for_model("cloudflare:@cf/zai-org/glm-5.3"),
            Err(EvaluationError::UnsupportedModel { .. })
        ));
        assert!(matches!(
            EvaluationProviderFactory::parse_model("jev-latest"),
            Err(EvaluationError::InvalidModelFormat(_))
        ));
        assert!(EvaluationProviderFactory::parse_model("typesafe:").is_err());
    }
}
