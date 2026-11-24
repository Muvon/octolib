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

use super::*;

#[test]
fn provider_setup_covers_supported_providers_except_inception() {
	let setup = EvaluationProviderFactory::provider_setup_info();
	let expected: Vec<_> = EvaluationProviderFactory::supported_providers()
		.iter()
		.copied()
		.filter(|name| *name != "inception")
		.collect();
	let names: Vec<_> = setup.iter().map(|info| info.name).collect();
	assert_eq!(names, expected);
}

#[test]
fn provider_setup_models_are_supported() {
	for info in EvaluationProviderFactory::provider_setup_info() {
		let provider = EvaluationProviderFactory::create_provider(info.name).unwrap();
		for model in &info.models {
			assert!(
				provider.supports_model(model),
				"unsupported setup model: {}:{model}",
				info.name
			);
		}
	}
}

#[test]
fn routes_provider_model_pairs_to_their_native_model() {
	let (provider, model) = EvaluationProviderFactory::get_provider_for_model(" TypeSafe : jev-latest ").unwrap();
	assert_eq!(provider.name(), "typesafe");
	assert_eq!(model, "jev-latest");

	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("cloudflare:typesafe/jev").unwrap();
	assert_eq!(provider.name(), "cloudflare");
	assert_eq!(model, "typesafe/jev");

	let (provider, model) =
		EvaluationProviderFactory::get_provider_for_model("cloudflare:@cf/cloudflare/clef").unwrap();
	assert_eq!(provider.name(), "cloudflare");
	assert_eq!(model, "@cf/cloudflare/clef");

	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("inception:mercury-decide").unwrap();
	assert_eq!(provider.name(), "inception");
	assert_eq!(model, "mercury-decide");

	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("octohub:jev").unwrap();
	assert_eq!(provider.name(), "octohub");
	assert_eq!(model, "jev");

	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("ollama:clef").unwrap();
	assert_eq!(provider.name(), "ollama");
	assert_eq!(model, "clef");

	// A tag after the model name is part of the model, not a second separator.
	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("ollama:tev1:0.8b").unwrap();
	assert_eq!(provider.name(), "ollama");
	assert_eq!(model, "tev1:0.8b");

	let (provider, model) = EvaluationProviderFactory::get_provider_for_model("openai:gpt-6-luna").unwrap();
	assert_eq!(provider.name(), "openai");
	assert_eq!(model, "gpt-6-luna");

	let (provider, model) =
		EvaluationProviderFactory::get_provider_for_model("openrouter:openai/gpt-6-luna-decisions").unwrap();
	assert_eq!(provider.name(), "openrouter");
	assert_eq!(model, "openai/gpt-6-luna-decisions");
}

#[test]
fn rejects_unknown_providers_models_and_malformed_ids() {
	assert!(matches!(
		EvaluationProviderFactory::get_provider_for_model("unknown:jev-latest"),
		Err(EvaluationError::UnsupportedProvider(_))
	));
	assert!(matches!(
		EvaluationProviderFactory::get_provider_for_model("openai:jev-latest"),
		Err(EvaluationError::UnsupportedModel { .. })
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
