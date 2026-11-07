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
use crate::llm::providers::OpenRouterProvider;
use crate::llm::types::Message;

const DEEPSEEK: &str = "deepseek-v4.1-flash-1017-fast";
const GLM: &str = "glm-5.3-flash";

fn catalog_fixture() -> Vec<CatalogEntry> {
	let response: CatalogResponse = serde_json::from_value(serde_json::json!({
		"object": "list",
		"data": [
			{
				"id": "new-model",
				"isoquant": {
					"apis": {"chat_completions": {
						"input_modalities": ["text", "image"],
						"structured_outputs": ["json_object", "json_schema"],
						"tool_choice": ["auto", "required"]
					}},
					"context_length": 262144,
					"pricing": {
						"input_nano_usd_per_token": "35",
						"output_nano_usd_per_token": "370",
						"cached_input_nano_usd_per_token": "5"
					},
					"reasoning": {"efforts": ["none", "low", "future-level", "high"]}
				}
			},
			{
				"id": "plain-model",
				"isoquant": {
					"apis": {"chat_completions": {"input_modalities": ["text"]}},
					"context_length": 32768
				}
			},
			{
				"id": "messages-only",
				"isoquant": {"apis": {"messages": {}}, "context_length": 1000}
			},
			{"id": "no-metadata"}
		]
	}))
	.unwrap();
	response.data
}

#[test]
fn catalog_keeps_chat_models_with_their_facts() {
	let models = parse_catalog(catalog_fixture());
	let ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();
	assert_eq!(ids, ["new-model", "plain-model"]);

	let full = &models[0];
	assert_eq!(full.context_window, 262_144);
	assert!(full.vision);
	assert!(full.structured_output);
	assert!(full.required_tool_choice);
	let pricing = full.pricing.unwrap();
	assert_eq!(pricing.input_price_per_1m, 0.035);
	assert_eq!(pricing.output_price_per_1m, 0.37);
	assert_eq!(pricing.cache_write_price_per_1m, 0.035);
	assert_eq!(pricing.cache_read_price_per_1m, 0.005);
	let levels: Vec<&str> = full.efforts.iter().map(|(_, name)| *name).collect();
	assert_eq!(levels, ["none", "low", "high"]);

	let plain = &models[1];
	assert!(!plain.vision);
	assert!(!plain.structured_output);
	assert!(!plain.required_tool_choice);
	assert!(plain.pricing.is_none());
	assert!(plain.efforts.is_empty());
}

#[test]
fn cached_rate_defaults_to_input_rate() {
	let pricing = parse_pricing(&CatalogPricing {
		input_nano_usd_per_token: Some("70".to_string()),
		output_nano_usd_per_token: Some("150".to_string()),
		cached_input_nano_usd_per_token: None,
	})
	.unwrap();
	assert_eq!(pricing.cache_read_price_per_1m, 0.07);
}

#[test]
fn effort_requests_land_on_a_listed_level() {
	use ReasoningEffort::*;

	for (requested, deepseek, glm) in [
		(None, "none", "low"),
		(Low, "low", "low"),
		(Medium, "low", "low"),
		(High, "high", "high"),
		(XHigh, "high", "high"),
		(Max, "high", "max"),
	] {
		assert_eq!(
			reasoning_effort_value(DEEPSEEK, requested),
			Some(deepseek),
			"{requested:?}"
		);
		assert_eq!(reasoning_effort_value(GLM, requested), Some(glm), "{requested:?}");
	}
}

#[test]
fn models_without_levels_send_no_effort() {
	assert_eq!(select_effort(&[], ReasoningEffort::High), Option::None);
	assert_eq!(
		reasoning_effort_value("unknown-model", ReasoningEffort::High),
		Option::None
	);
}

#[test]
fn zero_data_retention_replaces_a_caller_value() {
	let messages = vec![Message::user("hi")];
	let mut params = ChatCompletionParams::new(&messages, GLM, 0.7, 1.0, 50, 64).with_extra_headers(
		[
			("isoquant-zdr".to_string(), "off".to_string()),
			("x-kept".to_string(), "yes".to_string()),
		]
		.into(),
	);

	require_zero_data_retention(&mut params);

	let headers = params.extra_headers.unwrap();
	assert_eq!(headers.len(), 2);
	assert_eq!(headers.get("Isoquant-ZDR").map(String::as_str), Some("required"));
	assert_eq!(headers.get("x-kept").map(String::as_str), Some("yes"));
}

#[test]
fn catalog_url_sits_beside_the_chat_endpoint() {
	assert_eq!(models_url(ISOQUANT_API_URL), "https://api.isoquant.ai/v1/models");
	assert_eq!(
		models_url("http://localhost:8080/v1"),
		"http://localhost:8080/v1/models"
	);
}

#[test]
fn known_models_work_before_the_catalog_loads() {
	let provider = IsoquantProvider::new();
	assert_eq!(provider.name(), "isoquant");
	assert!(!provider.supports_model(""));
	assert!(provider.supports_model("model-released-tomorrow"));

	for model in [DEEPSEEK, GLM] {
		assert!(provider.supports_model(model), "{model}");
		assert!(provider.supports_vision(model), "{model}");
		assert!(!provider.supports_video(model), "{model}");
		assert!(provider.supports_caching(model), "{model}");
		assert!(provider.supports_structured_output(model), "{model}");
		assert!(provider.enforces_response_schema(model), "{model}");
		assert!(provider.supports_required_tool_choice(model), "{model}");
		assert_eq!(provider.get_max_input_tokens(model), 1_048_576, "{model}");
		assert_eq!(
			provider.supported_sampling_params(model),
			SamplingSupport::TEMPERATURE_AND_TOP_P,
			"{model}"
		);
	}
}

#[test]
fn unprobed_models_do_not_claim_schema_enforcement() {
	let provider = IsoquantProvider::new();
	assert!(!provider.enforces_response_schema("model-released-tomorrow"));
	assert!(!provider.supports_required_tool_choice("model-released-tomorrow"));
}

#[test]
fn cache_lifetime_is_the_measured_hour_only_on_deepseek() {
	let provider = IsoquantProvider::new();
	assert_eq!(
		provider.cache_lifetime("deepseek-v4.1-flash-1017-fast", false),
		Some(std::time::Duration::from_secs(3600))
	);
	assert_eq!(provider.cache_lifetime("glm-5.3-flash", false), None);
	assert_eq!(provider.cache_lifetime("model-released-tomorrow", false), None);
}

#[test]
fn pricing_uses_isoquant_rates() {
	let provider = IsoquantProvider::new();
	let deepseek = provider.get_model_pricing(DEEPSEEK).unwrap();
	assert_eq!(deepseek.input_price_per_1m, 0.035);
	assert_eq!(deepseek.output_price_per_1m, 0.37);
	assert_eq!(deepseek.cache_read_price_per_1m, 0.005);

	let glm = provider.get_model_pricing(GLM).unwrap();
	assert_eq!(glm.input_price_per_1m, 0.07);
	assert_eq!(glm.output_price_per_1m, 0.15);
	assert_eq!(glm.cache_read_price_per_1m, 0.025);
}

#[test]
fn served_models_have_reference_prices() {
	assert!(get_reference_pricing(DEEPSEEK).is_some());
	assert!(get_reference_pricing(GLM).is_some());
	// The catalog lists only GLM as ready on OpenRouter.
	assert!(OpenRouterProvider::new()
		.get_model_pricing("z-ai/glm-5.3-flash")
		.is_some());
}

#[tokio::test]
#[ignore = "reads the public catalog over the network"]
async fn live_catalog_lists_known_models_with_their_facts() {
	let models = fetch_catalog(&models_url(ISOQUANT_API_URL)).await.unwrap();

	for (id, input, output, cached, efforts) in KNOWN_MODELS {
		let model = find_model(&models, id).unwrap();
		let pricing = model.pricing.unwrap();
		assert_eq!(pricing.input_price_per_1m, *input, "{id}");
		assert_eq!(pricing.output_price_per_1m, *output, "{id}");
		assert_eq!(pricing.cache_read_price_per_1m, *cached, "{id}");
		assert_eq!(model.context_window, KNOWN_CONTEXT_WINDOW, "{id}");
		assert!(model.vision, "{id}");
		assert!(model.structured_output, "{id}");
		assert!(model.required_tool_choice, "{id}");
		assert_eq!(model.efforts, listed_efforts(efforts), "{id}");
	}
}
