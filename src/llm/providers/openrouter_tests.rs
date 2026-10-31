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
use crate::test_support::env::EnvGuard;

fn catalog_fixture() -> Vec<CatalogModel> {
	serde_json::from_value::<CatalogResponse>(serde_json::json!({
		"data": [
			{
				"id": "google/gemini-3.5-flash-lite",
				"reasoning": {
					"mandatory": true,
					"default_enabled": true,
					"supported_efforts": ["high", "medium", "low", "minimal"],
					"default_effort": "minimal"
				}
			},
			{
				"id": "google/gemini-3.1-flash-lite",
				"reasoning": {
					"mandatory": false,
					"default_enabled": true,
					"supported_efforts": ["high", "medium", "low", "minimal"],
					"default_effort": "minimal"
				}
			},
			{
				"id": "deepseek/deepseek-v4.1-flash",
				"reasoning": {
					"mandatory": false,
					"default_enabled": true,
					"supported_efforts": ["max", "high", "low"],
					"default_effort": "high"
				}
			},
			{
				"id": "example/no-reasoning",
				"reasoning": null
			},
			{
				"id": "liquid/lfm-2.5-24b-a2b",
				"reasoning": {
					"mandatory": true
				}
			},
			{
				"id": "example/optional-reasoning-without-levels",
				"reasoning": {
					"mandatory": false
				}
			},
			{
				"id": "example/malformed-reasoning",
				"reasoning": {
					"mandatory": "unknown",
					"supported_efforts": "high"
				}
			}
		]
	}))
	.unwrap()
	.data
}

#[test]
fn catalog_effort_none_uses_mandatory_minimum_or_disables_reasoning() {
	let catalog = catalog_fixture();
	let mandatory = catalog_model(&catalog, "google/gemini-3.5-flash-lite").unwrap();
	assert_eq!(
		select_effort(mandatory, ReasoningEffort::None),
		Some(ReasoningSelection::Effort("minimal"))
	);

	let optional = catalog_model(&catalog, "google/gemini-3.1-flash-lite").unwrap();
	assert_eq!(
		select_effort(optional, ReasoningEffort::None),
		Some(ReasoningSelection::Disabled)
	);
}

#[test]
fn catalog_without_effort_levels_omits_or_disables_reasoning() {
	let catalog = catalog_fixture();
	assert_eq!(catalog.len(), 7);
	assert!(catalog_model(&catalog, "example/malformed-reasoning")
		.unwrap()
		.reasoning
		.is_none());

	let mut mandatory = serde_json::json!({});
	apply_reasoning_effort(
		&mut mandatory,
		Some(&catalog),
		"liquid/lfm-2.5-24b-a2b",
		ReasoningEffort::None,
	);
	assert_eq!(mandatory, serde_json::json!({}));

	let mut optional = serde_json::json!({});
	apply_reasoning_effort(
		&mut optional,
		Some(&catalog),
		"example/optional-reasoning-without-levels",
		ReasoningEffort::None,
	);
	assert_eq!(optional, serde_json::json!({"reasoning": {"enabled": false}}));

	let mut requested = serde_json::json!({});
	apply_reasoning_effort(
		&mut requested,
		Some(&catalog),
		"liquid/lfm-2.5-24b-a2b",
		ReasoningEffort::XHigh,
	);
	assert_eq!(requested, serde_json::json!({"reasoning_effort": "high"}));
}

#[test]
fn catalog_effort_maps_to_the_nearest_supported_level_ties_up() {
	let catalog = catalog_fixture();
	// low/high/max: the middle request lands on the middle level, not the lowest.
	let deepseek = catalog_model(&catalog, "deepseek/deepseek-v4.1-flash").unwrap();
	assert_eq!(
		select_effort(deepseek, ReasoningEffort::Medium),
		Some(ReasoningSelection::Effort("high"))
	);
	assert_eq!(
		select_effort(deepseek, ReasoningEffort::XHigh),
		Some(ReasoningSelection::Effort("max"))
	);
	// minimal..high: nothing above "high", so the top requests stay at it.
	let gemini = catalog_model(&catalog, "google/gemini-3.5-flash-lite").unwrap();
	assert_eq!(
		select_effort(gemini, ReasoningEffort::Max),
		Some(ReasoningSelection::Effort("high"))
	);
}

#[test]
fn catalog_lookup_strips_openrouter_routing_variant() {
	let catalog = catalog_fixture();
	assert_eq!(
		catalog_reasoning_selection(&catalog, "google/gemini-3.5-flash-lite:nitro", ReasoningEffort::None),
		Some(ReasoningSelection::Effort("minimal"))
	);
}

#[test]
fn absent_catalog_model_keeps_legacy_reasoning_mapping() {
	let catalog = catalog_fixture();
	let mut disabled = serde_json::json!({});
	apply_reasoning_effort(&mut disabled, Some(&catalog), "unknown/model", ReasoningEffort::None);
	assert_eq!(disabled, serde_json::json!({"reasoning": {"enabled": false}}));

	let mut capped = serde_json::json!({});
	apply_reasoning_effort(&mut capped, Some(&catalog), "unknown/model", ReasoningEffort::Max);
	assert_eq!(capped, serde_json::json!({"reasoning_effort": "high"}));
}

#[test]
fn null_supported_efforts_accepts_every_gateway_level() {
	// `supported_efforts: null` means all gateway efforts are accepted; it must
	// not drop the reasoning entry and with it the `mandatory` flag.
	let catalog = serde_json::from_value::<CatalogResponse>(serde_json::json!({
		"data": [
			{
				"id": "example/mandatory-any-effort",
				"reasoning": {"mandatory": true, "supported_efforts": null}
			},
			{
				"id": "example/optional-any-effort",
				"reasoning": {"mandatory": false, "supported_efforts": null}
			}
		]
	}))
	.unwrap()
	.data;

	let mandatory = catalog_model(&catalog, "example/mandatory-any-effort").unwrap();
	assert_eq!(mandatory.reasoning.as_ref().unwrap().supported_efforts, EFFORT_LADDER);
	assert_eq!(
		select_effort(mandatory, ReasoningEffort::None),
		Some(ReasoningSelection::Effort("minimal"))
	);
	assert_eq!(
		select_effort(mandatory, ReasoningEffort::Max),
		Some(ReasoningSelection::Effort("max"))
	);

	let optional = catalog_model(&catalog, "example/optional-any-effort").unwrap();
	assert_eq!(
		select_effort(optional, ReasoningEffort::None),
		Some(ReasoningSelection::Disabled)
	);
	assert_eq!(
		select_effort(optional, ReasoningEffort::XHigh),
		Some(ReasoningSelection::Effort("xhigh"))
	);
}

#[test]
fn october_2026_models_are_priced_on_the_aggregator_route() {
	let provider = OpenRouterProvider::new();

	let terra = provider
		.get_model_pricing("openai/gpt-6.1-terra")
		.expect("gpt-6.1-terra must be priced");
	assert_eq!(terra.input_price_per_1m, 1.00);
	assert_eq!(terra.output_price_per_1m, 5.00);
	assert_eq!(terra.cache_write_price_per_1m, 1.25);
	assert_eq!(terra.cache_read_price_per_1m, 0.05);

	let omni_plus = provider
		.get_model_pricing("qwen/qwen3.5-omni-plus")
		.expect("qwen3.5-omni-plus must be priced");
	assert_eq!(omni_plus.input_price_per_1m, 1.40);
	assert_eq!(omni_plus.output_price_per_1m, 8.30);
}

#[test]
fn test_supports_model() {
	let provider = OpenRouterProvider::new();

	// OpenRouter supports many models
	assert!(provider.supports_model("anthropic/claude-3.5-sonnet"));
	assert!(provider.supports_model("openai/gpt-4o"));
	assert!(provider.supports_model("meta/llama-3.1-70b"));
	assert!(provider.supports_model("deepseek-chat"));

	// Should accept any non-empty model string as fallback
	assert!(provider.supports_model("any-model-name"));
}

#[test]
fn test_supports_model_case_insensitive() {
	let provider = OpenRouterProvider::new();

	// Test uppercase
	assert!(provider.supports_model("ANTHROPIC/CLAUDE-3.5-SONNET"));
	assert!(provider.supports_model("OPENAI/GPT-4O"));
	assert!(provider.supports_model("META/LLAMA-3.1-70B"));
	// Test mixed case
	assert!(provider.supports_model("Anthropic/Claude-3.5-Sonnet"));
	assert!(provider.supports_model("DEEPSEEK-CHAT"));
}

#[test]
fn test_supports_vision_case_insensitive() {
	let provider = OpenRouterProvider::new();

	// Test lowercase
	assert!(provider.supports_vision("gpt-4o"));
	assert!(provider.supports_vision("claude-3-haiku"));

	// Test uppercase
	assert!(provider.supports_vision("GPT-4O"));
	assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
	// Test mixed case
	assert!(provider.supports_vision("Gemini-1.5-Pro"));
}

#[test]
fn test_supports_caching_case_insensitive() {
	let provider = OpenRouterProvider::new();

	// Test lowercase
	assert!(provider.supports_caching("anthropic/claude-3.5-sonnet"));
	assert!(provider.supports_caching("claude-3-haiku"));

	// Test uppercase
	assert!(provider.supports_caching("ANTHROPIC/CLAUDE-3.5-SONNET"));
	assert!(provider.supports_caching("CLAUDE-3-HAIKU"));
}

#[test]
fn test_anthropic_fast_route_doubles_pricing() {
	let provider = OpenRouterProvider::new();
	let base = provider.get_model_pricing("anthropic/claude-opus-5").unwrap();
	let fast = provider.get_model_pricing("anthropic/claude-opus-5-fast").unwrap();
	assert_eq!(fast.input_price_per_1m, base.input_price_per_1m * 2.0);
	assert_eq!(fast.output_price_per_1m, base.output_price_per_1m * 2.0);
	assert_eq!(fast.cache_write_price_per_1m, base.cache_write_price_per_1m * 2.0);
	assert_eq!(fast.cache_read_price_per_1m, base.cache_read_price_per_1m * 2.0);
}

#[test]
fn dotted_claude_versions_are_priced_as_their_own_model() {
	use crate::llm::providers::AnthropicProvider;
	let provider = OpenRouterProvider::new();
	let anthropic = AnthropicProvider::new();
	for (dotted, dashed) in [
		("anthropic/claude-sonnet-5.5", "claude-sonnet-5-5"),
		("anthropic/claude-opus-5.5", "claude-opus-5-5"),
	] {
		let routed = provider.get_model_pricing(dotted).unwrap();
		let direct = anthropic.get_model_pricing(dashed).unwrap();
		assert_eq!(routed.input_price_per_1m, direct.input_price_per_1m, "{dotted}");
		assert_eq!(
			routed.cache_read_price_per_1m, direct.cache_read_price_per_1m,
			"{dotted}"
		);
	}
}

#[test]
fn test_models_without_a_maker_table_price_from_the_reference_table() {
	let provider = OpenRouterProvider::new();
	let flash = provider
		.get_model_pricing("xiaomi/mimo-v2.6-flash")
		.expect("MiMo is priced through the reference table");
	assert_eq!(flash.input_price_per_1m, 0.14);
	assert_eq!(flash.output_price_per_1m, 0.28);
	assert_eq!(flash.cache_read_price_per_1m, 0.0028);
	// Longest-match ordering: UltraSpeed is not priced as plain Pro.
	let ultraspeed = provider.get_model_pricing("xiaomi/mimo-v2.6-pro-ultraspeed").unwrap();
	assert_eq!(ultraspeed.input_price_per_1m, 4.35);
	assert!(provider.get_model_pricing("nobody/unknown-model-x").is_none());
}

#[test]
fn schema_enforcement_follows_live_verified_routes() {
	let provider = OpenRouterProvider::new();
	for model in [
		"openai/gpt-6-luna",
		"anthropic/claude-haiku-5.5",
		"xiaomi/mimo-v2.6-flash",
		"deepseek/deepseek-v4.1-flash",
		"qwen/qwen3.7-plus",
	] {
		assert!(provider.enforces_response_schema(model), "{model}");
	}
	// Advertised structured_outputs that did not hold the shape, and unknowns.
	for model in [
		"inception/mercury-3",
		"qwen/qwen3.8-omni-flash",
		"thinkingmachines/inkling-small",
		"unknown/provider-model",
	] {
		assert!(!provider.enforces_response_schema(model), "{model}");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn schema_requests_omit_unsupported_sampling_and_keep_strict_routing() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![
		(
			200,
			serde_json::json!({"data": [
				{"id": "openai/gpt-6-luna", "supported_parameters": ["max_tokens", "response_format", "structured_outputs"]},
				{"id": "example/sampling", "supported_parameters": ["temperature", "top_p", "top_k", "repetition_penalty"]}
			]})
			.to_string(),
		),
		(
			200,
			crate::test_support::mock_http::chat_completion_body(r#"{"ok":true}"#),
		),
	]);
	let _env = EnvGuard::new().set("OPENROUTER_API_URL", mock.url.clone());

	let mut params =
		ChatCompletionParams::new(&[], "openai/gpt-6-luna", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	params.response_format = Some(crate::llm::types::StructuredOutputRequest {
		format: crate::llm::types::OutputFormat::JsonSchema,
		mode: crate::llm::types::ResponseMode::Strict,
		schema: Some(
			serde_json::json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}),
		),
	});
	OpenRouterProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");

	let requests = mock.requests();
	let chat_request = requests.last().expect("chat request recorded");
	let req_body: serde_json::Value = serde_json::from_str(&chat_request.body).expect("request json");
	assert_eq!(req_body["provider"]["require_parameters"], true);
	assert_eq!(req_body["response_format"]["json_schema"]["strict"], true);
	for parameter in ["temperature", "top_p", "top_k", "repetition_penalty"] {
		assert!(
			req_body.get(parameter).is_none(),
			"unsupported {parameter} prevents strict routing"
		);
	}
}

fn strict_schema_params(model: &str) -> ChatCompletionParams {
	let mut params = ChatCompletionParams::new(&[], model, 0.25, 0.5, 17, 10);
	params.response_format = Some(crate::llm::types::StructuredOutputRequest {
		format: crate::llm::types::OutputFormat::JsonSchema,
		mode: crate::llm::types::ResponseMode::Strict,
		schema: Some(serde_json::json!({
			"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]
		})),
	});
	params.tools = Some(vec![crate::llm::types::FunctionDefinition {
		name: "verify".to_string(),
		description: "Read verification evidence".to_string(),
		parameters: serde_json::json!({"type": "object", "properties": {}}),
		cache_control: None,
	}]);
	params.reasoning_effort = Some(ReasoningEffort::Low);
	params
}

fn assert_strict_requirements(body: &serde_json::Value) {
	assert_eq!(body["provider"]["require_parameters"], true);
	assert_eq!(body["response_format"]["type"], "json_schema");
	assert_eq!(body["response_format"]["json_schema"]["strict"], true);
	assert_eq!(
		body["response_format"]["json_schema"]["schema"],
		serde_json::json!({
			"type": "object", "properties": {"ok": {"type": "boolean"}},
			"required": ["ok"], "additionalProperties": false
		})
	);
	assert_eq!(body["tools"][0]["function"]["name"], "verify");
	assert_eq!(body["max_tokens"], 10);
}

fn assert_optional_hints_omitted(body: &serde_json::Value) {
	for parameter in [
		"temperature",
		"top_p",
		"top_k",
		"repetition_penalty",
		"reasoning",
		"reasoning_effort",
		"tool_choice",
		"parallel_tool_calls",
	] {
		assert!(
			body.get(parameter).is_none(),
			"unexpected routing constraint: {parameter}"
		);
	}
	assert_strict_requirements(body);
}

#[test]
fn strict_schema_without_catalog_omits_optional_hints() {
	for effort in [ReasoningEffort::Low, ReasoningEffort::None] {
		let mut params = strict_schema_params("example/model");
		params.reasoning_effort = Some(effort);
		assert_optional_hints_omitted(&build_request_body(&params, None).unwrap());
	}
}

#[tokio::test]
async fn strict_schema_after_catalog_fetch_failure_omits_optional_hints() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(503, "unavailable".to_owned())]);
	let catalog = fetch_catalog("mock-key", &mock.url, None).await;
	assert!(catalog.is_err());
	let body = build_request_body(&strict_schema_params("example/model"), catalog.as_deref().ok()).unwrap();
	assert_optional_hints_omitted(&body);
}

#[test]
fn strict_schema_with_model_absent_from_catalog_omits_optional_hints() {
	let catalog = catalog_fixture();
	let body = build_request_body(&strict_schema_params("example/unlisted"), Some(&catalog)).unwrap();
	assert_optional_hints_omitted(&body);
}

#[test]
fn strict_schema_with_unknown_or_unsupported_metadata_omits_optional_hints() {
	for metadata in [
		serde_json::json!({}),
		serde_json::json!({"supported_parameters": null}),
		serde_json::json!({"supported_parameters": []}),
	] {
		let mut entry = metadata;
		entry["id"] = serde_json::json!("example/model");
		let catalog = vec![serde_json::from_value::<CatalogModel>(entry).unwrap()];
		let body = build_request_body(&strict_schema_params("example/model"), Some(&catalog)).unwrap();
		assert_optional_hints_omitted(&body);
	}
}

#[test]
fn strict_schema_retains_only_supported_optional_parameters() {
	let optional = [
		"temperature",
		"top_p",
		"top_k",
		"repetition_penalty",
		"reasoning",
		"reasoning_effort",
		"tool_choice",
		"parallel_tool_calls",
	];
	// Exercise every hint independently so one advertised field cannot enable another.
	for supported in optional {
		let catalog = vec![serde_json::from_value::<CatalogModel>(serde_json::json!({
			"id": "example/model", "supported_parameters": [supported]
		}))
		.unwrap()];
		let mut params = strict_schema_params("EXAMPLE/MODEL:nitro");
		if supported == "reasoning" {
			params.reasoning_effort = Some(ReasoningEffort::None);
		}
		let body = build_request_body(&params, Some(&catalog)).unwrap();
		assert_strict_requirements(&body);
		for parameter in optional {
			assert_eq!(
				body.get(parameter).is_some(),
				parameter == supported,
				"{supported}: {parameter}"
			);
		}
	}
	let catalog = vec![serde_json::from_value::<CatalogModel>(serde_json::json!({
		"id": "example/model", "supported_parameters": optional
	}))
	.unwrap()];
	let body = build_request_body(&strict_schema_params("example/model"), Some(&catalog)).unwrap();
	assert_strict_requirements(&body);
	assert_eq!(body["temperature"], 0.25);
	assert_eq!(body["top_p"], 0.5);
	assert_eq!(body["top_k"], 17);
	assert_eq!(body["repetition_penalty"], DEFAULT_REPETITION_PENALTY);
	assert_eq!(body["reasoning_effort"], "low");
	assert_eq!(body["tool_choice"], "auto");
	assert_eq!(body["parallel_tool_calls"], true);
}

#[test]
fn ordinary_requests_keep_reasoning_and_tool_hints_without_strict_routing() {
	let mut params = strict_schema_params("example/model");
	params.response_format = None;
	let body = build_request_body(&params, None).unwrap();
	assert!(body["provider"].get("require_parameters").is_none());
	assert_eq!(body["reasoning_effort"], "low");
	assert_eq!(body["tool_choice"], "auto");
	assert_eq!(body["parallel_tool_calls"], true);
}

#[test]
fn cache_prewarm_requests_one_token_and_normal_requests_keep_their_cap() {
	let mut params = strict_schema_params("example/model");
	params.response_format = None;
	assert_eq!(build_request_body(&params, None).unwrap()["max_tokens"], 10);

	let prewarm = params.with_cache_prewarm(true);
	assert_eq!(build_request_body(&prewarm, None).unwrap()["max_tokens"], 1);
}

#[test]
fn catalog_parameter_support_distinguishes_missing_and_empty_lists() {
	let catalog = serde_json::from_value::<CatalogResponse>(serde_json::json!({"data": [
		{"id": "example/none", "supported_parameters": []},
		{"id": "example/missing"},
		{"id": "example/null", "supported_parameters": null},
		{"id": "example/partial", "supported_parameters": ["temperature"]}
	]}))
	.unwrap()
	.data;
	assert_eq!(
		supports_parameter(Some(&catalog), "example/none", "temperature"),
		Some(false)
	);
	assert_eq!(
		supports_parameter(Some(&catalog), "example/missing", "temperature"),
		None
	);
	assert_eq!(supports_parameter(Some(&catalog), "example/null", "temperature"), None);
	assert_eq!(
		supports_parameter(Some(&catalog), "example/unknown", "temperature"),
		None
	);
	assert_eq!(supports_parameter(None, "example/partial", "temperature"), None);
	assert_eq!(
		supports_parameter(Some(&catalog), "EXAMPLE/PARTIAL:nitro", "temperature"),
		Some(true)
	);
	assert_eq!(
		supports_parameter(Some(&catalog), "example/partial", "top_k"),
		Some(false)
	);
}

#[test]
fn tool_images_follow_tool_results_without_losing_call_association() {
	let messages = [shared::tests::tool_image_message("")];
	let value = serde_json::to_value(convert_messages(&messages).unwrap()).unwrap();
	assert_eq!(value[0]["role"], "tool");
	assert_eq!(value[0]["tool_call_id"], "call_image");
	assert_eq!(value[0]["content"], "");
	assert_eq!(value[1]["role"], "user");
	assert_eq!(
		value[1]["content"][1]["image_url"]["url"],
		"data:image/png;base64,aW1hZ2U="
	);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_parses_text_and_usage() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("mocked openrouter answer"),
	)]);
	let _env = EnvGuard::new().set("OPENROUTER_API_URL", mock.url.clone());

	let provider = OpenRouterProvider::new();
	let params = ChatCompletionParams::new(&[], "openai/gpt-4o", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "mocked openrouter answer");
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);
	assert!(usage.total_tokens >= 19);

	// The last request is the chat call (an earlier one may be the catalog fetch).
	let requests = mock.requests();
	let chat_request = requests.last().expect("chat request recorded");
	let body: serde_json::Value = serde_json::from_str(&chat_request.body).expect("request json");
	assert_eq!(body["model"], "openai/gpt-4o");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_handles_tools_thinking_and_cache() {
	let body = r#"{"id":"mock-or","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_details":[{"type":"text","text":"step one"},{"type":"text","text":"step two"}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}},{"id":"call_2","type":"custom","function":{"name":"dropped","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":40,"total_tokens":140,"prompt_tokens_details":{"cached_tokens":30},"completion_tokens_details":{"reasoning_tokens":10},"cost":0.0125}}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("OPENROUTER_API_URL", mock.url.clone());

	let provider = OpenRouterProvider::new();
	let params = ChatCompletionParams::new(&[], "openai/gpt-4o", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	// Only "function" tool calls survive; other types are dropped.
	let calls = response.tool_calls.expect("tool calls present");
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].id, "call_1");
	assert_eq!(calls[0].name, "get_weather");
	assert_eq!(calls[0].arguments["city"], "Oslo");

	// reasoning_details texts are joined into the thinking block.
	let thinking = response.thinking.expect("thinking present");
	assert_eq!(thinking.content, "step one\n\nstep two");

	// input excludes cache reads; output excludes reasoning; cost comes from the API.
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 70);
	assert_eq!(usage.cache_read_tokens, 30);
	assert_eq!(usage.output_tokens, 30);
	assert_eq!(usage.reasoning_tokens, 10);
	assert_eq!(usage.cost, Some(0.0125));
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	let _env = EnvGuard::new().set("OPENROUTER_API_URL", mock.url.clone());

	let provider = OpenRouterProvider::new();
	let params = ChatCompletionParams::new(&[], "openai/gpt-4o", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(
		err.to_string().contains("OpenRouter API error"),
		"unexpected error: {err}"
	);
}
