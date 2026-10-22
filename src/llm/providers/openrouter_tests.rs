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
fn test_schema_enforcement_proxy_policy() {
	let provider = OpenRouterProvider::new();
	assert!(provider.enforces_response_schema("deepseek-v4-pro"));
	assert!(provider.enforces_response_schema("openai/gpt-4o"));
	assert!(provider.enforces_response_schema("unknown/provider-model"));
	assert!(!provider.enforces_response_schema("mistral-7b"));
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
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(
		200,
		crate::llm::providers::mock_http::chat_completion_body("mocked openrouter answer"),
	)]);
	std::env::set_var("OPENROUTER_API_URL", mock.url.clone());

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

	std::env::remove_var("OPENROUTER_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_handles_tools_thinking_and_cache() {
	let body = r#"{"id":"mock-or","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_details":[{"type":"text","text":"step one"},{"type":"text","text":"step two"}],"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Oslo\"}"}},{"id":"call_2","type":"custom","function":{"name":"dropped","arguments":"{}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":100,"completion_tokens":40,"total_tokens":140,"prompt_tokens_details":{"cached_tokens":30},"completion_tokens_details":{"reasoning_tokens":10},"cost":0.0125}}"#;
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	std::env::set_var("OPENROUTER_API_URL", mock.url.clone());

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

	std::env::remove_var("OPENROUTER_API_URL");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = crate::llm::providers::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	std::env::set_var("OPENROUTER_API_URL", mock.url.clone());

	let provider = OpenRouterProvider::new();
	let params = ChatCompletionParams::new(&[], "openai/gpt-4o", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(
		err.to_string().contains("OpenRouter API error"),
		"unexpected error: {err}"
	);

	std::env::remove_var("OPENROUTER_API_URL");
}
