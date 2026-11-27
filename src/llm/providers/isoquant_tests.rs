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
use crate::llm::types::{FunctionDefinition, ImageAttachment, ImageData, Message, SourceType};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::{self, MockHttp};
use serde_json::{json, Value};

const DEEPSEEK: &str = "deepseek-v4.1-flash-1017-fast";
const GLM: &str = "glm-5.3-flash";

/// The catalog's shape, as fetched 2026-11-27: an `apis` block per protocol.
fn catalog_fixture() -> Vec<CatalogEntry> {
	let response: CatalogResponse = serde_json::from_value(json!({
		"object": "list",
		"data": [
			{
				"id": "chat-only",
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
			{
				"id": "dual-model",
				"isoquant": {"apis": {
					"chat_completions": {"input_modalities": ["text"], "tool_choice": ["auto", "required"]},
					"messages": {
						"input_modalities": ["text", "image"],
						"structured_outputs": ["json_schema"],
						"tool_choice": ["auto", "none", "any", "tool"]
					}
				}, "context_length": 4096}
			},
			{"id": "no-protocol", "isoquant": {"apis": {}, "context_length": 10}},
			{"id": "no-metadata"}
		]
	}))
	.unwrap();
	response.data
}

fn image(data: ImageData) -> ImageAttachment {
	ImageAttachment {
		data,
		media_type: "image/png".to_owned(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}
}

/// A complete Messages response: a text answer and a thinking block, billed as
/// 100 input and 50 output tokens.
fn message_body() -> String {
	json!({
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": DEEPSEEK,
		"content": [
			{"type": "thinking", "thinking": "let me think"},
			{"type": "text", "text": "ok"}
		],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 100, "output_tokens": 50}
	})
	.to_string()
}

/// Messages SSE events framed on the wire, as the accumulator reads them.
fn sse(events: &[Value]) -> String {
	events
		.iter()
		.map(|event| {
			format!(
				"event: {}\ndata: {event}\n\n",
				event["type"].as_str().unwrap_or_default()
			)
		})
		.collect()
}

/// The single completion POST. The catalog fetch may precede it and repeats the
/// canned reply, which is not a catalog, so `KNOWN_MODELS` stays in charge.
fn completion_request(mock: &MockHttp) -> mock_http::RecordedRequest {
	mock.requests()
		.into_iter()
		.find(|request| request.method == "POST")
		.expect("completion request")
}

fn body_of(request: &mock_http::RecordedRequest) -> Value {
	serde_json::from_str(&request.body).expect("request json")
}

fn messages_url(mock: &MockHttp) -> String {
	format!("{}/v1/messages", mock.base_url())
}

/// Paths of the POSTs received so far.
fn posted_paths(mock: &MockHttp) -> Vec<String> {
	mock.requests()
		.into_iter()
		.filter(|request| request.method == "POST")
		.map(|request| request.path)
		.collect()
}

#[test]
fn catalog_keeps_served_models_with_what_each_protocol_takes() {
	let models = parse_catalog(catalog_fixture());
	let ids: Vec<&str> = models.iter().map(|model| model.id.as_str()).collect();
	assert_eq!(ids, ["chat-only", "plain-model", "messages-only", "dual-model"]);

	let chat_only = &models[0];
	assert_eq!(chat_only.context_window, 262_144);
	assert_eq!(chat_only.messages, None);
	assert_eq!(
		chat_only.chat,
		Some(IsoquantApi {
			vision: true,
			structured_output: true,
			required_tool_choice: true
		})
	);
	let pricing = chat_only.pricing.unwrap();
	assert_eq!(pricing.input_price_per_1m, 0.035);
	assert_eq!(pricing.output_price_per_1m, 0.37);
	assert_eq!(pricing.cache_write_price_per_1m, 0.035);
	assert_eq!(pricing.cache_read_price_per_1m, 0.005);
	let levels: Vec<&str> = chat_only.efforts.iter().map(|(_, name)| *name).collect();
	assert_eq!(levels, ["none", "low", "high"]);

	let nothing = IsoquantApi {
		vision: false,
		structured_output: false,
		required_tool_choice: false,
	};
	let plain = &models[1];
	assert_eq!(plain.chat, Some(nothing));
	assert!(plain.pricing.is_none());
	assert!(plain.efforts.is_empty());

	let messages_only = &models[2];
	assert_eq!(messages_only.chat, None);
	assert_eq!(messages_only.messages, Some(nothing));

	// One model can differ per protocol; each is read on its own.
	let dual = &models[3];
	assert_eq!(
		dual.chat,
		Some(IsoquantApi {
			vision: false,
			structured_output: false,
			required_tool_choice: true
		})
	);
	assert_eq!(
		dual.messages,
		Some(IsoquantApi {
			vision: true,
			structured_output: true,
			// The catalog lists auto, none, any and tool for Messages.
			required_tool_choice: false
		})
	);
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
		(Max, "max", "max"),
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
fn provider_headers_cannot_be_overridden_or_duplicated() {
	let mut params = ChatCompletionParams::new(&[], GLM, 64).with_extra_headers(
		[
			("isoquant-zdr".to_string(), "off".to_string()),
			("Authorization".to_string(), "Bearer caller".to_string()),
			("X-Api-Key".to_string(), "caller".to_string()),
			("Anthropic-Version".to_string(), "2020-01-01".to_string()),
			("x-kept".to_string(), "yes".to_string()),
		]
		.into(),
	);

	protect_request_headers(&mut params, ApiMode::Anthropic);

	let headers = params.extra_headers.unwrap();
	assert_eq!(headers.len(), 2);
	assert_eq!(headers.get("Isoquant-ZDR").map(String::as_str), Some("required"));
	assert_eq!(headers.get("x-kept").map(String::as_str), Some("yes"));

	// Chat Completions sets no protocol version, so a caller's is not touched there.
	let mut params = ChatCompletionParams::new(&[], GLM, 64).with_extra_headers(
		[
			("x-api-key".to_string(), "caller".to_string()),
			("authorization".to_string(), "Bearer caller".to_string()),
			("anthropic-version".to_string(), "2020-01-01".to_string()),
		]
		.into(),
	);
	protect_request_headers(&mut params, ApiMode::OpenAi);
	let headers = params.extra_headers.unwrap();
	assert_eq!(headers.len(), 2);
	assert_eq!(headers.get("Isoquant-ZDR").map(String::as_str), Some("required"));
	assert_eq!(headers.get("anthropic-version").map(String::as_str), Some("2020-01-01"));
}

#[test]
fn catalog_url_sits_beside_either_endpoint() {
	assert_eq!(models_url(CHAT_COMPLETIONS_URL), "https://api.isoquant.ai/v1/models");
	assert_eq!(models_url(MESSAGES_URL), "https://api.isoquant.ai/v1/models");
	assert_eq!(
		models_url("http://localhost:8080/v1/messages"),
		"http://localhost:8080/v1/models"
	);
	assert_eq!(
		models_url("http://localhost:8080/v1"),
		"http://localhost:8080/v1/models"
	);
	// Only the path counts: trailing slashes are trimmed and the query, with
	// whatever credentials it carries, stays off the catalog request.
	assert_eq!(
		models_url("http://localhost:8080/v1/messages/?api_key=secret"),
		"http://localhost:8080/v1/models"
	);
	assert_eq!(
		models_url("http://localhost:8080/v1/chat/completions/?key=secret#frag"),
		"http://localhost:8080/v1/models"
	);
	assert_eq!(models_url("http://localhost:8080/"), "http://localhost:8080/models");
	assert_eq!(
		models_url("http://localhost:8080/custom?x=1"),
		"http://localhost:8080/custom/models"
	);
}

#[test]
fn endpoint_follows_the_mode_then_the_url() {
	let messages = "http://localhost:8080/v1/messages?beta=true";
	let completions = "http://localhost:8080/v1/chat/completions/";
	let custom = "http://localhost:8080/inference";

	for (mode, api_url, expected_mode, expected_url) in [
		(None, None, ApiMode::Anthropic, MESSAGES_URL),
		(Some(ApiMode::OpenAi), None, ApiMode::OpenAi, CHAT_COMPLETIONS_URL),
		(Some(ApiMode::Anthropic), None, ApiMode::Anthropic, MESSAGES_URL),
		(None, Some(messages), ApiMode::Anthropic, messages),
		(None, Some(completions), ApiMode::OpenAi, completions),
		// A custom URL naming neither endpoint keeps the protocol it always had.
		(None, Some(custom), ApiMode::OpenAi, custom),
		// The mode overrides what the URL names; the URL is sent as written.
		(
			Some(ApiMode::Anthropic),
			Some(completions),
			ApiMode::Anthropic,
			completions,
		),
		(Some(ApiMode::OpenAi), Some(messages), ApiMode::OpenAi, messages),
	] {
		assert_eq!(
			select_endpoint(mode, api_url, DEEPSEEK),
			(expected_mode, expected_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
}

#[test]
fn messages_never_claims_a_required_tool_choice() {
	let models = parse_catalog(catalog_fixture());
	let dual = &models[3];
	let listed = dual.chat.expect("chat api");
	let unlisted = dual.messages.expect("messages api");
	assert!(takes_required_tool_choice(listed, ApiMode::OpenAi));
	assert!(!takes_required_tool_choice(listed, ApiMode::Anthropic));
	// A catalog entry listing `required` for Messages is not honored by the transport.
	let claiming = IsoquantApi {
		required_tool_choice: true,
		..unlisted
	};
	assert!(takes_required_tool_choice(claiming, ApiMode::OpenAi));
	assert!(!takes_required_tool_choice(claiming, ApiMode::Anthropic));
	assert!(!takes_required_tool_choice(unlisted, ApiMode::Anthropic));
}

#[tokio::test]
#[serial_test::serial]
async fn an_invalid_mode_makes_no_request() {
	let mock = MockHttp::start(vec![]);
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "claude")
		.set("ISOQUANT_API_URL", messages_url(&mock));
	let refused = "Configuration error: ISOQUANT_API_MODE must be `anthropic` or `openai`, got `claude`";

	let params = ChatCompletionParams::new(&[], DEEPSEEK, 10).with_api_key("mock-key".to_owned());
	let error = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect_err("refused");
	assert_eq!(error.to_string(), refused);

	let error = preload().await.expect_err("refused");
	assert_eq!(error.to_string(), refused);

	assert!(
		mock.requests().is_empty(),
		"neither the catalog nor a completion was requested: {:?}",
		mock.requests()
	);
}

#[tokio::test]
#[serial_test::serial]
async fn a_completion_url_goes_out_as_written_and_discovery_takes_only_its_path() {
	let mock = MockHttp::start(vec![(200, message_body())]);
	// A trailing slash and a query: the completion keeps both, the catalog is
	// read beside the path alone, so the query's credential is not forwarded.
	let _env = EnvGuard::new().set(
		"ISOQUANT_API_URL",
		format!("{}/v1/messages/?api_key=secret", mock.base_url()),
	);
	let params = ChatCompletionParams::new(&[], DEEPSEEK, 4096).with_api_key("mock-key".to_owned());
	IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	let request = completion_request(&mock);
	assert_eq!(request.path, "/v1/messages/?api_key=secret");
	assert_eq!(request.header("x-api-key"), Some("mock-key"));
	let catalog_paths: Vec<String> = mock
		.requests()
		.into_iter()
		.filter(|request| request.method == "GET")
		.map(|request| request.path)
		.collect();
	assert_eq!(catalog_paths, ["/v1/models"]);
}

#[test]
fn an_unpinned_standard_endpoint_prefers_messages_unless_the_catalog_serves_chat_only() {
	let models = parse_catalog(catalog_fixture());
	assert_eq!(preferred_mode(Some(&models[0])), ApiMode::OpenAi);
	assert_eq!(preferred_mode(Some(&models[2])), ApiMode::Anthropic);
	assert_eq!(preferred_mode(Some(&models[3])), ApiMode::Anthropic);
	// A model nobody knows yet is not assumed to be Chat-only.
	assert_eq!(preferred_mode(None), ApiMode::Anthropic);

	for model in [DEEPSEEK, GLM] {
		assert_eq!(
			select_endpoint(None, None, model),
			(ApiMode::Anthropic, MESSAGES_URL.to_owned()),
			"{model}"
		);
	}
}

#[test]
#[serial_test::serial]
fn known_models_report_their_facts_per_route() {
	let provider = IsoquantProvider::new();
	assert_eq!(provider.name(), "isoquant");
	assert!(!provider.supports_model(""));
	assert!(provider.supports_model("model-released-tomorrow"));

	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "openai");
	for model in [DEEPSEEK, GLM] {
		assert!(provider.supports_vision(model), "{model}");
		assert!(!provider.supports_video(model), "{model}");
		assert!(provider.supports_caching(model), "{model}");
		assert!(provider.supports_structured_output(model), "{model}");
		assert!(provider.enforces_response_schema(model), "{model}");
		assert!(provider.supports_required_tool_choice(model), "{model}");
		assert_eq!(provider.get_max_input_tokens(model), 1_048_576, "{model}");
	}

	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "anthropic");
	for model in [DEEPSEEK, GLM] {
		assert!(provider.supports_vision(model), "{model}");
		assert!(!provider.supports_video(model), "{model}");
		assert!(!provider.supports_caching(model), "{model}");
		assert!(provider.supports_structured_output(model), "{model}");
		// Proven on Chat Completions only, and Messages lists no required choice.
		assert!(!provider.enforces_response_schema(model), "{model}");
		assert!(!provider.supports_required_tool_choice(model), "{model}");
		assert_eq!(provider.get_max_input_tokens(model), 1_048_576, "{model}");
	}
}

#[test]
#[serial_test::serial]
fn messages_takes_nothing_from_a_model_name() {
	// Priced and described for other hosts, absent from Isoquant's catalog.
	let model = "glm-5.3-flash-fast";
	assert!(get_reference_pricing(model).is_some());
	assert!(get_reference_capabilities(model).is_some_and(|capabilities| capabilities.vision));

	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "anthropic");
	let provider = IsoquantProvider::new();
	assert!(!provider.supports_vision(model));
	assert!(!provider.supports_structured_output(model));
	assert!(!provider.supports_required_tool_choice(model));
	assert!(!provider.enforces_response_schema(model));
	// The context window and the prices are the model's, not the protocol's.
	assert!(provider.get_max_input_tokens(model) > 0);
	assert!(provider.get_model_pricing(model).is_some());

	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "openai");
	assert!(provider.supports_vision(model));
}

#[test]
fn unprobed_models_do_not_claim_schema_enforcement() {
	let provider = IsoquantProvider::new();
	assert!(!provider.enforces_response_schema("model-released-tomorrow"));
	assert!(!provider.supports_required_tool_choice("model-released-tomorrow"));
}

#[test]
#[serial_test::serial]
fn cache_lifetime_is_the_measured_hour_only_on_deepseek_on_chat_completions() {
	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "openai");
	let provider = IsoquantProvider::new();
	assert_eq!(
		provider.cache_lifetime(DEEPSEEK, false),
		Some(std::time::Duration::from_secs(3600))
	);
	assert_eq!(provider.cache_lifetime(GLM, false), None);
	assert_eq!(provider.cache_lifetime("model-released-tomorrow", false), None);

	let _env = EnvGuard::new().set("ISOQUANT_API_MODE", "anthropic");
	assert_eq!(provider.cache_lifetime(DEEPSEEK, false), None);
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
	let models = fetch_catalog(&models_url(CHAT_COMPLETIONS_URL)).await.unwrap();

	for (id, input, output, cached, efforts) in KNOWN_MODELS {
		let model = find_model(&models, id).unwrap();
		let pricing = model.pricing.unwrap();
		assert_eq!(pricing.input_price_per_1m, *input, "{id}");
		assert_eq!(pricing.output_price_per_1m, *output, "{id}");
		assert_eq!(pricing.cache_read_price_per_1m, *cached, "{id}");
		assert_eq!(model.context_window, KNOWN_CONTEXT_WINDOW, "{id}");
		assert_eq!(model.chat, Some(KNOWN_CHAT_API), "{id}");
		assert_eq!(model.messages, Some(KNOWN_MESSAGES_API), "{id}");
		assert_eq!(model.efforts, listed_efforts(efforts), "{id}");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn messages_request_carries_x_api_key_zdr_effort_and_model() {
	let mock = MockHttp::start(vec![(200, message_body())]);
	// The mode overrides the URL's own path, and the URL is used as written.
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "anthropic")
		.set("ISOQUANT_API_URL", mock.url.clone());
	let params = ChatCompletionParams::new(&[Message::user("hi")], DEEPSEEK, 4096)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::None)
		.with_extra_headers(
			[
				("x-api-key".to_string(), "caller".to_string()),
				("authorization".to_string(), "Bearer caller".to_string()),
				("isoquant-zdr".to_string(), "off".to_string()),
			]
			.into(),
		);

	let response = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	let request = completion_request(&mock);
	assert_eq!(request.path, "/v1/chat/completions", "the configured URL, as written");
	// The key travels in x-api-key only; the caller's values are replaced.
	assert_eq!(request.header("x-api-key"), Some("mock-key"));
	assert_eq!(request.header("authorization"), None);
	assert_eq!(request.header("isoquant-zdr"), Some("required"));
	assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
	assert_eq!(
		request.headers.iter().filter(|(name, _)| name == "x-api-key").count(),
		1
	);

	let body = body_of(&request);
	assert_eq!(body["model"], DEEPSEEK);
	assert_eq!(body["max_tokens"], 4096);
	assert_eq!(body["output_config"]["effort"], "none");
	assert_eq!(
		body["messages"],
		json!([{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }])
	);
	assert!(body.get("reasoning_effort").is_none(), "no Chat Completions knobs");
	assert_eq!(body["stream"], true);

	assert_eq!(response.content, "ok");
	assert_eq!(response.id.as_deref(), Some("msg_1"));
	assert_eq!(response.finish_reason.as_deref(), Some("end_turn"));
	assert_eq!(response.exchange.provider, "isoquant");
	// Thinking is carved out of the output counter and billed inside it.
	assert_eq!(response.thinking.expect("thinking").content, "let me think");
	let usage = response.exchange.usage.expect("usage");
	// Messages reports no thinking counter here: its size is estimated from the
	// emitted text and carved out of the output counter.
	assert_eq!(
		(usage.input_tokens, usage.output_tokens, usage.reasoning_tokens),
		(100, 47, 3)
	);
	assert_eq!(usage.total_tokens, 150);
	let cost = provider_cost(DEEPSEEK, 100, 0, 0, 50);
	assert_eq!(usage.cost, Some(cost));
	assert!((cost - 2.2e-5).abs() < 1e-12, "{cost}");

	assert_eq!(
		mock.requests()
			.iter()
			.filter(|request| request.method == "POST")
			.count(),
		1,
		"one request, no second protocol"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn messages_stream_assembles_thinking_tools_and_images() {
	let events = [
		json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":20}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}),
		json!({"type":"content_block_stop","index":1}),
		json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}),
		json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}),
		json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"x\"}"}}),
		json!({"type":"content_block_stop","index":2}),
		// Usage is a snapshot: the final counter replaces the start one.
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":50}}),
		json!({"type":"message_stop"}),
	];
	let mock = MockHttp::start(vec![(200, sse(&events))]);
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "anthropic")
		.set("ISOQUANT_API_URL", messages_url(&mock));

	let picture = Message::user("what is this?").with_images(vec![image(ImageData::Base64("aW1hZ2U=".to_owned()))]);
	let params = ChatCompletionParams::new(&[Message::system("be brief"), picture], GLM, 0)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::XHigh)
		.with_tools(vec![FunctionDefinition {
			name: "lookup".to_owned(),
			description: String::new(),
			parameters: json!({ "type": "object" }),
			cache_control: None,
			freeform: None,
		}]);

	let response = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect("messages stream");

	let body = body_of(&completion_request(&mock));
	assert_eq!(body["model"], GLM);
	assert_eq!(body["system"], "be brief");
	// max_tokens is required on Messages; 0 stands in for the endpoint's default.
	assert_eq!(body["max_tokens"], 16_384);
	// XHigh is not listed for GLM: the highest listed level not above it goes.
	assert_eq!(body["output_config"]["effort"], "high");
	assert_eq!(body["tools"][0]["name"], "lookup");
	assert_eq!(body["tools"][0]["input_schema"], json!({ "type": "object" }));
	assert_eq!(
		body["messages"][0]["content"][1],
		json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=" } })
	);

	assert_eq!(response.content, "ok");
	assert_eq!(response.thinking.expect("thinking").content, "think");
	let call = &response.tool_calls.expect("tool calls")[0];
	assert_eq!(
		(call.id.as_str(), call.name.as_str(), &call.arguments),
		("toolu_1", "lookup", &json!({ "q": "x" }))
	);
	assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(
			usage.input_tokens,
			usage.cache_read_tokens,
			usage.output_tokens,
			usage.reasoning_tokens
		),
		(100, 20, 49, 1)
	);
	// Cache reads bill at the cached rate, thinking at the output rate.
	assert_eq!(usage.cost, Some(provider_cost(GLM, 100, 0, 20, 50)));
}

#[tokio::test]
#[serial_test::serial]
async fn messages_refuses_what_the_model_is_not_known_to_take() {
	let mock = MockHttp::start(vec![(200, message_body())]);
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "anthropic")
		.set("ISOQUANT_API_URL", messages_url(&mock));
	let provider = IsoquantProvider::new();

	// Messages documents an image as inline data only.
	let url =
		Message::user("what is this?").with_images(vec![image(ImageData::Url("https://example.com/a.png".to_owned()))]);
	let error = provider
		.chat_completion(ChatCompletionParams::new(&[url], DEEPSEEK, 64).with_api_key("k".to_owned()))
		.await
		.expect_err("URL image refused");
	assert!(
		error
			.to_string()
			.contains("isoquant model deepseek-v4.1-flash-1017-fast does not accept URL images"),
		"unexpected error: {error}"
	);

	// Nothing is known about a model outside the catalog: no image is sent.
	let base64 = Message::user("what is this?").with_images(vec![image(ImageData::Base64("aW1hZ2U=".to_owned()))]);
	let error = provider
		.chat_completion(
			ChatCompletionParams::new(&[base64], "model-released-tomorrow", 64).with_api_key("k".to_owned()),
		)
		.await
		.expect_err("image on an unknown model refused");
	assert!(
		error
			.to_string()
			.contains("isoquant model model-released-tomorrow does not accept image input"),
		"unexpected error: {error}"
	);

	assert!(
		mock.requests().iter().all(|request| request.method != "POST"),
		"refused before any request"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_request_keeps_bearer_and_reasoning_effort() {
	let mock = MockHttp::start(vec![(200, mock_http::chat_completion_body("ok"))]);
	// The mode overrides the URL's own path, and the URL is used as written.
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "openai")
		.set("ISOQUANT_API_URL", messages_url(&mock));
	let params = ChatCompletionParams::new(&[Message::user("hi")], GLM, 2048)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::Medium)
		.with_extra_headers(
			[
				("x-api-key".to_string(), "caller".to_string()),
				("Authorization".to_string(), "Bearer caller".to_string()),
			]
			.into(),
		);

	let response = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect("chat completion");

	let request = completion_request(&mock);
	assert_eq!(request.path, "/v1/messages", "the configured URL, as written");
	assert_eq!(request.header("authorization"), Some("Bearer mock-key"));
	assert_eq!(request.header("x-api-key"), None);
	assert_eq!(request.header("isoquant-zdr"), Some("required"));

	let body = body_of(&request);
	assert_eq!(body["model"], GLM);
	assert_eq!(body["reasoning_effort"], "low");
	assert!(body.get("output_config").is_none(), "no Messages knobs");

	let usage = response.exchange.usage.expect("usage");
	let cost = provider_cost(
		GLM,
		usage.input_tokens,
		usage.cache_write_tokens,
		usage.cache_read_tokens,
		usage.billable_output_tokens(),
	);
	assert_eq!(usage.cost, Some(cost));
}

#[tokio::test]
#[serial_test::serial]
async fn a_failed_turn_stays_on_its_protocol() {
	let boom = (500, "{\"error\":\"boom\"}".to_string());

	let messages = MockHttp::start(vec![boom.clone()]);
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "anthropic")
		.set("ISOQUANT_API_URL", messages_url(&messages));
	let params = ChatCompletionParams::new(&[], DEEPSEEK, 10)
		.with_api_key("mock-key".to_owned())
		.with_max_retries(0);
	let error = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect_err("upstream failure");
	assert!(
		error.to_string().contains("isoquant API error 500"),
		"unexpected error: {error}"
	);
	let posts = posted_paths(&messages);
	assert_eq!(posts, ["/v1/messages"], "no Chat Completions attempt");

	let chat = MockHttp::start(vec![boom]);
	let _env = EnvGuard::new()
		.set("ISOQUANT_API_MODE", "openai")
		.set("ISOQUANT_API_URL", chat.url.clone());
	let params = ChatCompletionParams::new(&[], DEEPSEEK, 10)
		.with_api_key("mock-key".to_owned())
		.with_max_retries(0);
	let error = IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect_err("upstream failure");
	assert!(
		error.to_string().contains("isoquant API error 500"),
		"unexpected error: {error}"
	);
	let posts = posted_paths(&chat);
	assert_eq!(posts, ["/v1/chat/completions"], "no Messages attempt");
}

fn provider_cost(model: &str, input: u64, cache_write: u64, cache_read: u64, output: u64) -> f64 {
	IsoquantProvider::new()
		.get_model_pricing(model)
		.expect("catalog pricing")
		.calculate_cost(input, cache_write, cache_read, output)
}

#[tokio::test]
#[serial_test::serial]
async fn request_omits_sampling_fields() {
	let mock = MockHttp::start(vec![(200, mock_http::chat_completion_body("ok"))]);
	let _env = EnvGuard::new().set("ISOQUANT_API_URL", mock.url.clone());
	let params = ChatCompletionParams::new(&[], DEEPSEEK, 10).with_api_key("mock-key".to_owned());
	IsoquantProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");
	// A URL ending in /chat/completions names the Chat Completions route.
	let body = body_of(&completion_request(&mock));
	for field in ["temperature", "top_p", "top_k"] {
		assert!(body.get(field).is_none(), "{field} must not be sent");
	}
	assert_eq!(completion_request(&mock).path, "/v1/chat/completions");
}
