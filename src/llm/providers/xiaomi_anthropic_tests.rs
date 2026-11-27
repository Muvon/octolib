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

//! Xiaomi's Messages route and its endpoint resolution, through local HTTP
//! fixtures that speak the contract Xiaomi documents.

use super::*;
use crate::llm::tool_calls::GenericToolCall;
use crate::llm::types::{
	FunctionDefinition, ImageAttachment, ImageData, Message, SourceType, StructuredOutputRequest, VideoAttachment,
	VideoData,
};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::{self, MockHttp};
use serde_json::{json, Value};

const PRO: &str = "mimo-v2.6-pro";

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

/// A thinking block, a text answer and one tool call: 100 input tokens (20 read
/// from the cache) and 50 output tokens, thinking included.
fn stream() -> String {
	sse(&[
		json!({"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":PRO,"content":[],"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":20}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}),
		json!({"type":"content_block_stop","index":1}),
		json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}),
		json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}),
		json!({"type":"content_block_stop","index":2}),
		// Usage is a snapshot: the final counter replaces the start one.
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":50}}),
		json!({"type":"message_stop"}),
	])
}

/// A plain text answer.
fn text_stream(text: &str) -> String {
	sse(&[
		json!({"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}),
		json!({"type":"message_stop"}),
	])
}

/// The schema path's answer: a call to the synthetic tool whose arguments are
/// `input`.
fn tool_stream(input: &Value) -> String {
	sse(&[
		json!({"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"emit_structured_response","input":{}}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":input.to_string()}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":20}}),
		json!({"type":"message_stop"}),
	])
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

fn body_of(request: &mock_http::RecordedRequest) -> Value {
	serde_json::from_str(&request.body).expect("request json")
}

/// Paths of the POSTs received so far.
fn posted_paths(mock: &MockHttp) -> Vec<String> {
	mock.requests()
		.into_iter()
		.filter(|request| request.method == "POST")
		.map(|request| request.path)
		.collect()
}

/// A request with no tools, no schema and no effort.
fn ask(messages: Vec<Message>) -> ChatCompletionParams {
	ChatCompletionParams::new(&messages, PRO, 0).with_api_key("mock-key".to_owned())
}

#[test]
fn the_endpoint_follows_the_mode_then_the_url() {
	let plan_messages = "https://token-plan-sgp.xiaomimimo.com/anthropic";
	let plan_chat = "https://token-plan-cn.xiaomimimo.com/v1";
	let full_messages = "https://api.xiaomimimo.com/anthropic/v1/messages/";
	let full_chat = "http://localhost:8080/v1/chat/completions/?key=secret";
	let custom = "http://localhost:8080/inference";
	let chat = "https://api.xiaomimimo.com/v1/chat/completions";
	let messages = "https://api.xiaomimimo.com/anthropic/v1/messages";

	for (mode, api_url, expected_mode, expected_url) in [
		(None, None, ApiMode::OpenAi, chat),
		(Some(ApiMode::OpenAi), None, ApiMode::OpenAi, chat),
		(Some(ApiMode::Anthropic), None, ApiMode::Anthropic, messages),
		// An empty variable is an unset one.
		(None, Some(""), ApiMode::OpenAi, chat),
		// A documented base names the host; each protocol's endpoint is built on it.
		(
			None,
			Some(plan_messages),
			ApiMode::Anthropic,
			"https://token-plan-sgp.xiaomimimo.com/anthropic/v1/messages",
		),
		(
			None,
			Some(plan_chat),
			ApiMode::OpenAi,
			"https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
		),
		// A trailing slash on a base changes nothing.
		(
			None,
			Some("https://token-plan-sgp.xiaomimimo.com/anthropic/"),
			ApiMode::Anthropic,
			"https://token-plan-sgp.xiaomimimo.com/anthropic/v1/messages",
		),
		(
			None,
			Some("https://token-plan-cn.xiaomimimo.com/v1/"),
			ApiMode::OpenAi,
			"https://token-plan-cn.xiaomimimo.com/v1/chat/completions",
		),
		// A bare origin names no base: as written, on the default route.
		(
			None,
			Some("https://token-plan-sgp.xiaomimimo.com"),
			ApiMode::OpenAi,
			"https://token-plan-sgp.xiaomimimo.com",
		),
		(
			Some(ApiMode::OpenAi),
			Some(plan_messages),
			ApiMode::OpenAi,
			"https://token-plan-sgp.xiaomimimo.com/v1/chat/completions",
		),
		(
			Some(ApiMode::Anthropic),
			Some(plan_chat),
			ApiMode::Anthropic,
			"https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages",
		),
		// A full endpoint or an opaque URL is sent as written; the mode overrides its path.
		(None, Some(full_messages), ApiMode::Anthropic, full_messages),
		(None, Some(full_chat), ApiMode::OpenAi, full_chat),
		(Some(ApiMode::Anthropic), Some(full_chat), ApiMode::Anthropic, full_chat),
		(
			Some(ApiMode::OpenAi),
			Some(full_messages),
			ApiMode::OpenAi,
			full_messages,
		),
		(None, Some(custom), ApiMode::OpenAi, custom),
	] {
		assert_eq!(
			select_endpoint(mode, api_url),
			(expected_mode, expected_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn an_unknown_mode_is_refused_before_any_request() {
	let mock = MockHttp::start(vec![]);
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "claude")
		.set("XIAOMI_API_URL", format!("{}/anthropic", mock.base_url()));

	let error = XiaomiProvider::new()
		.chat_completion(ask(vec![Message::user("hi")]))
		.await
		.expect_err("refused");
	assert_eq!(
		error.to_string(),
		"Configuration error: XIAOMI_API_MODE must be `anthropic` or `openai`, got `claude`"
	);
	assert!(mock.requests().is_empty(), "no request: {:?}", mock.requests());
}

#[tokio::test]
#[serial_test::serial]
async fn messages_speaks_the_documented_contract() {
	let mock = MockHttp::start(vec![(200, stream())]);
	// The documented Token Plan base: the Messages endpoint is built on it.
	let _env = EnvGuard::new().set("XIAOMI_API_URL", format!("{}/anthropic", mock.base_url()));

	let media = Message::user("what is this?")
		.with_images(vec![
			image(ImageData::Url("https://example.com/a.png".to_owned())),
			image(ImageData::Base64("aW1hZ2U=".to_owned())),
		])
		.with_videos(vec![VideoAttachment {
			data: VideoData::Url("https://example.com/clip.mp4".to_owned()),
			media_type: "video/mp4".to_owned(),
			source_type: SourceType::Url,
			dimensions: None,
			size_bytes: None,
			duration_secs: None,
		}])
		.with_cache_marker();
	let params = ChatCompletionParams::new(
		&[Message::system("be brief").with_cache_marker(), media],
		"MiMo-V2.6-Pro",
		0,
	)
	.with_api_key("mock-key".to_owned())
	.with_reasoning_effort(ReasoningEffort::High)
	.with_tools(vec![FunctionDefinition {
		name: "lookup".to_owned(),
		description: String::new(),
		parameters: json!({ "type": "object" }),
		cache_control: Some(json!({ "type": "ephemeral" })),
		freeform: None,
	}]);

	let response = XiaomiProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	let request = &mock.requests()[0];
	assert_eq!(request.method, "POST");
	assert_eq!(request.path, "/anthropic/v1/messages");
	// The contract names `Authorization: Bearer`, and no other key header.
	assert_eq!(request.header("authorization"), Some("Bearer mock-key"));
	assert_eq!(request.header("x-api-key"), None);

	let body = body_of(request);
	assert_eq!(body["model"], PRO, "the catalogue's exact id");
	assert_eq!(body["stream"], true);
	assert_eq!(body["system"], "be brief");
	assert!(body.get("max_tokens").is_none(), "0 is the provider default");
	assert_eq!(body["thinking"], json!({ "type": "enabled" }));
	assert_eq!(body["tools"][0]["name"], "lookup");
	assert_eq!(body["tools"][0]["input_schema"], json!({ "type": "object" }));
	// The contract names none of these.
	for field in [
		"temperature",
		"top_p",
		"top_k",
		"reasoning_effort",
		"response_format",
		"output_config",
		"tool_choice",
	] {
		assert!(body.get(field).is_none(), "{field} must not be sent");
	}
	let blocks = body["messages"][0]["content"].as_array().expect("user blocks");
	assert_eq!(blocks.len(), 4, "text, two images, one video");
	assert_eq!(blocks[0], json!({ "type": "text", "text": "what is this?" }));
	assert_eq!(
		blocks[1]["source"],
		json!({ "type": "url", "url": "https://example.com/a.png" })
	);
	assert_eq!(
		blocks[2]["source"],
		json!({ "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=" })
	);
	assert_eq!(
		blocks[3],
		json!({ "type": "video", "source": { "type": "url", "url": "https://example.com/clip.mp4" } })
	);
	assert!(
		!request.body.contains("cache_control"),
		"the contract has no breakpoint field"
	);

	assert_eq!(response.content, "ok");
	assert_eq!(response.thinking.expect("thinking").content, "think");
	let call = &response.tool_calls.expect("tool calls")[0];
	assert_eq!(
		(call.id.as_str(), call.name.as_str(), &call.arguments),
		("toolu_1", "lookup", &json!({ "q": "x" }))
	);
	assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
	assert_eq!(response.exchange.provider, "xiaomi");
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(
			usage.input_tokens,
			usage.cache_read_tokens,
			usage.cache_write_tokens,
			usage.output_tokens,
			usage.reasoning_tokens
		),
		(100, 20, 0, 49, 1)
	);
	assert_eq!(usage.total_tokens, 170);
	// Cache reads bill at the cached rate and thinking at the output rate.
	let cost = pricing(PRO).expect("priced").calculate_cost(100, 0, 20, 50);
	assert_eq!(usage.cost, Some(cost));
	assert!((cost - 8.7072e-5).abs() < 1e-12, "{cost}");
	assert_eq!(mock.requests().len(), 1, "one request, no second protocol");
}

#[tokio::test]
#[serial_test::serial]
async fn thinking_follows_the_documented_switch() {
	let mock = MockHttp::start(vec![(200, stream())]);
	// The mode overrides the URL's own Chat Completions path; the URL is sent as written.
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "anthropic")
		.set("XIAOMI_API_URL", mock.url.clone());

	for (effort, expected) in [
		(Some(ReasoningEffort::None), json!({ "type": "disabled" })),
		(Some(ReasoningEffort::Max), json!({ "type": "enabled" })),
		(None, Value::Null),
	] {
		let params = match effort {
			Some(effort) => ask(vec![Message::user("hi")]).with_reasoning_effort(effort),
			None => ask(vec![Message::user("hi")]),
		};
		XiaomiProvider::new()
			.chat_completion(params)
			.await
			.expect("messages completion");

		let requests = mock.requests();
		let request = requests.last().expect("request");
		assert_eq!(request.path, "/v1/chat/completions", "as written");
		let body = body_of(request);
		assert_eq!(
			body["messages"],
			json!([{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }])
		);
		// No effort leaves the switch out: thinking is on by default upstream.
		assert_eq!(
			body.get("thinking"),
			if expected.is_null() { None } else { Some(&expected) },
			"{effort:?}"
		);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn a_schema_takes_the_forced_tool_path_and_is_validated() {
	let schema = json!({
		"type": "object",
		"properties": { "colors": { "type": "array", "items": { "type": "string" } } },
		"required": ["colors"],
		"additionalProperties": false
	});
	let mock = MockHttp::start(vec![
		(200, tool_stream(&json!({ "colors": ["red", "green", "blue"] }))),
		(200, tool_stream(&json!({ "colors": "red" }))),
		(200, tool_stream(&json!({ "other": 1 }))),
	]);
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "anthropic")
		.set("XIAOMI_API_URL", format!("{}/anthropic", mock.base_url()));
	let request = || {
		ask(vec![Message::user("three colors")])
			.with_structured_output(StructuredOutputRequest::json_schema(schema.clone()))
	};

	let response = XiaomiProvider::new()
		.chat_completion(request())
		.await
		.expect("schema answer");

	let body = body_of(&mock.requests()[0]);
	// No response format is documented on Messages: none is sent.
	assert!(body.get("output_config").is_none());
	assert!(body.get("response_format").is_none());
	assert_eq!(body["tools"][0]["name"], "emit_structured_response");
	assert_eq!(body["tools"][0]["input_schema"], schema);
	assert!(body["system"]
		.as_str()
		.unwrap_or_default()
		.contains("emit_structured_response"));
	assert_eq!(
		response.structured_output,
		Some(json!({ "colors": ["red", "green", "blue"] }))
	);
	assert_eq!(response.content, r#"{"colors":["red","green","blue"]}"#);
	assert!(response.tool_calls.is_none(), "the synthetic call never leaks");

	// A wrong shape, or a foreign object, fails closed; the model is not asked again.
	for _ in 1..3 {
		let error = XiaomiProvider::new()
			.chat_completion(request())
			.await
			.expect_err("fails closed");
		assert!(
			error.to_string().contains("does not match the schema"),
			"unexpected error: {error}"
		);
	}
	assert_eq!(posted_paths(&mock), ["/anthropic/v1/messages"; 3]);
}

#[tokio::test]
#[serial_test::serial]
async fn a_messages_failure_never_falls_back_to_chat() {
	let mock = MockHttp::start(vec![(
		400,
		r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad request"}}"#.to_owned(),
	)]);
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "anthropic")
		.set("XIAOMI_API_URL", format!("{}/anthropic", mock.base_url()));

	let error = XiaomiProvider::new()
		.chat_completion(ask(vec![Message::user("hi")]))
		.await
		.expect_err("upstream failure");
	assert!(
		error.to_string().contains("xiaomi API error 400"),
		"unexpected error: {error}"
	);
	assert_eq!(posted_paths(&mock), ["/anthropic/v1/messages".to_owned()]);
}

#[tokio::test]
#[serial_test::serial]
async fn a_chat_base_builds_the_chat_endpoint() {
	let mock = MockHttp::start(vec![(200, mock_http::chat_completion_body("ok"))]);
	// The documented Token Plan Chat Completions base, on the route it keeps.
	let _env = EnvGuard::new().set("XIAOMI_API_URL", format!("{}/v1", mock.base_url()));

	let response = XiaomiProvider::new()
		.chat_completion(ask(vec![Message::user("hi")]))
		.await
		.expect("chat completion");

	assert_eq!(posted_paths(&mock), ["/v1/chat/completions".to_owned()]);
	let body = body_of(&mock.requests()[0]);
	assert_eq!(body["model"], PRO);
	assert!(body.get("thinking").is_none(), "no effort, no switch: on by default");
	assert_eq!(response.content, "ok");
}

#[test]
#[serial_test::serial]
fn capabilities_follow_the_route() {
	let provider = XiaomiProvider::new();
	let _env = EnvGuard::new().remove("XIAOMI_API_MODE").remove("XIAOMI_API_URL");
	// Chat Completions by default: native schema decoding and the measured cache hour.
	assert!(provider.enforces_response_schema(PRO));
	assert_eq!(
		provider.cache_lifetime(PRO, false),
		Some(std::time::Duration::from_secs(3600))
	);

	let _env = EnvGuard::new().set("XIAOMI_API_MODE", "anthropic");
	for (model, _) in MODELS {
		assert!(provider.supports_structured_output(model), "{model}");
		assert!(
			!provider.enforces_response_schema(model),
			"{model}: no response format is documented"
		);
		assert!(
			provider.supports_caching(model),
			"{model}: cache_read_input_tokens is reported"
		);
		assert_eq!(
			provider.cache_lifetime(model, true),
			None,
			"{model}: not measured on Messages"
		);
		assert!(!provider.supports_required_tool_choice(model), "{model}");
		assert!(
			provider.supports_vision(model) && provider.supports_video(model),
			"{model}"
		);
	}

	// The route decides, not the variable: a Messages base with no mode says the same.
	let _env = EnvGuard::new()
		.remove("XIAOMI_API_MODE")
		.set("XIAOMI_API_URL", "https://token-plan-sgp.xiaomimimo.com/anthropic");
	assert!(!provider.enforces_response_schema(PRO));
	assert_eq!(provider.cache_lifetime(PRO, true), None);

	// An empty mode variable is an unset one.
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "")
		.set("XIAOMI_API_URL", "https://token-plan-sgp.xiaomimimo.com/anthropic");
	assert!(!provider.enforces_response_schema(PRO));

	// ...and the mode overrides the base back to Chat Completions on that host.
	let _env = EnvGuard::new().set("XIAOMI_API_MODE", "openai");
	assert!(provider.enforces_response_schema(PRO));
	assert_eq!(
		provider.cache_lifetime(PRO, true),
		Some(std::time::Duration::from_secs(3600))
	);
}

#[tokio::test]
#[serial_test::serial]
async fn thinking_blocks_go_back_as_captured() {
	let mock = MockHttp::start(vec![(200, stream()), (200, text_stream("done"))]);
	let _env = EnvGuard::new()
		.set("XIAOMI_API_MODE", "anthropic")
		.set("XIAOMI_API_URL", format!("{}/v1/messages", mock.base_url()));

	let first = XiaomiProvider::new()
		.chat_completion(ask(vec![Message::user("lookup x")]))
		.await
		.expect("tool call");
	// The next turn, as a caller stores it: the unified tool calls and the
	// blocks the exchange kept for replay.
	let calls: Vec<GenericToolCall> = first
		.tool_calls
		.expect("calls")
		.into_iter()
		.map(|call| GenericToolCall {
			id: call.id,
			name: call.name,
			arguments: call.arguments,
			meta: None,
		})
		.collect();
	let mut turn = Message::assistant("");
	turn.tool_calls = Some(serde_json::to_value(calls).expect("unified tool calls"));
	turn.provider_items = Some(first.exchange.response["output_items"].clone());

	let response = XiaomiProvider::new()
		.chat_completion(ask(vec![
			Message::user("lookup x"),
			turn,
			Message::tool("x", "toolu_1", "lookup"),
		]))
		.await
		.expect("answer");

	assert_eq!(response.content, "done");
	let body = body_of(&mock.requests()[1]);
	assert_eq!(
		body["messages"][0],
		json!({ "role": "user", "content": [{ "type": "text", "text": "lookup x" }] })
	);
	// Every block goes back as it came: thinking, text, then the tool call.
	assert_eq!(
		body["messages"][1]["content"],
		json!([
			{ "type": "thinking", "thinking": "think" },
			{ "type": "text", "text": "ok" },
			{ "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": { "q": "x" } }
		])
	);
	assert_eq!(
		body["messages"][2]["content"][0],
		json!({ "type": "tool_result", "tool_use_id": "toolu_1", "content": "x" })
	);
}
