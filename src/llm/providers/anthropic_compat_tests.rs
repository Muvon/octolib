// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;
use crate::llm::types::{
	FunctionDefinition, ImageAttachment, OutputFormat, ResponseMode, SourceType, StructuredOutputRequest,
	VideoAttachment,
};
use crate::test_support::mock_http::MockHttp;

const TEST: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "test",
	images: |model| match model {
		"text" => Images::None,
		"m-vision" => Images::Base64AndUrl,
		_ => Images::Base64,
	},
	videos: |model| model == "m-vision",
	enforces_response_schema: false,
	thinking: |_, effort, request| {
		if effort.is_some() {
			request["thinking"] = json!({ "type": "enabled" });
		}
	},
	// Every counter at $1 per 1M, so the cost is the token total.
	cost: |_, usage| {
		Some(
			(usage.input_tokens + usage.cache_write_tokens() + usage.cache_read_tokens + usage.output_tokens) as f64
				/ 1_000_000.0,
		)
	},
	default_system: "",
};

/// TEST with the fallback prompt MiniMax sends for a conversation without a
/// system turn.
const WITH_FALLBACK: AnthropicCompatConfig = AnthropicCompatConfig {
	default_system: "You are a helpful assistant.",
	..TEST
};

/// TEST with a schema-constraining endpoint.
const ENFORCING: AnthropicCompatConfig = AnthropicCompatConfig {
	enforces_response_schema: true,
	..TEST
};

fn image(data: ImageData) -> ImageAttachment {
	ImageAttachment {
		data,
		media_type: "image/png".to_owned(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}
}

fn video(data: VideoData) -> VideoAttachment {
	VideoAttachment {
		data,
		media_type: "video/mp4".to_owned(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
		duration_secs: None,
	}
}

/// A complete Messages response with the given content blocks and usage.
fn message_body(blocks: Value, usage: Value, stop_reason: &str) -> String {
	json!({
		"id": "msg_1",
		"type": "message",
		"role": "assistant",
		"model": "m",
		"content": blocks,
		"stop_reason": stop_reason,
		"stop_sequence": null,
		"usage": usage,
	})
	.to_string()
}

fn text_body(text: &str) -> String {
	message_body(
		json!([{ "type": "text", "text": text }]),
		json!({ "input_tokens": 10, "output_tokens": 5 }),
		"end_turn",
	)
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

fn content_of(message: &Value) -> Vec<Value> {
	message["content"].as_array().expect("content array").clone()
}

fn convert(messages: &[Message], config: AnthropicCompatConfig, model: &str) -> Vec<Value> {
	convert_messages(messages, config, model).expect("messages convert")
}

async fn complete(
	config: AnthropicCompatConfig,
	mock: &MockHttp,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	chat_completion(config, "secret-key".to_owned(), mock.url.clone(), params).await
}

#[test]
fn the_url_path_names_the_protocol_or_nothing() {
	assert_eq!(
		url_api_mode("https://api.z.ai/api/anthropic/v1/messages"),
		Some(ApiMode::Anthropic)
	);
	assert_eq!(
		url_api_mode("http://localhost:1234/v1/messages?beta=true"),
		Some(ApiMode::Anthropic)
	);
	assert_eq!(
		url_api_mode("http://localhost:1234/v1/messages/"),
		Some(ApiMode::Anthropic)
	);
	assert_eq!(
		url_api_mode("https://api.z.ai/api/paas/v4/chat/completions?x=1"),
		Some(ApiMode::OpenAi)
	);
	assert_eq!(
		url_api_mode("http://localhost:1234/v1/chat/completions/"),
		Some(ApiMode::OpenAi)
	);
	// Only the terminal path counts: not the host, the query, or a prefix.
	assert_eq!(
		url_api_mode("https://api.deepseek.com/anthropic/v1/chat/completions"),
		Some(ApiMode::OpenAi)
	);
	assert_eq!(url_api_mode("http://localhost:1234/v1"), None);
	assert_eq!(
		url_api_mode("https://chat/completions.example/v1"),
		None,
		"the host names nothing"
	);
	assert_eq!(
		url_api_mode("http://localhost:1234/v1/chat/completions/extra"),
		None,
		"only the terminal path counts"
	);
	assert_eq!(
		url_api_mode("http://localhost:1234/v1/Chat/Completions"),
		None,
		"paths are case-sensitive"
	);
	assert_eq!(url_api_mode("not a url"), None);
}

#[test]
fn an_unknown_mode_is_refused_before_any_request() {
	// The value is trimmed before it is judged and before it is shown, so a
	// padded or whitespace-only variable reports what the user meant to write,
	// not the padding.
	for (mode_env, raw, shown) in [
		("DEEPSEEK_API_MODE", "anthropic2", "anthropic2"),
		("MINIMAX_API_MODE", "claude", "claude"),
		("ZAI_API_MODE", "OPENAI_COMPAT", "OPENAI_COMPAT"),
		("LOCAL_API_MODE", " ", ""),
	] {
		let error = parse_api_mode(mode_env, raw).expect_err("refused");
		assert_eq!(
			error.to_string(),
			format!("Configuration error: {mode_env} must be `anthropic` or `openai`, got `{shown}`"),
			"unexpected error: {error}"
		);
	}
	assert_eq!(
		parse_api_mode("X_API_MODE", " OpenAI ").expect("parsed"),
		ApiMode::OpenAi
	);
	assert_eq!(
		parse_api_mode("X_API_MODE", "Anthropic").expect("parsed"),
		ApiMode::Anthropic
	);
}

#[test]
fn the_configuration_guide_names_each_mode_variable() {
	let guide = include_str!("../../../doc/04-advanced-guide.md");
	for mode_env in [
		"DEEPSEEK_API_MODE",
		"MINIMAX_API_MODE",
		"ZAI_API_MODE",
		"ISOQUANT_API_MODE",
		"LOCAL_API_MODE",
	] {
		assert!(guide.contains(&format!("`{mode_env}`")), "{mode_env} is undocumented");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn the_fallback_prompt_is_sent_only_when_no_system_turn_is_given() {
	let mock = MockHttp::start(vec![
		(200, text_body("ok")),
		(200, text_body("ok")),
		(200, text_body("ok")),
		(200, text_body("ok")),
	]);
	let params = ChatCompletionParams::new(&[Message::user("hi")], "m", 10);

	complete(WITH_FALLBACK, &mock, params.clone())
		.await
		.expect("completion");
	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(request["system"], "You are a helpful assistant.");

	// A system turn of its own is sent instead of the fallback.
	complete(
		WITH_FALLBACK,
		&mock,
		ChatCompletionParams::new(&[Message::system("Be terse."), Message::user("hi")], "m", 10),
	)
	.await
	.expect("completion");
	let request: Value = serde_json::from_str(&mock.requests()[1].body).expect("request json");
	assert_eq!(request["system"], "Be terse.");

	// An explicitly empty system turn is the caller's empty prompt: no
	// prompt content is invented in its place.
	complete(
		WITH_FALLBACK,
		&mock,
		ChatCompletionParams::new(&[Message::system(""), Message::user("hi")], "m", 10),
	)
	.await
	.expect("completion");
	let request: Value = serde_json::from_str(&mock.requests()[2].body).expect("request json");
	assert_eq!(request["system"], "");

	// A provider without a fallback sends no system prompt at all.
	complete(TEST, &mock, params).await.expect("completion");
	let request: Value = serde_json::from_str(&mock.requests()[3].body).expect("request json");
	assert!(request.get("system").is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn request_shape_keeps_system_turns_verbatim_and_sends_bearer_auth() {
	let mock = MockHttp::start(vec![(200, text_body("ok")), (200, text_body("ok"))]);

	let mut first = Message::system("  first half  ");
	first.cached = true;
	let messages = vec![first, Message::system("second half"), Message::user("hi")];
	let params = ChatCompletionParams::new(&messages, "m", 0).with_tools(vec![
		FunctionDefinition {
			name: "zeta".to_owned(),
			description: String::new(),
			parameters: json!({ "type": "object" }),
			cache_control: None,
			freeform: None,
		},
		FunctionDefinition {
			name: "alpha".to_owned(),
			description: String::new(),
			parameters: json!({ "type": "object" }),
			cache_control: None,
			freeform: None,
		},
	]);

	let response = complete(TEST, &mock, params).await.expect("completion");
	assert_eq!(response.content, "ok");

	let requests = mock.requests();
	let request: Value = serde_json::from_str(&requests[0].body).expect("request json");
	// Each system turn is its own block, verbatim, with the breakpoint only
	// on the turn that carries it.
	assert_eq!(
		request["system"],
		json!([
			{ "type": "text", "text": "  first half  ", "cache_control": { "type": "ephemeral" } },
			{ "type": "text", "text": "second half" }
		])
	);
	assert_eq!(
		request["messages"],
		json!([{ "role": "user", "content": [{ "type": "text", "text": "hi" }] }])
	);
	assert_eq!(
		request["tools"][0]["name"], "alpha",
		"tools are a stable, sorted cache prefix"
	);
	assert_eq!(request["tools"][1]["name"], "zeta");
	assert_eq!(request["stream"], true);
	assert!(request.get("max_tokens").is_none(), "0 is the provider default");
	assert!(request.get("thinking").is_none(), "no effort, no thinking knobs");

	assert_eq!(requests[0].header("authorization"), Some("Bearer secret-key"));
	assert_eq!(requests[0].header("anthropic-version"), Some(ANTHROPIC_VERSION));
	assert_eq!(requests[0].header("x-api-key"), None);
	assert_eq!(response.exchange.provider, "test");

	// One cached turn is the same single block, on its own.
	let mut only = Message::system("only turn");
	only.cached = true;
	complete(
		TEST,
		&mock,
		ChatCompletionParams::new(&[only, Message::user("hi")], "m", 0),
	)
	.await
	.expect("completion");
	// The log grows with every call: re-read it rather than indexing a
	// snapshot taken before this request was made.
	let requests = mock.requests();
	assert_eq!(requests.len(), 2, "one request per completion");
	let request: Value = serde_json::from_str(&requests[1].body).expect("request json");
	assert_eq!(
		request["system"],
		json!([{ "type": "text", "text": "only turn", "cache_control": { "type": "ephemeral" } }])
	);
}

#[tokio::test]
#[serial_test::serial]
async fn x_api_key_sends_the_key_in_its_own_header_only() {
	let mock = MockHttp::start(vec![(200, text_body("ok"))]);
	let params = ChatCompletionParams::new(&[Message::user("hi")], "m", 0);

	chat_completion_with_x_api_key(TEST, "secret-key".to_owned(), mock.url.clone(), params)
		.await
		.expect("completion");

	let requests = mock.requests();
	assert_eq!(requests[0].header("x-api-key"), Some("secret-key"));
	assert_eq!(requests[0].header("authorization"), None);
	assert_eq!(requests[0].header("anthropic-version"), Some(ANTHROPIC_VERSION));
	assert_eq!(
		requests[0]
			.headers
			.iter()
			.filter(|(name, _)| name == "x-api-key")
			.count(),
		1
	);
}

#[tokio::test]
#[serial_test::serial]
async fn empty_key_sends_no_authorization_header() {
	let mock = MockHttp::start(vec![(200, text_body("ok"))]);
	let params =
		ChatCompletionParams::new(&[Message::user("hi")], "m", 100).with_reasoning_effort(ReasoningEffort::Low);

	chat_completion(TEST, String::new(), mock.url.clone(), params)
		.await
		.expect("completion");

	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), None);
	let request: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["max_tokens"], 100);
	assert_eq!(
		request["thinking"],
		json!({ "type": "enabled" }),
		"the effort reaches the provider's knob"
	);
}

#[test]
fn images_follow_the_endpoint_policy_instead_of_being_dropped() {
	let base64 = Message::user("what is this?").with_images(vec![image(ImageData::Base64("aW1hZ2U=".to_owned()))]);
	assert_eq!(
		content_of(&convert(std::slice::from_ref(&base64), TEST, "m")[0])[1],
		json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": "aW1hZ2U=" } })
	);

	let url =
		Message::user("what is this?").with_images(vec![image(ImageData::Url("https://example.com/a.png".to_owned()))]);
	assert_eq!(
		content_of(&convert(std::slice::from_ref(&url), TEST, "m-vision")[0])[1],
		json!({ "type": "image", "source": { "type": "url", "url": "https://example.com/a.png" } })
	);
	for (messages, model, refused) in [
		(vec![url], "m", "URL images"),
		(vec![base64.clone()], "text", "image input"),
	] {
		let error = convert_messages(&messages, TEST, model).expect_err("unsupported image");
		assert!(
			error
				.to_string()
				.contains(&format!("test model {model} does not accept {refused}")),
			"unexpected error: {error}"
		);
	}

	let clip = Message::user("watch").with_videos(vec![video(VideoData::Url("https://example.com/a.mp4".to_owned()))]);
	assert_eq!(
		content_of(&convert(std::slice::from_ref(&clip), TEST, "m-vision")[0])[1],
		json!({ "type": "video", "source": { "type": "url", "url": "https://example.com/a.mp4" } })
	);
	assert!(convert_messages(&[clip], TEST, "m").is_err(), "no video on this model");
}

#[test]
fn parallel_tool_results_share_one_user_message() {
	let mut assistant = Message::assistant("");
	assistant.tool_calls = Some(json!([
		{ "id": "toolu_a", "name": "first", "arguments": {}, "meta": null },
		{ "id": "toolu_b", "name": "second", "arguments": {}, "meta": null }
	]));
	let messages = vec![
		assistant,
		Message::tool("result A", "toolu_a", "first"),
		Message::tool("result B", "toolu_b", "second"),
	];

	let converted = convert(&messages, TEST, "m");
	assert_eq!(converted.len(), 2);
	assert_eq!(converted[0]["role"], "assistant");
	assert_eq!(
		converted[0]["content"][0]["input"],
		json!({}),
		"tool arguments are an object"
	);
	assert_eq!(converted[1]["role"], "user");
	let blocks = content_of(&converted[1]);
	assert_eq!(blocks.len(), 2);
	assert_eq!(blocks[0]["type"], "tool_result");
	assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
	assert_eq!(blocks[0]["content"], "result A");
	assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
}

#[test]
fn tool_results_merge_following_user_hint() {
	let messages = vec![
		Message::tool("result A", "toolu_a", "first"),
		Message::tool("result B", "toolu_b", "second"),
		Message::user("Please use those results."),
	];

	let converted = convert(&messages, TEST, "m");
	assert_eq!(converted.len(), 1);
	let blocks = content_of(&converted[0]);
	assert_eq!(blocks.len(), 3);
	assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
	assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
	assert_eq!(blocks[2]["type"], "text");
	assert_eq!(blocks[2]["text"], "Please use those results.");
}

#[test]
fn tool_images_stay_inside_the_matching_tool_result() {
	for text in ["caption", ""] {
		let converted = convert(&[shared::tests::tool_image_message(text)], TEST, "m-vision");
		let result = &converted[0]["content"][0];
		assert_eq!(result["type"], "tool_result");
		assert_eq!(result["tool_use_id"], "call_image");
		let offset = usize::from(!text.is_empty());
		assert_eq!(result["content"][offset]["source"]["data"], "aW1hZ2U=");
		assert_eq!(
			result["content"][offset + 1]["source"]["url"],
			"https://example.com/screenshot.jpg"
		);
	}
}
/// A tool result is not a place to lose an attachment: media the endpoint does
/// not take is refused, exactly as in a user turn.
#[test]
fn tool_result_media_is_refused_when_the_endpoint_does_not_take_it() {
	let clip = Message::tool("watch", "call_video", "camera")
		.with_videos(vec![video(VideoData::Url("https://example.com/a.mp4".to_owned()))]);
	let error = convert_messages(std::slice::from_ref(&clip), TEST, "m").expect_err("no video on this model");
	assert!(
		error.to_string().contains("test model m does not accept video input"),
		"unexpected error: {error}"
	);

	let converted = convert(&[clip], TEST, "m-vision");
	assert_eq!(
		content_of(&converted[0])[0]["content"][1],
		json!({ "type": "video", "source": { "type": "url", "url": "https://example.com/a.mp4" } })
	);

	// An image the endpoint does not take is refused rather than checked and
	// then left out of the block.
	let error = convert_messages(&[shared::tests::tool_image_message("caption")], TEST, "text")
		.expect_err("no images on this model");
	assert!(
		error
			.to_string()
			.contains("test model text does not accept image input"),
		"unexpected error: {error}"
	);
}

#[test]
fn captured_thinking_blocks_are_replayed_only_when_they_still_match_the_turn() {
	let mut assistant = Message::assistant("I will check.");
	assistant.tool_calls = Some(json!([{ "id": "toolu_a", "name": "first", "arguments": { "x": 1 } }]));
	assistant.thinking = Some(ThinkingBlock {
		content: "stored text without the signature".to_owned(),
		tokens: 3,
	});
	let items = json!([
		{ "type": "thinking", "thinking": "reasoning", "signature": "signed" },
		{ "type": "text", "text": "I will check." },
		{ "type": "tool_use", "id": "toolu_a", "name": "first", "input": { "x": 1 } }
	]);

	assistant.provider_items = Some(json!({ "provider": "test", "items": items }));
	assert_eq!(
		content_of(&convert(&[assistant.clone()], TEST, "m")[0]),
		items.as_array().unwrap().clone()
	);

	// Another provider's blocks, or a turn whose tool calls have changed, are
	// rebuilt from the stored fields; the unsigned thinking text is not sent.
	for captured in [
		json!({ "provider": "other", "items": items }),
		json!({ "provider": "test", "items": [items[0], items[1]] }),
	] {
		assistant.provider_items = Some(captured);
		let content = content_of(&convert(&[assistant.clone()], TEST, "m")[0]);
		assert_eq!(content.len(), 2, "text and tool_use only");
		assert_eq!(content[0]["type"], "text");
		assert_eq!(content[1]["type"], "tool_use");
		assert_eq!(content[1]["id"], "toolu_a");
		assert_eq!(content[1]["input"], json!({ "x": 1 }));
	}
}

#[tokio::test]
#[serial_test::serial]
async fn json_response_splits_thinking_tools_usage_and_cost() {
	let body = message_body(
		json!([
			{ "type": "thinking", "thinking": "pondering deeply", "signature": "signed" },
			{ "type": "text", "text": "final answer" },
			{ "type": "tool_use", "id": "toolu_a", "name": "get_weather", "input": { "city": "Oslo" } }
		]),
		json!({
			"input_tokens": 10,
			"cache_read_input_tokens": 4,
			"cache_creation_input_tokens": 2,
			"cache_creation": { "ephemeral_5m_input_tokens": 2, "ephemeral_1h_input_tokens": 3 },
			"output_tokens": 30,
			"output_tokens_details": { "thinking_tokens": 6 }
		}),
		"tool_use",
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let response = complete(TEST, &mock, ChatCompletionParams::new(&[], "m", 10))
		.await
		.expect("completion");

	assert_eq!(response.content, "final answer");
	assert_eq!(
		response.thinking.as_ref().expect("thinking").content,
		"pondering deeply"
	);
	assert_eq!(
		response.thinking.as_ref().expect("thinking").tokens,
		6,
		"the reported counter, not the estimate"
	);
	assert_eq!(response.id.as_deref(), Some("msg_1"));
	assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
	let calls = response.tool_calls.expect("tool calls");
	assert_eq!(
		(calls[0].id.as_str(), calls[0].name.as_str()),
		("toolu_a", "get_weather")
	);
	assert_eq!(calls[0].arguments["city"], "Oslo");

	// Input stays clean, each cache TTL write is kept, thinking is carved out
	// of output and the whole output counter is priced.
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens),
		(10, 4, 5)
	);
	assert_eq!((usage.output_tokens, usage.reasoning_tokens), (24, 6));
	assert_eq!(usage.total_tokens, 49);
	assert!((usage.cost.expect("cost") - 49.0 / 1_000_000.0).abs() < 1e-15);
	assert!(usage.request_time_ms.is_some());

	// The blocks go back on the next turn with their signature.
	assert_eq!(
		response.exchange.response["output_items"],
		json!({ "provider": "test", "items": response.exchange.response["content"].clone() })
	);
}

#[tokio::test]
#[serial_test::serial]
async fn stream_fragments_assemble_signed_thinking_tool_arguments_and_final_usage() {
	let events = vec![
		json!({"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":20}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"run","input":{}}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
		json!({"type":"content_block_stop","index":1}),
		// Usage is a snapshot: the final counter replaces the start one.
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":40}}),
		json!({"type":"message_stop"}),
	];
	let mock = MockHttp::start(vec![(200, sse(&events))]);
	let response = complete(TEST, &mock, ChatCompletionParams::new(&[], "m", 10))
		.await
		.expect("stream assembles");

	assert_eq!(response.thinking.expect("thinking").content, "think");
	assert_eq!(response.tool_calls.expect("tool calls")[0].arguments, json!({ "x": 1 }));
	assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(usage.input_tokens, usage.cache_read_tokens, usage.output_tokens),
		(10, 20, 39)
	);
	assert_eq!(response.exchange.response["content"][0]["signature"], "signed");
	assert_eq!(response.exchange.provider, "test");
}

#[tokio::test]
#[serial_test::serial]
async fn unfinished_stream_is_an_error_not_a_partial_answer() {
	let events = vec![
		json!({"type":"message_start","message":{"id":"m1","type":"message","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"half an ans"}}),
	];
	let mock = MockHttp::start(vec![(200, sse(&events))]);
	let error = complete(TEST, &mock, ChatCompletionParams::new(&[], "m", 10).with_max_retries(1))
		.await
		.expect_err("no message_stop");
	assert!(
		!error.to_string().contains("half an ans"),
		"the partial text is not an answer"
	);
	// A stream cut before its terminal event is a transport interruption: it
	// is retried on a fresh accumulator, as on every other provider.
	assert_eq!(mock.requests().len(), 2, "one attempt plus one retry");
}

#[tokio::test]
#[serial_test::serial]
async fn retryable_status_is_retried_and_permanent_errors_surface() {
	let mock = MockHttp::start(vec![
		(503, r#"{"error":{"message":"overloaded"}}"#.to_string()),
		(200, text_body("ok")),
	]);
	let params = ChatCompletionParams::new(&[Message::user("hi")], "m", 10)
		.with_max_retries(1)
		.with_retry_timeout(std::time::Duration::from_millis(1));
	let response = complete(TEST, &mock, params).await.expect("retry succeeds");
	assert_eq!(response.content, "ok");
	assert_eq!(mock.requests().len(), 2, "one attempt plus one retry");

	let mock = MockHttp::start(vec![(400, r#"{"error":{"message":"bad request"}}"#.to_string())]);
	let params = ChatCompletionParams::new(&[Message::user("hi")], "m", 10).with_max_retries(2);
	let error = complete(TEST, &mock, params).await.expect_err("http error surfaces");
	assert!(
		error.to_string().contains("test API error 400"),
		"unexpected error: {error}"
	);
	assert_eq!(mock.requests().len(), 1, "400 is not retried");
}

#[tokio::test]
#[serial_test::serial]
async fn cancelled_request_fails_without_sending() {
	let mock = MockHttp::start(vec![(200, text_body("ok"))]);
	let (sender, token) = tokio::sync::watch::channel(false);
	sender.send(true).expect("cancel");
	let params = ChatCompletionParams::new(&[Message::user("hi")], "m", 10).with_cancellation_token(token);

	assert!(
		complete(TEST, &mock, params).await.is_err(),
		"a cancelled request does not answer"
	);
	assert!(mock.requests().is_empty(), "nothing is sent after cancellation");
}

#[tokio::test]
#[serial_test::serial]
async fn schema_goes_through_the_forced_tool_unless_the_endpoint_enforces_it() {
	let schema = json!({
		"type": "object",
		"properties": { "colors": { "type": "array", "items": { "type": "string" } } },
		"required": ["colors"],
		"additionalProperties": false
	});
	let structured = StructuredOutputRequest {
		format: OutputFormat::JsonSchema,
		mode: ResponseMode::Auto,
		schema: Some(schema.clone()),
	};
	let tool_answer = message_body(
		json!([{ "type": "tool_use", "id": "t1", "name": "emit_structured_response", "input": { "colors": ["Crimson", "Teal", "Amber"] } }]),
		json!({ "input_tokens": 10, "output_tokens": 5 }),
		"tool_use",
	);
	let params = ChatCompletionParams::new(&[Message::system("Be terse."), Message::user("three colors")], "m", 10)
		.with_structured_output(structured.clone());

	let mock = MockHttp::start(vec![(200, tool_answer)]);
	let response = complete(TEST, &mock, params.clone()).await.expect("forced tool answer");
	assert_eq!(
		response.structured_output,
		Some(json!({ "colors": ["Crimson", "Teal", "Amber"] }))
	);
	assert!(
		response.tool_calls.is_none(),
		"the synthetic tool call never reaches the client"
	);

	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(request["tools"][0]["name"], "emit_structured_response");
	assert_eq!(request["tools"][0]["input_schema"], schema);
	assert!(request.get("output_config").is_none());
	// The enforcement's trailing system turn follows the caller's own.
	assert_eq!(request["system"][0]["text"], "Be terse.");
	let enforcement = request["system"][1]["text"].as_str().expect("enforcement system text");
	assert!(
		enforcement.starts_with("Call the `emit_structured_response`"),
		"unexpected system: {enforcement}"
	);

	// Without the forced tool the endpoint takes the schema itself.
	let mock = MockHttp::start(vec![(200, text_body(r#"{"colors":["Crimson"]}"#))]);
	let response = complete(ENFORCING, &mock, params)
		.await
		.expect("endpoint-constrained answer");
	assert_eq!(response.structured_output, Some(json!({ "colors": ["Crimson"] })));
	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(
		request["output_config"]["format"],
		json!({ "type": "json_schema", "schema": schema })
	);
	assert!(request.get("tools").is_none());

	// A forced-tool answer that ignores the schema fails closed.
	let wrong = message_body(
		json!([{ "type": "tool_use", "id": "t1", "name": "emit_structured_response", "input": { "colors": "Crimson" } }]),
		json!({ "input_tokens": 10, "output_tokens": 5 }),
		"tool_use",
	);
	let mock = MockHttp::start(vec![(200, wrong)]);
	let params =
		ChatCompletionParams::new(&[Message::user("three colors")], "m", 10).with_structured_output(structured);
	assert!(
		complete(TEST, &mock, params).await.is_err(),
		"schema mismatch is an error"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn a_json_object_request_is_sent_as_an_object_schema() {
	// The forced-tool path: the object guarantee rides the schema machinery
	// instead of being dropped on the floor.
	let body = message_body(
		json!([{ "type": "tool_use", "id": "t1", "name": "emit_structured_response", "input": { "answer": 42 } }]),
		json!({ "input_tokens": 10, "output_tokens": 5 }),
		"tool_use",
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let params = ChatCompletionParams::new(&[Message::user("give me JSON")], "m", 10)
		.with_structured_output(StructuredOutputRequest::json());
	let response = complete(TEST, &mock, params).await.expect("object answer");

	assert_eq!(response.structured_output, Some(json!({ "answer": 42 })));
	assert_eq!(response.content, r#"{"answer":42}"#);
	assert!(response.tool_calls.is_none(), "the synthetic tool call never leaks");
	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(request["tools"][0]["name"], "emit_structured_response");
	assert_eq!(request["tools"][0]["input_schema"], json!({ "type": "object" }));
	assert!(request.get("output_config").is_none(), "not an enforcing endpoint");

	// Ordinary chat keeps its exact body: no tools, no added prompt.
	let mock = MockHttp::start(vec![(200, text_body("plain answer"))]);
	let response = complete(TEST, &mock, ChatCompletionParams::new(&[Message::user("hi")], "m", 10))
		.await
		.expect("plain completion");
	assert_eq!(response.content, "plain answer");
	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert!(request.get("tools").is_none(), "ordinary chat is not touched");
	assert!(request.get("system").is_none(), "no prompt is added");
}

#[tokio::test]
#[serial_test::serial]
async fn a_json_object_request_on_an_enforcing_endpoint_asks_for_an_object() {
	let mock = MockHttp::start(vec![(200, text_body("{\"a\":1}"))]);
	let params = ChatCompletionParams::new(&[Message::user("give me JSON")], "m", 10)
		.with_structured_output(StructuredOutputRequest::json());
	let response = complete(ENFORCING, &mock, params).await.expect("object answer");

	assert_eq!(response.structured_output, Some(json!({ "a": 1 })));
	let request: Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(
		request["output_config"]["format"],
		json!({ "type": "json_schema", "schema": { "type": "object" } })
	);
}

#[tokio::test]
#[serial_test::serial]
async fn a_json_object_request_fails_closed_on_prose() {
	let mock = MockHttp::start(vec![(200, text_body("Sure, here are three colors."))]);
	let params = ChatCompletionParams::new(&[Message::user("three colors")], "m", 10)
		.with_structured_output(StructuredOutputRequest::json());
	let error = complete(TEST, &mock, params)
		.await
		.expect_err("prose is not a JSON object");
	assert!(
		error.to_string().contains("no parseable structured output"),
		"unexpected error: {error}"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn tool_use_blocks_need_a_non_blank_string_id_and_name() {
	// Every broken shape is refused, and the valid blocks beside it are not
	// returned as a partial answer.
	for block in [
		json!({ "type": "tool_use", "name": "f", "input": {} }),
		json!({ "type": "tool_use", "id": "", "name": "f", "input": {} }),
		json!({ "type": "tool_use", "id": "   ", "name": "f", "input": {} }),
		json!({ "type": "tool_use", "id": 7, "name": "f", "input": {} }),
		json!({ "type": "tool_use", "id": "toolu_1", "input": {} }),
		json!({ "type": "tool_use", "id": "toolu_1", "name": "", "input": {} }),
		json!({ "type": "tool_use", "id": "toolu_1", "name": null, "input": {} }),
		json!({ "type": "tool_use", "id": "toolu_1", "name": ["f"], "input": {} }),
	] {
		let body = message_body(
			json!([{ "type": "text", "text": "partial answer" }, block.clone()]),
			json!({ "input_tokens": 1, "output_tokens": 1 }),
			"tool_use",
		);
		let mock = MockHttp::start(vec![(200, body)]);
		let error = complete(TEST, &mock, ChatCompletionParams::new(&[Message::user("hi")], "m", 10))
			.await
			.expect_err("broken tool_use");
		assert!(error.to_string().contains("without a string"), "{block}: {error}");
		assert!(
			!error.to_string().contains("partial answer"),
			"nothing of the partial answer is handed back"
		);
	}

	// A complete block is still a tool call.
	let body = message_body(
		json!([{ "type": "tool_use", "id": "toolu_1", "name": "f", "input": { "a": 1 } }]),
		json!({ "input_tokens": 1, "output_tokens": 1 }),
		"tool_use",
	);
	let mock = MockHttp::start(vec![(200, body)]);
	let response = complete(TEST, &mock, ChatCompletionParams::new(&[Message::user("hi")], "m", 10))
		.await
		.expect("valid tool call");
	assert_eq!(response.tool_calls.expect("tool calls")[0].arguments, json!({ "a": 1 }));
}

#[tokio::test]
#[serial_test::serial]
async fn a_streamed_tool_use_without_an_id_is_refused_too() {
	let events = vec![
		json!({"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"calling"}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"","name":"f","input":{}}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}),
		json!({"type":"content_block_stop","index":1}),
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}),
		json!({"type":"message_stop"}),
	];
	let mock = MockHttp::start(vec![(200, sse(&events))]);
	let error = complete(TEST, &mock, ChatCompletionParams::new(&[Message::user("hi")], "m", 10))
		.await
		.expect_err("broken streamed tool call");
	assert!(
		error.to_string().contains("without a string id"),
		"unexpected error: {error}"
	);
}
