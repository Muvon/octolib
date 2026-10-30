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
use crate::llm::factory::ProviderFactory;
use crate::llm::providers::shared;
use crate::llm::tool_calls::GenericToolCall;
use crate::llm::types::FunctionDefinition;

fn assistant_with_call(meta: Option<serde_json::Map<String, Value>>) -> Message {
	let mut message = Message::assistant("");
	message.tool_calls = Some(
		serde_json::to_value(vec![GenericToolCall {
			id: "call_1".to_string(),
			name: "view".to_string(),
			arguments: json!({"path": "a.rs"}),
			meta,
		}])
		.unwrap(),
	);
	message
}

#[test]
fn factory_resolves_chatgpt_models() {
	let (provider, model) = ProviderFactory::get_provider_for_model("chatgpt:gpt-6.1-sol").unwrap();
	assert_eq!(provider.name(), "chatgpt");
	assert_eq!(model, "gpt-6.1-sol");
	assert!(provider.get_model_pricing(&model).is_some());
}

#[test]
fn request_follows_the_plan_route_contract() {
	let messages = vec![Message::system("rules"), Message::user("hi")];
	let mut params = ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000).with_long_cache(true);
	params.previous_id = Some("resp_1".to_string());
	params.tools = Some(vec![FunctionDefinition {
		name: "view".to_string(),
		description: "Read a file".to_string(),
		parameters: json!({"type": "object"}),
		cache_control: None,
	}]);

	let request = build_request(&params);

	assert_eq!(request["store"], false);
	assert_eq!(params.chatgpt_transport, crate::ChatGptTransport::Http);
	assert_eq!(request["stream"], true);
	assert!(request.get("type").is_none());
	assert_eq!(request["include"], json!(["reasoning.encrypted_content"]));
	for rejected in [
		"previous_response_id",
		"max_output_tokens",
		"temperature",
		"top_p",
		"prompt_cache_retention",
	] {
		assert!(request.get(rejected).is_none(), "{} must not be sent", rejected);
	}
	assert_eq!(request["input"][0]["role"], "developer");
	assert_eq!(request["input"][1]["role"], "user");
	assert_eq!(request["tools"][0]["type"], "namespace");
	assert_eq!(request["tools"][0]["name"], TOOL_NAMESPACE);
	assert_eq!(request["tools"][0]["tools"][0]["name"], "view");
	assert_eq!(request["tools"][0]["tools"][0]["strict"], false);
}

#[cfg(feature = "websockets")]
#[test]
fn cache_key_names_the_session_so_the_plan_route_keeps_it() {
	let messages = vec![Message::user("hi")];
	let params = ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000);
	assert!(!request_headers(&params).unwrap().contains_key(SESSION_HEADER));
	let params = params.with_prompt_cache_key("s-1");
	assert_eq!(build_request(&params)["prompt_cache_key"], "s-1");
	assert_eq!(request_headers(&params).unwrap()[SESSION_HEADER], "s-1");
	let caller = HashMap::from([("Session-Id".to_string(), "caller".to_string())]);
	assert_eq!(
		request_headers(&params.with_extra_headers(caller)).unwrap()[SESSION_HEADER],
		"caller"
	);
}

#[cfg(feature = "websockets")]
#[test]
fn malformed_handshake_headers_fail_before_connecting() {
	let params = ChatCompletionParams::new(&[], "gpt-6.1-sol", 0.7, 1.0, 50, 1000)
		.with_extra_headers(HashMap::from([("x-test".to_string(), "bad\nvalue".to_string())]));
	assert!(request_headers(&params).is_err());
}

fn captured(items: Value) -> Option<Value> {
	Some(json!({"provider": "chatgpt", "items": items}))
}

#[test]
fn captured_items_replay_verbatim_and_in_order() {
	let items = json!([
		{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-1"},
		{"type": "message", "id": "msg_1", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "checking"}]},
		{"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc-2"},
		{"type": "function_call", "id": "fc_1", "call_id": "call_1", "status": "completed", "name": "view", "namespace": TOOL_NAMESPACE, "arguments": "{\"path\":\"a.rs\"}"}
	]);
	let mut assistant = assistant_with_call(None);
	assistant.content = "flattened text is not replayed".to_string();
	assistant.provider_items = captured(items.clone());

	assert_eq!(build_input(&[assistant]), items.as_array().unwrap().clone());
}

#[test]
fn text_only_response_round_trips_through_captured_items() {
	let response_json = json!({
		"id": "resp_text",
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-1"},
			{"type": "message", "id": "msg_1", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "waiting"}]},
			{"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "enc-2"},
			{"type": "message", "id": "msg_2", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "done"}]}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
	});
	let expected = response_json["output"].as_array().unwrap().clone();
	let response = openai::parse_responses_api_response(
		json!({"model": "gpt-6.1-sol", "store": false}),
		response_json,
		"chatgpt",
		1,
		HashMap::new(),
	)
	.unwrap();
	let mut assistant = Message::assistant(&response.content);
	assistant.provider_items = Some(response.exchange.response["output_items"].clone());

	assert_eq!(build_input(&[assistant]), expected);
}

#[test]
fn captured_items_that_do_not_match_the_message_are_not_replayed() {
	let mut assistant = assistant_with_call(None);
	assistant.content = "working".to_string();
	let fallback = build_input(std::slice::from_ref(&assistant));
	assert_eq!(
		fallback[0],
		json!({"role": "assistant", "content": "working"}),
		"no phase is invented"
	);
	assert_eq!(fallback[1]["call_id"], "call_1");

	let call = json!({"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{}"});
	for provider_items in [
		Some(json!({"provider": "openai", "items": [call.clone()]})),
		captured(json!({"type": "reasoning"})),
		captured(json!([{"type": "reasoning", "id": "rs_1"}])),
		captured(json!([{"type": "function_call", "call_id": "other"}])),
		captured(json!([call.clone(), call])),
	] {
		assistant.provider_items = provider_items;
		assert_eq!(build_input(std::slice::from_ref(&assistant)), fallback);
	}
}
// The plan route caches by rendered prefix: a later request of the same session
// must start with the earlier request byte-for-byte, or every turn pays its whole
// history again.
#[test]
fn a_grown_transcript_extends_the_previous_request_unchanged() {
	let mut first_assistant = assistant_with_call(None);
	first_assistant.provider_items = captured(json!([
		{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc"},
		{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{\"path\":\"a.rs\"}"}
	]));
	let mut messages = vec![
		Message::system("rules"),
		Message::user("go"),
		first_assistant,
		Message::tool("ok", "call_1", "view"),
	];
	let request_at = |messages: &[Message]| {
		let mut params =
			ChatCompletionParams::new(messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000).with_prompt_cache_key("s-1");
		params.tools = Some(vec![FunctionDefinition {
			name: "view".to_string(),
			description: "Read a file".to_string(),
			parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
			cache_control: None,
		}]);
		build_request(&params)
	};

	let first = request_at(&messages);
	assert_eq!(first, request_at(&messages), "request building is deterministic");

	let mut next_call = assistant_with_call(None);
	next_call.content = "checking".to_string();
	next_call.provider_items = captured(json!([
		{"type": "message", "id": "msg_2", "role": "assistant", "phase": "commentary", "content": [{"type": "output_text", "text": "checking"}]},
		{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{\"path\":\"a.rs\"}"}
	]));
	messages.push(next_call);
	messages.push(Message::tool("done", "call_1", "view"));
	let second = request_at(&messages);

	let (first_input, second_input) = (first["input"].as_array().unwrap(), second["input"].as_array().unwrap());
	assert!(second_input.len() > first_input.len());
	assert_eq!(&second_input[..first_input.len()], first_input.as_slice());
	for field in first.as_object().unwrap().keys().filter(|key| *key != "input") {
		assert_eq!(first[field], second[field], "{field} changed between turns");
	}
}

#[tokio::test]
async fn stream_folds_into_a_response_that_keeps_reasoning_for_replay() {
	let body = [
        "event: response.created",
        r#"data: {"type":"response.created","response":{"id":"resp_1"}}"#,
        "",
        "event: response.output_item.done",
        r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"enc"}}"#,
        "",
        "event: response.output_item.done",
        r#"data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"view","namespace":"tools","arguments":"{\"path\":\"a.rs\"}"}}"#,
        "",
        "event: response.completed",
        r#"data: {"type":"response.completed","response":{"id":"resp_1","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}"#,
    ]
    .join("\n");

	let merged = merge_stream(&body).await.unwrap();
	assert_eq!(merged["output"].as_array().unwrap().len(), 2);

	let response = openai::parse_responses_api_response(
		json!({"model": "gpt-6.1-sol", "store": false}),
		merged,
		"chatgpt",
		1,
		HashMap::new(),
	)
	.unwrap();

	let calls = response.tool_calls.expect("tool call parsed");
	assert_eq!(calls[0].name, "view");
	assert_eq!(calls[0].arguments, json!({"path": "a.rs"}));
	let cost = response.exchange.usage.expect("usage").cost;
	assert!(cost.is_some_and(|cost| cost > 0.0), "{:?}", cost);
	assert_eq!(response.exchange.provider, "chatgpt");

	// The streamed output items replay in order, reasoning first.
	let mut assistant = Message::assistant("");
	assistant.tool_calls = Some(response.exchange.response["tool_calls"].clone());
	assistant.provider_items = Some(response.exchange.response["output_items"].clone());
	let input = build_input(&[assistant]);
	assert_eq!(input[0]["encrypted_content"], "enc");
	assert_eq!(input[1]["call_id"], "call_1");
	assert_eq!(input[1]["namespace"], TOOL_NAMESPACE);
}

#[tokio::test]
async fn failed_and_unfinished_streams_are_errors_and_only_transient_ones_are_retried() {
	let failed = r#"data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"weekly cap reached"}}}"#;
	let error = merge_stream(failed).await.unwrap_err();
	assert!(!error.is::<RetryableStreamError>());
	assert!(
		error.to_string().contains("subscription_sharing_usage_limit_exceeded"),
		"{}",
		error
	);

	let unfinished =
		r#"data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":[]}}"#;
	let error = merge_stream(unfinished).await.unwrap_err();
	assert!(error.is::<RetryableStreamError>());
	assert!(error.to_string().contains("without response.completed"), "{}", error);
}

#[tokio::test]
async fn server_faults_reported_mid_stream_are_retried() {
	// Verbatim shape of a plan-route `error` event observed in a live session.
	let stream_error = r#"data: {"type":"error","error":{"type":"server_error","code":"server_error","message":"An error occurred while processing your request. You can retry your request.","param":null},"sequence_number":7}"#;
	let error = merge_stream(stream_error).await.unwrap_err();
	assert!(error.is::<RetryableStreamError>());
	assert!(error.to_string().starts_with("ChatGPT stream error: "), "{}", error);
	assert!(error.to_string().contains("server_error"), "{}", error);

	for code in TRANSIENT_ERROR_CODES {
		let failed = format!(
			r#"data: {{"type":"response.failed","response":{{"error":{{"code":"{}","message":"try again"}}}}}}"#,
			code
		);
		let error = merge_stream(&failed).await.unwrap_err();
		assert!(error.is::<RetryableStreamError>(), "{}", code);
		assert!(error.to_string().starts_with("ChatGPT response failed: "), "{}", error);
	}
}

#[tokio::test]
async fn request_faults_reported_mid_stream_are_not_retried() {
	let stream_error = r#"data: {"type":"error","error":{"type":"invalid_request_error","code":"invalid_prompt","message":"Invalid prompt.","param":null},"sequence_number":3}"#;
	let error = merge_stream(stream_error).await.unwrap_err();
	assert!(!error.is::<RetryableStreamError>());
	assert!(error.to_string().contains("invalid_prompt"), "{}", error);

	let uncoded = r#"data: {"type":"error","error":{"message":"no code"},"sequence_number":1}"#;
	assert!(!merge_stream(uncoded).await.unwrap_err().is::<RetryableStreamError>());
}

#[tokio::test]
async fn plan_usage_errors_lead_with_guidance_and_keep_the_raw_error() {
	// Verbatim shape of the plan-route event observed when the app limit was hit.
	let limit = r#"data: {"type":"error","error":{"type":"invalid_request_error","code":"subscription_sharing_usage_limit_exceeded","message":"The ChatGPT user has reached their Subscription Sharing usage limit.","param":null},"sequence_number":2}"#;
	let error = merge_stream(limit).await.unwrap_err();
	assert!(!error.is::<RetryableStreamError>());
	let text = error.to_string();
	assert!(
		text.starts_with("ChatGPT plan usage limit reached for this app."),
		"{}",
		text
	);
	assert!(text.contains("ChatGPT settings → Usage"), "{}", text);
	assert!(text.contains("subscription_sharing_usage_limit_exceeded"), "{}", text);

	let unavailable = with_usage_guidance(
		Some("subscription_sharing_usage_unavailable"),
		"ChatGPT API error 503 Service Unavailable: {}".to_string(),
	);
	assert!(unavailable.to_string().starts_with("ChatGPT could not check"));
	assert!(unavailable.to_string().ends_with("503 Service Unavailable: {}"));

	let other = with_usage_guidance(Some("invalid_prompt"), "raw".to_string());
	assert_eq!(other.to_string(), "raw");
}

#[test]
fn tool_images_reach_the_chatgpt_plan_route() {
	let input = build_input(&[shared::tests::tool_image_message("caption")]);
	assert_eq!(input.len(), 1);
	assert_eq!(input[0]["type"], "function_call_output");
	assert_eq!(input[0]["call_id"], "call_image");
	assert_eq!(input[0]["output"][1]["image_url"], "data:image/png;base64,aW1hZ2U=");
}

#[cfg(feature = "websockets")]
#[test]
fn websocket_selection_is_explicit_and_http_can_be_selected_again() {
	let params = ChatCompletionParams::new(&[Message::user("hello")], "gpt-6.1-sol", 0.0, 1.0, 0, 0);
	assert_eq!(params.chatgpt_transport, crate::ChatGptTransport::Http);
	let http = build_request(&params);
	let params = params.with_chatgpt_transport(crate::ChatGptTransport::WebSocket);
	let ws = build_request(&params);
	assert_eq!(ws["type"], "response.create");
	assert!(ws.get("stream").is_none());
	for field in ["model", "input", "store", "include"] {
		assert_eq!(http[field], ws[field]);
	}
	let params = params.with_chatgpt_transport(crate::ChatGptTransport::Http);
	assert_eq!(build_request(&params), http);
}

async fn merge_stream(body: &str) -> Result<Value> {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, format!("{body}\n\n"))]);
	let captured = super::http::read_response(shared::http_client().get(&mock.url), None, None, None).await?;
	Ok(serde_json::from_str(&captured.body)?)
}
