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
    let mut params = ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000)
        .with_long_cache(true);
    params.previous_id = Some("resp_1".to_string());
    params.tools = Some(vec![FunctionDefinition {
        name: "view".to_string(),
        description: "Read a file".to_string(),
        parameters: json!({"type": "object"}),
        cache_control: None,
    }]);

    let request = build_request(&params);

    assert_eq!(request["store"], false);
    assert_eq!(request["stream"], true);
    assert_eq!(request["include"], json!(["reasoning.encrypted_content"]));
    for rejected in [
        "previous_response_id",
        "max_output_tokens",
        "temperature",
        "top_p",
        "prompt_cache_retention",
    ] {
        assert!(
            request.get(rejected).is_none(),
            "{} must not be sent",
            rejected
        );
    }
    assert_eq!(request["input"][0]["role"], "developer");
    assert_eq!(request["input"][1]["role"], "user");
    assert_eq!(request["tools"][0]["type"], "namespace");
    assert_eq!(request["tools"][0]["name"], TOOL_NAMESPACE);
    assert_eq!(request["tools"][0]["tools"][0]["name"], "view");
}

#[test]
fn replay_puts_encrypted_reasoning_before_namespaced_calls() {
    let reasoning = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "enc"
    });
    let meta = serde_json::Map::from_iter([(
        openai::REASONING_META_KEY.to_string(),
        json!([reasoning.clone()]),
    )]);
    let messages = vec![
        Message::user("go"),
        assistant_with_call(Some(meta)),
        Message::tool("ok", "call_1", "view"),
    ];

    let input = build_input(&messages);

    assert_eq!(input.len(), 4);
    assert_eq!(input[1], reasoning);
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["namespace"], TOOL_NAMESPACE);
    assert_eq!(input[3]["type"], "function_call_output");
}

#[test]
fn stream_folds_into_a_response_that_keeps_reasoning_for_replay() {
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

    let merged = merge_stream(&body).unwrap();
    assert_eq!(merged["output"].as_array().unwrap().len(), 2);

    let response = openai::parse_responses_api_response(
        json!({"model": "gpt-6.1-sol"}),
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

    // Stored on the tool call, the reasoning is replayed by the next request.
    let mut assistant = Message::assistant("");
    assistant.tool_calls = Some(response.exchange.response["tool_calls"].clone());
    assert_eq!(
        stored_reasoning_items(&assistant)[0]["encrypted_content"],
        "enc"
    );
}

#[test]
fn failed_and_unfinished_streams_are_errors_and_only_transient_ones_are_retried() {
    let failed = r#"data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"weekly cap reached"}}}"#;
    let error = merge_stream(failed).unwrap_err();
    assert!(!error.is::<RetryableStreamError>());
    assert!(
        error
            .to_string()
            .contains("subscription_sharing_usage_limit_exceeded"),
        "{}",
        error
    );

    let unfinished =
        r#"data: {"type":"response.output_item.done","item":{"type":"message","content":[]}}"#;
    let error = merge_stream(unfinished).unwrap_err();
    assert!(error.is::<RetryableStreamError>());
    assert!(
        error.to_string().contains("without response.completed"),
        "{}",
        error
    );
}

#[test]
fn server_faults_reported_mid_stream_are_retried() {
    // Verbatim shape of a plan-route `error` event observed in a live session.
    let stream_error = r#"data: {"type":"error","error":{"type":"server_error","code":"server_error","message":"An error occurred while processing your request. You can retry your request.","param":null},"sequence_number":7}"#;
    let error = merge_stream(stream_error).unwrap_err();
    assert!(error.is::<RetryableStreamError>());
    assert!(
        error.to_string().starts_with("ChatGPT stream error: "),
        "{}",
        error
    );
    assert!(error.to_string().contains("server_error"), "{}", error);

    for code in TRANSIENT_ERROR_CODES {
        let failed = format!(
            r#"data: {{"type":"response.failed","response":{{"error":{{"code":"{}","message":"try again"}}}}}}"#,
            code
        );
        let error = merge_stream(&failed).unwrap_err();
        assert!(error.is::<RetryableStreamError>(), "{}", code);
        assert!(
            error.to_string().starts_with("ChatGPT response failed: "),
            "{}",
            error
        );
    }
}

#[test]
fn request_faults_reported_mid_stream_are_not_retried() {
    let stream_error = r#"data: {"type":"error","error":{"type":"invalid_request_error","code":"invalid_prompt","message":"Invalid prompt.","param":null},"sequence_number":3}"#;
    let error = merge_stream(stream_error).unwrap_err();
    assert!(!error.is::<RetryableStreamError>());
    assert!(error.to_string().contains("invalid_prompt"), "{}", error);

    let uncoded = r#"data: {"type":"error","error":{"message":"no code"},"sequence_number":1}"#;
    assert!(!merge_stream(uncoded)
        .unwrap_err()
        .is::<RetryableStreamError>());
}
