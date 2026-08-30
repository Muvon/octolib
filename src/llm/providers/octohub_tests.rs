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
fn test_provider_name() {
    let provider = OctoHubProvider::new();
    assert_eq!(provider.name(), "octohub");
}

#[test]
fn test_supports_any_model() {
    let provider = OctoHubProvider::new();
    assert!(provider.supports_model("gpt-4o"));
    assert!(provider.supports_model("claude-sonnet-4-20250514"));
    assert!(provider.supports_model("any-model-name"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_capabilities() {
    let provider = OctoHubProvider::new();
    assert!(provider.supports_caching("any"));
    assert!(provider.supports_vision("any"));
    assert!(provider.supports_video("any"));
    assert!(provider.supports_structured_output("any"));
    assert!(provider.enforces_response_schema("unknown-model"));
    assert!(provider.enforces_response_schema("deepseek-v4-pro"));
    assert!(!provider.enforces_response_schema("mistral-7b"));
    assert_eq!(provider.get_max_input_tokens("any"), 1_048_576);
}

#[test]
fn test_extract_instructions_single() {
    let messages = vec![Message::system("You are helpful."), Message::user("Hello")];
    let instr = extract_instructions(&messages).unwrap();
    assert_eq!(instr, serde_json::json!("You are helpful."));
}

#[test]
fn test_extract_instructions_none() {
    let messages = vec![Message::user("Hello")];
    assert_eq!(extract_instructions(&messages), None);
}

#[test]
fn test_extract_instructions_cached() {
    let messages = vec![
        Message::system("You are helpful.").with_cache_marker(),
        Message::user("Hello"),
    ];
    let instr = extract_instructions(&messages).unwrap();
    let arr = instr.as_array().expect("should be array when cached");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["type"], "text");
    assert_eq!(arr[0]["text"], "You are helpful.");
    assert_eq!(arr[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_user_message_value_plain() {
    let msg = Message::user("Hello");
    let val = user_message_value(&msg);
    assert_eq!(val["content"], "Hello");
}

#[test]
fn test_user_message_value_cached() {
    let msg = Message::user("Hello").with_cache_marker();
    let val = user_message_value(&msg);
    let content = val["content"].as_array().expect("should be array");
    assert_eq!(content.len(), 1);
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["text"], "Hello");
    assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_messages_to_input_initial() {
    let messages = vec![Message::system("You are helpful."), Message::user("Hello!")];

    let input = messages_to_input(&messages, None);
    // System messages go to instructions, not input
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"], "Hello!");
}

#[test]
fn test_messages_to_input_continuation_user() {
    let mut assistant = Message::assistant("Rust is a systems language.");
    assistant.id = Some("resp_abc".to_string());
    let messages = vec![
        Message::user("What is Rust?"),
        assistant,
        Message::user("Tell me more."),
    ];

    let input = messages_to_input(&messages, Some("resp_abc"));
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"], "Tell me more.");
}

#[test]
fn test_messages_to_input_tool_results() {
    let mut assistant_msg = Message::assistant("");
    assistant_msg.tool_calls = Some(serde_json::json!([{
        "id": "call_xyz",
        "name": "get_weather",
        "arguments": {"location": "NYC"}
    }]));
    assistant_msg.id = Some("resp_123".to_string());
    let messages = vec![
        Message::user("What is the weather?"),
        assistant_msg,
        Message::tool("72°F sunny", "call_xyz", "get_weather"),
    ];

    let input = messages_to_input(&messages, Some("resp_123"));
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["type"], "function_call_output");
    assert_eq!(input[0]["call_id"], "call_xyz");
    assert_eq!(input[0]["output"], "72°F sunny");
}

#[test]
fn test_messages_to_input_rebased_tool_call_and_result() {
    let mut assistant_msg = Message::assistant("");
    assistant_msg.tool_calls = Some(serde_json::json!([{
        "id": "call_xyz",
        "name": "get_weather",
        "arguments": {"location": "NYC"}
    }]));
    let messages = vec![
        Message::user("What is the weather?"),
        assistant_msg,
        Message::tool("72°F sunny", "call_xyz", "get_weather"),
    ];

    let input = messages_to_input(&messages, None);
    assert_eq!(input.len(), 3);
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["type"], "function_call");
    assert_eq!(input[1]["call_id"], "call_xyz");
    assert_eq!(input[1]["name"], "get_weather");
    assert_eq!(input[1]["arguments"], r#"{"location":"NYC"}"#);
    assert_eq!(input[2]["type"], "function_call_output");
    assert_eq!(input[2]["call_id"], "call_xyz");
    assert_eq!(input[2]["output"], "72°F sunny");
}

/// Regression: after a multi-turn cancel that leaves a tool_result without
/// a follow-up assistant response, the next user message must be sent
/// alongside the tool_result — not dropped. Previously this case returned
/// only the tool_result and the user's question was silently lost,
/// causing the next assistant turn to reply to the tool result and the
/// model to drift "off-track" for the rest of the conversation.
#[test]
fn test_messages_to_input_tool_result_then_user_after_cancel() {
    let mut assistant_msg = Message::assistant("");
    assistant_msg.tool_calls = Some(serde_json::json!([{
        "id": "call_xyz",
        "name": "get_weather",
        "arguments": {"location": "NYC"}
    }]));
    assistant_msg.id = Some("resp_123".to_string());
    let messages = vec![
        Message::user("What is the weather?"),
        assistant_msg,
        Message::tool("72°F sunny", "call_xyz", "get_weather"),
        Message::user("Now write me a poem about it."),
    ];

    let input = messages_to_input(&messages, Some("resp_123"));
    assert_eq!(
        input.len(),
        2,
        "tool_result and follow-up user must both be sent"
    );
    assert_eq!(input[0]["type"], "function_call_output");
    assert_eq!(input[0]["call_id"], "call_xyz");
    assert_eq!(input[0]["output"], "72°F sunny");
    assert_eq!(input[1]["type"], "message");
    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[1]["content"], "Now write me a poem about it.");
}

#[test]
fn test_parse_response() {
    let json = r#"{
        "id": "resp_abc123",
        "object": "response",
        "model": "gpt-4o",
        "output": [
            {
                "type": "message",
                "id": "msg_001",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello!"}
                ]
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0001,
            "request_time_ms": 500
        },
        "created_at": 1700000000
    }"#;

    let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.id, Some("resp_abc123".to_string()));
    assert_eq!(resp.output.len(), 1);
    assert_eq!(resp.output[0].output_type, "message");
    assert_eq!(resp.usage.input_tokens, 10);
    assert_eq!(resp.usage.output_tokens, 5);
    assert_eq!(resp.usage.cost, Some(0.0001));
    assert_eq!(resp.usage.request_time_ms, Some(500));
}

#[test]
fn test_parse_function_call_response() {
    let json = r#"{
        "id": "resp_xyz",
        "output": [
            {
                "type": "function_call",
                "id": "fc_001",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": "{\"location\":\"NYC\"}"
            }
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30
        }
    }"#;

    let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.output.len(), 1);
    assert_eq!(resp.output[0].output_type, "function_call");
    assert_eq!(resp.output[0].name, Some("get_weather".to_string()));
    assert_eq!(resp.output[0].call_id, Some("call_abc".to_string()));
}

#[test]
fn test_parse_usage_with_cache() {
    let json = r#"{
        "id": "resp_cache",
        "output": [],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "cache_read_tokens": 80,
            "cache_write_tokens": 20,
            "cost": 0.005,
            "request_time_ms": 200
        }
    }"#;

    let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.usage.cache_read_tokens, Some(80));
    assert_eq!(resp.usage.cache_write_tokens, Some(20));
    assert_eq!(resp.usage.cost, Some(0.005));
}
