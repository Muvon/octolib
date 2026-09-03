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
fn tool_choice_is_explicitly_mapped() {
    assert_eq!(
        openai_tool_choice_value(Some(&ToolChoice::Required)),
        serde_json::json!("required")
    );
    assert_eq!(
        openai_tool_choice_value(Some(&ToolChoice::Function("emit".to_string()))),
        serde_json::json!({"type": "function", "function": {"name": "emit"}})
    );
    assert_eq!(openai_tool_choice_value(None), serde_json::json!("auto"));
}

#[test]
fn test_gemini_thought_signature_round_trip() {
    // Response side: extra_content survives deserialization
    let call: OpenAiCompatToolCall = serde_json::from_value(serde_json::json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "sig123"}}
    }))
    .unwrap();
    assert!(call.extra_content.is_some());

    // Replay side: meta.extra_content is emitted back on the tool call
    let tool_calls_json = serde_json::to_value(vec![crate::llm::tool_calls::GenericToolCall {
        id: "call_1".to_string(),
        name: "search".to_string(),
        arguments: serde_json::json!({}),
        meta: serde_json::json!({
            "extra_content": {"google": {"thought_signature": "sig123"}}
        })
        .as_object()
        .cloned(),
    }])
    .unwrap();

    let msg = Message {
        role: "assistant".to_string(),
        content: String::new(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(tool_calls_json),
        images: None,
        videos: None,
        thinking: None,
        id: None,
    };
    let converted = convert_messages(std::slice::from_ref(&msg), "local", "test-model");
    let json = serde_json::to_value(&converted[0]).unwrap();
    assert_eq!(
        json["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
        "sig123"
    );

    // Tool calls without meta must not gain an extra_content field
    let plain = serde_json::to_value(vec![crate::llm::tool_calls::GenericToolCall {
        id: "call_2".to_string(),
        name: "search".to_string(),
        arguments: serde_json::json!({}),
        meta: None,
    }])
    .unwrap();
    let msg_plain = Message {
        tool_calls: Some(plain),
        ..msg
    };
    let converted = convert_messages(std::slice::from_ref(&msg_plain), "local", "test-model");
    let json = serde_json::to_value(&converted[0]).unwrap();
    assert!(json["tool_calls"][0].get("extra_content").is_none());
}

#[test]
fn test_ollama_reasoning_is_stored_and_replayed() {
    let response_message: OpenAiCompatResponseMessage = serde_json::from_value(serde_json::json!({
        "content": "I need the repository state.",
        "reasoning": "The first step is to inspect git status.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "git_status", "arguments": "{}"}
        }]
    }))
    .unwrap();

    let thinking = extract_thinking(&response_message).expect("reasoning must be stored");
    assert_eq!(thinking.content, "The first step is to inspect git status.");

    let tool_calls = serde_json::to_value(vec![crate::llm::tool_calls::GenericToolCall {
        id: "call_1".to_string(),
        name: "git_status".to_string(),
        arguments: serde_json::json!({}),
        meta: None,
    }])
    .unwrap();
    let history = Message {
        role: "assistant".to_string(),
        content: "I need the repository state.".to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(tool_calls),
        images: None,
        videos: None,
        thinking: Some(thinking),
        id: None,
    };

    let converted = convert_messages(&[history], "ollama", "glm-5.2");
    let json = serde_json::to_value(&converted[0]).unwrap();
    assert_eq!(
        json.get("reasoning").and_then(serde_json::Value::as_str),
        Some("The first step is to inspect git status.")
    );
}

#[test]
fn test_ollama_reasoning_replays_only_on_last_assistant() {
    let assistant = |text: &str| Message {
        role: "assistant".to_string(),
        content: text.to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: None,
        images: None,
        videos: None,
        thinking: Some(ThinkingBlock {
            content: format!("thinking for {text}"),
            tokens: 4,
        }),
        id: None,
    };
    let messages = [assistant("one"), assistant("two")];

    let converted = convert_messages(&messages, "ollama", "glm-5.2");
    assert!(converted[0].reasoning.is_none());
    assert_eq!(converted[1].reasoning.as_deref(), Some("thinking for two"));

    // Kimi preserve-thinking models keep reasoning on every assistant turn.
    let converted = convert_messages(&messages, "ollama", "kimi-k2.7-code");
    assert_eq!(converted[0].reasoning.as_deref(), Some("thinking for one"));
    assert_eq!(converted[1].reasoning.as_deref(), Some("thinking for two"));
}

#[test]
fn test_openai_compat_reasoning_field_variants_are_stored() {
    for (field, value) in [
        ("reasoning_content", "reasoning-content trace"),
        ("reasoning", "reasoning trace"),
        ("thinking", "native Ollama trace"),
    ] {
        let mut json = serde_json::json!({"content": "answer"});
        json[field] = serde_json::json!(value);
        let message: OpenAiCompatResponseMessage = serde_json::from_value(json).unwrap();
        assert_eq!(
            extract_thinking(&message).map(|thinking| thinking.content),
            Some(value.to_string())
        );
    }
}

/// Model Studio ignores historical `reasoning_content` for DeepSeek (its
/// `preserve_thinking` mechanism covers only Qwen/Kimi), so no assistant turn
/// carries reasoning fields on the Alibaba route.
#[test]
fn test_alibaba_deepseek_sends_no_reasoning_fields() {
    let assistant = |text: &str| {
        Message::assistant(text).with_thinking(ThinkingBlock {
            content: format!("thinking for {text}"),
            tokens: 4,
        })
    };
    let messages = [assistant("one"), assistant("two")];

    let converted = convert_messages(&messages, "alibaba", "deepseek-v4-flash-0731");
    assert!(converted.iter().all(|message| message.reasoning.is_none()));
}

#[test]
fn test_alibaba_deepseek_uses_json_object_guidance_not_json_schema() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"]
    });
    let response_format = crate::llm::types::StructuredOutputRequest::json_schema(schema.clone());
    let mut request_body = serde_json::json!({
        "messages": [{"role": "user", "content": "answer"}]
    });

    apply_response_format(
        &mut request_body,
        "alibaba",
        "deepseek-v4-flash-0731",
        &response_format,
    );

    assert_eq!(
        request_body["response_format"],
        serde_json::json!({"type": "json_object"})
    );
    assert!(request_body["response_format"].get("json_schema").is_none());
    let guidance = request_body["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .expect("schema guidance message");
    assert!(guidance.contains("JSON schema"));
    assert!(guidance.contains(&schema.to_string()));
}

/// Model Studio rejects json_object requests unless the word "JSON" appears
/// in the messages, so the Alibaba route injects a guidance line.
#[test]
fn test_alibaba_json_mode_injects_json_keyword() {
    let response_format = crate::llm::types::StructuredOutputRequest::json();
    let mut request_body = serde_json::json!({
        "messages": [{"role": "user", "content": "answer"}]
    });

    apply_response_format(
        &mut request_body,
        "alibaba",
        "qwen3.8-flash",
        &response_format,
    );

    assert_eq!(
        request_body["response_format"],
        serde_json::json!({"type": "json_object"})
    );
    let guidance = request_body["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_str())
        .expect("JSON guidance message");
    assert!(guidance.contains("JSON"));

    // Other providers keep the request untouched.
    let mut other_body = serde_json::json!({
        "messages": [{"role": "user", "content": "answer"}]
    });
    apply_response_format(&mut other_body, "groq", "some-model", &response_format);
    assert_eq!(other_body["messages"].as_array().map(Vec::len), Some(1));
}

#[test]
fn test_alibaba_deepseek_reasoning_effort_preserves_supported_values() {
    use crate::llm::types::ReasoningEffort;

    // Model Studio accepts the standard values and performs each model's
    // documented equivalence mapping itself. Preserve caller intent verbatim.
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash", ReasoningEffort::Low),
        "low"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash", ReasoningEffort::Medium),
        "medium"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash", ReasoningEffort::XHigh),
        "xhigh"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-pro", ReasoningEffort::Max),
        "max"
    );

    // Snapshot aliases receive the same accepted values; Model Studio owns any
    // snapshot-specific equivalence between them.
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash-0731", ReasoningEffort::Low),
        "low"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash-0731", ReasoningEffort::Medium),
        "medium"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash-0731", ReasoningEffort::XHigh),
        "xhigh"
    );
    assert_eq!(
        reasoning_effort_value("alibaba", "deepseek-v4-flash-0731", ReasoningEffort::Max),
        "max"
    );
}

#[test]
fn test_ollama_max_reasoning_effort_is_not_downgraded() {
    assert_eq!(
        reasoning_effort_value("ollama", "qwen3", crate::llm::types::ReasoningEffort::Max),
        "max"
    );
    assert_eq!(
        reasoning_effort_value(
            "nvidia",
            "nemotron",
            crate::llm::types::ReasoningEffort::Max
        ),
        "high"
    );
}

#[test]
fn test_opencode_kimi_k3_max_reasoning_effort_is_kept() {
    // OpenCode forwards verbatim to Moonshot: kimi-k3's top tier is "max"
    assert_eq!(
        reasoning_effort_value(
            "opencode-go",
            "kimi-k3",
            crate::llm::types::ReasoningEffort::Max
        ),
        "max"
    );
    assert_eq!(
        reasoning_effort_value(
            "opencode-zen",
            "kimi-k3",
            crate::llm::types::ReasoningEffort::Max
        ),
        "max"
    );
    // Non-Kimi models through OpenCode keep the generic downgrade
    assert_eq!(
        reasoning_effort_value(
            "opencode-go",
            "gpt-5.5",
            crate::llm::types::ReasoningEffort::Max
        ),
        "high"
    );
}

#[test]
fn test_meta_reasoning_effort_xhigh_ceiling() {
    // Muse Spark accepts minimal..xhigh ("none" is a 400): XHigh maps
    // through and Max collapses onto the same xhigh ceiling.
    assert_eq!(
        reasoning_effort_value(
            "meta",
            "muse-spark-1.3",
            crate::llm::types::ReasoningEffort::Low
        ),
        "low"
    );
    assert_eq!(
        reasoning_effort_value(
            "meta",
            "muse-spark-1.3",
            crate::llm::types::ReasoningEffort::XHigh
        ),
        "xhigh"
    );
    assert_eq!(
        reasoning_effort_value(
            "meta",
            "muse-spark-1.3",
            crate::llm::types::ReasoningEffort::Max
        ),
        "xhigh"
    );
}
