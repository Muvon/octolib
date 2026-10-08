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
fn test_maybe_ephemeral_cache_control() {
    assert!(maybe_ephemeral_cache_control(false).is_none());
    assert_eq!(
        maybe_ephemeral_cache_control(true),
        Some(serde_json::json!({"type": "ephemeral"}))
    );
}

/// GLM-5.3 has no off level on Model Studio or Z.AI; every other model and host
/// keeps the requested effort.
#[test]
fn test_supported_reasoning_effort_floors_only_glm_5_3_on_verified_hosts() {
    let off = ReasoningEffort::None;
    assert_eq!(
        supported_reasoning_effort("alibaba", "glm-5.3", off),
        ReasoningEffort::Low
    );
    assert_eq!(
        supported_reasoning_effort("zai", "GLM-5.3-Flash", off),
        ReasoningEffort::Low
    );
    assert_eq!(
        supported_reasoning_effort("alibaba", "glm-5.3", ReasoningEffort::High),
        ReasoningEffort::High
    );
    assert_eq!(supported_reasoning_effort("alibaba", "glm-5.2", off), off);
    assert_eq!(
        supported_reasoning_effort("together", "zai-org/glm-5.3", off),
        off
    );
}

#[test]
fn test_parse_generic_tool_calls_lossy() {
    let calls = serde_json::json!([{
        "id": "call_1",
        "name": "lookup",
        "arguments": { "q": "rust" },
        "meta": null
    }]);
    assert_eq!(
        parse_generic_tool_calls_lossy(Some(&calls), "test").len(),
        1
    );
    assert!(
        parse_generic_tool_calls_lossy(Some(&serde_json::json!({"bad": true})), "test").is_empty()
    );
}

#[test]
fn test_parse_generic_tool_calls_strict() {
    let calls = serde_json::json!([{
        "id": "call_1",
        "name": "lookup",
        "arguments": { "q": "rust" },
        "meta": null
    }]);
    assert!(parse_generic_tool_calls_strict(&calls, "test").is_ok());
    assert!(parse_generic_tool_calls_strict(&serde_json::json!({"bad": true}), "test").is_err());
}

#[test]
fn test_set_response_tool_calls() {
    let calls = vec![ToolCall {
        id: "call_1".to_string(),
        name: "lookup".to_string(),
        arguments: serde_json::json!({"q": "rust"}),
    }];
    let mut response = serde_json::json!({});
    set_response_tool_calls(&mut response, &calls, None);
    assert!(response.get("tool_calls").is_some());
}

#[test]
fn test_parse_structured_output_from_text() {
    assert!(parse_structured_output_from_text("{\"x\":1}").is_some());
    assert!(parse_structured_output_from_text("[1,2]").is_some());
    assert_eq!(
        parse_structured_output_from_text("```json\n{\"x\":1}\n```"),
        Some(serde_json::json!({"x": 1}))
    );
    assert_eq!(
        parse_structured_output_from_text("```\n[1,2]\n```"),
        Some(serde_json::json!([1, 2]))
    );
    assert!(parse_structured_output_from_text("not json").is_none());
    assert!(parse_structured_output_from_text("{not-json").is_none());
    assert!(parse_structured_output_from_text("before\n```json\n{}\n```").is_none());
    assert!(parse_structured_output_from_text("```rust\n{}\n```").is_none());
    assert!(parse_structured_output_from_text("```json\n{}").is_none());
}

#[test]
fn test_apply_extra_headers_upserts_and_preserves() {
    let mut extra = std::collections::HashMap::new();
    extra.insert("X-Model-Purpose".to_string(), "compression".to_string());
    extra.insert("Authorization".to_string(), "Bearer override".to_string());
    extra.insert("bad name!".to_string(), "ignored".to_string());

    let builder = http_client()
        .post("http://localhost/never-sent")
        .header("Authorization", "Bearer original")
        .header("Content-Type", "application/json");
    let req = apply_extra_headers(builder, Some(&extra)).build().unwrap();

    // Override wins, without duplicating the header.
    let auth: Vec<_> = req.headers().get_all("Authorization").iter().collect();
    assert_eq!(auth.len(), 1);
    assert_eq!(auth[0], "Bearer override");
    // New name lands; provider-set names not in the map survive.
    assert_eq!(req.headers()["X-Model-Purpose"], "compression");
    assert_eq!(req.headers()["Content-Type"], "application/json");
    // Invalid names are skipped, not fatal.
    assert!(req.headers().get("bad name!").is_none());

    // None / empty map are no-ops.
    let untouched = apply_extra_headers(
        http_client().post("http://localhost/x").header("A", "1"),
        None,
    )
    .build()
    .unwrap();
    assert_eq!(untouched.headers()["A"], "1");
}

#[test]
fn test_parse_tool_call_arguments_lossy() {
    assert_eq!(
        parse_tool_call_arguments_lossy("{\"a\":1}"),
        serde_json::json!({"a": 1})
    );
    assert_eq!(
        parse_tool_call_arguments_lossy("{invalid"),
        serde_json::json!({"raw_arguments": "{invalid"})
    );
}

// Central fixture keeps provider contract regressions on the same image payload.
pub(crate) fn tool_image_message(text: &str) -> crate::llm::types::Message {
    use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};
    Message::tool(text, "call_image", "screenshot").with_images(vec![
        ImageAttachment {
            data: ImageData::Base64("aW1hZ2U=".into()),
            media_type: "image/png".into(),
            source_type: SourceType::Url,
            dimensions: None,
            size_bytes: None,
        },
        ImageAttachment {
            data: ImageData::Url("https://example.com/screenshot.jpg".into()),
            media_type: "image/jpeg".into(),
            source_type: SourceType::Url,
            dimensions: None,
            size_bytes: None,
        },
    ])
}

#[test]
fn chat_images_follow_all_parallel_tool_results_and_keep_attribution() {
    use crate::llm::types::Message;
    let messages = vec![
        Message::assistant(""),
        tool_image_message("caption"),
        Message::tool("second result", "call_second", "view"),
        Message::user("continue"),
    ];
    let converted = chat_completion_messages(&messages);
    assert_eq!(converted.len(), 5);
    assert_eq!(converted[1].tool_call_id.as_deref(), Some("call_image"));
    assert!(converted[1].images.is_none());
    assert_eq!(converted[2].tool_call_id.as_deref(), Some("call_second"));
    assert_eq!(converted[3].role, "user");
    assert!(converted[3].content.contains("screenshot"));
    assert!(converted[3].content.contains("call_image"));
    assert_eq!(converted[3].images.as_ref().unwrap().len(), 2);
    assert_eq!(converted[4].content, "continue");
    assert_eq!(messages[1].images.as_ref().unwrap().len(), 2);
    let text_only = [Message::tool("text", "call_text", "view")];
    assert!(matches!(
        chat_completion_messages(&text_only),
        std::borrow::Cow::Borrowed(_)
    ));
}

#[test]
fn anthropic_tool_blocks_preserve_images_and_text_only_shape() {
    use crate::llm::types::Message;
    assert_eq!(
        anthropic_tool_content(&Message::tool("text", "id", "view")),
        serde_json::json!("text")
    );
    for text in ["caption", ""] {
        let blocks = anthropic_tool_content(&tool_image_message(text));
        let offset = usize::from(!text.is_empty());
        assert_eq!(blocks.as_array().unwrap().len(), offset + 2);
        assert_eq!(blocks[offset]["source"]["type"], "base64");
        assert_eq!(blocks[offset]["source"]["media_type"], "image/png");
        assert_eq!(blocks[offset]["source"]["data"], "aW1hZ2U=");
        assert_eq!(blocks[offset + 1]["source"]["type"], "url");
        assert_eq!(
            blocks[offset + 1]["source"]["url"],
            "https://example.com/screenshot.jpg"
        );
    }
}
