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
