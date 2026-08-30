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
fn test_normalize_model_name() {
    assert_eq!(normalize_model_name("GPT-4o"), "gpt-4o");
    assert_eq!(normalize_model_name("claude-3-haiku"), "claude-3-haiku");
    assert_eq!(normalize_model_name("MiniMax-M2.1"), "minimax-m2.1");
}

#[test]
fn test_normalize_model_name_edge_cases() {
    // Empty string
    assert_eq!(normalize_model_name(""), "");
    // Numbers and special chars
    assert_eq!(normalize_model_name("GPT-4.5-TURBO"), "gpt-4.5-turbo");
    assert_eq!(normalize_model_name("o1-preview"), "o1-preview");
    // Colons (common in Bedrock model IDs)
    assert_eq!(
        normalize_model_name("ANTHROPIC.CLAUDE-3-HAIKU-V1:0"),
        "anthropic.claude-3-haiku-v1:0"
    );
}

#[test]
fn test_starts_with_ignore_ascii_case() {
    assert!(starts_with_ignore_ascii_case("GPT-4o-mini", "gpt-4o"));
    assert!(starts_with_ignore_ascii_case("gpt-4o", "GPT-4O"));
    assert!(!starts_with_ignore_ascii_case("gpt-3.5", "gpt-4"));
    assert!(!starts_with_ignore_ascii_case("gpt", "gpt-4"));
}

#[test]
fn test_starts_with_ignore_ascii_case_edge_cases() {
    // Empty prefix
    assert!(starts_with_ignore_ascii_case("gpt-4", ""));
    // Prefix longer than model
    assert!(!starts_with_ignore_ascii_case("gpt", "gpt-4o-mini"));
    // Exact match
    assert!(starts_with_ignore_ascii_case("GPT-4O", "GPT-4O"));
}

#[test]
fn test_contains_ignore_ascii_case() {
    assert!(contains_ignore_ascii_case(
        "anthropic.claude-3-haiku-v1:0",
        "claude"
    ));
    assert!(contains_ignore_ascii_case("CLAUDE-3", "claude"));
    assert!(!contains_ignore_ascii_case("gpt-4o", "claude"));
}

#[test]
fn test_contains_ignore_ascii_case_edge_cases() {
    // Empty substring
    assert!(contains_ignore_ascii_case("gpt-4o", ""));
    // Empty model
    assert!(!contains_ignore_ascii_case("", "gpt"));
    // Both empty
    assert!(contains_ignore_ascii_case("", ""));
}

#[test]
fn test_sanitize_model_name() {
    assert_eq!(sanitize_model_name("llama3.3:70b"), "llama-3.3-70-b");
    assert_eq!(sanitize_model_name("qwen2.5-72b"), "qwen-2.5-72-b");
    assert_eq!(
        sanitize_model_name("meta-llama/llama-3.3-70b-instruct"),
        "meta-llama/llama-3.3-70-b-instruct"
    );
    assert_eq!(sanitize_model_name("phi4"), "phi-4");
    assert_eq!(sanitize_model_name("deepseek-r1"), "deepseek-r-1");
}

#[test]
fn test_is_model_in_pricing_table() {
    let pricing: &[PricingTuple] = &[
        ("gpt-4o", 2.50, 10.0, 2.50, 1.25),
        ("gpt-4o-mini", 0.15, 0.60, 0.15, 0.075),
    ];
    assert!(is_model_in_pricing_table("gpt-4o", pricing));
    assert!(is_model_in_pricing_table("GPT-4O", pricing));
    assert!(is_model_in_pricing_table("gpt-4o-mini", pricing));
    assert!(!is_model_in_pricing_table("gpt-5", pricing));
    assert!(!is_model_in_pricing_table("unknown-model", pricing));
}

#[test]
fn test_normalize_strict_schema_adds_additional_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["file_id", "description"]
                }
            }
        },
        "required": ["descriptions"]
    });

    let normalized = normalize_strict_schema(&schema, crate::llm::types::ResponseMode::Strict);
    let obj = normalized.as_object().unwrap();
    assert_eq!(
        obj.get("additionalProperties"),
        Some(&serde_json::Value::Bool(false))
    );

    let items = obj["properties"]["descriptions"]["items"]
        .as_object()
        .unwrap();
    assert_eq!(
        items.get("additionalProperties"),
        Some(&serde_json::Value::Bool(false))
    );
}

#[test]
fn test_normalize_strict_schema_preserves_existing_additional_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "additionalProperties": true,
        "properties": {
            "name": {"type": "string"}
        }
    });

    let normalized = normalize_strict_schema(&schema, crate::llm::types::ResponseMode::Strict);
    let obj = normalized.as_object().unwrap();
    assert_eq!(
        obj.get("additionalProperties"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn test_normalize_strict_schema_skips_non_object() {
    let schema = serde_json::json!({"type": "string"});
    let normalized = normalize_strict_schema(&schema, crate::llm::types::ResponseMode::Strict);
    assert_eq!(normalized, serde_json::json!({"type": "string"}));
}

#[test]
fn test_normalize_strict_schema_noop_when_not_strict() {
    // Non-strict modes must pass the schema through untouched.
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"name": {"type": "string"}}
    });
    let normalized = normalize_strict_schema(&schema, crate::llm::types::ResponseMode::Auto);
    assert_eq!(normalized, schema);
}

#[test]
fn test_normalize_tool_schema_collapses_nullable_types() {
    // Shape schemars emits for Option<f32> / Option<Vec<String>> /
    // Option<enum> — what octobrain advertises for `memorize`.
    let mut schema = serde_json::json!({
        "type": "object",
        "required": ["title"],
        "properties": {
            "title": {"type": "string"},
            "importance": {"type": ["number", "null"], "format": "float"},
            "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            "memory_type": {
                "anyOf": [
                    {"type": "string", "enum": ["code", "decision"]},
                    {"type": "null"}
                ],
                "description": "Memory category"
            }
        }
    });

    crate::llm::utils::normalize_tool_schema(&mut schema);

    let props = &schema["properties"];
    assert_eq!(props["importance"]["type"], "number");
    assert_eq!(props["importance"]["format"], "float");
    assert_eq!(props["tags"]["type"], "array");
    assert_eq!(props["tags"]["items"]["type"], "string");
    // anyOf collapses into the parent, field description preserved
    assert!(props["memory_type"].get("anyOf").is_none());
    assert_eq!(props["memory_type"]["type"], "string");
    assert_eq!(props["memory_type"]["enum"][1], "decision");
    assert_eq!(props["memory_type"]["description"], "Memory category");
    assert_eq!(schema["type"], "object");
    assert_eq!(props["title"]["type"], "string");
}

#[test]
fn test_normalize_tool_schema_keeps_real_unions() {
    // A genuine multi-branch union has nothing to collapse to and is left
    // as authored (octobrain's `remember.query`: string or array).
    let mut schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array"}]}
        }
    });
    let before = schema.clone();
    crate::llm::utils::normalize_tool_schema(&mut schema);
    assert_eq!(schema, before);
}
