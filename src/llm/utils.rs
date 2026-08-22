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

//! Utility functions for LLM providers
//!
//! This module provides case-insensitive string matching utilities optimized
//! for model name comparisons. All functions use ASCII-lowercase conversion
//! which is faster than full Unicode to_lowercase() since model names are
//! typically ASCII-only.

/// Convert a model name to lowercase for case-insensitive matching.
///
/// Uses ASCII-lowercase since model names are typically ASCII-only,
/// which is faster than full Unicode to_lowercase().
#[inline]
pub fn normalize_model_name(model: &str) -> String {
    model.to_ascii_lowercase()
}

/// Check if a model name starts with a given prefix (case-insensitive).
///
/// More efficient than creating a normalized string when checking a single prefix.
#[inline]
pub fn starts_with_ignore_ascii_case(model: &str, prefix: &str) -> bool {
    if model.len() < prefix.len() {
        return false;
    }
    model[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Check if a model name contains a given substring (case-insensitive).
///
/// Optimized to avoid double allocation by only converting the model name once.
#[inline]
pub fn contains_ignore_ascii_case(model: &str, substring: &str) -> bool {
    model
        .to_ascii_lowercase()
        .contains(&substring.to_ascii_lowercase())
}

/// Sanitize provider-specific model name formats into a canonical form
/// for matching against reference patterns.
///
/// Handles:
/// - Ollama format: `llama3.3:70b` → `llama-3.3-70b`
/// - HuggingFace/Together: `meta-llama/llama-3.3-70b-instruct` → strips org prefix irrelevant to matching
/// - Version dots without dashes: `qwen2.5` → `qwen-2.5`
pub(crate) fn sanitize_model_name(name: &str) -> String {
    let mut s = name.to_string();
    // Replace colons with dashes (Ollama uses `model:size`)
    s = s.replace(':', "-");
    // Insert dashes between letters and digits where missing (e.g., `llama3` → `llama-3`)
    let mut result = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        result.push(chars[i]);
        if i + 1 < chars.len() {
            let curr = chars[i];
            let next = chars[i + 1];
            // letter→digit or digit→letter boundary, but NOT around dots/dashes
            if (curr.is_ascii_alphabetic() && next.is_ascii_digit())
                || (curr.is_ascii_digit() && next.is_ascii_alphabetic())
            {
                // Only insert dash if there isn't already a separator
                if curr != '-' && curr != '.' && next != '-' && next != '.' {
                    result.push('-');
                }
            }
        }
    }
    result
}

fn is_model_in_pricing_names<'a, I>(model: &str, pricing_names: I) -> bool
where
    I: IntoIterator<Item = &'a str>,
{
    let normalized = normalize_model_name(model);
    pricing_names
        .into_iter()
        .any(|name| normalized.contains(&normalize_model_name(name)))
}

/// Pricing tuple: (model, input, output, cache_write, cache_read)
/// All prices are per 1M tokens in USD
pub type PricingTuple = (&'static str, f64, f64, f64, f64);

/// Check if a model is supported based on the pricing table.
///
/// For known providers (OpenAI, Anthropic, DeepSeek, Moonshot, MiniMax, Google, Zai),
/// we have a complete pricing table. If a model is not in the pricing table,
/// it's not supported by that provider.
pub fn is_model_in_pricing_table(model: &str, pricing: &[PricingTuple]) -> bool {
    is_model_in_pricing_names(model, pricing.iter().map(|(name, _, _, _, _)| *name))
}

/// Get pricing for a model from the pricing table.
/// Returns None if model not found.
pub fn get_model_pricing(model: &str, pricing: &[PricingTuple]) -> Option<(f64, f64, f64, f64)> {
    let normalized = normalize_model_name(model);
    pricing
        .iter()
        .find(|(name, _, _, _, _)| normalized.contains(&normalize_model_name(name)))
        .map(|(_, input, output, cache_write, cache_read)| {
            (*input, *output, *cache_write, *cache_read)
        })
}

/// Calculate cost using the pricing table.
pub fn calculate_cost_from_pricing_table(
    model: &str,
    pricing: &[PricingTuple],
    regular_input_tokens: u64,
    cache_write_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let (input, output, cache_write, cache_read) = get_model_pricing(model, pricing)?;

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input;
    let cache_write_cost = (cache_write_tokens as f64 / 1_000_000.0) * cache_write;
    let cache_read_cost = (cache_read_tokens as f64 / 1_000_000.0) * cache_read;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output;

    Some(regular_input_cost + cache_write_cost + cache_read_cost + output_cost)
}

/// Normalize a tool-parameter JSON schema to scalar `"type"` keywords.
///
/// Collapses `"type": [T, "null"]` to `T` and `anyOf: [X, {"type": "null"}]`
/// to `X`, merging the surviving branch into the parent so field-level keys
/// (description) win.
///
/// Serving stacks build their tool-call grammar from `"type"` and some read it
/// as a plain string: given the array form they fall through to the string
/// branch and constrain the model to emit that argument as text — `"0.6"`
/// instead of `0.6`, `"[\"a\"]"` instead of `["a"]` — which MCP servers then
/// reject while deserializing. Measured on Alibaba Model Studio's Qwen path and
/// on CoreWeave (via OpenRouter); the same Qwen weights on Parasail, Chutes and
/// DeepInfra, and glm-5.2 / deepseek-v4-pro on Alibaba's own endpoint, are
/// unaffected. Because the defect is per serving stack — not per provider, and
/// not visible from the model id when a gateway routes the request — this runs
/// for every provider.
///
/// Lossless for tool calling: optionality is carried by `required`, and no
/// provider sends tool definitions in strict mode (where the null branch would
/// be the idiom for an optional property).
pub fn normalize_tool_schema(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(obj) => {
            for nested in obj.values_mut() {
                normalize_tool_schema(nested);
            }

            let collapsed_type =
                obj.get_mut("type")
                    .and_then(|t| t.as_array_mut())
                    .and_then(|types| {
                        types.retain(|t| t.as_str() != Some("null"));
                        (types.len() == 1).then(|| types[0].clone())
                    });
            if let Some(single) = collapsed_type {
                obj.insert("type".to_string(), single);
            }

            for key in ["anyOf", "oneOf"] {
                let only = obj
                    .get_mut(key)
                    .and_then(|v| v.as_array_mut())
                    .and_then(|variants| {
                        variants.retain(|v| v.get("type").and_then(|t| t.as_str()) != Some("null"));
                        (variants.len() == 1).then(|| variants[0].clone())
                    });
                if let Some(serde_json::Value::Object(only)) = only {
                    obj.remove(key);
                    for (k, v) in only {
                        obj.entry(k).or_insert(v);
                    }
                }
            }
        }
        serde_json::Value::Array(items) => {
            for item in items.iter_mut() {
                normalize_tool_schema(item);
            }
        }
        _ => {}
    }
}

/// Normalize a JSON schema for OpenAI strict structured output.
///
/// OpenAI's strict mode requires `additionalProperties: false` on every
/// object in the schema. When `mode` is `ResponseMode::Strict` this
/// recursively walks the schema and injects the flag into any
/// `type: "object"` node missing it; otherwise the schema is returned
/// unchanged. Providers call this before sending the schema to
/// OpenAI-compatible endpoints (including OpenRouter, which forwards strict
/// mode to OpenAI/Azure). Existing `additionalProperties` values are
/// preserved.
pub fn normalize_strict_schema(
    schema: &serde_json::Value,
    mode: crate::llm::types::ResponseMode,
) -> serde_json::Value {
    if matches!(mode, crate::llm::types::ResponseMode::Strict) {
        inject_additional_properties(schema.clone())
    } else {
        schema.clone()
    }
}

/// Recursively inject `additionalProperties: false` into every object node.
fn inject_additional_properties(mut schema: serde_json::Value) -> serde_json::Value {
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };

    let is_object = obj
        .get("type")
        .and_then(|t| t.as_str())
        .map(|t| t == "object")
        .unwrap_or(false);

    if is_object && !obj.contains_key("additionalProperties") {
        obj.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(false),
        );
    }

    // Recurse into properties
    if let Some(properties) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        for child in properties.values_mut() {
            *child = inject_additional_properties(child.take());
        }
    }

    // Recurse into array items
    if let Some(items) = obj.get_mut("items") {
        *items = inject_additional_properties(items.take());
    }

    // Handle allOf/anyOf/oneOf composition
    for keyword in ["allOf", "anyOf", "oneOf"] {
        if let Some(arr) = obj.get_mut(keyword).and_then(|a| a.as_array_mut()) {
            for child in arr.iter_mut() {
                *child = inject_additional_properties(child.take());
            }
        }
    }

    schema
}

#[cfg(test)]
mod tests {
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
}
