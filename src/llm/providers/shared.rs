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

//! Shared helpers used by multiple provider adapters.

use crate::errors::ToolCallError;
use crate::llm::tool_calls::GenericToolCall;
use crate::llm::types::ToolCall;

const MAX_JSON_INPUT_BYTES: usize = 1_000_000;

/// Standard cache marker used by providers that support ephemeral prompt caching.
pub(super) fn ephemeral_cache_control() -> serde_json::Value {
    serde_json::json!({ "type": "ephemeral" })
}

/// Return ephemeral cache control metadata when a message is marked cached.
pub(super) fn maybe_ephemeral_cache_control(cached: bool) -> Option<serde_json::Value> {
    cached.then(ephemeral_cache_control)
}

/// Parse stored tool calls in our unified history format.
/// Returns empty vec on parse failures to preserve legacy lossy behavior.
pub(super) fn parse_generic_tool_calls_lossy(
    tool_calls: Option<&serde_json::Value>,
    provider: &str,
) -> Vec<GenericToolCall> {
    let Some(tool_calls) = tool_calls else {
        return Vec::new();
    };

    match serde_json::from_value::<Vec<GenericToolCall>>(tool_calls.clone()) {
        Ok(calls) => calls,
        Err(err) => {
            tracing::warn!(
                provider = provider,
                error = %err,
                "Failed to parse GenericToolCall list; dropping malformed tool_calls"
            );
            Vec::new()
        }
    }
}

/// Parse stored tool calls in our unified history format with strict validation.
pub(super) fn parse_generic_tool_calls_strict(
    tool_calls: &serde_json::Value,
    provider: &str,
) -> Result<Vec<GenericToolCall>, ToolCallError> {
    serde_json::from_value::<Vec<GenericToolCall>>(tool_calls.clone()).map_err(|_| {
        ToolCallError::InvalidFormat {
            provider: provider.to_string(),
            reason: "tool_calls must be Vec<GenericToolCall>".to_string(),
        }
    })
}

/// Convert runtime tool calls into unified history format with shared meta.
pub(super) fn to_generic_tool_calls_with_meta(
    calls: &[ToolCall],
    meta: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Vec<GenericToolCall> {
    let meta = meta.cloned();
    calls
        .iter()
        .map(|call| GenericToolCall {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            meta: meta.clone(),
        })
        .collect()
}

/// Persist tool calls into provider exchange response JSON in unified format.
pub(super) fn set_response_tool_calls(
    response_json: &mut serde_json::Value,
    calls: &[ToolCall],
    meta: Option<&serde_json::Map<String, serde_json::Value>>,
) {
    if calls.is_empty() {
        return;
    }

    let generic_calls = to_generic_tool_calls_with_meta(calls, meta);
    match serde_json::to_value(&generic_calls) {
        Ok(value) => response_json["tool_calls"] = value,
        Err(err) => tracing::warn!(error = %err, "Failed to serialize tool_calls for response"),
    }
}

/// Serialize JSON arguments to function-call argument strings.
pub(super) fn arguments_to_json_string(arguments: &serde_json::Value) -> String {
    match serde_json::to_string(arguments) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(error = %err, "Failed to serialize tool-call arguments");
            String::new()
        }
    }
}

/// Parse function-call arguments from provider responses.
/// Falls back to preserving the raw argument string to avoid silent data loss.
pub(super) fn parse_tool_call_arguments_lossy(raw_arguments: &str) -> serde_json::Value {
    if raw_arguments.len() > MAX_JSON_INPUT_BYTES {
        tracing::warn!(
            length = raw_arguments.len(),
            "Tool-call arguments exceed size limit; preserving as raw string"
        );
        return serde_json::json!({ "raw_arguments": raw_arguments });
    }

    match serde_json::from_str(raw_arguments) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!(error = %err, "Failed to parse tool-call arguments JSON");
            serde_json::json!({ "raw_arguments": raw_arguments })
        }
    }
}

/// Parse structured output directly from textual model content.
pub(super) fn parse_structured_output_from_text(content: &str) -> Option<serde_json::Value> {
    let trimmed = content.trim();
    if trimmed.len() > MAX_JSON_INPUT_BYTES {
        tracing::warn!(
            length = trimmed.len(),
            "Structured output candidate exceeds size limit; skipping JSON parse"
        );
        return None;
    }

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        match serde_json::from_str(trimmed) {
            Ok(value) => Some(value),
            Err(err) => {
                tracing::debug!(error = %err, "Failed to parse structured output JSON");
                None
            }
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
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
            parse_generic_tool_calls_lossy(Some(&serde_json::json!({"bad": true})), "test")
                .is_empty()
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
        assert!(
            parse_generic_tool_calls_strict(&serde_json::json!({"bad": true}), "test").is_err()
        );
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
        assert!(parse_structured_output_from_text("not json").is_none());
        assert!(parse_structured_output_from_text("{not-json").is_none());
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
}
