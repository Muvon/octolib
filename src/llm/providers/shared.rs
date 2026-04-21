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
use arc_swap::ArcSwap;
use std::sync::LazyLock;
use std::time::Duration;

/// Process-wide shared HTTP client, swappable on connection errors.
///
/// `reqwest::Client` holds a connection pool internally — reusing it across
/// all provider requests enables TCP keep-alive, HTTP/2 multiplexing, and
/// avoids the per-request TLS handshake overhead that causes connection-reset
/// errors under load.
///
/// When a connection error is detected (DNS failure, TCP reset, TLS handshake
/// failure, network unreachable), `refresh_http_client()` atomically swaps
/// in a fresh client with a new connection pool, so subsequent retries don't
/// reuse stale/broken connections.
///
/// No request timeout is set — LLM responses can legitimately take minutes.
/// Instead we configure:
/// - `tcp_keepalive`: OS-level probes detect dead connections before reuse
/// - `pool_idle_timeout`: evict idle pooled connections before NAT/firewall
///   silently drops them, preventing hangs on stale sockets
static HTTP_CLIENT: LazyLock<ArcSwap<reqwest::Client>> =
    LazyLock::new(|| ArcSwap::from_pointee(build_http_client()));

fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .tcp_keepalive(Duration::from_secs(60))
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("failed to build HTTP client")
}

/// Returns a cloned handle to the process-wide shared HTTP client.
///
/// `reqwest::Client` is internally `Arc`-based, so cloning is cheap and
/// always points to the current client (even after `refresh_http_client()`
/// swaps the global).
pub(super) fn http_client() -> reqwest::Client {
    // load_full() clones the Arc (cheap atomic increment),
    // then dereference and clone the Client (cheap — Client is Arc internally)
    (*HTTP_CLIENT.load_full()).clone()
}

/// Atomically replace the shared HTTP client with a fresh instance.
///
/// Call this when a connection error is detected (DNS failure, TCP reset,
/// TLS handshake failure, network unreachable). The new client gets a fresh
/// connection pool, so subsequent requests — including retry attempts —
/// won't reuse stale/broken connections from the old pool.
///
/// The old client is dropped once all outstanding references to it are gone,
/// which closes its idle connections.
pub(crate) fn refresh_http_client() {
    let fresh = build_http_client();
    HTTP_CLIENT.store(std::sync::Arc::new(fresh));
    tracing::debug!("HTTP client refreshed with new connection pool");
}

/// Returns true if the error is a connection-level failure that indicates
/// the HTTP client's connection pool may contain stale/broken connections.
///
/// Such errors include: DNS resolution failure, TCP connection refused/reset,
/// TLS handshake failure, network unreachable, and similar transport errors.
/// Retrying on the same client may reuse the same broken connection, so
/// callers should call `refresh_http_client()` before retrying.
pub(crate) fn is_connection_error(err: &anyhow::Error) -> bool {
    err.downcast_ref::<reqwest::Error>()
        .is_some_and(|e| e.is_connect())
}

/// Apply an optional per-request timeout to a RequestBuilder.
///
/// `None` leaves the builder unchanged (no timeout — LLM responses may take minutes).
/// `Some(d)` sets a hard timeout on the whole HTTP request; exceeding it aborts with
/// a reqwest timeout error that surfaces as a retryable failure.
pub(super) fn apply_request_timeout(
    req: reqwest::RequestBuilder,
    timeout: Option<Duration>,
) -> reqwest::RequestBuilder {
    match timeout {
        Some(d) => req.timeout(d),
        None => req,
    }
}

const MAX_JSON_INPUT_BYTES: usize = 1_000_000;

/// Standard cache marker used by providers that support ephemeral prompt caching.
pub(super) fn ephemeral_cache_control() -> serde_json::Value {
    serde_json::json!({ "type": "ephemeral" })
}

/// Return ephemeral cache control with optional TTL override.
/// When `ttl` is Some (e.g. "1h"), includes it in the cache_control block.
/// Only Anthropic supports TTL — other providers ignore the field.
pub(super) fn ephemeral_cache_control_with_ttl(ttl: Option<&str>) -> serde_json::Value {
    match ttl {
        Some(t) => serde_json::json!({ "type": "ephemeral", "ttl": t }),
        None => serde_json::json!({ "type": "ephemeral" }),
    }
}

/// Return ephemeral cache control metadata when a message is marked cached.
pub(super) fn maybe_ephemeral_cache_control(cached: bool) -> Option<serde_json::Value> {
    cached.then(ephemeral_cache_control)
}

/// Return cache control with optional TTL when message is cached.
pub(super) fn maybe_cache_control_with_ttl(
    cached: bool,
    ttl: Option<&str>,
) -> Option<serde_json::Value> {
    if cached {
        Some(ephemeral_cache_control_with_ttl(ttl))
    } else {
        None
    }
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
