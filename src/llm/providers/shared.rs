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
use crate::llm::types::{ReasoningEffort, ToolCall};
use std::time::Duration;

pub(crate) use crate::http::http_client;
pub use crate::http::set_user_agent;

/// Resolve request credentials before accessing any process-wide credentials.
pub(super) fn resolve_api_key(
	provider: &dyn crate::llm::traits::AiProvider,
	params: &crate::llm::types::ChatCompletionParams,
) -> anyhow::Result<String> {
	match &params.api_key {
		Some(key) => Ok(key.as_str().to_owned()),
		None => provider.get_api_key(),
	}
}

pub(super) fn reject_api_key(provider: &str, params: &crate::llm::types::ChatCompletionParams) -> anyhow::Result<()> {
	if params.api_key.is_some() {
		return Err(crate::errors::ProviderError::UnsupportedOperation {
			provider: provider.to_owned(),
			operation: "per-call API key authentication".to_owned(),
		}
		.into());
	}
	Ok(())
}

pub(super) fn sensitive_header(value: &str) -> anyhow::Result<reqwest::header::HeaderValue> {
	let mut header = reqwest::header::HeaderValue::from_str(value)?;
	header.set_sensitive(true);
	Ok(header)
}

pub(crate) fn redact_text(text: &str, api_key: Option<&crate::llm::types::PerCallApiKey>) -> String {
	match api_key.filter(|key| !key.as_str().is_empty()) {
		Some(key) => text.replace(key.as_str(), "[redacted]"),
		None => text.to_owned(),
	}
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct RedactedError {
	message: String,
	transport_error: Option<TransportErrorKind>,
	terminal_stream_error: bool,
}

pub(crate) fn redact_error(error: anyhow::Error, api_key: Option<&crate::llm::types::PerCallApiKey>) -> anyhow::Error {
	let Some(key) = api_key.filter(|key| !key.as_str().is_empty()) else {
		return error;
	};
	let message = format!("{error:#}");
	if !message.contains(key.as_str()) {
		return error;
	}
	// A retryable status keeps its type: the retry loop reads its status and
	// `Retry-After`, and it has no source chain to leak through.
	if let Some(status) = error.downcast_ref::<crate::llm::retry::RetryableStatusError>() {
		return status.with_message(redact_text(&message, Some(key))).into();
	}
	// Retaining the original source would expose the credential through error chains.
	RedactedError {
		message: redact_text(&message, Some(key)),
		transport_error: transport_error_kind(&error),
		terminal_stream_error: is_terminal_stream_error(&error),
	}
	.into()
}

pub(crate) fn is_terminal_stream_error(error: &anyhow::Error) -> bool {
	error.is::<super::streaming::TerminalStreamError>()
		|| error
			.downcast_ref::<RedactedError>()
			.is_some_and(|error| error.terminal_stream_error)
}

pub(super) fn parse_response<T: serde::de::DeserializeOwned>(
	body: &str,
	api_key: Option<&crate::llm::types::PerCallApiKey>,
) -> anyhow::Result<T> {
	serde_json::from_str(body).map_err(|error| redact_error(error.into(), api_key))
}

/// Chat Completions tool content is text-only. Deliver images as attributed user
/// turns after the entire tool-result batch, never between a call and its results.
pub(super) fn chat_completion_messages(
	messages: &[crate::llm::types::Message],
) -> std::borrow::Cow<'_, [crate::llm::types::Message]> {
	use crate::llm::types::Message;
	use std::borrow::Cow;
	if !messages
		.iter()
		.any(|message| message.role == "tool" && message.images.as_ref().is_some_and(|images| !images.is_empty()))
	{
		return Cow::Borrowed(messages);
	}
	let mut result = Vec::with_capacity(messages.len());
	let mut image_turns = Vec::new();
	for message in messages {
		if message.role != "tool" {
			result.append(&mut image_turns);
		}
		let mut message = message.clone();
		if message.role == "tool" && message.images.as_ref().is_some_and(|images| !images.is_empty()) {
			let mut image_turn = Message::user(&format!(
				"Images returned by tool {} (tool_call_id: {}):",
				message.name.as_deref().unwrap_or_default(),
				message.tool_call_id.as_deref().unwrap_or_default(),
			));
			image_turn.timestamp = message.timestamp;
			image_turn.images = message.images.take();
			image_turns.push(image_turn);
		}
		result.push(message);
	}
	result.append(&mut image_turns);
	Cow::Owned(result)
}

/// Anthropic-compatible tool results accept text or nested image content blocks.
pub(super) fn anthropic_tool_content(message: &crate::llm::types::Message) -> serde_json::Value {
	use crate::llm::types::ImageData;
	let images = message.images.as_deref().unwrap_or_default();
	if images.is_empty() {
		return serde_json::json!(message.content);
	}
	let mut blocks = Vec::new();
	if !message.content.trim().is_empty() {
		blocks.push(serde_json::json!({"type": "text", "text": message.content}));
	}
	for image in images {
		let source = match &image.data {
			ImageData::Base64(data) => serde_json::json!({
				"type": "base64", "media_type": image.media_type, "data": data,
			}),
			ImageData::Url(url) => serde_json::json!({"type": "url", "url": url}),
		};
		blocks.push(serde_json::json!({"type": "image", "source": source}));
	}
	serde_json::Value::Array(blocks)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TransportErrorKind {
	Timeout,
	Connect,
	Request,
	Body,
	Decode,
}

impl TransportErrorKind {
	pub(crate) fn reason(self) -> &'static str {
		match self {
			Self::Timeout => "request timed out",
			Self::Connect => "connection establishment failed",
			Self::Request => "request transport failed",
			Self::Body => "body transport failed",
			Self::Decode => "response decoding failed",
		}
	}
}

pub(crate) fn transport_error_kind(error: &anyhow::Error) -> Option<TransportErrorKind> {
	if error.is::<super::streaming::IncompleteStream>() {
		return Some(TransportErrorKind::Body);
	}
	if let Some(error) = error.downcast_ref::<RedactedError>() {
		return error.transport_error;
	}
	if matches!(
		error.downcast_ref::<crate::errors::ProviderError>(),
		Some(crate::errors::ProviderError::TimeoutError { .. })
	) {
		return Some(TransportErrorKind::Timeout);
	}
	let error = error.chain().find_map(|cause| cause.downcast_ref::<reqwest::Error>())?;
	if error.is_timeout() {
		Some(TransportErrorKind::Timeout)
	} else if error.is_connect() {
		Some(TransportErrorKind::Connect)
	} else if error.is_request() {
		Some(TransportErrorKind::Request)
	} else if error.is_body() {
		Some(TransportErrorKind::Body)
	} else if error.is_decode() {
		Some(TransportErrorKind::Decode)
	} else {
		None
	}
}

/// A complete HTTP response: status, headers, and a buffered or accumulated body.
///
/// Produced by [`send_and_read`] or the SSE reader, which consume the body INSIDE the retried unit
/// so that a mid-stream connection reset surfaces as a retryable error instead
/// of escaping the retry loop. Callers inspect `status`/`headers` and parse
/// `body` without any further `.await` on the (already consumed) connection.
pub(crate) struct CapturedResponse {
	pub status: reqwest::StatusCode,
	pub headers: reqwest::header::HeaderMap,
	pub body: String,
}

/// Fold caller-supplied extra headers into a request builder, LAST, with
/// upsert semantics: `RequestBuilder::headers` replaces any existing value for
/// a name present in the map (reqwest's `replace_headers`), so a caller can
/// override even a provider-set header, while every untouched name survives.
/// Invalid header names/values are skipped with a warning — a malformed extra
/// header must not fail the request it was riding on.
pub(super) fn apply_extra_headers(
	request: reqwest::RequestBuilder,
	extra: Option<&std::collections::HashMap<String, String>>,
) -> reqwest::RequestBuilder {
	let Some(map) = extra.filter(|m| !m.is_empty()) else {
		return request;
	};
	let mut headers = reqwest::header::HeaderMap::new();
	for (name, value) in map {
		match (
			name.parse::<reqwest::header::HeaderName>(),
			value.parse::<reqwest::header::HeaderValue>(),
		) {
			(Ok(n), Ok(v)) => {
				headers.insert(n, v);
			}
			_ => {
				tracing::warn!(header = %name, "skipping invalid extra header");
			}
		}
	}
	request.headers(headers)
}

/// Send a request and read its entire body, all within one awaited unit.
///
/// Both the `send()` (connect/headers) and the body read happen here, so any
/// transport failure — including a mid-body RST that only manifests during
/// `text()` — is returned as an `Err` to the surrounding retry loop, where
/// transport errors are classified before the next attempt. Reqwest manages
/// failed connections. Status and headers are cloned before consuming the body.
///
/// `extra_headers` ([`ChatCompletionParams::extra_headers`]) are folded in here
/// through [`apply_extra_headers`], also used by the SSE reader, so the upsert
/// semantics hold uniformly no matter which provider built the request.
///
/// [`ChatCompletionParams::extra_headers`]: crate::llm::types::ChatCompletionParams
pub(super) async fn send_and_read(
	request: reqwest::RequestBuilder,
	timeout: Option<Duration>,
	extra_headers: Option<&std::collections::HashMap<String, String>>,
) -> anyhow::Result<CapturedResponse> {
	let request = apply_extra_headers(request, extra_headers);
	let response = apply_request_timeout(request, timeout)
		.send()
		.await
		.map_err(anyhow::Error::from)?;

	let status = response.status();
	let headers = response.headers().clone();
	let body = response.text().await.map_err(anyhow::Error::from)?;

	Ok(CapturedResponse { status, headers, body })
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
pub(super) fn maybe_cache_control_with_ttl(cached: bool, ttl: Option<&str>) -> Option<serde_json::Value> {
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
	serde_json::from_value::<Vec<GenericToolCall>>(tool_calls.clone()).map_err(|_| ToolCallError::InvalidFormat {
		provider: provider.to_string(),
		reason: "tool_calls must be Vec<GenericToolCall>".to_string(),
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
///
/// `pub(crate)` (rather than `pub(super)`): also reused by
/// `llm::schema_enforcement`'s best-effort fallback candidate extraction.
pub(crate) fn parse_structured_output_from_text(content: &str) -> Option<serde_json::Value> {
	let trimmed = content.trim();
	if trimmed.len() > MAX_JSON_INPUT_BYTES {
		tracing::warn!(
			length = trimmed.len(),
			"Structured output candidate exceeds size limit; skipping JSON parse"
		);
		return None;
	}

	let candidate = if let Some(fenced) = trimmed.strip_prefix("```") {
		let newline = fenced.find('\n')?;
		let language = fenced[..newline].trim();
		if !language.is_empty() && !language.eq_ignore_ascii_case("json") {
			return None;
		}
		fenced[newline + 1..].strip_suffix("```")?.trim()
	} else {
		trimmed
	};

	if candidate.starts_with('{') || candidate.starts_with('[') {
		match serde_json::from_str(candidate) {
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

/// `(host, model substring)` pairs whose thinking cannot be switched off.
///
/// GLM-5.3: Model Studio rejects `enable_thinking=false` ("restricted to
/// True") and `reasoning_effort="none"`, and Z.AI's `thinking: disabled`
/// still reasons (verified live, Sep 2026). Ollama Cloud lists no `false`
/// level for it and runs an omitted effort at its `max` default (`/api/show`,
/// Oct 2026); NVIDIA NIM documents `low|high|max` with default `max`
/// (z-ai-glm-5-3 reference, Oct 2026); models.dev lists the same levels with
/// no off toggle on OpenCode Zen/Go and Fireworks (Oct 2026).
///
/// Model Studio (deep-thinking and Kimi pages, Oct 2026): kimi-k3 and
/// qwen3.8-2.4t-a95b are thinking-only. Fireworks' qwen3p8-2p4t-a95b lists
/// `low|medium|xhigh` with no off toggle (models.dev, Oct 2026).
const ALWAYS_THINKS: &[(&str, &str)] = &[
	("alibaba", "glm-5.3"),
	("alibaba", "kimi-k3"),
	("alibaba", "qwen3.8-2.4t-a95b"),
	("zai", "glm-5.3"),
	("ollama", "glm-5.3"),
	("nvidia", "glm-5.3"),
	("opencode-zen", "glm-5.3"),
	("opencode-go", "glm-5.3"),
	("fireworks", "glm-5p3"),
	("fireworks", "qwen3p8-2p4t"),
];

/// The effort `provider_name` can run `model` at. On a model in
/// [`ALWAYS_THINKS`] a request for no reasoning runs at the lowest level
/// instead.
pub(super) fn supported_reasoning_effort(provider_name: &str, model: &str, effort: ReasoningEffort) -> ReasoningEffort {
	let always_thinks = ALWAYS_THINKS.iter().any(|(host, pattern)| {
		provider_name.eq_ignore_ascii_case(host) && crate::llm::utils::contains_ignore_ascii_case(model, pattern)
	});
	if always_thinks && effort == ReasoningEffort::None {
		ReasoningEffort::Low
	} else {
		effort
	}
}

#[cfg(test)]
#[path = "shared_tests.rs"]
pub(super) mod tests;

#[cfg(test)]
#[path = "shared_transport_tests.rs"]
mod transport_tests;
