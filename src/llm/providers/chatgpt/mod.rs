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

//! ChatGPT plan usage: Responses API requests billed to the user's ChatGPT
//! subscription through Sign in with ChatGPT (see [`start_login`]).
//!
//! Requests default to HTTP/SSE. The optional `websockets` feature enables
//! per-request WebSocket selection through `ChatGptTransport`. Both transports
//! preserve tool and reasoning replay and require `store: false`.
//! See <https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations>

mod auth;
mod http;
#[cfg(feature = "websockets")]
mod websocket;

pub use auth::{current_account, list_models, start_login, Account, ChatGptModel, PendingLogin};

use super::openai::{self, OpenAiProvider};
use crate::llm::traits::AiProvider;
use crate::llm::types::{
	ChatCompletionParams, ChatGptTransport, Message, ModelPricing, ProviderResponse, SamplingSupport,
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// Replayed function calls carry the namespace used to declare their tools.
const TOOL_NAMESPACE: &str = "tools";
const TOOL_NAMESPACE_DESCRIPTION: &str = "Tools available in this session.";
const SESSION_HEADER: &str = "session-id";

/// ChatGPT subscription provider (`chatgpt:<model>`).
#[derive(Debug, Clone, Default)]
pub struct ChatGptProvider;

impl ChatGptProvider {
	pub fn new() -> Self {
		Self
	}
}

#[async_trait::async_trait]
impl AiProvider for ChatGptProvider {
	fn name(&self) -> &str {
		"chatgpt"
	}

	/// The signed-in account's catalog decides which models exist ([`list_models`]).
	fn supports_model(&self, _model: &str) -> bool {
		true
	}

	fn get_api_key(&self) -> Result<String> {
		auth::stored_access_token()
	}

	fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
		SamplingSupport::NONE
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		OpenAiProvider.get_max_input_tokens(model)
	}

	fn supports_vision(&self, model: &str) -> bool {
		OpenAiProvider.supports_vision(model)
	}

	fn supports_structured_output(&self, model: &str) -> bool {
		OpenAiProvider.supports_structured_output(model)
	}

	/// Real per-token API prices, whatever the subscription covers, so cost
	/// reports actual model consumption.
	fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
		OpenAiProvider.get_model_pricing(model)
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		crate::llm::providers::shared::reject_api_key(self.name(), &params)?;
		let request_body = build_request(&params);
		let start_time = std::time::Instant::now();
		let (request_body, response_json) = match params.chatgpt_transport {
			ChatGptTransport::Http => {
				let response = http::complete(&params, &request_body).await?;
				(request_body, response)
			}
			#[cfg(feature = "websockets")]
			ChatGptTransport::WebSocket => websocket::complete(&params, request_body).await?,
		};
		let request_time_ms = start_time.elapsed().as_millis() as u64;

		openai::parse_responses_api_response(request_body, response_json, "chatgpt", request_time_ms, HashMap::new())
	}
}

fn build_request(params: &ChatCompletionParams) -> Value {
	let mut request = json!({
		"model": params.model,
		"input": build_input(&params.messages),
		"store": false,
		"include": ["reasoning.encrypted_content"],
	});

	match params.chatgpt_transport {
		ChatGptTransport::Http => request["stream"] = json!(true),
		#[cfg(feature = "websockets")]
		ChatGptTransport::WebSocket => request["type"] = json!("response.create"),
	}

	if let Some(effort) = openai::reasoning_effort(&params.model, params.reasoning_effort) {
		request["reasoning"] = json!({ "effort": effort });
	}

	if let Some(tools) = openai::function_tools(params.tools.as_deref()) {
		request["tools"] = json!([{
			"type": "namespace",
			"name": TOOL_NAMESPACE,
			"description": TOOL_NAMESPACE_DESCRIPTION,
			"tools": tools,
		}]);
	}

	if let Some(text) = params.response_format.as_ref().and_then(openai::text_format) {
		request["text"] = text;
	}

	if let Some(cache_key) = &params.prompt_cache_key {
		request["prompt_cache_key"] = json!(cache_key);
	}

	request
}

#[cfg(feature = "websockets")]
fn request_headers(params: &ChatCompletionParams) -> Result<reqwest::header::HeaderMap> {
	let mut headers = reqwest::header::HeaderMap::new();
	// Include the configured identity in the pool key as well as the handshake.
	headers.insert(
		reqwest::header::USER_AGENT,
		crate::http::user_agent()
			.parse()
			.context("Invalid ChatGPT User-Agent")?,
	);
	if let Some(cache_key) = &params.prompt_cache_key {
		headers.insert(
			SESSION_HEADER,
			cache_key.parse().context("Invalid ChatGPT session header")?,
		);
	}
	if let Some(extra) = &params.extra_headers {
		for (name, value) in extra {
			let name: reqwest::header::HeaderName = name.parse().context("Invalid ChatGPT header name")?;
			let value = value
				.parse()
				.with_context(|| format!("Invalid ChatGPT header {name}"))?;
			headers.insert(name, value);
		}
	}
	Ok(headers)
}

/// The full transcript as Responses API input: assistant turns replay their
/// captured output items verbatim, system messages are sent as developer
/// messages, and function calls are tagged with the namespace they were declared in.
fn build_input(messages: &[Message]) -> Vec<Value> {
	messages
		.iter()
		.flat_map(|message| {
			openai::replay_provider_items(message, "chatgpt")
				.unwrap_or_else(|| openai::messages_to_input(std::slice::from_ref(message), None, false))
		})
		.map(normalize_input_item)
		.collect()
}

fn normalize_input_item(mut item: Value) -> Value {
	if item["role"] == "system" {
		item["role"] = json!("developer");
	}
	if item["type"] == "function_call" {
		item["namespace"] = json!(TOOL_NAMESPACE);
	}
	item
}

/// Server faults, throttling and lost connection-local state can recover on a
/// fresh socket. An exhausted plan allowance cannot recover by retrying.
const TRANSIENT_ERROR_CODES: [&str; 6] = [
	"server_error",
	"server_is_overloaded",
	"rate_limit_exceeded",
	"slow_down",
	"previous_response_not_found",
	"websocket_connection_limit_reached",
];

/// A stream failure a retry can clear. Every other terminal event is the
/// outcome of the request, which a retry would not change.
#[derive(Debug, thiserror::Error)]
enum RetryableStreamError {
	/// The stream closed before any terminal event.
	#[error("ChatGPT stream ended without response.completed")]
	CutOff,
	#[cfg(feature = "websockets")]
	#[error("ChatGPT WebSocket request timed out")]
	Timeout,
	/// A terminal event carrying one of [`TRANSIENT_ERROR_CODES`].
	#[error("{0}")]
	Transient(String),
}

/// A terminal failure event as an error, retryable when its code is transient.
fn terminal_error(error: &Value, message: String) -> anyhow::Error {
	let code = error["code"].as_str();
	if code.is_some_and(|code| TRANSIENT_ERROR_CODES.contains(&code)) {
		RetryableStreamError::Transient(message).into()
	} else {
		with_usage_guidance(code, message)
	}
}

/// Lead a plan-usage error with what the user can act on; the raw provider
/// error follows unchanged. The code does not say whether the whole plan or an
/// app-specific limit is exhausted, and signing in again clears neither.
fn with_usage_guidance(code: Option<&str>, message: String) -> anyhow::Error {
	let guidance = match code {
		Some("subscription_sharing_usage_limit_exceeded") => {
			"ChatGPT plan usage limit reached for this app. It can be an app-specific limit \
             while the account still has usage in ChatGPT or Codex, and signing in again does \
             not clear it. Check ChatGPT settings → Usage for your limits and reset time, or \
             use an API-key provider until then."
		}
		Some("subscription_sharing_usage_unavailable") => {
			"ChatGPT could not check this plan's usage availability; try again later. \
             Your sign-in is still valid."
		}
		_ => return anyhow::Error::msg(message),
	};
	anyhow::Error::msg(format!("{guidance}\n{message}"))
}

#[derive(Default)]
struct ResponseEvents {
	output: BTreeMap<usize, Value>,
}

impl ResponseEvents {
	fn push(&mut self, event: Value) -> Result<Option<Value>> {
		match event["type"].as_str().context("ChatGPT event is missing type")? {
			"response.output_item.done" => {
				let index: usize = serde_json::from_value(event["output_index"].clone())
					.context("ChatGPT output item is missing output_index")?;
				let item = event
					.get("item")
					.filter(|item| item.is_object())
					.context("ChatGPT output event is missing item")?;
				anyhow::ensure!(
					self.output.insert(index, item.clone()).is_none(),
					"Duplicate ChatGPT output_index {index}"
				);
			}
			"response.completed" => {
				let mut response = event
					.get("response")
					.filter(|response| response.is_object())
					.context("ChatGPT completed event is missing response")?
					.clone();
				if !self.output.is_empty() {
					let output = std::mem::take(&mut self.output);
					let mut items = Vec::with_capacity(output.len());
					for (expected, (index, item)) in output.into_iter().enumerate() {
						anyhow::ensure!(index == expected, "ChatGPT response is missing output_index {expected}");
						items.push(item);
					}
					response["output"] = json!(items);
				}
				return Ok(Some(response));
			}
			"response.failed" => {
				return Err(terminal_error(
					&event["response"]["error"],
					format!("ChatGPT response failed: {}", event["response"]["error"]),
				))
			}
			"response.incomplete" => anyhow::bail!(
				"ChatGPT response incomplete: {}",
				event["response"]["incomplete_details"]
			),
			"error" => {
				return Err(terminal_error(
					&event["error"],
					format!("ChatGPT stream error: {event}"),
				))
			}
			_ => {}
		}
		Ok(None)
	}
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
