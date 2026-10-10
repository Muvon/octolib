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
//! The plan route is a restricted Responses API: `store: false` and
//! `stream: true` are mandatory, there is no server-side conversation state
//! (`previous_response_id`), sampling/limit/retention fields are rejected,
//! system items must be developer messages, and function tools must sit in a
//! namespace. Reasoning survives tool calls only by replaying its encrypted
//! form, which this provider requests and replays.
//! See <https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations>

mod auth;

pub use auth::{current_account, list_models, start_login, Account, ChatGptModel, PendingLogin};

use super::openai::{self, OpenAiProvider};
use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ModelPricing, ProviderResponse, SamplingSupport,
};
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
/// Every function tool is declared in this one namespace, so replayed
/// `function_call` items carry it too.
const TOOL_NAMESPACE: &str = "tools";
const TOOL_NAMESPACE_DESCRIPTION: &str = "Tools available in this session.";
/// Error codes for an exhausted plan allowance or per-app weekly cap
/// (`subscription_sharing_usage_limit_exceeded`, `…_usage_unavailable`).
const USAGE_LIMIT_ERROR_PREFIX: &str = "subscription_sharing_usage";

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
        let request_body = build_request(&params);
        let start_time = std::time::Instant::now();

        // Errors are retried; `Ok` holds the outcome of a delivered response,
        // which a retry would not change.
        let response_json = retry::retry_with_exponential_backoff(
            || {
                let client = shared::http_client();
                let request_body = request_body.clone();
                let request_timeout = params.request_timeout;
                let extra_headers = params.extra_headers.clone();
                Box::pin(async move {
                    let request = client.post(RESPONSES_URL).json(&request_body);
                    let captured = match auth::send_authenticated(
                        request,
                        request_timeout,
                        extra_headers.as_ref(),
                    )
                    .await
                    {
                        Ok(captured) => captured,
                        Err(error) if error.is::<reqwest::Error>() => return Err(error),
                        // Credential failures need sign-in, not another refresh attempt.
                        Err(error) => return Ok(Err(error)),
                    };

                    // An exhausted plan allowance or app cap does not clear on retry.
                    if retry::is_retryable_status(captured.status.as_u16())
                        && !captured.body.contains(USAGE_LIMIT_ERROR_PREFIX)
                    {
                        return Err(anyhow::anyhow!(
                            "ChatGPT API error {}: {}",
                            captured.status,
                            captured.body
                        ));
                    }

                    if !captured.status.is_success() {
                        let code = serde_json::from_str::<Value>(&captured.body)
                            .ok()
                            .and_then(|body| body["error"]["code"].as_str().map(str::to_owned));
                        return Ok(Err(with_usage_guidance(
                            code.as_deref(),
                            format!("ChatGPT API error {}: {}", captured.status, captured.body),
                        )));
                    }

                    match merge_stream(&captured.body) {
                        Err(e) if e.is::<RetryableStreamError>() => Err(e),
                        merged => Ok(merged),
                    }
                })
            },
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
            |e| {
                matches!(
                    e.downcast_ref::<ProviderError>(),
                    Some(ProviderError::Cancelled)
                )
            },
            |e: &anyhow::Error| shared::is_connection_error(e),
        )
        .await??;

        let request_time_ms = start_time.elapsed().as_millis() as u64;

        openai::parse_responses_api_response(
            request_body,
            response_json,
            "chatgpt",
            request_time_ms,
            HashMap::new(),
        )
    }
}

fn build_request(params: &ChatCompletionParams) -> Value {
    let mut request = json!({
        "model": params.model,
        "input": build_input(&params.messages),
        "store": false,
        "stream": true,
        "include": ["reasoning.encrypted_content"],
    });

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

    if let Some(text) = params
        .response_format
        .as_ref()
        .and_then(openai::text_format)
    {
        request["text"] = text;
    }

    if let Some(cache_key) = &params.prompt_cache_key {
        request["prompt_cache_key"] = json!(cache_key);
    }

    request
}

/// The full transcript as Responses API input: each assistant turn preceded by
/// its replayed encrypted reasoning, system messages sent as developer
/// messages, and function calls tagged with the namespace they were declared in.
fn build_input(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .flat_map(|message| {
            let mut items = stored_reasoning_items(message);
            items.extend(openai::messages_to_input(
                std::slice::from_ref(message),
                None,
                false,
            ));
            items
        })
        .map(|mut item| {
            if item["role"] == "system" {
                item["role"] = json!("developer");
            }
            if item["type"] == "function_call" {
                item["namespace"] = json!(TOOL_NAMESPACE);
            }
            item
        })
        .collect()
}

fn stored_reasoning_items(message: &Message) -> Vec<Value> {
    shared::parse_generic_tool_calls_lossy(message.tool_calls.as_ref(), "chatgpt")
        .first()
        .and_then(|call| call.meta.as_ref())
        .and_then(|meta| meta.get(openai::REASONING_META_KEY))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Responses API error codes for a server-side fault or throttling: the stream
/// counterparts of the HTTP 5xx and 429 that [`retry::is_retryable_status`]
/// retries. Usage-limit codes are absent — an exhausted allowance does not
/// clear on retry.
const TRANSIENT_ERROR_CODES: [&str; 4] = [
    "server_error",
    "server_is_overloaded",
    "rate_limit_exceeded",
    "slow_down",
];

/// A stream failure a retry can clear. Every other terminal event is the
/// outcome of the request, which a retry would not change.
#[derive(Debug, thiserror::Error)]
enum RetryableStreamError {
    /// The stream closed before any terminal event.
    #[error("ChatGPT stream ended without response.completed")]
    CutOff,
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

/// Fold the buffered SSE body back into one Responses API response object.
///
/// Output items arrive in `response.output_item.done` events; the terminal
/// `response.completed` event carries the response id and usage. Only a
/// completed stream is a successful response.
fn merge_stream(body: &str) -> Result<Value> {
    let mut output = Vec::new();

    for line in body.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let event: Value = serde_json::from_str(data.trim())
            .with_context(|| format!("ChatGPT stream: malformed event: {}", data))?;

        match event["type"].as_str() {
            Some("response.output_item.done") => output.push(event["item"].clone()),
            Some("response.completed") => {
                let mut response = event["response"].clone();
                response["output"] = Value::Array(output);
                return Ok(response);
            }
            Some("response.failed") => {
                let error = &event["response"]["error"];
                return Err(terminal_error(
                    error,
                    format!("ChatGPT response failed: {}", error),
                ));
            }
            Some("response.incomplete") => anyhow::bail!(
                "ChatGPT response incomplete: {}",
                event["response"]["incomplete_details"]
            ),
            // The plan route nests the error object (`{"type":"error",
            // "error":{"code":…}}`), unlike the flat `ResponseErrorEvent` of
            // the API reference.
            Some("error") => {
                return Err(terminal_error(
                    &event["error"],
                    format!("ChatGPT stream error: {}", event),
                ));
            }
            _ => {}
        }
    }

    Err(RetryableStreamError::CutOff.into())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
