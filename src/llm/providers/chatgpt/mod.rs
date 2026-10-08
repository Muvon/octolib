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
        let access_token = auth::access_token().await?;
        let request_body = build_request(&params);
        let start_time = std::time::Instant::now();

        // Errors are retried; `Ok` holds the outcome of a delivered response,
        // which a retry would not change.
        let response_json = retry::retry_with_exponential_backoff(
            || {
                let client = shared::http_client();
                let access_token = access_token.clone();
                let request_body = request_body.clone();
                let request_timeout = params.request_timeout;
                let extra_headers = params.extra_headers.clone();
                Box::pin(async move {
                    let request = client
                        .post(RESPONSES_URL)
                        .bearer_auth(access_token)
                        .json(&request_body);
                    let captured =
                        shared::send_and_read(request, request_timeout, extra_headers.as_ref())
                            .await?;

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
                        return Ok(Err(anyhow::anyhow!(
                            "ChatGPT API error {}: {}",
                            captured.status,
                            captured.body
                        )));
                    }

                    match merge_stream(&captured.body) {
                        Err(e) if e.is::<StreamCutOff>() => Err(e),
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

/// The stream closed before any terminal event. Unlike the failures terminal
/// events report, this cut-off is transient, so it is the one retried.
#[derive(Debug, thiserror::Error)]
#[error("ChatGPT stream ended without response.completed")]
struct StreamCutOff;

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
                anyhow::bail!("ChatGPT response failed: {}", event["response"]["error"])
            }
            Some("response.incomplete") => anyhow::bail!(
                "ChatGPT response incomplete: {}",
                event["response"]["incomplete_details"]
            ),
            Some("error") => anyhow::bail!("ChatGPT stream error: {}", event),
            _ => {}
        }
    }

    Err(StreamCutOff.into())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
