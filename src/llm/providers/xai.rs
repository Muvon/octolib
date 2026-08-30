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

//! Native xAI Responses API provider.
//!
//! The Responses API is used instead of the compatibility Chat Completions
//! endpoint so `previous_response_id` and encrypted reasoning items remain
//! available across tool rounds. When a caller rebases from local history,
//! encrypted reasoning items are replayed from unified tool-call metadata.

use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ModelPricing, OutputFormat, ProviderExchange, ProviderResponse,
    ReasoningEffort, ResponseMode, SamplingSupport, ThinkingBlock, TokenUsage, ToolCall,
};
use crate::llm::utils::{get_model_pricing, normalize_strict_schema, PricingTuple};
use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::env;

const XAI_API_KEY_ENV: &str = "XAI_API_KEY";
const XAI_API_URL_ENV: &str = "XAI_API_URL";
const XAI_API_URL: &str = "https://api.x.ai/v1/responses";
const XAI_RESPONSE_ID_PREFIX: &str = "xai_response:";
const LONG_CONTEXT_THRESHOLD: u64 = 200_000;
const USD_TICKS_PER_USD: f64 = 10_000_000_000.0;
const REASONING_META_KEY: &str = "xai_reasoning_items";

/// Current language-model pricing per 1M tokens, verified July 31, 2026.
/// Format: (model, uncached input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
    ("grok-4.20-multi-agent-0309", 1.25, 2.50, 1.25, 0.20),
    ("grok-4.20-0309-non-reasoning", 1.25, 2.50, 1.25, 0.20),
    ("grok-4.20-0309-reasoning", 1.25, 2.50, 1.25, 0.20),
    ("grok-build-0.1", 1.00, 2.00, 1.00, 0.20),
    ("grok-4.6", 2.00, 6.00, 2.00, 0.50),
    ("grok-4.5", 2.00, 6.00, 2.00, 0.30),
    ("grok-4.3", 1.25, 2.50, 1.25, 0.20),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFamily {
    Grok46,
    Grok45,
    Build,
    Grok43,
    Grok420,
}

fn model_family(model: &str) -> Option<ModelFamily> {
    let model = model.to_ascii_lowercase();
    match model.as_str() {
        "grok-4.6" | "grok-4.6-latest" => Some(ModelFamily::Grok46),
        "grok-4.5" | "grok-4.5-latest" | "grok-build-latest" => Some(ModelFamily::Grok45),
        "grok-build-0.1" | "grok-code-fast" | "grok-code-fast-1" | "grok-code-fast-1-0825" => {
            Some(ModelFamily::Build)
        }
        "grok-4.3"
        | "grok-4.3-latest"
        | "grok-latest"
        | "grok-4-1-fast-reasoning"
        | "grok-4-1-fast-non-reasoning"
        | "grok-4-fast-reasoning"
        | "grok-4-fast-non-reasoning"
        | "grok-4-0709"
        | "grok-3" => Some(ModelFamily::Grok43),
        "grok-4.20-0309-reasoning"
        | "grok-4.20-reasoning-latest"
        | "grok-4.20"
        | "grok-4.20-reasoning"
        | "grok-4.20-0309"
        | "grok-4.20-beta-0309-reasoning"
        | "grok-4.20-beta"
        | "grok-4.20-beta-0309"
        | "grok-4.20-beta-latest"
        | "grok-4.20-beta-latest-reasoning"
        | "grok-4.20-beta-reasoning"
        | "grok-4.20-experimental-beta-0304-reasoning"
        | "grok-4.20-experimental-beta-0304"
        | "grok-4.20-experimental-beta-reasoning-latest"
        | "grok-4.20-experimental-beta-latest"
        | "grok-4.20-reasoning-gv2"
        | "grok-4.20-0309-non-reasoning"
        | "grok-4.20-non-reasoning"
        | "grok-4.20-non-reasoning-latest"
        | "grok-4.20-beta-non-reasoning"
        | "grok-4.20-beta-latest-non-reasoning"
        | "grok-4.20-experimental-beta-0304-non-reasoning"
        | "grok-4.20-experimental-beta-non-reasoning-latest"
        | "grok-4.20-beta-0309-non-reasoning"
        | "grok-4.20-non-reasoning-gv2"
        | "grok-4.20-multi-agent-0309"
        | "grok-4.20-multi-agent"
        | "grok-4.20-multi-agent-latest"
        | "grok-4.20-multi-agent-beta-latest"
        | "grok-4.20-multi-agent-experimental-beta-0304"
        | "grok-4.20-multi-agent-experimental-beta-latest"
        | "grok-4.20-multi-agent-beta-0309" => Some(ModelFamily::Grok420),
        _ => None,
    }
}

fn canonical_model(model: &str) -> Option<&'static str> {
    match model_family(model)? {
        ModelFamily::Grok46 => Some("grok-4.6"),
        ModelFamily::Grok45 => Some("grok-4.5"),
        ModelFamily::Build => Some("grok-build-0.1"),
        ModelFamily::Grok43 => Some("grok-4.3"),
        ModelFamily::Grok420 if model.to_ascii_lowercase().contains("multi-agent") => {
            Some("grok-4.20-multi-agent-0309")
        }
        ModelFamily::Grok420 if model.to_ascii_lowercase().contains("non-reasoning") => {
            Some("grok-4.20-0309-non-reasoning")
        }
        ModelFamily::Grok420 => Some("grok-4.20-0309-reasoning"),
    }
}

fn usage_pricing(model: &str, total_input_tokens: u64) -> Option<ModelPricing> {
    let canonical = canonical_model(model)?;
    let (mut input, mut output, mut cache_write, mut cache_read) =
        get_model_pricing(canonical, PRICING)?;
    if total_input_tokens >= LONG_CONTEXT_THRESHOLD {
        input *= 2.0;
        output *= 2.0;
        cache_write *= 2.0;
        cache_read *= 2.0;
    }
    Some(ModelPricing::new(input, output, cache_write, cache_read))
}

fn calculate_cost(model: &str, clean_input: u64, cache_read: u64, output: u64) -> Option<f64> {
    let pricing = usage_pricing(model, clean_input.saturating_add(cache_read))?;
    Some(pricing.calculate_cost(clean_input, 0, cache_read, output))
}

fn ticks_to_usd(ticks: u64) -> f64 {
    ticks as f64 / USD_TICKS_PER_USD
}

fn reasoning_effort(model: &str, effort: Option<ReasoningEffort>) -> Option<&'static str> {
    let effort = effort?;
    let family = model_family(model)?;
    let normalized = model.to_ascii_lowercase();
    let is_multi_agent = normalized.contains("multi-agent");
    let is_configurable_single_agent = matches!(
        normalized.as_str(),
        "grok-4.6"
            | "grok-4.6-latest"
            | "grok-4.5"
            | "grok-4.5-latest"
            | "grok-build-latest"
            | "grok-4.3"
            | "grok-4.3-latest"
            | "grok-latest"
    );
    match (family, is_multi_agent, effort) {
        (
            ModelFamily::Grok46 | ModelFamily::Grok45 | ModelFamily::Grok43,
            _,
            ReasoningEffort::Low,
        ) if is_configurable_single_agent => Some("low"),
        (
            ModelFamily::Grok46 | ModelFamily::Grok45 | ModelFamily::Grok43,
            _,
            ReasoningEffort::Medium,
        ) if is_configurable_single_agent => Some("medium"),
        (ModelFamily::Grok46 | ModelFamily::Grok45 | ModelFamily::Grok43, _, _)
            if is_configurable_single_agent =>
        {
            Some("high")
        }
        (ModelFamily::Grok420, true, ReasoningEffort::Low) => Some("low"),
        (ModelFamily::Grok420, true, ReasoningEffort::Medium) => Some("medium"),
        (ModelFamily::Grok420, true, ReasoningEffort::High) => Some("high"),
        (ModelFamily::Grok420, true, ReasoningEffort::XHigh | ReasoningEffort::Max) => {
            Some("xhigh")
        }
        // Fixed reasoning/non-reasoning 4.20 and Build do not expose a documented effort knob.
        _ => None,
    }
}

fn raw_response_id(id: &str) -> Option<&str> {
    id.strip_prefix(XAI_RESPONSE_ID_PREFIX)
}

fn resolve_previous_response_id(messages: &[Message], explicit: Option<String>) -> Option<String> {
    match explicit {
        // Explicit IDs are trusted for callers that obtained a raw xAI ID independently.
        Some(id) => Some(raw_response_id(&id).unwrap_or(&id).to_string()),
        None => messages
            .iter()
            .rev()
            .find(|message| message.role == "assistant")
            .and_then(|message| message.id.as_deref())
            .and_then(raw_response_id)
            .map(str::to_string),
    }
}

fn user_content(message: &Message) -> Value {
    let has_images = message
        .images
        .as_ref()
        .is_some_and(|images| !images.is_empty());
    if !has_images {
        return json!(message.content);
    }

    let mut parts = vec![json!({"type": "input_text", "text": message.content})];
    if let Some(images) = &message.images {
        for image in images {
            let url = match &image.data {
                crate::llm::types::ImageData::Base64(data) => {
                    format!("data:{};base64,{}", image.media_type, data)
                }
                crate::llm::types::ImageData::Url(url) => url.clone(),
            };
            parts.push(json!({"type": "input_image", "image_url": url}));
        }
    }
    Value::Array(parts)
}

fn stored_reasoning_items(message: &Message) -> Vec<Value> {
    shared::parse_generic_tool_calls_lossy(message.tool_calls.as_ref(), "xai")
        .first()
        .and_then(|call| call.meta.as_ref())
        .and_then(|meta| meta.get(REASONING_META_KEY))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn message_to_input(message: &Message, replay_reasoning: bool) -> Vec<Value> {
    let mut items = Vec::new();
    match message.role.as_str() {
        "tool" => items.push(json!({
            "type": "function_call_output",
            "call_id": message.tool_call_id.clone().unwrap_or_default(),
            "output": message.content,
        })),
        "user" => items.push(json!({"role": "user", "content": user_content(message)})),
        "system" => items.push(json!({"role": "system", "content": message.content})),
        "assistant" => {
            if replay_reasoning {
                items.extend(stored_reasoning_items(message));
            }
            if !message.content.is_empty() {
                items.push(json!({"role": "assistant", "content": message.content}));
            }
            for call in shared::parse_generic_tool_calls_lossy(message.tool_calls.as_ref(), "xai") {
                items.push(json!({
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": shared::arguments_to_json_string(&call.arguments),
                }));
            }
        }
        _ => {}
    }
    items
}

fn messages_to_input(messages: &[Message], previous_response_id: Option<&str>) -> Vec<Value> {
    let start = previous_response_id
        .and_then(|raw_id| {
            let stored_id = format!("{XAI_RESPONSE_ID_PREFIX}{raw_id}");
            messages
                .iter()
                .rposition(|message| {
                    message.role == "assistant" && message.id.as_deref() == Some(&stored_id)
                })
                .map(|index| index + 1)
        })
        .unwrap_or(0);
    let replay_reasoning = previous_response_id.is_none();
    messages
        .iter()
        .skip(start)
        .flat_map(|message| message_to_input(message, replay_reasoning))
        .collect()
}

fn add_tools(request: &mut Value, params: &ChatCompletionParams) {
    let Some(tools) = params.tools.as_ref().filter(|tools| !tools.is_empty()) else {
        return;
    };
    let mut tools = tools.clone();
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    request["tools"] = Value::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect(),
    );
}

fn add_response_format(request: &mut Value, params: &ChatCompletionParams) {
    let Some(format) = &params.response_format else {
        return;
    };
    match format.format {
        OutputFormat::Json => {
            request["text"] = json!({"format": {"type": "json_object"}});
        }
        OutputFormat::JsonSchema => {
            if let Some(schema) = &format.schema {
                let mut object = json!({
                    "type": "json_schema",
                    "name": "response_schema",
                    "schema": normalize_strict_schema(schema, format.mode),
                });
                if matches!(format.mode, ResponseMode::Strict) {
                    object["strict"] = json!(true);
                }
                request["text"] = json!({"format": object});
            }
        }
    }
}

fn build_request(params: &ChatCompletionParams, previous_response_id: Option<&str>) -> Value {
    let mut request = json!({
        "model": params.model,
        "input": messages_to_input(&params.messages, previous_response_id),
        "include": ["reasoning.encrypted_content"],
    });
    if let Some(previous) = previous_response_id {
        request["previous_response_id"] = json!(previous);
    }
    if params.max_tokens > 0 {
        request["max_output_tokens"] = json!(params.max_tokens);
    }
    if let Some(effort) = reasoning_effort(&params.model, params.reasoning_effort) {
        request["reasoning"] = json!({"effort": effort});
    }
    add_tools(&mut request, params);
    add_response_format(&mut request, params);
    request
}

#[derive(Debug, Deserialize, Default)]
struct XaiUsage {
    #[serde(default, alias = "prompt_tokens")]
    input_tokens: u64,
    #[serde(default, alias = "completion_tokens")]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    #[serde(default, alias = "prompt_tokens_details")]
    input_tokens_details: Option<InputTokenDetails>,
    #[serde(default, alias = "completion_tokens_details")]
    output_tokens_details: Option<OutputTokenDetails>,
    #[serde(default)]
    cost_in_usd_ticks: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
struct InputTokenDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Deserialize, Default)]
struct OutputTokenDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

fn normalize_usage(model: &str, usage: XaiUsage, request_time_ms: u64) -> TokenUsage {
    let cache_read_tokens = usage
        .input_tokens_details
        .as_ref()
        .map(|details| details.cached_tokens)
        .unwrap_or(0);
    let clean_input_tokens = usage.input_tokens.saturating_sub(cache_read_tokens);
    let reasoning_tokens = usage
        .output_tokens_details
        .as_ref()
        .map(|details| details.reasoning_tokens)
        .unwrap_or(0);
    let cost = usage.cost_in_usd_ticks.map(ticks_to_usd).or_else(|| {
        calculate_cost(
            model,
            clean_input_tokens,
            cache_read_tokens,
            usage.output_tokens,
        )
    });
    let (output_tokens, reasoning_tokens) =
        TokenUsage::split_output(usage.output_tokens, reasoning_tokens);
    let total_tokens = if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input_tokens.saturating_add(usage.output_tokens)
    };

    TokenUsage {
        input_tokens: clean_input_tokens,
        cache_read_tokens,
        cache_write_tokens: 0,
        output_tokens,
        reasoning_tokens,
        total_tokens,
        cost,
        request_time_ms: Some(request_time_ms),
    }
}

fn text_from_output(output: &[Value]) -> String {
    let mut text = Vec::new();
    for item in output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
    {
        if let Some(parts) = item.get("content").and_then(Value::as_array) {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("output_text") {
                    if let Some(value) = part.get("text").and_then(Value::as_str) {
                        text.push(value);
                    }
                }
            }
        }
    }
    text.join("\n")
}

fn reasoning_from_output(output: &[Value]) -> Option<String> {
    let mut text = Vec::new();
    for item in output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
    {
        for field in ["summary", "content"] {
            if let Some(parts) = item.get(field).and_then(Value::as_array) {
                for part in parts {
                    if let Some(value) = part.get("text").and_then(Value::as_str) {
                        text.push(value);
                    }
                }
            }
        }
    }
    (!text.is_empty()).then(|| text.join("\n"))
}

fn tool_calls_from_output(output: &[Value]) -> Vec<ToolCall> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .filter_map(|item| {
            Some(ToolCall {
                id: item.get("call_id")?.as_str()?.to_string(),
                name: item.get("name")?.as_str()?.to_string(),
                arguments: item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(shared::parse_tool_call_arguments_lossy)
                    .unwrap_or_else(|| item.get("arguments").cloned().unwrap_or_else(|| json!({}))),
            })
        })
        .collect()
}

fn reasoning_meta(output: &[Value]) -> Option<Map<String, Value>> {
    let reasoning: Vec<Value> = output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("reasoning"))
        .cloned()
        .collect();
    if reasoning.is_empty() {
        None
    } else {
        Some(Map::from_iter([(
            REASONING_META_KEY.to_string(),
            Value::Array(reasoning),
        )]))
    }
}

fn rate_limit_headers(
    headers: &reqwest::header::HeaderMap,
) -> std::collections::HashMap<String, String> {
    let mut result = std::collections::HashMap::new();
    for name in [
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
    ] {
        if let Some(value) = headers.get(name).and_then(|value| value.to_str().ok()) {
            result.insert(name.to_string(), value.to_string());
        }
    }
    result
}

async fn execute_request(
    api_key: String,
    api_url: String,
    model: String,
    request: Value,
    params: &ChatCompletionParams,
) -> Result<ProviderResponse> {
    let started = std::time::Instant::now();
    let response = retry::retry_with_exponential_backoff(
        || {
            let client = shared::http_client();
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let request = request.clone();
            let extra_headers = params.extra_headers.clone();
            let request_timeout = params.request_timeout;
            Box::pin(async move {
                let captured = shared::send_and_read(
                    client
                        .post(api_url)
                        .bearer_auth(api_key)
                        .header("Content-Type", "application/json")
                        .json(&request),
                    request_timeout,
                    extra_headers.as_ref(),
                )
                .await?;
                if retry::is_retryable_status(captured.status.as_u16()) {
                    return Err(anyhow::anyhow!(
                        "xAI API error {}: {}",
                        captured.status,
                        captured.body
                    ));
                }
                Ok(captured)
            })
        },
        params.max_retries,
        params.retry_timeout,
        params.cancellation_token.as_ref(),
        || ProviderError::Cancelled.into(),
        |error| {
            matches!(
                error.downcast_ref::<ProviderError>(),
                Some(ProviderError::Cancelled)
            )
        },
        shared::is_connection_error,
    )
    .await?;

    if !response.status.is_success() {
        return Err(anyhow::anyhow!(
            "xAI API error {}: {}",
            response.status,
            response.body
        ));
    }

    let mut response_json: Value = serde_json::from_str(&response.body)?;
    let output = response_json
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let usage: XaiUsage = serde_json::from_value(
        response_json
            .get("usage")
            .cloned()
            .unwrap_or_else(|| json!({})),
    )?;
    let normalized_usage = normalize_usage(&model, usage, started.elapsed().as_millis() as u64);

    let content = text_from_output(&output);
    let calls = tool_calls_from_output(&output);
    let meta = reasoning_meta(&output);
    if !calls.is_empty() {
        shared::set_response_tool_calls(&mut response_json, &calls, meta.as_ref());
    }
    let thinking = reasoning_from_output(&output)
        .map(|text| ThinkingBlock::with_tokens(&text, normalized_usage.reasoning_tokens));
    let structured_output = shared::parse_structured_output_from_text(&content);
    let id = response_json
        .get("id")
        .and_then(Value::as_str)
        .map(|id| format!("{XAI_RESPONSE_ID_PREFIX}{id}"));
    let headers = rate_limit_headers(&response.headers);
    let exchange = if headers.is_empty() {
        ProviderExchange::new(request, response_json, Some(normalized_usage), "xai")
    } else {
        ProviderExchange::with_rate_limit_headers(
            request,
            response_json,
            Some(normalized_usage),
            "xai",
            headers,
        )
    };

    Ok(ProviderResponse {
        content,
        thinking,
        exchange,
        tool_calls: (!calls.is_empty()).then_some(calls),
        finish_reason: None,
        structured_output,
        id,
    })
}

/// xAI provider using the native Responses API.
#[derive(Debug, Clone, Default)]
pub struct XaiProvider;

impl XaiProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AiProvider for XaiProvider {
    fn name(&self) -> &str {
        "xai"
    }

    fn supports_model(&self, model: &str) -> bool {
        model_family(model).is_some()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(XAI_API_KEY_ENV).map_err(|_| {
            ProviderError::ApiKeyNotFound {
                provider: "xai".to_string(),
            }
            .into()
        })
    }

    fn supports_caching(&self, model: &str) -> bool {
        self.supports_model(model)
    }

    fn supports_vision(&self, model: &str) -> bool {
        self.supports_model(model)
    }

    fn supports_video(&self, _model: &str) -> bool {
        false
    }

    fn supports_structured_output(&self, model: &str) -> bool {
        self.supports_model(model)
    }

    fn enforces_response_schema(&self, model: &str) -> bool {
        self.supports_model(model)
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        match model_family(model) {
            Some(ModelFamily::Grok46 | ModelFamily::Grok45) => 500_000,
            Some(ModelFamily::Build) => 256_000,
            Some(ModelFamily::Grok43 | ModelFamily::Grok420) => 1_000_000,
            None => 0,
        }
    }

    fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
        usage_pricing(model, 0)
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        if self.supports_model(model) {
            SamplingSupport::TEMPERATURE_AND_TOP_P
        } else {
            SamplingSupport::NONE
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        if !self.supports_model(&params.model) {
            return Err(ProviderError::ModelNotSupported {
                provider: "xai".to_string(),
                model: params.model.clone(),
            }
            .into());
        }
        let api_key = self.get_api_key()?;
        let previous = resolve_previous_response_id(&params.messages, params.previous_id.clone());
        let mut request = build_request(&params, previous.as_deref());
        let sampling = self.effective_sampling_params(&params);
        if let Some(temperature) = sampling.temperature {
            request["temperature"] = json!(temperature);
        }
        if let Some(top_p) = sampling.top_p {
            request["top_p"] = json!(top_p);
        }
        let api_url = env::var(XAI_API_URL_ENV).unwrap_or_else(|_| XAI_API_URL.to_string());
        execute_request(api_key, api_url, params.model.clone(), request, &params).await
    }
}

#[cfg(test)]
#[path = "xai_tests.rs"]
mod tests;
