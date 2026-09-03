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

use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, SamplingSupport,
    ThinkingBlock, TokenUsage, ToolCall, ToolChoice,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAiCompatConfig {
    pub provider_name: &'static str,
    pub usage_fallback_cost: Option<f64>,
    pub use_response_cost: bool,
    pub enforces_response_schema: bool,
    pub supports_required_tool_choice: bool,
}

pub(crate) fn get_optional_api_key(env_name: &str) -> String {
    std::env::var(env_name).unwrap_or_default()
}

pub(crate) fn get_api_url(env_name: &str, default_url: &str) -> String {
    std::env::var(env_name).unwrap_or_else(|_| default_url.to_string())
}

fn reasoning_effort_value(
    provider_name: &str,
    model: &str,
    effort: crate::llm::types::ReasoningEffort,
) -> &'static str {
    if is_alibaba_deepseek_v4(provider_name, model) {
        use crate::llm::types::ReasoningEffort;

        // Model Studio accepts the OpenAI-standard five values for DeepSeek V4
        // and owns the model-specific collapse between them. Preserve the
        // caller's level verbatim instead of applying the generic high ceiling.
        return match effort {
            ReasoningEffort::Low => "low",
            ReasoningEffort::Medium => "medium",
            ReasoningEffort::High => "high",
            ReasoningEffort::XHigh => "xhigh",
            ReasoningEffort::Max => "max",
        };
    }

    match effort {
        crate::llm::types::ReasoningEffort::Low => "low",
        crate::llm::types::ReasoningEffort::Medium => "medium",
        crate::llm::types::ReasoningEffort::High => "high",
        // Meta Model API accepts "xhigh" as its ceiling ("none" is a 400):
        // map XHigh through instead of the generic downgrade.
        crate::llm::types::ReasoningEffort::XHigh if provider_name.eq_ignore_ascii_case("meta") => {
            "xhigh"
        }
        crate::llm::types::ReasoningEffort::XHigh => "high",
        crate::llm::types::ReasoningEffort::Max if provider_name.eq_ignore_ascii_case("ollama") => {
            "max"
        }
        // OpenCode routers forward the field verbatim to Moonshot, whose K3
        // top tier is "max" — keep it instead of the generic "high" downgrade.
        // Intermediate levels are floored by opencode::adjust_reasoning_effort.
        crate::llm::types::ReasoningEffort::Max
            if provider_name.starts_with("opencode")
                && crate::llm::utils::contains_ignore_ascii_case(model, "kimi-k3") =>
        {
            "max"
        }
        // Meta has no level above xhigh; collapse Max onto that ceiling.
        crate::llm::types::ReasoningEffort::Max if provider_name.eq_ignore_ascii_case("meta") => {
            "xhigh"
        }
        crate::llm::types::ReasoningEffort::Max => "high",
    }
}

fn is_alibaba_deepseek_v4(provider_name: &str, model: &str) -> bool {
    provider_name.eq_ignore_ascii_case("alibaba")
        && crate::llm::utils::contains_ignore_ascii_case(model, "deepseek-v4")
}

pub(crate) fn openai_tool_choice_value(choice: Option<&ToolChoice>) -> serde_json::Value {
    match choice {
        Some(ToolChoice::Required) => serde_json::json!("required"),
        Some(ToolChoice::None) => serde_json::json!("none"),
        Some(ToolChoice::Function(name)) => serde_json::json!({
            "type": "function",
            "function": {"name": name}
        }),
        Some(ToolChoice::Auto) | None => serde_json::json!("auto"),
    }
}

pub(crate) async fn chat_completion(
    config: OpenAiCompatConfig,
    api_key: String,
    api_url: String,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    chat_completion_with_sampling(config, SamplingSupport::ALL, api_key, api_url, params).await
}

/// Like [`chat_completion`], but omits sampling parameters the model rejects.
/// Proxy providers whose upstreams pin temperature/top_p (e.g. OpenCode routing
/// to Kimi K2.7/K3) must use this entry point.
pub(crate) async fn chat_completion_with_sampling(
    config: OpenAiCompatConfig,
    sampling: SamplingSupport,
    api_key: String,
    api_url: String,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    crate::llm::schema_enforcement::chat_completion_enforced(
        &OpenAiCompatTransport {
            config,
            sampling,
            api_key,
            api_url,
        },
        params,
    )
    .await
}

struct OpenAiCompatTransport {
    config: OpenAiCompatConfig,
    sampling: SamplingSupport,
    api_key: String,
    api_url: String,
}

#[async_trait::async_trait]
impl AiProvider for OpenAiCompatTransport {
    fn name(&self) -> &str {
        self.config.provider_name
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        Ok(self.api_key.clone())
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        self.config.enforces_response_schema
    }

    fn supports_required_tool_choice(&self, _model: &str) -> bool {
        self.config.supports_required_tool_choice
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        chat_completion_raw(
            self.config,
            self.sampling,
            self.api_key.clone(),
            self.api_url.clone(),
            None,
            params,
        )
        .await
    }

    async fn chat_completion_with_tool_choice(
        &self,
        params: ChatCompletionParams,
        tool_choice: ToolChoice,
    ) -> Result<ProviderResponse> {
        chat_completion_raw(
            self.config,
            self.sampling,
            self.api_key.clone(),
            self.api_url.clone(),
            Some(tool_choice),
            params,
        )
        .await
    }
}

async fn chat_completion_raw(
    config: OpenAiCompatConfig,
    sampling: SamplingSupport,
    api_key: String,
    api_url: String,
    tool_choice: Option<ToolChoice>,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let messages = convert_messages(&params.messages, config.provider_name, &params.model);

    let mut request_body = serde_json::json!({
        "model": params.model,
        "messages": messages,
    });

    let effective = sampling.effective(params.temperature, params.top_p, params.top_k);
    if let Some(temperature) = effective.temperature {
        request_body["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_p) = effective.top_p {
        request_body["top_p"] = serde_json::json!(top_p);
    }

    if config.provider_name.eq_ignore_ascii_case("ollama") {
        request_body["stream"] = serde_json::json!(false);
    }

    if params.max_tokens > 0 {
        request_body["max_tokens"] = serde_json::json!(params.max_tokens);
    }

    // Pass-through reasoning_effort as a top-level OpenAI-compat field.
    // Many providers (NVIDIA, Cerebras, Together, Groq, Fireworks, Cloudflare, Featherless,
    // OpenRouter, OctoHub, Google Vertex via OpenAI-compat) accept the standard
    // `reasoning_effort` parameter with values: "low" | "medium" | "high".
    // Providers that don't recognize it will ignore it.
    if let Some(effort) = params.reasoning_effort {
        let s = reasoning_effort_value(config.provider_name, &params.model, effort);
        request_body["reasoning_effort"] = serde_json::json!(s);
    }

    // DeepSeek V4 is a hybrid-thinking family. An explicit effort means the
    // caller selected thinking, so preserve that intent on Model Studio rather
    // than relying only on the moving alias' current default.
    if params.reasoning_effort.is_some()
        && is_alibaba_deepseek_v4(config.provider_name, &params.model)
    {
        request_body["enable_thinking"] = serde_json::json!(true);
    }

    if let Some(tools) = &params.tools {
        if !tools.is_empty() {
            let mut sorted_tools = tools.clone();
            sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

            let openai_tools = sorted_tools
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": f.name,
                            "description": f.description,
                            "parameters": f.parameters
                        }
                    })
                })
                .collect::<Vec<_>>();

            request_body["tools"] = serde_json::json!(openai_tools);
            request_body["tool_choice"] = openai_tool_choice_value(tool_choice.as_ref());
            // Explicit: ensures proxied backends honor parallel function calling
            // rather than falling back to their own defaults (which may differ).
            request_body["parallel_tool_calls"] = serde_json::json!(true);
        }
    }

    if let Some(response_format) = &params.response_format {
        apply_response_format(
            &mut request_body,
            config.provider_name,
            &params.model,
            response_format,
        );
    }

    execute_request(config, api_key, api_url, request_body, params).await
}

fn apply_response_format(
    request_body: &mut serde_json::Value,
    provider_name: &str,
    model: &str,
    response_format: &crate::llm::types::StructuredOutputRequest,
) {
    // Ollama and local servers use a top-level "format" key instead of
    // "response_format": "json" for object mode, or the schema itself.
    let is_ollama_like =
        provider_name.eq_ignore_ascii_case("ollama") || provider_name.eq_ignore_ascii_case("local");

    match &response_format.format {
        crate::llm::types::OutputFormat::Json => {
            if is_ollama_like {
                request_body["format"] = serde_json::json!("json");
            } else {
                request_body["response_format"] = serde_json::json!({
                    "type": "json_object"
                });
                // Model Studio rejects json_object requests unless the word
                // "JSON" appears somewhere in the messages.
                if provider_name.eq_ignore_ascii_case("alibaba") {
                    if let Some(messages) = request_body["messages"].as_array_mut() {
                        messages.push(serde_json::json!({
                            "role": "system",
                            "content": "Respond with a single JSON object."
                        }));
                    }
                }
            }
        }
        crate::llm::types::OutputFormat::JsonSchema => {
            if is_ollama_like {
                if let Some(schema) = &response_format.schema {
                    request_body["format"] = schema.clone();
                } else {
                    request_body["format"] = serde_json::json!("json");
                }
            } else if is_alibaba_deepseek_v4(provider_name, model) {
                // Model Studio exposes JSON Object, not JSON Schema, for
                // DeepSeek V4. Keep the requested schema in prompt guidance;
                // schema_enforcement validates it locally and falls back to the
                // synthetic schema tool when guidance is ignored.
                if let Some(schema) = &response_format.schema {
                    request_body["response_format"] = serde_json::json!({
                        "type": "json_object"
                    });
                    if let Some(messages) = request_body["messages"].as_array_mut() {
                        messages.push(serde_json::json!({
                            "role": "system",
                            "content": format!(
                                "When you produce the final answer, return one JSON object that conforms exactly to this JSON schema, with no markdown fences or trailing prose:\n{schema}"
                            )
                        }));
                    }
                }
            } else if let Some(schema) = &response_format.schema {
                // Strict structured outputs need additionalProperties:false on
                // every nested object (no-op unless mode is Strict).
                let schema =
                    crate::llm::utils::normalize_strict_schema(schema, response_format.mode);

                let mut format_obj = serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema
                    }
                });

                if matches!(
                    response_format.mode,
                    crate::llm::types::ResponseMode::Strict
                ) {
                    format_obj["json_schema"]["strict"] = serde_json::json!(true);
                }

                request_body["response_format"] = format_obj;
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct OpenAiCompatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiCompatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    /// Ollama's OpenAI compatibility layer maps this field to its native
    /// `message.thinking` value. Replaying it is required by Kimi models that
    /// preserve reasoning across turns.
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAiCompatToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiCompatFunction,
    /// Gemini thought signatures (`extra_content.google.thought_signature`).
    /// Gemini 3 rejects replayed tool-call turns that omit them, so this field
    /// must round-trip through history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extra_content: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAiCompatFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenAiCompatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    choices: Vec<OpenAiCompatChoice>,
    #[serde(default)]
    usage: Option<OpenAiCompatUsage>,
    #[serde(default)]
    message: Option<OpenAiCompatResponseMessage>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    done_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiCompatChoice {
    message: OpenAiCompatResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Clone)]
struct OpenAiCompatResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiCompatToolCall>>,
    #[serde(default)]
    reasoning_details: Option<serde_json::Value>,
    /// OpenAI-compatible reasoning text. Ollama emits `reasoning`; Kimi/vLLM
    /// deployments commonly emit `reasoning_content`; Ollama's native chat
    /// response uses `thinking`.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiCompatUsage {
    #[serde(default)]
    input_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    output_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    #[serde(default)]
    reasoning_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    total_cost: Option<f64>,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    prompt_cost: Option<f64>,
    #[serde(default)]
    completion_cost: Option<f64>,
}

#[derive(Deserialize, Debug)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

fn convert_messages(
    messages: &[Message],
    provider_name: &str,
    model: &str,
) -> Vec<OpenAiCompatMessage> {
    let mut result = Vec::new();

    // Reasoning replay is per-request context the provider re-renders (and bills)
    // every call on cache-less endpoints, so only the trailing assistant message —
    // the chain the model is continuing — carries it. Kimi models are the
    // exception: they require their reasoning preserved across every turn.
    let last_assistant = messages.iter().rposition(|m| m.role == "assistant");
    let replays_reasoning = |idx: usize| {
        provider_name.eq_ignore_ascii_case("ollama")
            && (Some(idx) == last_assistant
                || crate::llm::providers::moonshot::preserves_historical_thinking(model))
    };

    for (msg_idx, message) in messages.iter().enumerate() {
        match message.role.as_str() {
            "tool" => {
                result.push(OpenAiCompatMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: message.tool_call_id.clone(),
                    reasoning: None,
                });
            }
            "assistant" if message.tool_calls.is_some() => {
                let content = if !message.content.trim().is_empty() {
                    Some(serde_json::json!(message.content))
                } else {
                    None
                };

                let tool_calls = if let Ok(generic_calls) =
                    serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(
                        message.tool_calls.clone().unwrap_or_default(),
                    ) {
                    Some(
                        generic_calls
                            .iter()
                            .map(|tc| OpenAiCompatToolCall {
                                id: tc.id.clone(),
                                tool_type: "function".to_string(),
                                function: OpenAiCompatFunction {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                                extra_content: tc
                                    .meta
                                    .as_ref()
                                    .and_then(|m| m.get("extra_content"))
                                    .cloned(),
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                result.push(OpenAiCompatMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                    reasoning: if replays_reasoning(msg_idx) {
                        message.thinking.as_ref().map(|t| t.content.clone())
                    } else {
                        None
                    },
                });
            }
            "user" | "assistant" | "system" => {
                let has_images = message.images.as_ref().is_some_and(|imgs| !imgs.is_empty());
                let has_videos = message.videos.as_ref().is_some_and(|vids| !vids.is_empty());

                let content = if has_images || has_videos {
                    let mut content_parts = vec![serde_json::json!({
                        "type": "text",
                        "text": message.content
                    })];

                    if let Some(images) = &message.images {
                        for image in images {
                            let url = match &image.data {
                                crate::llm::types::ImageData::Base64(data) => {
                                    format!("data:{};base64,{}", image.media_type, data)
                                }
                                crate::llm::types::ImageData::Url(u) => u.clone(),
                            };
                            content_parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": url
                                }
                            }));
                        }
                    }

                    if let Some(videos) = &message.videos {
                        for video in videos {
                            let url = match &video.data {
                                crate::llm::types::VideoData::Base64(data) => {
                                    format!("data:{};base64,{}", video.media_type, data)
                                }
                                crate::llm::types::VideoData::Url(u) => u.clone(),
                            };
                            content_parts.push(serde_json::json!({
                                "type": "video_url",
                                "video_url": {
                                    "url": url
                                }
                            }));
                        }
                    }

                    Some(serde_json::json!(content_parts))
                } else {
                    Some(serde_json::json!(message.content))
                };

                result.push(OpenAiCompatMessage {
                    role: message.role.clone(),
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                    reasoning: if message.role == "assistant" && replays_reasoning(msg_idx) {
                        message.thinking.as_ref().map(|t| t.content.clone())
                    } else {
                        None
                    },
                });
            }
            _ => {
                tracing::warn!("Unknown message role: {}", message.role);
            }
        }
    }

    result
}

fn extract_thinking(message: &OpenAiCompatResponseMessage) -> Option<ThinkingBlock> {
    let plain_text = message
        .reasoning_content
        .as_ref()
        .or(message.reasoning.as_ref())
        .or(message.thinking.as_ref());

    if let Some(content) = plain_text {
        return Some(ThinkingBlock {
            content: content.clone(),
            tokens: (content.len() / 4) as u64,
        });
    }

    message.reasoning_details.as_ref().map(|details| {
        let content = details
            .as_array()
            .and_then(|items| {
                let texts = items
                    .iter()
                    .filter_map(|item| item.get("text").and_then(serde_json::Value::as_str))
                    .collect::<Vec<_>>();
                (!texts.is_empty()).then(|| texts.join("\n\n"))
            })
            .unwrap_or_else(|| details.to_string());

        ThinkingBlock {
            tokens: (content.len() / 4) as u64,
            content,
        }
    })
}

async fn execute_request(
    config: OpenAiCompatConfig,
    api_key: String,
    api_url: String,
    request_body: serde_json::Value,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let start_time = std::time::Instant::now();
    let request_timeout = params.request_timeout;

    let response = retry::retry_with_exponential_backoff(
        || {
            let client = shared::http_client();
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let request_body = request_body.clone();
            let provider_name = config.provider_name.to_string();
            let extra_headers = params.extra_headers.clone();

            Box::pin(async move {
                let mut request = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .json(&request_body);

                if !api_key.is_empty() {
                    request = request.header("Authorization", format!("Bearer {}", api_key));
                }

                let captured =
                    shared::send_and_read(request, request_timeout, extra_headers.as_ref()).await?;

                // Return Err for retryable HTTP errors so the retry loop catches them
                if retry::is_retryable_status(captured.status.as_u16()) {
                    return Err(anyhow::anyhow!(
                        "{} API error {}: {}",
                        provider_name,
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
        |e| {
            matches!(
                e.downcast_ref::<ProviderError>(),
                Some(ProviderError::Cancelled)
            )
        },
        |e: &anyhow::Error| shared::is_connection_error(e),
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status.is_success() {
        return Err(anyhow::anyhow!(
            "{} API error {}: {}",
            config.provider_name,
            response.status,
            response.body
        ));
    }

    let response_text = response.body;
    let api_response: OpenAiCompatResponse = serde_json::from_str(&response_text)?;

    let (message, finish_reason) = if let Some(choice) = api_response.choices.into_iter().next() {
        (choice.message, choice.finish_reason)
    } else if let Some(message) = api_response.message.clone() {
        (message, api_response.done_reason.clone())
    } else {
        return Err(anyhow::anyhow!("No choices/message in response"));
    };

    let content = message.content.clone().unwrap_or_default();

    let reasoning_details = &message.reasoning_details;
    let thinking = extract_thinking(&message);

    // Per-call extra_content (Gemini thought signatures), filtered the same way
    // as the conversion below so indexes stay aligned with `tool_calls`.
    let tool_call_extras: Vec<Option<serde_json::Value>> = message
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .filter(|c| c.tool_type == "function")
                .map(|c| c.extra_content.clone())
                .collect()
        })
        .unwrap_or_default();

    let tool_calls: Option<Vec<ToolCall>> = message.tool_calls.map(|calls| {
        calls
            .into_iter()
            .filter_map(|call| {
                if call.tool_type != "function" {
                    tracing::warn!("Unexpected tool type: {}", call.tool_type);
                    return None;
                }

                let arguments: serde_json::Value =
                    serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::json!({}));

                Some(ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments,
                })
            })
            .collect()
    });

    let input_tokens = api_response
        .usage
        .as_ref()
        .and_then(|u| u.input_tokens.or(u.prompt_tokens))
        .or(api_response.prompt_eval_count)
        .unwrap_or(0);

    let output_tokens = api_response
        .usage
        .as_ref()
        .and_then(|u| u.completion_tokens.or(u.output_tokens))
        .or(api_response.eval_count)
        .unwrap_or(0);

    let total_tokens = api_response
        .usage
        .as_ref()
        .and_then(|u| u.total_tokens)
        .unwrap_or(input_tokens.saturating_add(output_tokens));

    let cache_read_tokens = api_response
        .usage
        .as_ref()
        .and_then(|u| u.prompt_tokens_details.as_ref().map(|d| d.cached_tokens))
        .unwrap_or(0);

    let reasoning_tokens = api_response
        .usage
        .as_ref()
        .and_then(|u| {
            u.reasoning_tokens.or(u
                .completion_tokens_details
                .as_ref()
                .map(|d| d.reasoning_tokens))
        })
        .or_else(|| thinking.as_ref().map(|t| t.tokens))
        .unwrap_or(0);

    let response_cost = api_response.usage.as_ref().and_then(|u| {
        u.total_cost
            .or(u.cost)
            .or(match (u.prompt_cost, u.completion_cost) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            })
    });
    let cost = if config.use_response_cost {
        response_cost.or(config.usage_fallback_cost)
    } else {
        config.usage_fallback_cost
    };

    let usage = if api_response.usage.is_some()
        || api_response.prompt_eval_count.is_some()
        || api_response.eval_count.is_some()
        || reasoning_tokens > 0
        || cost.is_some()
    {
        let (output_tokens, reasoning_tokens) =
            TokenUsage::split_output(output_tokens, reasoning_tokens);
        Some(TokenUsage {
            input_tokens: input_tokens.saturating_sub(cache_read_tokens),
            cache_read_tokens,
            cache_write_tokens: 0,
            output_tokens,
            reasoning_tokens,
            total_tokens,
            cost,
            request_time_ms: Some(request_time_ms),
        })
    } else {
        None
    };

    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;
    if let Some(ref tc) = tool_calls {
        let reasoning_meta = if config.provider_name.eq_ignore_ascii_case("local") {
            reasoning_details.as_ref().map(|rd| {
                let mut m = serde_json::Map::new();
                m.insert("reasoning_details".to_string(), rd.clone());
                m
            })
        } else {
            None
        };
        shared::set_response_tool_calls(&mut response_json, tc, reasoning_meta.as_ref());

        // Attach per-call extra_content to the unified tool_calls meta so history
        // replays can send Gemini thought signatures back (Gemini 3 rejects
        // tool-call turns without them).
        if tool_call_extras.iter().any(Option::is_some) {
            if let Some(arr) = response_json["tool_calls"].as_array_mut() {
                for (call, extra) in arr.iter_mut().zip(&tool_call_extras) {
                    if let Some(extra) = extra {
                        call["meta"]["extra_content"] = extra.clone();
                    }
                }
            }
        }
    }

    let exchange = ProviderExchange::new(request_body, response_json, usage, config.provider_name);

    let structured_output = shared::parse_structured_output_from_text(&content);

    Ok(ProviderResponse {
        content,
        thinking,
        exchange,
        tool_calls,
        finish_reason,
        structured_output,
        id: api_response.id,
    })
}

#[cfg(test)]
#[path = "openai_compat_tests.rs"]
mod tests;
