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

//! Together.ai provider implementation
//!
//! Together.ai provides access to open-source models via OpenAI-compatible API.

use super::shared;
use crate::errors::ProviderError;
use crate::errors::ToolCallError;
use crate::llm::reference_models::proxy_route_enforces_response_schema;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, SamplingSupport,
    ThinkingBlock, TokenUsage, ToolCall,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

const TOGETHER_API_KEY_ENV: &str = "TOGETHER_API_KEY";
const TOGETHER_API_URL: &str = "https://api.together.xyz/v1/chat/completions";

#[derive(Debug, Clone, Default)]
pub struct TogetherProvider;

impl TogetherProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AiProvider for TogetherProvider {
    fn name(&self) -> &str {
        "together"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }
    fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
        // Together uses OpenAI-compatible API — supports temperature and top_p, not top_k.
        SamplingSupport::TEMPERATURE_AND_TOP_P
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(TOGETHER_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Together.ai API key not found. Set {} environment variable.",
                TOGETHER_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Together performs automatic prompt-prefix caching (no opt-in param)
        // and reports `cached_tokens` in usage on supported models.
        true
    }

    // supports_vision, supports_video, supports_structured_output
    // are resolved via reference capabilities (trait defaults)

    fn enforces_response_schema(&self, model: &str) -> bool {
        proxy_route_enforces_response_schema(model)
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try reference pricing for cost estimation based on the underlying model
        crate::llm::reference_models::get_reference_pricing(model)
    }

    // get_max_input_tokens resolved via reference capabilities (trait default)

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let messages = convert_messages(&params.messages)?;

        // Apply sampling parameters based on model support
        let sampling = self.effective_sampling_params(&params);

        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": messages,
        });
        if let Some(temp) = sampling.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = sampling.top_p {
            request_body["top_p"] = serde_json::json!(top_p);
        }
        // Note: Together doesn't support top_k

        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Together returns Server-Sent Events, and some models (e.g. large MoE
        // deployments) ONLY support streaming and reject non-streamed requests with
        // a 400 `streaming_required`. So always stream and merge the chunks back into
        // a single response. `include_usage` appends a final usage chunk so token
        // accounting and cost still work.
        request_body["stream"] = serde_json::json!(true);
        request_body["stream_options"] = serde_json::json!({ "include_usage": true });

        // Pass-through reasoning_effort for Together's OpenAI-compatible thinking models
        // (e.g. DeepSeek-R1, Qwen3-Thinking). Models without it ignore the field.
        if let Some(effort) = params.reasoning_effort {
            let s = match effort {
                crate::llm::types::ReasoningEffort::Low => "low",
                crate::llm::types::ReasoningEffort::Medium => "medium",
                crate::llm::types::ReasoningEffort::High => "high",
                crate::llm::types::ReasoningEffort::XHigh => "high",
                crate::llm::types::ReasoningEffort::Max => "high",
            };
            request_body["reasoning_effort"] = serde_json::json!(s);
        }

        // Tools (OpenAI-compatible)
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let mut sorted_tools = tools.clone();
                sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));
                let openai_tools: Vec<serde_json::Value> = sorted_tools
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
                    .collect();
                request_body["tools"] = serde_json::json!(openai_tools);
                request_body["tool_choice"] = serde_json::json!("auto");
            }
        }

        // Structured output (OpenAI-compatible)
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request_body["response_format"] = serde_json::json!({"type": "json_object"});
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    if let Some(schema) = &response_format.schema {
                        // Strict structured outputs need additionalProperties:false on
                        // every nested object (no-op unless mode is Strict).
                        let schema = crate::llm::utils::normalize_strict_schema(
                            schema,
                            response_format.mode,
                        );

                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": { "name": "response", "schema": schema }
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

        execute_together_request(
            api_key,
            request_body,
            params.max_retries,
            params.retry_timeout,
            params.request_timeout,
            params.cancellation_token.as_ref(),
        )
        .await
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct TogetherMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>,
}

/// One Server-Sent-Events chunk from Together's streaming response.
#[derive(Deserialize, Debug)]
struct TogetherStreamChunk {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    choices: Vec<TogetherStreamChoice>,
    /// Present only in the final chunk (stream_options.include_usage = true).
    #[serde(default)]
    usage: Option<TogetherUsage>,
}

#[derive(Deserialize, Debug)]
struct TogetherStreamChoice {
    #[serde(default)]
    delta: TogetherStreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct TogetherStreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Dedicated chain-of-thought field on reasoning models (Qwen3, GPT-OSS, GLM…).
    /// DeepSeek-R1 instead embeds it as `<think>…</think>` inside `content`.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<TogetherStreamToolCall>>,
}

/// A streamed tool-call fragment: `id`/`name` arrive once, `arguments` in pieces.
#[derive(Deserialize, Debug)]
struct TogetherStreamToolCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<TogetherStreamFunction>,
}

#[derive(Deserialize, Debug)]
struct TogetherStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct TogetherUsage {
    #[serde(default)]
    prompt_tokens: Option<u64>,
    #[serde(default)]
    completion_tokens: Option<u64>,
    #[serde(default)]
    total_tokens: Option<u64>,
    /// Together-only top-level field; may also arrive OpenAI-style nested
    #[serde(default)]
    cached_tokens: Option<u64>,
    #[serde(default)]
    prompt_tokens_details: Option<TogetherPromptTokensDetails>,
}

#[derive(Deserialize, Debug, Default)]
struct TogetherPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

/// A tool call being assembled from streaming deltas.
#[derive(Default)]
struct ToolCallBuilder {
    id: String,
    name: String,
    arguments: String,
}

/// All streaming deltas folded into a single logical response.
#[derive(Default)]
struct MergedStream {
    id: Option<String>,
    content: String,
    reasoning: String,
    finish_reason: Option<String>,
    tool_calls: Vec<ToolCallBuilder>,
    usage: Option<TogetherUsage>,
    parsed_any: bool,
}

/// Parse Together's SSE body (`data: {json}` lines, terminated by `data: [DONE]`)
/// and fold every delta into one [`MergedStream`]. Unparseable / keep-alive lines
/// are skipped; `parsed_any` reports whether at least one real chunk was seen.
fn merge_sse_stream(body: &str) -> MergedStream {
    let mut merged = MergedStream::default();

    for line in body.lines() {
        // SSE event lines look like `data: {...}`; skip blanks, comments (`:`), etc.
        let Some(data) = line.trim_start().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<TogetherStreamChunk>(data) else {
            continue;
        };
        merged.parsed_any = true;

        if merged.id.is_none() {
            merged.id = chunk.id;
        }
        if chunk.usage.is_some() {
            merged.usage = chunk.usage;
        }

        for choice in chunk.choices {
            if let Some(c) = choice.delta.content {
                merged.content.push_str(&c);
            }
            if let Some(r) = choice.delta.reasoning {
                merged.reasoning.push_str(&r);
            }
            if choice.finish_reason.is_some() {
                merged.finish_reason = choice.finish_reason;
            }
            for tc in choice.delta.tool_calls.into_iter().flatten() {
                if merged.tool_calls.len() <= tc.index {
                    merged
                        .tool_calls
                        .resize_with(tc.index + 1, ToolCallBuilder::default);
                }
                let slot = &mut merged.tool_calls[tc.index];
                if let Some(id) = tc.id {
                    if !id.is_empty() {
                        slot.id = id;
                    }
                }
                if let Some(f) = tc.function {
                    if let Some(name) = f.name {
                        if !name.is_empty() {
                            slot.name = name;
                        }
                    }
                    if let Some(args) = f.arguments {
                        slot.arguments.push_str(&args);
                    }
                }
            }
        }
    }

    merged
}

fn convert_messages(messages: &[Message]) -> Result<Vec<TogetherMessage>, ToolCallError> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                result.push(TogetherMessage {
                    role: message.role.clone(),
                    content: serde_json::json!(message.content),
                    tool_call_id: message.tool_call_id.clone(),
                    name: message.name.clone(),
                    tool_calls: None,
                });
            }
            "assistant" if message.tool_calls.is_some() => {
                let content = if message.content.trim().is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(message.content)
                };

                let Some(tool_calls_value) = message.tool_calls.as_ref() else {
                    return Err(ToolCallError::MissingField {
                        field: "tool_calls".to_string(),
                    });
                };
                let generic_calls =
                    shared::parse_generic_tool_calls_strict(tool_calls_value, "together")?;

                let together_calls: Vec<serde_json::Value> = generic_calls
                    .into_iter()
                    .map(|call| {
                        serde_json::json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": shared::arguments_to_json_string(&call.arguments)
                            }
                        })
                    })
                    .collect();

                result.push(TogetherMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls: Some(serde_json::Value::Array(together_calls)),
                });
            }
            _ => {
                let mut content_parts = vec![serde_json::json!({
                    "type": "text",
                    "text": message.content
                })];

                if let Some(images) = &message.images {
                    for image in images {
                        if let crate::llm::types::ImageData::Base64(data) = &image.data {
                            content_parts.push(serde_json::json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": format!("data:{};base64,{}", image.media_type, data)
                                }
                            }));
                        }
                    }
                }

                if let Some(videos) = &message.videos {
                    for video in videos {
                        match &video.data {
                            crate::llm::types::VideoData::Base64(data) => {
                                content_parts.push(serde_json::json!({
                                    "type": "video_url",
                                    "video_url": {
                                        "url": format!("data:{};base64,{}", video.media_type, data)
                                    }
                                }));
                            }
                            crate::llm::types::VideoData::Url(url) => {
                                content_parts.push(serde_json::json!({
                                    "type": "video_url",
                                    "video_url": { "url": url }
                                }));
                            }
                        }
                    }
                }

                let content = if content_parts.len() == 1 {
                    content_parts[0]["text"].clone()
                } else {
                    serde_json::json!(content_parts)
                };

                result.push(TogetherMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls: None,
                });
            }
        }
    }

    Ok(result)
}

async fn execute_together_request(
    api_key: String,
    request_body: serde_json::Value,
    max_retries: u32,
    base_timeout: std::time::Duration,
    request_timeout: Option<std::time::Duration>,
    cancellation_token: Option<&tokio::sync::watch::Receiver<bool>>,
) -> Result<ProviderResponse> {
    let start_time = std::time::Instant::now();

    let response = retry::retry_with_exponential_backoff(
        || {
            let client = shared::http_client();
            let api_key = api_key.clone();
            let request_body = request_body.clone();

            Box::pin(async move {
                let req = client
                    .post(TOGETHER_API_URL)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .json(&request_body);

                let captured = shared::send_and_read(req, request_timeout).await?;

                if retry::is_retryable_status(captured.status.as_u16()) {
                    return Err(anyhow::anyhow!(
                        "Together.ai API error {}: {}",
                        captured.status,
                        captured.body
                    ));
                }

                Ok(captured)
            })
        },
        max_retries,
        base_timeout,
        cancellation_token,
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
            "Together.ai API error {}: {}",
            response.status,
            response.body
        ));
    }

    let merged = merge_sse_stream(&response.body);
    if !merged.parsed_any {
        return Err(anyhow::anyhow!(
            "Together.ai: could not parse streaming response: {}",
            response.body
        ));
    }

    // Reasoning models surface chain-of-thought in a dedicated `reasoning` stream or
    // as inline <think>…</think> tags (DeepSeek-R1). Pull it out, keep content clean.
    let reasoning = (!merged.reasoning.trim().is_empty()).then_some(merged.reasoning);
    let (thinking, content) = extract_thinking(&merged.content, reasoning);

    let tool_calls: Option<Vec<ToolCall>> = {
        let calls: Vec<ToolCall> = merged
            .tool_calls
            .into_iter()
            .filter(|b| !b.name.is_empty())
            .map(|b| ToolCall {
                id: b.id,
                name: b.name,
                arguments: shared::parse_tool_call_arguments_lossy(&b.arguments),
            })
            .collect();
        (!calls.is_empty()).then_some(calls)
    };

    let usage_data = merged.usage.unwrap_or_default();
    let prompt_tokens = usage_data.prompt_tokens.unwrap_or(0);
    let output_tokens = usage_data.completion_tokens.unwrap_or(0);
    let total_tokens = usage_data
        .total_tokens
        .unwrap_or(prompt_tokens.saturating_add(output_tokens));
    let cache_read_tokens = usage_data
        .cached_tokens
        .or_else(|| {
            usage_data
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
        })
        .unwrap_or(0);
    let input_tokens = prompt_tokens.saturating_sub(cache_read_tokens);

    let usage = TokenUsage {
        input_tokens,
        output_tokens,
        cache_write_tokens: 0,
        cache_read_tokens,
        reasoning_tokens: thinking.as_ref().map(|t| t.tokens).unwrap_or(0),
        total_tokens,
        cost: request_body
            .get("model")
            .and_then(|m| m.as_str())
            .and_then(crate::llm::reference_models::get_reference_pricing)
            .map(|pricing| {
                pricing.calculate_cost(input_tokens, 0, cache_read_tokens, output_tokens)
            }),
        request_time_ms: Some(request_time_ms),
    };

    // Reconstruct a normalized (non-streamed) chat-completion shape for the exchange log.
    let mut response_json = serde_json::json!({
        "id": merged.id.clone(),
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content.clone(),
                "reasoning": thinking.as_ref().map(|t| t.content.clone()),
            },
            "finish_reason": merged.finish_reason.clone(),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cache_read_tokens,
        },
    });

    if let Some(ref tc) = tool_calls {
        shared::set_response_tool_calls(&mut response_json, tc, None);
    }

    let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "together");
    let structured_output = shared::parse_structured_output_from_text(&content);

    Ok(ProviderResponse {
        content,
        thinking,
        exchange,
        tool_calls,
        finish_reason: merged.finish_reason,
        structured_output,
        id: merged.id,
    })
}

/// Extract thinking content from a dedicated `reasoning` field or inline
/// `<think>…</think>` tags, returning the thinking block (if any) and the
/// cleaned content. Mirrors the Z.ai provider's reasoning handling.
fn extract_thinking(content: &str, reasoning: Option<String>) -> (Option<ThinkingBlock>, String) {
    // Priority 1: dedicated `reasoning` field (most Together reasoning models)
    if let Some(ref thinking_str) = reasoning {
        if !thinking_str.trim().is_empty() {
            let tokens = (thinking_str.len() / 4) as u64;
            return (
                Some(ThinkingBlock {
                    content: thinking_str.clone(),
                    tokens,
                }),
                content.to_string(),
            );
        }
    }

    // Priority 2: inline <think>…</think> tags (DeepSeek-R1 on Together)
    let think_start = "<think>";
    let think_end = "</think>";
    if let Some(start_idx) = content.find(think_start) {
        if let Some(end_idx) = content.find(think_end) {
            let thinking_content = &content[start_idx + think_start.len()..end_idx];
            let before = &content[..start_idx];
            let after = &content[end_idx + think_end.len()..];
            let clean = format!("{}{}", before.trim(), after.trim())
                .trim()
                .to_string();
            let tokens = (thinking_content.len() / 4) as u64;
            return (
                Some(ThinkingBlock {
                    content: thinking_content.to_string(),
                    tokens,
                }),
                clean,
            );
        }
    }

    (None, content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = TogetherProvider::new();
        assert!(provider.supports_model("meta-llama/Llama-3.3-70B-Instruct-Turbo"));
        assert!(provider.supports_model("moonshotai/Kimi-K2.5"));
        assert!(provider.supports_model("any-model-name"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_extract_thinking_reasoning_field() {
        let (thinking, content) =
            extract_thinking("the answer is 42", Some("let me think...".to_string()));
        assert_eq!(thinking.unwrap().content, "let me think...");
        assert_eq!(content, "the answer is 42"); // content untouched
    }

    #[test]
    fn test_extract_thinking_think_tags() {
        let (thinking, content) = extract_thinking("<think>internal</think>visible answer", None);
        assert_eq!(thinking.unwrap().content, "internal");
        assert_eq!(content, "visible answer"); // tags stripped
    }

    #[test]
    fn test_merge_sse_stream_content_and_usage() {
        let body = "\
data: {\"id\":\"abc\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\
data: {\"choices\":[{\"delta\":{\"reasoning\":\"think \"}}]}\n\
data: {\"choices\":[{\"delta\":{\"reasoning\":\"more\"}}]}\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\
data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"cached_tokens\":4}}\n\
data: [DONE]\n";
        let m = merge_sse_stream(body);
        assert!(m.parsed_any);
        assert_eq!(m.id.as_deref(), Some("abc"));
        assert_eq!(m.content, "Hello world");
        assert_eq!(m.reasoning, "think more");
        assert_eq!(m.finish_reason.as_deref(), Some("stop"));
        let u = m.usage.unwrap();
        assert_eq!(u.prompt_tokens, Some(10));
        assert_eq!(u.cached_tokens, Some(4));
    }

    #[test]
    fn test_merge_sse_stream_tool_call_fragments() {
        // id+name arrive once; arguments stream as fragments and must concatenate.
        let body = "\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ty\\\":\\\"NYC\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
        let m = merge_sse_stream(body);
        assert_eq!(m.tool_calls.len(), 1);
        assert_eq!(m.tool_calls[0].id, "call_1");
        assert_eq!(m.tool_calls[0].name, "get_weather");
        assert_eq!(m.tool_calls[0].arguments, "{\"city\":\"NYC\"}");
        assert_eq!(m.finish_reason.as_deref(), Some("tool_calls"));
    }

    #[test]
    fn test_merge_sse_stream_ignores_garbage() {
        let m = merge_sse_stream(": keep-alive\n\ndata: not-json\n");
        assert!(!m.parsed_any);
        assert_eq!(m.content, "");
    }

    #[test]
    fn test_extract_thinking_none() {
        let (thinking, content) = extract_thinking("plain answer", None);
        assert!(thinking.is_none());
        assert_eq!(content, "plain answer");
        // empty/whitespace reasoning is ignored
        let (thinking, _) = extract_thinking("x", Some("  ".to_string()));
        assert!(thinking.is_none());
    }
}
