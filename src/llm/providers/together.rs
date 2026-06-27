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

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Try reference pricing for cost estimation based on the underlying model
        crate::llm::reference_pricing::get_reference_pricing(model)
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

#[derive(Deserialize, Debug)]
struct TogetherResponse {
    id: String,
    choices: Vec<TogetherChoice>,
    usage: TogetherUsage,
}

#[derive(Deserialize, Debug)]
struct TogetherChoice {
    message: TogetherResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct TogetherResponseMessage {
    content: Option<String>,
    /// Reasoning models (Qwen3, GPT-OSS, GLM, …) return their chain-of-thought
    /// in a dedicated nullable `reasoning` field alongside `content`.
    /// DeepSeek-R1 instead embeds it as `<think>…</think>` inside `content`.
    #[serde(default)]
    reasoning: Option<String>,
    tool_calls: Option<Vec<TogetherToolCall>>,
}

#[derive(Deserialize, Debug)]
struct TogetherToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: TogetherFunction,
}

#[derive(Deserialize, Debug)]
struct TogetherFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug)]
struct TogetherPromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
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

    let response_text = response.body;
    let together_response: TogetherResponse = serde_json::from_str(&response_text)?;

    let choice = together_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in Together.ai response"))?;

    // Together reasoning models surface chain-of-thought either in a dedicated
    // `reasoning` field or as inline <think>…</think> tags (DeepSeek-R1). Pull it
    // out so it isn't silently dropped, and keep `content` clean.
    let raw_content = choice.message.content.unwrap_or_default();
    let (thinking, content) = extract_thinking(&raw_content, choice.message.reasoning);

    let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
        calls
            .into_iter()
            .filter_map(|call| {
                if call.tool_type != "function" {
                    tracing::warn!(
                        "Unexpected tool type '{}' from Together.ai API",
                        call.tool_type
                    );
                    return None;
                }
                let arguments = shared::parse_tool_call_arguments_lossy(&call.function.arguments);
                Some(ToolCall {
                    id: call.id,
                    name: call.function.name,
                    arguments,
                })
            })
            .collect()
    });

    let prompt_tokens = together_response.usage.prompt_tokens.unwrap_or(0);
    let output_tokens = together_response.usage.completion_tokens.unwrap_or(0);
    let total_tokens = together_response
        .usage
        .total_tokens
        .unwrap_or(prompt_tokens.saturating_add(output_tokens));
    let cache_read_tokens = together_response
        .usage
        .cached_tokens
        .or_else(|| {
            together_response
                .usage
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
            .and_then(crate::llm::reference_pricing::get_reference_pricing)
            .map(|pricing| {
                pricing.calculate_cost(input_tokens, 0, cache_read_tokens, output_tokens)
            }),
        request_time_ms: Some(request_time_ms),
    };

    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

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
        finish_reason: choice.finish_reason,
        structured_output,
        id: Some(together_response.id),
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
    fn test_extract_thinking_none() {
        let (thinking, content) = extract_thinking("plain answer", None);
        assert!(thinking.is_none());
        assert_eq!(content, "plain answer");
        // empty/whitespace reasoning is ignored
        let (thinking, _) = extract_thinking("x", Some("  ".to_string()));
        assert!(thinking.is_none());
    }
}
