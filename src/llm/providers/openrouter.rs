// Copyright 2025 Muvon Un Limited
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

//! OpenRouter provider implementation

use super::shared;
use crate::errors::ProviderError;
use crate::errors::ToolCallError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
use crate::llm::utils::normalize_model_name;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// OpenRouter provider (uses OpenAI-compatible API)
#[derive(Debug, Clone)]
pub struct OpenRouterProvider;

impl Default for OpenRouterProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenRouterProvider {
    pub fn new() -> Self {
        Self
    }
}

const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const OPENROUTER_API_URL_ENV: &str = "OPENROUTER_API_URL";
const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";
#[async_trait::async_trait]
impl AiProvider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn supports_model(&self, model: &str) -> bool {
        // OpenRouter supports many models from different providers (case-insensitive)
        // Accept models with provider prefixes (anthropic/, openai/, meta/, google/, etc.)
        // or direct model names
        let normalized = normalize_model_name(model);
        normalized.starts_with("anthropic/")
            || normalized.starts_with("openai/")
            || normalized.starts_with("meta/")
            || normalized.starts_with("google/")
            || normalized.starts_with("mistral/")
            || normalized.starts_with("cohere/")
            || normalized.contains("claude")
            || normalized.contains("gpt-")
            || normalized.contains("llama")
            || normalized.contains("gemini")
            || normalized.contains("mistral")
            || !model.is_empty() // Accept any non-empty model string as fallback
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(OPENROUTER_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "OpenRouter API key not found in environment variable: {}",
                OPENROUTER_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, model: &str) -> bool {
        // OpenRouter supports caching for Anthropic models (case-insensitive)
        let normalized = normalize_model_name(model);
        normalized.starts_with("anthropic") || normalized.starts_with("claude")
    }

    fn supports_vision(&self, model: &str) -> bool {
        // OpenRouter is an aggregator - we can't know which models support vision
        // Default to true and let the underlying provider handle it
        let normalized = normalize_model_name(model);
        // For known non-vision models, return false as optimization
        let known_non_vision = normalized.starts_with("gpt-3.5")
            || normalized.starts_with("text-")
            || normalized == "o1-preview"
            || normalized == "o1-mini";
        !known_non_vision
    }

    fn supports_video(&self, _model: &str) -> bool {
        // OpenRouter is an aggregator - support all by default
        // The underlying provider/model will handle the actual capability
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Auto-generated from OpenRouter API (case-insensitive)
        let normalized = normalize_model_name(model);
        match normalized.as_str() {
            // claude models
            _ if normalized.starts_with("claude") => 200_000,
            // gpt-4o models
            _ if normalized.starts_with("gpt-4o") => 128_000,
            // gpt-4-turbo models
            _ if normalized.starts_with("gpt-4-turbo") => 128_000,
            // o1/o3 models
            _ if normalized.starts_with("o1") || normalized.starts_with("o3") => 200_000,
            // gpt-4 models
            _ if normalized.starts_with("gpt-4") && !normalized.starts_with("gpt-4o") => 8_192,
            // gpt-3.5-turbo models
            _ if normalized.starts_with("gpt-3.5-turbo") => 16_384,
            // llama models
            _ if normalized.starts_with("llama-3") => 131_072,
            _ if normalized.starts_with("llama-4") => 200_000,
            // gemini models
            _ if normalized.starts_with("gemini-1.5-pro") => 2_000_000,
            _ if normalized.starts_with("gemini-1.5-flash") => 1_000_000,
            _ if normalized.starts_with("gemini-2") => 1_048_576,
            // mistral models
            _ if normalized.starts_with("mistral-large") => 128_000,
            _ if normalized.starts_with("mistral-small") => 32_000,
            // deepseek models
            _ if normalized.starts_with("deepseek") => 128_000,
            // Fallback
            _ => 2_000_000,
        }
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true // All OpenRouter models support structured output
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // OpenRouter proxies to underlying providers
        // Try to detect provider from model name and delegate to their pricing
        let normalized = normalize_model_name(model);

        // Anthropic models (claude)
        if normalized.starts_with("anthropic/") || normalized.contains("claude") {
            // Delegate to Anthropic provider pricing
            let anthropic = crate::llm::providers::AnthropicProvider::new();
            return anthropic.get_model_pricing(model);
        }

        // OpenAI models (gpt)
        if normalized.starts_with("openai/") || normalized.contains("gpt-") {
            let openai = crate::llm::providers::OpenAiProvider::new();
            return openai.get_model_pricing(model);
        }

        // DeepSeek models
        if normalized.starts_with("deepseek") {
            let deepseek = crate::llm::providers::DeepSeekProvider::new();
            return deepseek.get_model_pricing(model);
        }

        // Google models (gemini)
        if normalized.starts_with("google/") || normalized.contains("gemini") {
            let google = crate::llm::providers::GoogleVertexProvider::new();
            return google.get_model_pricing(model);
        }

        // Unknown provider - no pricing available
        None
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to OpenRouter format (same as OpenAI)
        let messages = convert_messages(&params.messages)?;

        // Create the request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repetition_penalty": 1.1,
            "usage": {
                "include": true  // Always enable usage tracking for all requests
            },
            "provider": {
                "order": [
                    "Anthropic",
                    "OpenAI",
                    "Amazon Bedrock",
                    "Azure",
                    "Cloudflare",
                    "Google Vertex",
                    "xAI",
                ],
                "allow_fallbacks": true,
            },
        });

        // Add max_tokens if specified (0 means don't include it in request)
        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add max_tokens if specified
        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add tools if available (OpenRouter supports OpenAI-compatible tools)
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                // Sort tools by name for consistent ordering
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
                request_body["tool_choice"] = serde_json::json!("auto");
            }
        }

        // Add structured output format if specified (OpenRouter supports OpenAI-compatible format)
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request_body["response_format"] = serde_json::json!({
                        "type": "json_object"
                    });
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    if let Some(schema) = &response_format.schema {
                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema
                            }
                        });

                        // Add strict mode if specified
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

        // Execute the request
        let api_url =
            env::var(OPENROUTER_API_URL_ENV).unwrap_or_else(|_| OPENROUTER_API_URL.to_string());

        let response = execute_openrouter_request(
            api_key,
            api_url,
            request_body,
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        Ok(response)
    }
}

// Reuse OpenAI structures since OpenRouter is compatible
#[derive(Serialize, Deserialize, Debug)]
struct OpenRouterMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>, // For tool messages: the ID of the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>, // For tool messages: the name of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>, // For assistant messages: array of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_details: Option<serde_json::Value>, // For Gemini thought signatures preservation
}

#[derive(Deserialize, Debug)]
struct OpenRouterResponse {
    id: String,
    choices: Vec<OpenRouterChoice>,
    usage: OpenRouterUsage,
}

#[derive(Deserialize, Debug)]
struct OpenRouterChoice {
    message: OpenRouterResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    reasoning_details: Option<serde_json::Value>, // Gemini thought signatures
}

#[derive(Deserialize, Debug)]
struct OpenRouterToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenRouterFunction,
}

#[derive(Deserialize, Debug)]
struct OpenRouterFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenRouterUsage {
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
    prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
    #[serde(default)]
    completion_tokens_details: Option<OpenRouterCompletionTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct OpenRouterCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

// Convert messages to OpenRouter format (same as OpenAI)
fn convert_messages(messages: &[Message]) -> Result<Vec<OpenRouterMessage>, ToolCallError> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                // Tool messages in OpenRouter format - MUST include tool_call_id and name
                let tool_call_id = message.tool_call_id.clone();
                let name = message.name.clone();

                let content = if message.cached {
                    let mut text_content = serde_json::json!({
                        "type": "text",
                        "text": message.content
                    });
                    text_content["cache_control"] = shared::ephemeral_cache_control();
                    serde_json::json!([text_content])
                } else {
                    serde_json::json!(message.content)
                };

                result.push(OpenRouterMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id,
                    name,
                    tool_calls: None,
                    reasoning_details: None,
                });
            }
            "assistant" if message.tool_calls.is_some() => {
                // Assistant message with tool calls - convert from unified GenericToolCall format
                let mut content_parts = Vec::new();

                // Add text content if not empty
                if !message.content.trim().is_empty() {
                    let mut text_content = serde_json::json!({
                        "type": "text",
                        "text": message.content
                    });

                    if message.cached {
                        text_content["cache_control"] = shared::ephemeral_cache_control();
                    }

                    content_parts.push(text_content);
                }

                let content = if content_parts.len() == 1 && !message.cached {
                    content_parts[0]["text"].clone()
                } else if content_parts.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(content_parts)
                };

                // Convert unified GenericToolCall format to OpenRouter format
                let Some(tool_calls_value) = message.tool_calls.as_ref() else {
                    return Err(ToolCallError::MissingField {
                        field: "tool_calls".to_string(),
                    });
                };
                let generic_calls =
                    shared::parse_generic_tool_calls_strict(tool_calls_value, "openrouter")?;

                // Extract reasoning_details from first tool call's meta (Gemini thought signatures)
                let reasoning_details = generic_calls
                    .first()
                    .and_then(|call| call.meta.as_ref())
                    .and_then(|meta| meta.get("reasoning_details"))
                    .cloned();

                // Convert GenericToolCall to OpenRouter format
                let openrouter_calls: Vec<serde_json::Value> = generic_calls
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

                let tool_calls = Some(serde_json::Value::Array(openrouter_calls));

                result.push(OpenRouterMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls,
                    reasoning_details, // Add reasoning_details at message level
                });
            }
            _ => {
                // Handle other message types with cache support
                let mut content_parts = vec![{
                    let mut text_content = serde_json::json!({
                        "type": "text",
                        "text": message.content
                    });

                    // Add cache_control if needed
                    if message.cached {
                        text_content["cache_control"] = shared::ephemeral_cache_control();
                    }

                    text_content
                }];

                // Add images if present
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

                // Add videos if present
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
                                    "video_url": {
                                        "url": url
                                    }
                                }));
                            }
                        }
                    }
                }

                let content = if content_parts.len() == 1 && !message.cached {
                    content_parts[0]["text"].clone()
                } else {
                    serde_json::json!(content_parts)
                };

                result.push(OpenRouterMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls: None,
                    reasoning_details: None,
                });
            }
        }
    }

    Ok(result)
}

// Execute OpenRouter HTTP request
async fn execute_openrouter_request(
    api_key: String,
    api_url: String,
    request_body: serde_json::Value,
    max_retries: u32,
    base_timeout: std::time::Duration,
    cancellation_token: Option<&tokio::sync::watch::Receiver<bool>>,
) -> Result<ProviderResponse> {
    let client = Client::new();
    let start_time = std::time::Instant::now();

    let response = retry::retry_with_exponential_backoff(
        || {
            let client = client.clone();
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let request_body = request_body.clone();
            let openrouter_app_title =
                std::env::var("OPENROUTER_APP_TITLE").unwrap_or_else(|_| "octolib".to_string());
            let openrouter_http_referer = std::env::var("OPENROUTER_HTTP_REFERER")
                .unwrap_or_else(|_| "https://octolib.muvon.io".to_string());

            Box::pin(async move {
                let response = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("HTTP-Referer", openrouter_http_referer)
                    .header("X-Title", openrouter_app_title)
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(anyhow::Error::from)?;

                // Return Err for retryable HTTP errors so the retry loop catches them
                if retry::is_retryable_status(response.status().as_u16()) {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!(
                        "OpenRouter API error {}: {}",
                        status,
                        error_text
                    ));
                }

                Ok(response)
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
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = retry::cancellable(
            async { response.text().await.map_err(anyhow::Error::from) },
            cancellation_token,
            || ProviderError::Cancelled.into(),
        )
        .await?;
        return Err(anyhow::anyhow!(
            "OpenRouter API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = retry::cancellable(
        async { response.text().await.map_err(anyhow::Error::from) },
        cancellation_token,
        || ProviderError::Cancelled.into(),
    )
    .await?;
    let openrouter_response: OpenRouterResponse = serde_json::from_str(&response_text)?;

    let choice = openrouter_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in OpenRouter response"))?;

    let content = choice.message.content.unwrap_or_default();

    // Extract reasoning_details as thinking (for Gemini and other providers)
    let reasoning_details = &choice.message.reasoning_details;

    // Calculate thinking content and extract tokens
    let thinking = match reasoning_details.as_ref() {
        Some(rd) => {
            // Extract text content from reasoning_details array
            let thinking_text = rd
                .as_array()
                .and_then(|arr| {
                    let texts: Vec<String> = arr
                        .iter()
                        .filter_map(|item| {
                            item.get("text")
                                .and_then(|t| t.as_str().map(|s| s.to_string()))
                        })
                        .collect();
                    if texts.is_empty() {
                        None
                    } else {
                        Some(texts)
                    }
                })
                .map(|texts| texts.join("\n\n"))
                .unwrap_or_else(|| rd.to_string());

            // Estimate reasoning tokens from content length (4 chars per token)
            let estimated = (thinking_text.len() / 4) as u64;

            Some(ThinkingBlock {
                content: thinking_text,
                tokens: estimated,
            })
        }
        None => None,
    };

    // Convert tool calls if present
    let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
        calls
            .into_iter()
            .filter_map(|call| {
                // Validate tool type - OpenRouter should only have "function" type
                if call.tool_type != "function" {
                    tracing::warn!(
                        "Unexpected tool type '{}' from OpenRouter API",
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

    // Prefer usage reasoning tokens if present; fallback to estimation from reasoning_details
    let reasoning_tokens = openrouter_response
        .usage
        .completion_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .filter(|v| *v > 0)
        .or_else(|| thinking.as_ref().map(|t| t.tokens))
        .unwrap_or(0);

    let input_tokens_raw = openrouter_response
        .usage
        .input_tokens
        .or(openrouter_response.usage.prompt_tokens)
        .unwrap_or(0);
    let output_tokens = openrouter_response
        .usage
        .completion_tokens
        .or(openrouter_response.usage.output_tokens)
        .unwrap_or(0);
    let cache_read_tokens = openrouter_response
        .usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);
    let total_tokens = openrouter_response
        .usage
        .total_tokens
        .unwrap_or(input_tokens_raw.saturating_add(output_tokens));
    let input_tokens_clean = input_tokens_raw.saturating_sub(cache_read_tokens);

    // Octolib semantic: input_tokens excludes cache reads
    let usage = TokenUsage {
        input_tokens: input_tokens_clean,
        cache_read_tokens,
        cache_write_tokens: 0,
        output_tokens,
        reasoning_tokens,
        total_tokens,
        cost: None, // OpenRouter doesn't provide cost
        request_time_ms: Some(request_time_ms),
    };

    // Create response JSON and store tool_calls in unified format
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    // Extract reasoning_details from response for Gemini thought signatures
    if let Some(ref tc) = tool_calls {
        let reasoning_details = choice.message.reasoning_details.clone();

        let reasoning_meta = reasoning_details.as_ref().map(|rd| {
            let mut meta_map = serde_json::Map::new();
            meta_map.insert("reasoning_details".to_string(), rd.clone());
            meta_map
        });
        shared::set_response_tool_calls(&mut response_json, tc, reasoning_meta.as_ref());
    }

    let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "openrouter");

    // Try to parse structured output if it was requested
    let structured_output = shared::parse_structured_output_from_text(&content);

    Ok(ProviderResponse {
        content,
        thinking, // Add thinking from reasoning_details
        exchange,
        tool_calls,
        finish_reason: choice.finish_reason,
        structured_output,
        id: Some(openrouter_response.id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = OpenRouterProvider::new();

        // OpenRouter supports many models
        assert!(provider.supports_model("anthropic/claude-3.5-sonnet"));
        assert!(provider.supports_model("openai/gpt-4o"));
        assert!(provider.supports_model("meta/llama-3.1-70b"));
        assert!(provider.supports_model("deepseek-chat"));

        // Should accept any non-empty model string as fallback
        assert!(provider.supports_model("any-model-name"));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = OpenRouterProvider::new();

        // Test uppercase
        assert!(provider.supports_model("ANTHROPIC/CLAUDE-3.5-SONNET"));
        assert!(provider.supports_model("OPENAI/GPT-4O"));
        assert!(provider.supports_model("META/LLAMA-3.1-70B"));
        // Test mixed case
        assert!(provider.supports_model("Anthropic/Claude-3.5-Sonnet"));
        assert!(provider.supports_model("DEEPSEEK-CHAT"));
    }

    #[test]
    fn test_supports_vision_case_insensitive() {
        let provider = OpenRouterProvider::new();

        // Test lowercase
        assert!(provider.supports_vision("gpt-4o"));
        assert!(provider.supports_vision("claude-3-haiku"));

        // Test uppercase
        assert!(provider.supports_vision("GPT-4O"));
        assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
        // Test mixed case
        assert!(provider.supports_vision("Gemini-1.5-Pro"));
    }

    #[test]
    fn test_supports_caching_case_insensitive() {
        let provider = OpenRouterProvider::new();

        // Test lowercase
        assert!(provider.supports_caching("anthropic/claude-3.5-sonnet"));
        assert!(provider.supports_caching("claude-3-haiku"));

        // Test uppercase
        assert!(provider.supports_caching("ANTHROPIC/CLAUDE-3.5-SONNET"));
        assert!(provider.supports_caching("CLAUDE-3-HAIKU"));
    }
}
