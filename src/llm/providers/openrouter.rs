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

use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
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
        // OpenRouter supports many models from different providers
        // Accept models with provider prefixes (anthropic/, openai/, meta/, google/, etc.)
        // or direct model names
        model.contains("anthropic/")
            || model.contains("openai/")
            || model.contains("meta/")
            || model.contains("google/")
            || model.contains("mistral/")
            || model.contains("cohere/")
            || model.contains("claude")
            || model.contains("gpt-")
            || model.contains("llama")
            || model.contains("gemini")
            || model.contains("mistral")
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
        // OpenRouter supports caching for Anthropic models
        model.contains("anthropic") || model.contains("claude")
    }

    fn supports_vision(&self, model: &str) -> bool {
        model.contains("gpt-4o")
            || model.contains("gpt-4-turbo")
            || model.contains("claude-3")
            || model.contains("claude-4")
            || model.contains("gemini")
            || model.contains("llava")
            || model.contains("qwen-vl")
            || model.contains("vision")
            || model.contains("anthropic/")
            || model.contains("google/")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // All OpenRouter models support structured output as requested
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Auto-generated from OpenRouter API
        match model {
            // claude models
            _ if model.contains("claude") => 200_000,
            // gpt-4o models
            _ if model.contains("gpt-4o") => 128_000,
            // gpt-4-turbo models
            _ if model.contains("gpt-4-turbo") => 128_000,
            // o1/o3 models
            _ if model.contains("o1") || model.contains("o3") => 200_000,
            // gpt-4 models
            _ if model.contains("gpt-4") && !model.contains("gpt-4o") => 8_192,
            // gpt-3.5-turbo models
            _ if model.contains("gpt-3.5-turbo") => 16_384,
            // llama models
            _ if model.contains("llama-3") => 131_072,
            _ if model.contains("llama-4") => 200_000,
            // gemini models
            _ if model.contains("gemini-1.5-pro") => 2_000_000,
            _ if model.contains("gemini-1.5-flash") => 1_000_000,
            _ if model.contains("gemini-2") => 1_048_576,
            // mistral models
            _ if model.contains("mistral-large") => 128_000,
            _ if model.contains("mistral-small") => 32_000,
            // deepseek models
            _ if model.contains("deepseek") => 128_000,
            // Fallback
            _ => 2_000_000,
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to OpenAI-compatible format (OpenRouter uses OpenAI API)
        let messages = convert_messages(&params.messages);

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
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// Convert messages to OpenRouter format (same as OpenAI)
fn convert_messages(messages: &[Message]) -> Vec<OpenRouterMessage> {
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
                    text_content["cache_control"] = serde_json::json!({
                        "type": "ephemeral"
                    });
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
                        text_content["cache_control"] = serde_json::json!({
                            "type": "ephemeral"
                        });
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
                let (tool_calls, reasoning_details) = if let Ok(generic_calls) =
                    serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(
                        message.tool_calls.clone().unwrap(),
                    ) {
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
                                    "arguments": serde_json::to_string(&call.arguments).unwrap_or_default()
                                }
                            })
                        })
                        .collect();
                    (
                        Some(serde_json::Value::Array(openrouter_calls)),
                        reasoning_details,
                    )
                } else {
                    panic!("Invalid tool_calls format - must be Vec<GenericToolCall>");
                };

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
                        text_content["cache_control"] = serde_json::json!({
                            "type": "ephemeral"
                        });
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

    result
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
                client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("HTTP-Referer", openrouter_http_referer)
                    .header("X-Title", openrouter_app_title)
                    .json(&request_body)
                    .send()
                    .await
            })
        },
        max_retries,
        base_timeout,
        cancellation_token,
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "OpenRouter API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = response.text().await?;
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
                    eprintln!(
                        "Warning: Unexpected tool type '{}' from OpenRouter API",
                        call.tool_type
                    );
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

    // Estimate reasoning tokens from thinking content length (4 chars per token)
    let reasoning_tokens = thinking.as_ref().map(|t| t.tokens).unwrap_or(0);

    let usage = TokenUsage {
        prompt_tokens: openrouter_response.usage.prompt_tokens,
        output_tokens: openrouter_response.usage.completion_tokens,
        reasoning_tokens,
        total_tokens: openrouter_response.usage.total_tokens,
        cached_tokens: 0, // OpenRouter doesn't provide cache info in usage
        cost: None,       // OpenRouter doesn't provide cost info directly
        request_time_ms: Some(request_time_ms),
    };

    // Create response JSON and store tool_calls in unified format
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    // Extract reasoning_details from response for Gemini thought signatures
    if let Some(ref tc) = tool_calls {
        let reasoning_details = choice.message.reasoning_details.clone();

        let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = tc
            .iter()
            .map(|call| {
                // Store reasoning_details in meta if present (for Gemini thought signatures)
                let meta = reasoning_details.as_ref().map(|rd| {
                    let mut meta_map = serde_json::Map::new();
                    meta_map.insert("reasoning_details".to_string(), rd.clone());
                    meta_map
                });

                crate::llm::tool_calls::GenericToolCall {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                    meta,
                }
            })
            .collect();

        response_json["tool_calls"] = serde_json::to_value(&generic_calls).unwrap_or_default();
    }

    let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "openrouter");

    // Try to parse structured output if it was requested
    let structured_output = if content.trim().starts_with('{') || content.trim().starts_with('[') {
        serde_json::from_str(&content).ok()
    } else {
        None
    };

    Ok(ProviderResponse {
        content,
        thinking, // Add thinking from reasoning_details
        exchange,
        tool_calls,
        finish_reason: choice.finish_reason,
        structured_output,
    })
}
