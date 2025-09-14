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

//! OpenAI provider implementation

use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, TokenUsage, ToolCall,
};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// OpenAI pricing constants (per 1M tokens in USD)
/// Source: https://platform.openai.com/docs/pricing (as of January 2025)
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    // Latest models (2025)
    // GPT-4.1 and variants
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4.1-2025-04-14", 2.00, 8.00),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1-mini-2025-04-14", 0.40, 1.60),
    ("gpt-4.1-nano", 0.10, 0.40),
    ("gpt-4.1-nano-2025-04-14", 0.10, 0.40),
    // GPT-4.5
    ("gpt-4.5-preview", 75.00, 150.00),
    ("gpt-4.5-preview-2025-02-27", 75.00, 150.00),
    // GPT-5
    ("gpt-5", 1.25, 10.00),
    ("gpt-5-2025-08-07", 1.25, 10.00),
    ("gpt-5-mini", 0.25, 2.0),
    ("gpt-5-mini-2025-08-07", 0.25, 2.0),
    ("gpt-5-nano", 0.05, 0.40),
    ("gpt-5-nano-2025-08-07", 0.05, 0.40),
    // GPT-4o series
    ("gpt-4o", 2.50, 10.00),
    ("gpt-4o-2024-08-06", 2.50, 10.00),
    ("gpt-4o-realtime-preview", 5.00, 20.00),
    ("gpt-4o-realtime-preview-2025-06-03", 5.00, 20.00),
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4o-mini-2024-07-18", 0.15, 0.60),
    ("gpt-4o-mini-realtime-preview", 0.60, 2.40),
    ("gpt-4o-mini-realtime-preview-2024-12-17", 0.60, 2.40),
    ("gpt-4o-mini-search-preview", 0.15, 0.60),
    ("gpt-4o-mini-search-preview-2025-03-11", 0.15, 0.60),
    ("gpt-4o-search-preview", 2.50, 10.00),
    ("gpt-4o-search-preview-2025-03-11", 2.50, 10.00),
    // O-series and variants
    ("o1", 15.00, 60.00),
    ("o1-2024-12-17", 15.00, 60.00),
    ("o1-pro", 150.00, 600.00),
    ("o1-pro-2025-03-19", 150.00, 600.00),
    ("o1-mini", 1.10, 4.40),
    ("o1-mini-2024-09-12", 1.10, 4.40),
    ("o3", 2.00, 8.00),
    ("o3-2025-04-16", 2.00, 8.00),
    ("o3-pro", 20.00, 80.00),
    ("o3-pro-2025-06-10", 20.00, 80.00),
    ("o3-mini", 1.10, 4.40),
    ("o3-mini-2025-01-31", 1.10, 4.40),
    ("o3-deep-research", 10.00, 40.00),
    ("o3-deep-research-2025-06-26", 10.00, 40.00),
    ("o4-mini", 1.10, 4.40),
    ("o4-mini-2025-04-16", 1.10, 4.40),
    ("o4-mini-deep-research", 2.00, 8.00),
    ("o4-mini-deep-research-2025-06-26", 2.00, 8.00),
    // GPT-4 Turbo
    ("gpt-4-turbo", 10.00, 30.00),
    ("gpt-4-turbo-2024-04-09", 10.00, 30.00),
    // GPT-4
    ("gpt-4", 30.00, 60.00),
    ("gpt-4-0613", 30.00, 60.00),
    ("gpt-4-32k", 60.00, 120.00),
    // GPT-3.5 Turbo
    ("gpt-3.5-turbo", 0.50, 1.50),
    ("gpt-3.5-turbo-0125", 0.50, 1.50),
    ("gpt-3.5-turbo-instruct", 1.50, 2.00),
    ("gpt-3.5-turbo-16k-0613", 3.00, 4.00),
];

/// Calculate cost for OpenAI models with basic pricing
fn calculate_cost(model: &str, prompt_tokens: u64, completion_tokens: u64) -> Option<f64> {
    for (pricing_model, input_price, output_price) in PRICING {
        if model.contains(pricing_model) {
            let input_cost = (prompt_tokens as f64 / 1_000_000.0) * input_price;
            let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;
            return Some(input_cost + output_cost);
        }
    }
    None
}

/// Check if a model supports the temperature parameter
/// O1, O2, O3, O4 and GPT-5 series models don't support temperature
fn supports_temperature(model: &str) -> bool {
    !model.starts_with("o1")
        && !model.starts_with("o2")
        && !model.starts_with("o3")
        && !model.starts_with("o4")
        && !model.starts_with("gpt-5")
}

/// Check if a model uses max_completion_tokens instead of max_tokens
/// GPT-5 models use the new max_completion_tokens parameter
fn uses_max_completion_tokens(model: &str) -> bool {
    model.starts_with("gpt-5")
}

/// Get cache pricing multiplier based on model
/// GPT-5 models have 0.1x cache pricing (90% cheaper)
/// Other models have 0.25x cache pricing (75% cheaper)
fn get_cache_multiplier(model: &str) -> f64 {
    if model.starts_with("gpt-5") {
        0.1 // GPT-5 models: 10% of normal price for cache reads
    } else {
        0.25 // Other models: 25% of normal price for cache reads
    }
}

/// Calculate cost with cache-aware pricing
/// This function handles the different pricing tiers for cached vs non-cached tokens:
/// - cache_read_tokens: charged at model-specific multiplier (GPT-5: 0.1x, others: 0.25x)
/// - regular_input_tokens: charged at normal price (includes cache write tokens)
/// - output_tokens: charged at normal price
fn calculate_cost_with_cache(
    model: &str,
    regular_input_tokens: u64,
    cache_read_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    for (pricing_model, input_price, output_price) in PRICING {
        if model.contains(pricing_model) {
            // Regular input tokens at normal price (includes cache write - no additional cost)
            let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input_price;

            // Cache read tokens at model-specific multiplier
            let cache_multiplier = get_cache_multiplier(model);
            let cache_read_cost =
                (cache_read_tokens as f64 / 1_000_000.0) * input_price * cache_multiplier;

            // Output tokens at normal price (never cached)
            let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

            return Some(regular_input_cost + cache_read_cost + output_cost);
        }
    }
    None
}

/// OpenAI provider
#[derive(Debug, Clone)]
pub struct OpenAiProvider;

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self
    }
}

const OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

#[async_trait::async_trait]
impl AiProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &str) -> bool {
        // OpenAI models - current lineup
        model.starts_with("gpt-5")
            || model.starts_with("gpt-4o")
            || model.starts_with("gpt-4.5")
            || model.starts_with("gpt-4.1")
            || model.starts_with("gpt-4")
            || model.starts_with("gpt-3.5")
            || model.starts_with("o1")
            || model.starts_with("o3")
            || model.starts_with("o4")
            || model == "chatgpt-4o-latest"
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(OPENAI_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "OpenAI API key not found in environment variable: {}",
                OPENAI_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, model: &str) -> bool {
        // OpenAI supports automatic prompt caching for these models (as of Oct 2024)
        model.contains("gpt-4o")
            || model.contains("gpt-4.1")
            || model.contains("gpt-5")
            || model.contains("o1-preview")
            || model.contains("o1-mini")
            || model.contains("o1")
            || model.contains("o3")
            || model.contains("o4")
    }

    fn supports_vision(&self, model: &str) -> bool {
        // OpenAI vision-capable models
        model.contains("gpt-4o")
            || model.contains("gpt-4.1")
            || model.contains("gpt-4-turbo")
            || model.contains("gpt-4-vision-preview")
            || model.starts_with("gpt-4o-")
            || model.starts_with("gpt-5-")
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // OpenAI model context window limits (what we can send as input)
        // These are the actual context windows - API handles output limits

        // GPT-5 models: 128K context window
        if model.starts_with("gpt-5") {
            return 128_000;
        }
        // GPT-4o models: 128K context window
        if model.contains("gpt-4o") {
            return 128_000;
        }
        // GPT-4 models: varies by version
        if model.contains("gpt-4-turbo") || model.contains("gpt-4.5") || model.contains("gpt-4.1") {
            return 128_000;
        }
        if model.contains("gpt-4") && !model.contains("gpt-4o") {
            return 8_192; // Old GPT-4: 8K context window
        }
        // O-series models: 128K context window
        if model.starts_with("o1") || model.starts_with("o2") || model.starts_with("o3") {
            return 128_000;
        }
        // GPT-3.5: 16K context window
        if model.contains("gpt-3.5") {
            return 16_384;
        }
        // Default conservative limit
        8_192
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to OpenAI format
        let openai_messages = convert_messages(&params.messages);

        // Create the request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": openai_messages,
        });

        // Only add temperature for models that support it
        // O1/O2/O3/O4 and GPT-5 series models don't support temperature parameter
        if supports_temperature(&params.model) {
            request_body["temperature"] = serde_json::json!(params.temperature);
            request_body["top_p"] = serde_json::json!(params.top_p);
            // Note: OpenAI doesn't have top_k parameter, but has similar "top_logprobs"
            // We'll skip top_k for OpenAI as it's not directly supported
        }

        // Add max_tokens if specified (0 means don't include it in request)
        if params.max_tokens > 0 {
            if uses_max_completion_tokens(&params.model) {
                // GPT-5 models use max_completion_tokens
                request_body["max_completion_tokens"] = serde_json::json!(params.max_tokens);
            } else {
                // Other models use max_tokens
                request_body["max_tokens"] = serde_json::json!(params.max_tokens);
            }
        }

        // Add tools if available
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

        // Execute the request with retry logic
        let response = execute_openai_request(
            api_key,
            request_body,
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        Ok(response)
    }
}

// OpenAI API structures
#[derive(Serialize, Deserialize, Debug)]
struct OpenAiMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>, // For tool messages: the ID of the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>, // For tool messages: the name of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<serde_json::Value>, // For assistant messages: array of tool calls
}

#[derive(Deserialize, Debug)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: OpenAiUsage,
}

#[derive(Deserialize, Debug)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenAiToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiFunction,
}

#[derive(Deserialize, Debug)]
struct OpenAiFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenAiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// Convert our session messages to OpenAI format
fn convert_messages(messages: &[Message]) -> Vec<OpenAiMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                // Tool messages MUST include tool_call_id and name
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

                result.push(OpenAiMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id,
                    name,
                    tool_calls: None,
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

                // Convert unified GenericToolCall format to OpenAI format
                let tool_calls = if let Ok(generic_calls) =
                    serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(
                        message.tool_calls.clone().unwrap(),
                    ) {
                    // Convert GenericToolCall to OpenAI format
                    let openai_calls: Vec<serde_json::Value> = generic_calls
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
                    Some(serde_json::Value::Array(openai_calls))
                } else {
                    panic!("Invalid tool_calls format - must be Vec<GenericToolCall>");
                };

                result.push(OpenAiMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls,
                });
            }
            _ => {
                // Handle regular messages (user, system)
                let mut content_parts = vec![{
                    let mut text_content = serde_json::json!({
                        "type": "text",
                        "text": message.content
                    });

                    // Add cache_control if needed (OpenAI format - currently not supported but prepared)
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

                result.push(OpenAiMessage {
                    role: message.role.clone(),
                    content,
                    tool_call_id: None,
                    name: None,
                    tool_calls: None,
                });
            }
        }
    }

    result
}

// Execute OpenAI HTTP request
async fn execute_openai_request(
    api_key: String,
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
            let request_body = request_body.clone();
            Box::pin(async move {
                client
                    .post(OPENAI_API_URL)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
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

    // Extract rate limit headers before consuming response
    let mut rate_limit_headers = std::collections::HashMap::new();
    let headers = response.headers();

    // Check for cache hit headers first
    let cache_creation_input_tokens = headers
        .get("x-cache-creation-input-tokens")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let cache_read_input_tokens = headers
        .get("x-cache-read-input-tokens")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    // OpenAI rate limit headers
    if let Some(requests_limit) = headers
        .get("x-ratelimit-limit-requests")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("requests_limit".to_string(), requests_limit.to_string());
    }
    if let Some(requests_remaining) = headers
        .get("x-ratelimit-remaining-requests")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert(
            "requests_remaining".to_string(),
            requests_remaining.to_string(),
        );
    }
    if let Some(tokens_limit) = headers
        .get("x-ratelimit-limit-tokens")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("tokens_limit".to_string(), tokens_limit.to_string());
    }
    if let Some(tokens_remaining) = headers
        .get("x-ratelimit-remaining-tokens")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("tokens_remaining".to_string(), tokens_remaining.to_string());
    }
    if let Some(request_reset) = headers
        .get("x-ratelimit-reset-requests")
        .and_then(|h| h.to_str().ok())
    {
        rate_limit_headers.insert("request_reset".to_string(), request_reset.to_string());
    }

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "OpenAI API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = response.text().await?;
    let openai_response: OpenAiResponse = serde_json::from_str(&response_text)?;

    let choice = openai_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in OpenAI response"))?;

    let content = choice.message.content.unwrap_or_default();

    // Convert tool calls if present
    let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
        calls
            .into_iter()
            .filter_map(|call| {
                // Validate tool type - OpenAI should only have "function" type
                if call.tool_type != "function" {
                    eprintln!(
                        "Warning: Unexpected tool type '{}' from OpenAI API",
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

    // Calculate cost using local pricing tables if model is available
    let cost = request_body
        .get("model")
        .and_then(|m| m.as_str())
        .and_then(|model| {
            if cache_creation_input_tokens > 0 || cache_read_input_tokens > 0 {
                // Use cache-aware pricing when cache tokens are present
                let regular_input_tokens = openai_response
                    .usage
                    .prompt_tokens
                    .saturating_sub(cache_read_input_tokens as u64);
                calculate_cost_with_cache(
                    model,
                    regular_input_tokens,
                    cache_read_input_tokens as u64,
                    openai_response.usage.completion_tokens,
                )
            } else {
                // Use basic pricing when no cache tokens
                calculate_cost(
                    model,
                    openai_response.usage.prompt_tokens,
                    openai_response.usage.completion_tokens,
                )
            }
        });

    let usage = TokenUsage {
        prompt_tokens: openai_response.usage.prompt_tokens,
        output_tokens: openai_response.usage.completion_tokens,
        total_tokens: openai_response.usage.total_tokens,
        cached_tokens: cache_read_input_tokens as u64,
        cost,
        request_time_ms: Some(request_time_ms),
    };

    // Create response JSON and store tool_calls in unified format
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    if let Some(ref tc) = tool_calls {
        let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = tc
            .iter()
            .map(|call| crate::llm::tool_calls::GenericToolCall {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect();

        response_json["tool_calls"] = serde_json::to_value(&generic_calls).unwrap_or_default();
    }

    let exchange = if rate_limit_headers.is_empty() {
        ProviderExchange::new(request_body, response_json, Some(usage), "openai")
    } else {
        ProviderExchange::with_rate_limit_headers(
            request_body,
            response_json,
            Some(usage),
            "openai",
            rate_limit_headers,
        )
    };

    Ok(ProviderResponse {
        content,
        exchange,
        tool_calls,
        finish_reason: choice.finish_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_cache_multiplier() {
        // GPT-5 models should have 0.1x cache multiplier (10% of normal price)
        assert_eq!(get_cache_multiplier("gpt-5"), 0.1);
        assert_eq!(get_cache_multiplier("gpt-5-2025-08-07"), 0.1);
        assert_eq!(get_cache_multiplier("gpt-5-mini"), 0.1);
        assert_eq!(get_cache_multiplier("gpt-5-mini-2025-08-07"), 0.1);
        assert_eq!(get_cache_multiplier("gpt-5-nano"), 0.1);
        assert_eq!(get_cache_multiplier("gpt-5-nano-2025-08-07"), 0.1);

        // Other models should have 0.25x cache multiplier (25% of normal price)
        assert_eq!(get_cache_multiplier("gpt-4o"), 0.25);
        assert_eq!(get_cache_multiplier("gpt-4o-mini"), 0.25);
        assert_eq!(get_cache_multiplier("gpt-4.1"), 0.25);
        assert_eq!(get_cache_multiplier("gpt-4"), 0.25);
        assert_eq!(get_cache_multiplier("gpt-3.5-turbo"), 0.25);
        assert_eq!(get_cache_multiplier("o1"), 0.25);
        assert_eq!(get_cache_multiplier("o3"), 0.25);
    }

    #[test]
    fn test_calculate_cost_with_cache() {
        // Test GPT-5 model with cache (0.1x multiplier)
        let cost = calculate_cost_with_cache("gpt-5", 1000, 500, 200);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();
        // Expected: (1000/1M * 1.25) + (500/1M * 1.25 * 0.1) + (200/1M * 10.0)
        // = 0.00125 + 0.0000625 + 0.002 = 0.0033125
        assert!((cost_value - 0.0033125).abs() < 0.0000001);

        // Test GPT-4o model with cache (0.25x multiplier)
        let cost = calculate_cost_with_cache("gpt-4o", 1000, 500, 200);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();
        // Expected: (1000/1M * 2.50) + (500/1M * 2.50 * 0.25) + (200/1M * 10.0)
        // = 0.0025 + 0.0003125 + 0.002 = 0.0048125
        assert!((cost_value - 0.0048125).abs() < 0.0000001);

        // Test unknown model
        let cost = calculate_cost_with_cache("unknown-model", 1000, 500, 200);
        assert!(cost.is_none());
    }

    #[test]
    fn test_supports_temperature() {
        // Models that should support temperature
        assert!(supports_temperature("gpt-4"));
        assert!(supports_temperature("gpt-4o"));
        assert!(supports_temperature("gpt-4o-mini"));
        assert!(supports_temperature("gpt-3.5-turbo"));
        assert!(supports_temperature("chatgpt-4o-latest"));

        // Models that should NOT support temperature (o1/o2/o3/o4 and gpt-5 series)
        assert!(!supports_temperature("o1"));
        assert!(!supports_temperature("o1-preview"));
        assert!(!supports_temperature("o1-mini"));
        assert!(!supports_temperature("o2"));
        assert!(!supports_temperature("o3"));
        assert!(!supports_temperature("o3-mini"));
        assert!(!supports_temperature("o4"));
        assert!(!supports_temperature("gpt-5"));
        assert!(!supports_temperature("gpt-5-mini"));
        assert!(!supports_temperature("gpt-5-nano"));
    }

    #[test]
    fn test_uses_max_completion_tokens() {
        // GPT-5 models should use max_completion_tokens
        assert!(uses_max_completion_tokens("gpt-5"));
        assert!(uses_max_completion_tokens("gpt-5-2025-08-07"));
        assert!(uses_max_completion_tokens("gpt-5-mini"));
        assert!(uses_max_completion_tokens("gpt-5-mini-2025-08-07"));
        assert!(uses_max_completion_tokens("gpt-5-nano"));
        assert!(uses_max_completion_tokens("gpt-5-nano-2025-08-07"));

        // Other models should use max_tokens (return false)
        assert!(!uses_max_completion_tokens("gpt-4o"));
        assert!(!uses_max_completion_tokens("gpt-4o-mini"));
        assert!(!uses_max_completion_tokens("gpt-4.1"));
        assert!(!uses_max_completion_tokens("gpt-4"));
        assert!(!uses_max_completion_tokens("gpt-3.5-turbo"));
        assert!(!uses_max_completion_tokens("o1"));
        assert!(!uses_max_completion_tokens("o3"));
    }

    #[test]
    fn test_supports_model_gpt5() {
        let provider = OpenAiProvider::new();

        // GPT-5 models should be supported
        assert!(provider.supports_model("gpt-5"));
        assert!(provider.supports_model("gpt-5-2025-08-07"));
        assert!(provider.supports_model("gpt-5-mini"));
        assert!(provider.supports_model("gpt-5-mini-2025-08-07"));
        assert!(provider.supports_model("gpt-5-nano"));
        assert!(provider.supports_model("gpt-5-nano-2025-08-07"));

        // Other models should still be supported
        assert!(provider.supports_model("gpt-4o"));
        assert!(provider.supports_model("gpt-4"));
        assert!(provider.supports_model("gpt-3.5-turbo"));
        assert!(provider.supports_model("o1"));

        // Unsupported models
        assert!(!provider.supports_model("claude-3"));
        assert!(!provider.supports_model("llama-2"));
    }

    #[test]
    fn test_get_max_input_tokens_gpt5() {
        let provider = OpenAiProvider::new();

        // GPT-5 models should have 128K context window
        assert_eq!(provider.get_max_input_tokens("gpt-5"), 128_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-2025-08-07"), 128_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-mini"), 128_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-nano"), 128_000);

        // Other models should maintain their existing limits
        assert_eq!(provider.get_max_input_tokens("gpt-4o"), 128_000);
        assert_eq!(provider.get_max_input_tokens("gpt-4"), 8_192);
        assert_eq!(provider.get_max_input_tokens("gpt-3.5-turbo"), 16_384);
    }

    #[test]
    fn test_supports_vision() {
        let provider = OpenAiProvider::new();

        // Models that should support vision
        assert!(provider.supports_vision("gpt-4o"));
        assert!(provider.supports_vision("gpt-4o-mini"));
        assert!(provider.supports_vision("gpt-4o-2024-05-13"));
        assert!(provider.supports_vision("gpt-4-turbo"));
        assert!(provider.supports_vision("gpt-4-vision-preview"));
        assert!(provider.supports_vision("gpt-4.1"));
        assert!(provider.supports_vision("gpt-5-mini"));

        // Models that should NOT support vision
        assert!(!provider.supports_vision("gpt-3.5-turbo"));
        assert!(!provider.supports_vision("gpt-4"));
        assert!(!provider.supports_vision("o1-preview"));
        assert!(!provider.supports_vision("o1-mini"));
        assert!(!provider.supports_vision("text-davinci-003"));
    }
}
