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

//! Z.ai (Zhipu AI) provider implementation
//!
//! PRICING UPDATE: January 2026
//!
//! GLM-4.5 series:
//! - Input: $0.35
//! - Output: $1.55
//!
//! GLM-4.6 series:
//! - Input: $0.30
//! - Output: $0.90
//!
//! GLM-4.7 series:
//! - Input: $0.10-0.16
//! - Output: $2.20
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, ResponseMode, TokenUsage, ToolCall,
};
use crate::llm::utils::{normalize_model_name, starts_with_ignore_ascii_case};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// Z.ai pricing constants (per 1M tokens in USD)
/// Source: https://docs.z.ai/guides/overview/pricing (as of Jan 2026)
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    // IMPORTANT: More specific model names must come first (using contains matching)
    // GLM-4.7 series (latest flagship) - more specific variants first
    ("glm-4.7-battle", 0.14, 2.20),
    ("glm-4.7-flash", 0.10, 2.20),
    ("glm-4.7", 0.14, 2.20),
    // GLM-4.6 series
    ("glm-4.6-flash", 0.30, 0.90),
    ("glm-4.6", 0.30, 0.90),
    // GLM-4.5 series
    ("glm-4.5-air-plus", 0.35, 1.55),
    ("glm-4.5-air", 0.35, 1.55),
    ("glm-4.5-flash", 0.35, 1.55),
    ("glm-4.5", 0.35, 1.55),
    // GLM-4 series
    ("glm-4-flash", 0.35, 1.55),
    ("glm-4", 0.35, 1.55),
];

/// Calculate cost for Z.ai models
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

/// Z.ai provider
#[derive(Debug, Clone, Default)]
pub struct ZaiProvider;

impl ZaiProvider {
    pub fn new() -> Self {
        Self
    }
}

// Constants
const ZAI_API_KEY_ENV: &str = "ZAI_API_KEY";
const ZAI_API_URL_ENV: &str = "ZAI_API_URL";
const ZAI_API_URL: &str = "https://api.z.ai/api/paas/v4/chat/completions";
// Z.ai API request/response structures
#[derive(Serialize, Debug)]
struct ZaiRequest {
    model: String,
    messages: Vec<ZaiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>, // Changed to f64 for better precision control
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>, // Changed to f64 for better precision control
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_messages: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    do_meta: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZaiMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ZaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZaiToolCall {
    id: String,
    #[serde(rename = "type")]
    type_field: String,
    function: ZaiFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZaiFunction {
    name: String,
    arguments: String, // Changed from serde_json::Value to String - Z.ai expects JSON string like OpenAI
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct ZaiResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<ZaiChoice>,
    usage: Option<ZaiUsage>,
    #[serde(default)]
    web_search: Vec<ZaiWebSearch>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ZaiChoice {
    message: ZaiMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ZaiUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_tokens_details: ZaiPromptTokensDetails,
}

#[derive(Deserialize, Debug, Default)]
struct ZaiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug)]
struct ZaiWebSearch {
    title: String,
    content: String,
    link: String,
    media: String,
    icon: String,
    refer: String,
    publish_date: String,
}

#[async_trait]
impl AiProvider for ZaiProvider {
    fn name(&self) -> &str {
        "zai"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Z.ai GLM models (case-insensitive)
        starts_with_ignore_ascii_case(model, "glm-")
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(ZAI_API_KEY_ENV)
            .map_err(|_| anyhow::anyhow!("{} not found in environment", ZAI_API_KEY_ENV))
    }
    fn supports_caching(&self, _model: &str) -> bool {
        true // Z.ai supports prompt caching
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false // Z.ai GLM-4 series does not support vision
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // Z.ai supports structured output via response_format
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Z.ai model context window limits (case-insensitive)
        let model_lower = normalize_model_name(model);
        if model_lower.contains("glm-4.7") {
            200_000 // 200K context window for GLM-4.7
        } else if model_lower.contains("glm-4.6") {
            128_000 // 128K context window for GLM-4.6
        } else if model_lower.contains("glm-4.5") {
            131_072 // ~128K context window for GLM-4.5
        } else {
            128_000 // Covers glm-4 and any other model
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let (api_key, api_url) = get_api_key_and_url()?;

        // Convert messages to Z.ai format
        let messages: Vec<ZaiMessage> = params
            .messages
            .iter()
            .map(|msg| ZaiMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
                tool_calls: msg.tool_calls.as_ref().map(convert_tool_calls),
                tool_calls_id: None,
            })
            .collect();

        // Build request
        // Z.ai API is strict about floating point precision - convert f32 to f64 and round to 2 decimal places
        let temperature = (params.temperature as f64 * 100.0).round() / 100.0;
        let top_p = (params.top_p as f64 * 100.0).round() / 100.0;

        let request = ZaiRequest {
            model: params.model.clone(),
            messages,
            do_sample: Some(params.temperature > 0.0),
            temperature: Some(temperature),
            top_p: Some(top_p),
            max_tokens: Some(params.max_tokens),
            stream: Some(false),
            stop: None,
            tools: params.tools.as_ref().map(|t| convert_tools(t)),
            tool_choice: None,
            return_messages: Some(true),
            request_id: None,
            do_meta: None,
            web_search: None,
            response_format: params.response_format.as_ref().map(|so| {
                let mode_str = match so.mode {
                    ResponseMode::Auto => "auto",
                    ResponseMode::Strict => "json_object",
                };
                serde_json::json!({
                    "type": mode_str
                })
            }),
        };

        // Execute request with retry logic
        let response = execute_zai_request(
            api_key,
            api_url,
            request,
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        Ok(response)
    }
}

/// Convert tool calls from unified format to Z.ai format
fn convert_tool_calls(tool_calls: &serde_json::Value) -> Vec<ZaiToolCall> {
    // Parse as GenericToolCall format
    if let Ok(calls) =
        serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(tool_calls.clone())
    {
        calls
            .iter()
            .map(|call| ZaiToolCall {
                id: call.id.clone(),
                type_field: "function".to_string(),
                function: ZaiFunction {
                    name: call.name.clone(),
                    // Z.ai expects arguments as a JSON string, not a JSON object (like OpenAI)
                    arguments: serde_json::to_string(&call.arguments).unwrap_or_default(),
                },
            })
            .collect()
    } else {
        vec![]
    }
}

/// Convert tools to Z.ai format
fn convert_tools(tools: &[crate::llm::types::FunctionDefinition]) -> serde_json::Value {
    serde_json::json!(tools
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
        .collect::<Vec<_>>())
}

/// Get API key and endpoint URL based on available configuration
/// Returns (api_key, api_url) tuple
fn get_api_key_and_url() -> Result<(String, String)> {
    let api_key = env::var(ZAI_API_KEY_ENV)
        .map_err(|_| anyhow::anyhow!("{} not found in environment", ZAI_API_KEY_ENV))?;

    // Use custom URL if configured, otherwise use default
    let api_url = env::var(ZAI_API_URL_ENV).unwrap_or_else(|_| ZAI_API_URL.to_string());

    Ok((api_key, api_url))
}

/// Execute a single Z.ai HTTP request with retry logic
async fn execute_zai_request(
    api_key: String,
    api_url: String,
    request: ZaiRequest,
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
            let request_body = serde_json::to_value(&request).unwrap();

            Box::pin(async move {
                client
                    .post(&api_url)
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

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!("Z.ai API error {}: {}", status, error_text));
    }

    let response_text = response.text().await?;
    let zai_response: ZaiResponse = serde_json::from_str(&response_text)?;

    // Extract content and tool calls
    let content = zai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .unwrap_or_default();

    let finish_reason = zai_response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.clone());

    // Extract tool calls if present
    let tool_calls: Option<Vec<ToolCall>> = zai_response.choices.first().and_then(|choice| {
        choice.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| {
                    // Parse the JSON string arguments into a Value
                    let arguments: serde_json::Value = if tc.function.arguments.trim().is_empty() {
                        serde_json::json!({})
                    } else {
                        serde_json::from_str(&tc.function.arguments).unwrap_or_else(
                            |_| serde_json::json!({"raw_arguments": tc.function.arguments}),
                        )
                    };

                    ToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments,
                    }
                })
                .collect()
        })
    });

    // Calculate cost
    let usage = zai_response.usage.as_ref();
    let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
    let completion_tokens = usage.map(|u| u.completion_tokens).unwrap_or(0);
    let cached_tokens = usage
        .map(|u| u.prompt_tokens_details.cached_tokens)
        .unwrap_or(0);

    let cost = calculate_cost(
        zai_response.model.as_str(),
        prompt_tokens.saturating_sub(cached_tokens),
        completion_tokens,
    );

    let token_usage = TokenUsage {
        prompt_tokens,
        output_tokens: completion_tokens,
        reasoning_tokens: 0, // Z.ai doesn't provide reasoning token count
        total_tokens: usage.map(|u| u.total_tokens).unwrap_or(0),
        cached_tokens,
        cost,
        request_time_ms: Some(request_time_ms),
    };

    // Build response JSON for exchange
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    if let Some(ref calls) = tool_calls {
        let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = calls
            .iter()
            .map(|tc| crate::llm::tool_calls::GenericToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
                meta: None,
            })
            .collect();

        response_json["tool_calls"] = serde_json::to_value(&generic_calls).unwrap_or_default();
    }

    // Check for structured output in response
    let structured_output = extract_structured_output(&response_json);

    let exchange = ProviderExchange::new(
        serde_json::to_value(&request).unwrap_or_default(),
        response_json,
        Some(token_usage),
        "zai",
    );

    Ok(ProviderResponse {
        content,
        thinking: None, // Z.ai doesn't support thinking
        exchange,
        tool_calls,
        finish_reason,
        structured_output,
    })
}

/// Extract structured output from response if present
fn extract_structured_output(response: &serde_json::Value) -> Option<serde_json::Value> {
    // Z.ai may return structured output in the message content as JSON
    // Check if content is a JSON object
    if let Some(content) = response["choices"]
        .get(0)
        .and_then(|c| c["message"]["content"].as_str())
    {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
            if json.is_object()
                && (json.get("properties").is_some() || json.get("$schema").is_some())
            {
                return Some(json);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_support() {
        let provider = ZaiProvider::new();
        assert!(provider.supports_model("glm-4.7"));
        assert!(provider.supports_model("glm-4.7-flash"));
        assert!(provider.supports_model("glm-4.6"));
        assert!(provider.supports_model("glm-4.5"));
        assert!(provider.supports_model("glm-4"));
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("claude-3"));
    }

    #[test]
    fn test_model_support_case_insensitive() {
        let provider = ZaiProvider::new();
        // Test uppercase
        assert!(provider.supports_model("GLM-4.7"));
        assert!(provider.supports_model("GLM-4"));
        // Test mixed case
        assert!(provider.supports_model("Glm-4.7"));
        assert!(provider.supports_model("GLM-4.6"));
    }

    #[test]
    fn test_cost_calculation() {
        // Test GLM-4.5: $0.35 input, $1.55 output
        let cost = calculate_cost("glm-4.5", 1_000_000, 1_000_000);
        assert_eq!(cost, Some(1.90)); // 0.35 + 1.55

        // Test GLM-4.7: $0.14 input, $2.20 output
        let cost = calculate_cost("glm-4.7", 1_000_000, 1_000_000);
        // Use approximate comparison for floating point
        assert!((cost.unwrap() - 2.34).abs() < 0.01); // 0.14 + 2.20

        // Test GLM-4.6: $0.30 input, $0.90 output
        let cost = calculate_cost("glm-4.6", 1_000_000, 1_000_000);
        assert_eq!(cost, Some(1.20)); // 0.30 + 0.90

        // Test GLM-4.7-flash: $0.10 input, $2.20 output
        let cost = calculate_cost("glm-4.7-flash", 1_000_000, 1_000_000);
        // Use approximate comparison for floating point
        assert!((cost.unwrap() - 2.30).abs() < 0.01); // 0.10 + 2.20
    }

    #[test]
    fn test_cost_with_partial_tokens() {
        // Test with 500K tokens each
        let cost = calculate_cost("glm-4.5", 500_000, 500_000);
        assert_eq!(cost, Some(0.95)); // 0.35 * 0.5 + 1.55 * 0.5
    }

    #[test]
    fn test_unknown_model() {
        let cost = calculate_cost("unknown-model", 1_000_000, 1_000_000);
        assert_eq!(cost, None);
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = ZaiProvider::new();
        assert!(provider.supports_caching("glm-4.7"));
        assert!(!provider.supports_vision("glm-4.7"));
        assert!(provider.supports_structured_output("glm-4.7"));
    }
}
