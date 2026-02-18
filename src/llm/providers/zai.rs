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
//! PRICING UPDATE: February 2026 (from <https://docs.z.ai/guides/overview/pricing>)
//!
//! GLM-5 series (NEW - Feb 2026):
//! - GLM-5: Input $1.00/1M, Output $3.20/1M
//! - GLM-5-Code: Input $1.20/1M, Output $5.00/1M
//!
//! GLM-4.7 series:
//! - GLM-4.7: Input $0.60/1M, Output $2.20/1M
//! - GLM-4.7-Flash: Free model
//! - GLM-4.7-FlashX: Free model
//!
//! GLM-4.6 series:
//! - GLM-4.6: Input $0.60/1M, Output $2.20/1M
//! - GLM-4.6V: Input $0.30/1M, Output $0.90/1M
//! - GLM-4.6V-Flash: Input $0.04/1M, Output $0.40/1M
//! - GLM-4.6V-FlashX: Input $0.04/1M, Output $0.40/1M
//!
//! GLM-4.5 series:
//! - GLM-4.5: Input $0.60/1M, Output $2.20/1M
//! - GLM-4.5V: Input $0.60/1M, Output $1.80/1M
//! - GLM-4.5-X: Input $2.20/1M, Output $8.90/1M
//! - GLM-4.5-Air: Input $0.20/1M, Output $1.10/1M
//! - GLM-4.5-AirX: Input $1.10/1M, Output $4.50/1M
//!
//! GLM-4 series:
//! - GLM-4-32B-0414-128K: Input $0.10/1M, Output $0.10/1M
use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, ResponseMode, ThinkingBlock,
    TokenUsage, ToolCall,
};
use crate::llm::utils::{
    calculate_cost_from_pricing_table, get_model_pricing, is_model_in_pricing_table,
    normalize_model_name, PricingTuple,
};
use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// Z.ai pricing constants (per 1M tokens in USD)
/// Source: https://docs.z.ai/guides/overview/pricing (verified Feb 18, 2026)
/// Format: (model, input, output, cache_write, cache_read)
const PRICING: &[PricingTuple] = &[
    // GLM-5 series (latest generation - Feb 2026)
    ("glm-5-code", 1.20, 5.00, 1.20, 0.12),
    ("glm-5", 1.00, 3.20, 1.00, 0.10),
    // GLM-4.7 series (flagship) - more specific variants first
    ("glm-4.7-flashx", 0.00, 0.00, 0.00, 0.00), // free model
    ("glm-4.7-flash", 0.00, 0.00, 0.00, 0.00),  // free model
    ("glm-4.7-battle", 0.60, 2.20, 0.60, 0.06),
    ("glm-4.7", 0.60, 2.20, 0.60, 0.30),
    // GLM-4.6 series
    ("glm-4.6v-flashx", 0.04, 0.40, 0.04, 0.004),
    ("glm-4.6v-flash", 0.04, 0.40, 0.04, 0.004),
    ("glm-4.6v", 0.30, 0.90, 0.30, 0.03),
    ("glm-4.6-flash", 0.60, 2.20, 0.60, 0.06),
    ("glm-4.6", 0.60, 2.20, 0.60, 0.06),
    // GLM-4.5 series - most specific first
    ("glm-4.5-airx", 1.10, 4.50, 1.10, 0.50),
    ("glm-4.5-air-plus", 0.20, 1.10, 0.20, 0.10),
    ("glm-4.5-air", 0.20, 1.10, 0.20, 0.10),
    ("glm-4.5v", 0.60, 1.80, 0.60, 0.30),
    ("glm-4.5-x", 2.20, 8.90, 2.20, 1.10),
    ("glm-4.5-flash", 0.60, 2.20, 0.60, 0.30),
    ("glm-4.5", 0.60, 2.20, 0.60, 0.30),
    // GLM-4 series
    ("glm-4-32b-0414-128k", 0.10, 0.10, 0.10, 0.01),
    ("glm-4-32b", 0.10, 0.10, 0.10, 0.01),
    ("glm-4-flash", 0.60, 2.20, 0.60, 0.06),
    ("glm-4", 0.60, 2.20, 0.60, 0.06),
];

/// Calculate cost for Z.ai models (case-insensitive)
fn calculate_cost(
    model: &str,
    regular_input_tokens: u64,
    cache_read_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    calculate_cost_from_pricing_table(
        model,
        PRICING,
        regular_input_tokens,
        0,
        cache_read_tokens,
        completion_tokens,
    )
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
    reasoning_content: Option<String>, // For thinking mode
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
        // Z.ai (GLM) models - check against pricing table (strict)
        is_model_in_pricing_table(model, PRICING)
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

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            get_model_pricing(model, PRICING)?;

        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
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
                reasoning_content: msg.thinking.as_ref().map(|t| t.content.clone()),
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
                let response = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", api_key))
                    .json(&request_body)
                    .send()
                    .await
                    .map_err(anyhow::Error::from)?;

                // Return Err for retryable HTTP errors so the retry loop catches them
                if retry::is_retryable_status(response.status().as_u16()) {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!("Z.ai API error {}: {}", status, error_text));
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
        return Err(anyhow::anyhow!("Z.ai API error {}: {}", status, error_text));
    }

    let response_text = retry::cancellable(
        async { response.text().await.map_err(anyhow::Error::from) },
        cancellation_token,
        || ProviderError::Cancelled.into(),
    )
    .await?;
    let zai_response: ZaiResponse = serde_json::from_str(&response_text)?;

    // Extract content and tool calls
    let raw_content = zai_response
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .unwrap_or_default();

    // Extract thinking from reasoning_content field first, then fall back to tags
    let (thinking, content) = extract_thinking(
        &raw_content,
        zai_response
            .choices
            .first()
            .and_then(|c| c.message.reasoning_content.clone()),
    );

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

    // Extract reasoning tokens from thinking block
    // Z.ai doesn't provide reasoning_tokens in usage response, so we estimate from thinking content length
    let reasoning_tokens = thinking.as_ref().map(|t| t.tokens).unwrap_or(0);

    // Calculate cost
    let usage = zai_response.usage.as_ref();
    // Z.ai returns prompt_tokens; this is RAW input and may include cached reads.
    let input_tokens_raw = usage.map(|u| u.prompt_tokens).unwrap_or(0);
    let completion_tokens = usage.map(|u| u.completion_tokens).unwrap_or(0);

    // Z.ai reports cached_tokens in prompt_tokens_details (these are cache READ tokens)
    let cache_read_tokens = usage
        .map(|u| u.prompt_tokens_details.cached_tokens)
        .unwrap_or(0);

    // Z.ai doesn't expose cache_write separately
    let cache_write_tokens = 0_u64;

    // Cost needs regular (non-cached) input split from cache reads.
    let regular_input_tokens = input_tokens_raw.saturating_sub(cache_read_tokens);

    let cost = calculate_cost(
        zai_response.model.as_str(),
        regular_input_tokens,
        cache_read_tokens,
        completion_tokens,
    );

    let token_usage = TokenUsage {
        // CLEAN input tokens - excludes cached reads (as per TokenUsage contract)
        input_tokens: regular_input_tokens,
        cache_read_tokens,  // Tokens read from cache
        cache_write_tokens, // Z.ai doesn't expose this (0)
        output_tokens: completion_tokens,
        reasoning_tokens, // Extracted from thinking block content
        total_tokens: usage.map(|u| u.total_tokens).unwrap_or(0),
        cost,
        request_time_ms: Some(request_time_ms),
    };

    // Build response JSON for exchange
    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified GenericToolCall format for conversation history
    if let Some(ref calls) = tool_calls {
        shared::set_response_tool_calls(&mut response_json, calls, None);
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
        thinking,
        exchange,
        tool_calls,
        finish_reason,
        structured_output,
        id: Some(zai_response.id),
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

/// Extract thinking content from reasoning_content field or <think>...</think> tags
fn extract_thinking(
    content: &str,
    reasoning_content: Option<String>,
) -> (Option<ThinkingBlock>, String) {
    // Priority 1: reasoning_content field (new API format for streaming)
    if let Some(ref thinking_str) = reasoning_content {
        if !thinking_str.trim().is_empty() {
            let tokens = (thinking_str.len() / 4) as u64;
            let thinking = Some(ThinkingBlock {
                content: thinking_str.clone(),
                tokens,
            });
            return (thinking, content.to_string());
        }
    }

    // Priority 2: <think>...</think> tags (legacy format)
    let think_start = "<think>";
    let think_end = "</think>";

    if let Some(start_idx) = content.find(think_start) {
        if let Some(end_idx) = content.find(think_end) {
            let thinking_content = &content[start_idx + think_start.len()..end_idx];
            let before_think = &content[..start_idx];
            let after_think = &content[end_idx + think_end.len()..];
            let clean_content = format!("{}{}", before_think.trim(), after_think.trim())
                .trim()
                .to_string();
            let tokens = (thinking_content.len() / 4) as u64;
            let thinking = Some(ThinkingBlock {
                content: thinking_content.to_string(),
                tokens,
            });
            return (thinking, clean_content);
        }
    }

    (None, content.to_string())
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
        // Test GLM-4.5: $0.60 input, $2.20 output (UPDATED PRICING)
        let cost = calculate_cost("glm-4.5", 1_000_000, 0, 1_000_000);
        assert!((cost.unwrap() - 2.80).abs() < 0.01); // 0.60 + 2.20

        // Test GLM-4.7: $0.60 input, $2.20 output (UPDATED PRICING)
        let cost = calculate_cost("glm-4.7", 1_000_000, 0, 1_000_000);
        assert!((cost.unwrap() - 2.80).abs() < 0.01); // 0.60 + 2.20

        // Test GLM-4.6: $0.60 input, $2.20 output (UPDATED PRICING)
        let cost = calculate_cost("glm-4.6", 1_000_000, 0, 1_000_000);
        assert!((cost.unwrap() - 2.80).abs() < 0.01); // 0.60 + 2.20

        // Test GLM-4.7-flash: free model
        let cost = calculate_cost("glm-4.7-flash", 1_000_000, 0, 1_000_000);
        assert_eq!(cost.unwrap(), 0.0);
    }

    #[test]
    fn test_extract_thinking_from_reasoning_content() {
        let content = "Final answer";
        let (thinking, clean) = extract_thinking(content, Some("step by step".to_string()));

        assert_eq!(clean, "Final answer");
        assert!(thinking.is_some());
        assert_eq!(thinking.as_ref().unwrap().content, "step by step");
    }

    #[test]
    fn test_extract_thinking_from_think_tags() {
        let content = "before <think>internal reasoning</think> after";
        let (thinking, clean) = extract_thinking(content, None);

        assert_eq!(clean, "beforeafter");
        assert!(thinking.is_some());
        assert_eq!(thinking.as_ref().unwrap().content, "internal reasoning");
    }

    #[test]
    fn test_cost_calculation_case_insensitive() {
        // Test mixed case model names
        let cost = calculate_cost("GLM-4.7", 1_000_000, 0, 1_000_000);
        assert!((cost.unwrap() - 2.80).abs() < 0.01); // Should work with uppercase

        let cost = calculate_cost("gLm-4.7-FlAsH", 1_000_000, 0, 1_000_000);
        assert_eq!(cost.unwrap(), 0.0); // Should work with mixed case

        let cost = calculate_cost("glm-4.5-AIR", 1_000_000, 0, 1_000_000);
        assert!((cost.unwrap() - 1.30).abs() < 0.01); // 0.20 + 1.10
    }

    #[test]
    fn test_cost_with_partial_tokens() {
        // Test with 500K tokens each
        let cost = calculate_cost("glm-4.5", 500_000, 0, 500_000);
        assert!((cost.unwrap() - 1.40).abs() < 0.01); // 0.60 * 0.5 + 2.20 * 0.5
    }

    #[test]
    fn test_unknown_model() {
        let cost = calculate_cost("unknown-model", 1_000_000, 0, 1_000_000);
        assert_eq!(cost, None);
    }

    #[test]
    fn test_cost_with_cache_read_tokens() {
        // GLM-4.7 pricing: input 0.60, cache_read 0.30, output 2.20
        // regular_input=100K => 0.06
        // cache_read=200K => 0.06
        // output=100K => 0.22
        // total => 0.34
        let cost = calculate_cost("glm-4.7", 100_000, 200_000, 100_000).unwrap();
        assert!((cost - 0.34).abs() < 0.0001);
    }

    #[test]
    fn test_zai_usage_deserialize_prompt_tokens_shape() {
        let parsed: ZaiUsage = serde_json::from_value(serde_json::json!({
            "prompt_tokens": 173,
            "completion_tokens": 104,
            "total_tokens": 277,
            "prompt_tokens_details": { "cached_tokens": 43 }
        }))
        .expect("prompt_tokens shape should deserialize");
        assert_eq!(parsed.prompt_tokens, 173);
        assert_eq!(parsed.prompt_tokens_details.cached_tokens, 43);
    }

    #[test]
    fn test_provider_capabilities() {
        let provider = ZaiProvider::new();
        assert!(provider.supports_caching("glm-4.7"));
        assert!(!provider.supports_vision("glm-4.7"));
        assert!(provider.supports_structured_output("glm-4.7"));
    }
}
