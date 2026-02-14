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

use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
use crate::llm::utils::{
    calculate_cost_from_pricing_table, get_model_pricing, is_model_in_pricing_table,
    normalize_model_name, PricingTuple,
};
use anyhow::Result;
use reqwest::Client;
use serde::Deserialize;
use std::env;

/// OpenAI pricing constants (per 1M tokens in USD)
/// Source: https://platform.openai.com/docs/pricing and model cards (verified Feb 13, 2026)
/// Format: (model, input, output, cache_write, cache_read)
/// Note: For models without caching, cache_write = input and cache_read = input
const PRICING: &[PricingTuple] = &[
    // GPT-5.2 family
    ("gpt-5.3-codex", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2-pro", 21.00, 168.00, 21.00, 21.00),
    ("gpt-5.2-codex", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2-chat-latest", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2", 1.75, 14.00, 1.75, 0.175),
    // GPT-5.1 family
    ("gpt-5.1-codex-mini", 0.25, 2.00, 0.25, 0.025),
    ("gpt-5.1-codex-max", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1-codex", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1-chat-latest", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1", 1.25, 10.00, 1.25, 0.125),
    // GPT-5 family
    ("gpt-5-pro", 15.00, 120.00, 15.00, 15.00),
    ("gpt-5-codex", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5-chat-latest", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5-mini", 0.25, 2.00, 0.25, 0.025),
    ("gpt-5-nano", 0.05, 0.40, 0.05, 0.005),
    ("gpt-5", 1.25, 10.00, 1.25, 0.125),
    // Codex CLI optimized model
    ("codex-mini-latest", 1.50, 6.00, 1.50, 0.375),
    // GPT-4.1 family
    ("gpt-4.1-mini", 0.40, 1.60, 0.40, 0.10),
    ("gpt-4.1-nano", 0.10, 0.40, 0.10, 0.025),
    ("gpt-4.1", 2.00, 8.00, 2.00, 0.50),
    // GPT-4o / realtime / audio
    ("gpt-realtime-mini", 0.60, 2.40, 0.60, 0.06),
    ("gpt-realtime", 4.00, 16.00, 4.00, 0.40),
    ("gpt-audio", 2.50, 10.00, 2.50, 2.50),
    ("gpt-4o-mini-realtime-preview", 0.60, 2.40, 0.60, 0.30),
    ("gpt-4o-realtime-preview", 5.00, 20.00, 5.00, 2.50),
    ("gpt-4o-mini", 0.15, 0.60, 0.15, 0.075),
    ("gpt-4o-2024-05-13", 5.00, 15.00, 5.00, 5.00),
    ("gpt-4o", 2.50, 10.00, 2.50, 1.25),
    // Legacy/long-tail models retained for compatibility
    ("gpt-4.5-preview", 75.00, 150.00, 75.00, 75.00),
    ("o1", 15.00, 60.00, 15.00, 7.50),
    ("o1-pro", 150.00, 600.00, 150.00, 150.00),
    ("o1-mini", 1.10, 4.40, 1.10, 0.55),
    ("o3", 2.00, 8.00, 2.00, 0.50),
    ("o3-pro", 20.00, 80.00, 20.00, 20.00),
    ("o3-mini", 1.10, 4.40, 1.10, 0.55),
    ("o3-deep-research", 10.00, 40.00, 10.00, 2.50),
    ("o4-mini", 1.10, 4.40, 1.10, 0.275),
    ("o4-mini-deep-research", 2.00, 8.00, 2.00, 0.50),
    ("gpt-4-turbo", 10.00, 30.00, 10.00, 10.00),
    ("gpt-4", 30.00, 60.00, 30.00, 30.00),
    ("gpt-4-32k", 60.00, 120.00, 60.00, 60.00),
    ("gpt-3.5-turbo-instruct", 1.50, 2.00, 1.50, 1.50),
    ("gpt-3.5-turbo-16k-0613", 3.00, 4.00, 3.00, 3.00),
    ("gpt-3.5-turbo", 0.50, 1.50, 0.50, 0.50),
];

/// Calculate cost for OpenAI models with basic pricing (case-insensitive)
fn calculate_cost(model: &str, input_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_from_pricing_table(model, PRICING, input_tokens, 0, 0, completion_tokens)
}

/// Calculate cost with cache-aware pricing (case-insensitive)
/// - regular_input_tokens: charged at normal price
/// - cache_read_tokens: charged at model-specific cached-input price
/// - output_tokens: charged at normal price
fn calculate_cost_with_cache(
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

/// Check if a model supports the temperature parameter
/// O1, O2, O3, O4 and GPT-5 series models don't support temperature
fn supports_temperature(model: &str) -> bool {
    !model.starts_with("o1")
        && !model.starts_with("o2")
        && !model.starts_with("o3")
        && !model.starts_with("o4")
        && !model.starts_with("gpt-5")
}

/// Convert messages to Responses API input format
///
/// The OpenAI Responses API maintains conversation history server-side via `previous_id`.
/// This means we only send NEW messages/tool results, not the full conversation history.
///
/// # Behavior
/// - **Initial request** (no previous_id): Send all user/system messages
/// - **Tool response**: Send ONLY new tool results after last assistant message as function_call_output
/// - **Continuation**: Send ONLY new user/system messages after last assistant message
///
/// # Arguments
/// * `messages` - Full conversation history
/// * `has_previous_response` - Whether we have a previous_id (continuation)
fn messages_to_input(messages: &[Message], has_previous_response: bool) -> Vec<serde_json::Value> {
    if has_previous_response {
        // Find the index of the last assistant message with an ID
        let last_assistant_idx = messages
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.role == "assistant" && m.id.is_some())
            .map(|(idx, _)| idx);

        if let Some(assistant_idx) = last_assistant_idx {
            // Check if there are any tool messages AFTER the last assistant message
            let new_tool_results: Vec<_> = messages
                .iter()
                .skip(assistant_idx + 1)
                .filter_map(|msg| {
                    if msg.role == "tool" {
                        let call_id_str = msg.tool_call_id.clone().unwrap_or_default();
                        Some(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": call_id_str,
                            "output": msg.content
                        }))
                    } else {
                        None
                    }
                })
                .collect();

            // If we have new tool results, send them
            if !new_tool_results.is_empty() {
                return new_tool_results;
            }
        }

        // No new tool results - send new user/system messages after the last assistant
        messages
            .iter()
            .skip(last_assistant_idx.map(|idx| idx + 1).unwrap_or(0))
            .filter_map(|msg| match msg.role.as_str() {
                "user" | "system" => Some(serde_json::json!({
                    "role": msg.role,
                    "content": msg.content
                })),
                _ => None,
            })
            .collect()
    } else {
        // Initial request: send all user/system messages (skip assistant messages)
        messages
            .iter()
            .filter_map(|msg| match msg.role.as_str() {
                "user" | "system" => Some(serde_json::json!({
                    "role": msg.role,
                    "content": msg.content
                })),
                _ => None,
            })
            .collect()
    }
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
const OPENAI_OAUTH_ACCESS_TOKEN_ENV: &str = "OPENAI_OAUTH_ACCESS_TOKEN";
const OPENAI_OAUTH_ACCOUNT_ID_ENV: &str = "OPENAI_OAUTH_ACCOUNT_ID";
const OPENAI_API_URL_ENV: &str = "OPENAI_API_URL";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/responses";
#[async_trait::async_trait]
impl AiProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn supports_model(&self, model: &str) -> bool {
        // OpenAI models - check against pricing table (strict, if not in pricing = not supported)
        is_model_in_pricing_table(model, PRICING)
    }

    fn get_api_key(&self) -> Result<String> {
        // Check for OAuth tokens first (priority)
        if env::var(OPENAI_OAUTH_ACCESS_TOKEN_ENV).is_ok() {
            return Err(anyhow::anyhow!(
                "Using OAuth authentication. API key not available when {} is set.",
                OPENAI_OAUTH_ACCESS_TOKEN_ENV
            ));
        }

        // Fall back to API key
        match env::var(OPENAI_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "OpenAI API key not found in environment variable: {}. Set either {} for API key auth or {} + {} for OAuth.",
                OPENAI_API_KEY_ENV,
                OPENAI_API_KEY_ENV,
                OPENAI_OAUTH_ACCESS_TOKEN_ENV,
                OPENAI_OAUTH_ACCOUNT_ID_ENV
            )),
        }
    }

    fn supports_caching(&self, model: &str) -> bool {
        // OpenAI supports automatic prompt caching for most text models.
        // Exclude known no-cache models (pro and audio variants).
        let model_lower = normalize_model_name(model);
        !(model_lower.starts_with("gpt-5-pro")
            || model_lower.starts_with("gpt-5.2-pro")
            || model_lower.starts_with("gpt-audio"))
            && (model_lower.contains("gpt-4o")
                || model_lower.contains("gpt-4.1")
                || model_lower.contains("gpt-5")
                || model_lower.contains("codex-mini")
                || model_lower.contains("gpt-realtime")
                || model_lower.contains("o1-preview")
                || model_lower.contains("o1-mini")
                || model_lower.contains("o1")
                || model_lower.contains("o3")
                || model_lower.contains("o4"))
    }

    fn supports_vision(&self, model: &str) -> bool {
        // OpenAI vision-capable models (case-insensitive)
        let normalized = normalize_model_name(model);
        normalized.starts_with("gpt-4o")
            || normalized.starts_with("gpt-4.1")
            || normalized.starts_with("gpt-4-turbo")
            || normalized.starts_with("gpt-4-vision-preview")
            || normalized.starts_with("gpt-4o-")
            || normalized.starts_with("gpt-5")
            || normalized.starts_with("codex-mini")
            || normalized.starts_with("gpt-realtime")
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // OpenAI model context window limits (case-insensitive)
        // These are the actual context windows - API handles output limits
        let normalized = normalize_model_name(model);

        // GPT-5 family: 400K context window
        if normalized.starts_with("gpt-5") {
            return 400_000;
        }
        // codex-mini-latest: 200K context window
        if normalized.starts_with("codex-mini") {
            return 200_000;
        }
        // Realtime models: 32K context window
        if normalized.starts_with("gpt-realtime") {
            return 32_000;
        }
        // GPT Audio: 128K context window
        if normalized.starts_with("gpt-audio") {
            return 128_000;
        }
        // GPT-4o models: 128K context window
        if normalized.starts_with("gpt-4o") {
            return 128_000;
        }
        // GPT-4 models: varies by version
        if normalized.starts_with("gpt-4-turbo")
            || normalized.starts_with("gpt-4.5")
            || normalized.starts_with("gpt-4.1")
        {
            return 128_000;
        }
        if normalized.starts_with("gpt-4") && !normalized.starts_with("gpt-4o") {
            return 8_192; // Old GPT-4: 8K context window
        }
        // O-series models: 128K context window
        if normalized.starts_with("o1")
            || normalized.starts_with("o2")
            || normalized.starts_with("o3")
        {
            return 128_000;
        }
        // GPT-3.5: 16K context window
        if normalized.starts_with("gpt-3.5") {
            return 16_384;
        }
        // Default conservative limit
        8_192
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true // All OpenAI models support structured output
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

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        // Check for OAuth tokens first (priority), otherwise use API key
        let (use_oauth, oauth_account_id) = if let (Ok(access_token), Ok(account_id)) = (
            env::var(OPENAI_OAUTH_ACCESS_TOKEN_ENV),
            env::var(OPENAI_OAUTH_ACCOUNT_ID_ENV),
        ) {
            (true, Some((access_token, account_id)))
        } else {
            (false, None)
        };

        let auth_token = if use_oauth {
            oauth_account_id.as_ref().unwrap().0.clone()
        } else {
            self.get_api_key()?
        };

        // Extract previous_id from messages if not explicitly provided
        // Find the LAST message with an ID - this is the most recent response from the API
        // The Responses API maintains conversation state server-side via this ID
        let previous_id = params.previous_id.clone().or_else(|| {
            params
                .messages
                .iter()
                .rev()
                .find(|m| m.id.is_some())
                .and_then(|m| m.id.clone())
        });

        // Convert messages to array input format for Responses API
        let input_array = messages_to_input(&params.messages, previous_id.is_some());

        // Create the request body for Responses API
        let mut request_body = serde_json::json!({
            "model": params.model,
            "input": input_array,
        });

        // Only add temperature/top_p for models that support it
        if supports_temperature(&params.model) {
            request_body["temperature"] = serde_json::json!(params.temperature);
            request_body["top_p"] = serde_json::json!(params.top_p);
        }

        // Add previous_id for multi-turn conversations
        if let Some(ref prev_id) = previous_id {
            request_body["previous_response_id"] = serde_json::json!(prev_id);
        }

        // Add max_output_tokens if specified
        if params.max_tokens > 0 {
            request_body["max_output_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add reasoning effort for reasoning models
        if params.model.starts_with("o1")
            || params.model.starts_with("o3")
            || params.model.starts_with("o4")
            || params.model.starts_with("gpt-5")
        {
            request_body["reasoning"] = serde_json::json!({
                "effort": "medium"
            });
        }

        // Add tools if available
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let mut sorted_tools = tools.clone();
                sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

                let openai_tools: Vec<serde_json::Value> = sorted_tools
                    .iter()
                    .map(|f| {
                        serde_json::json!({
                            "type": "function",
                            "name": f.name,
                            "description": f.description,
                            "parameters": f.parameters
                        })
                    })
                    .collect();

                request_body["tools"] = serde_json::json!(openai_tools);
            }
        }

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request_body["text"] = serde_json::json!({
                        "format": {
                            "type": "json_object"
                        }
                    });
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    if let Some(schema) = &response_format.schema {
                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "name": "response_schema",
                            "schema": schema
                        });

                        // Add strict mode if specified
                        if matches!(
                            response_format.mode,
                            crate::llm::types::ResponseMode::Strict
                        ) {
                            format_obj["strict"] = serde_json::json!(true);
                        }

                        request_body["text"] = serde_json::json!({
                            "format": format_obj
                        });
                    }
                }
            }
        }

        // Execute the request with retry logic
        let account_id_header = oauth_account_id.as_ref().map(|(_, id)| id.clone());
        let api_url = env::var(OPENAI_API_URL_ENV).unwrap_or_else(|_| OPENAI_API_URL.to_string());

        let response = execute_openai_request(
            auth_token,
            account_id_header,
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

// Responses API structures
#[derive(Deserialize, Debug)]
struct ResponsesApiResponse {
    #[serde(default)]
    id: Option<String>,
    output: Vec<ResponseOutput>,
    usage: ResponseUsage,
}
#[derive(Deserialize, Debug)]
struct ResponseOutput {
    #[serde(rename = "type")]
    output_type: String, // "message", "function_call", "reasoning"
    #[serde(default)]
    #[allow(dead_code)]
    // id field exists in API response but we use call_id for function calls
    id: Option<String>,

    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    content: Option<Vec<ResponseContent>>,
}

#[derive(Deserialize, Debug)]
struct ResponseContent {
    #[serde(rename = "type")]
    content_type: String, // "output_text"
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ResponseUsage {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
    #[serde(default)]
    output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Deserialize, Debug)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct OutputTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

// Execute OpenAI HTTP request
async fn execute_openai_request(
    auth_token: String,
    account_id: Option<String>,
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
            let auth_token = auth_token.clone();
            let account_id = account_id.clone();
            let api_url = api_url.clone();
            let request_body = request_body.clone();
            Box::pin(async move {
                let mut req = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .header("Authorization", format!("Bearer {}", auth_token));

                // Add ChatGPT-Account-ID header if using OAuth
                if let Some(id) = account_id {
                    req = req.header("ChatGPT-Account-ID", id);
                }

                req.json(&request_body)
                    .send()
                    .await
                    .map_err(anyhow::Error::from)
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

    // Extract rate limit headers before consuming response
    let mut rate_limit_headers = std::collections::HashMap::new();
    let headers = response.headers();

    // Check for cache hit headers first (fallback for older API versions)
    let _cache_creation_input_tokens = headers
        .get("x-cache-creation-input-tokens")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    let _cache_read_input_tokens = headers
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
        let error_text = retry::cancellable(
            async { response.text().await.map_err(anyhow::Error::from) },
            cancellation_token,
            || ProviderError::Cancelled.into(),
        )
        .await?;
        return Err(anyhow::anyhow!(
            "OpenAI API error {}: {}",
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
    let api_response: ResponsesApiResponse = serde_json::from_str(&response_text)?;

    // Extract content from output array
    let mut content = String::new();
    let mut tool_calls: Option<Vec<ToolCall>> = None;
    let mut reasoning_content: Option<String> = None;

    for output in &api_response.output {
        match output.output_type.as_str() {
            "message" => {
                if let Some(content_array) = &output.content {
                    for content_item in content_array {
                        if content_item.content_type == "output_text" {
                            if let Some(text) = &content_item.text {
                                if !content.is_empty() {
                                    content.push('\n');
                                }
                                content.push_str(text);
                            }
                        }
                    }
                }
            }
            "function_call" => {
                // Extract tool call from function_call output
                // Parse arguments to avoid double-escaping when serializing back
                if let (Some(name), Some(args), Some(call_id)) =
                    (&output.name, &output.arguments, &output.call_id)
                {
                    let arguments: serde_json::Value = if args.is_string() {
                        serde_json::from_str(args.as_str().unwrap_or("{}"))
                            .unwrap_or(serde_json::json!({}))
                    } else {
                        args.clone()
                    };

                    // CRITICAL: APPEND to tool_calls vector for parallel tool call support
                    let new_tool_call = ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments,
                    };

                    if let Some(ref mut calls) = tool_calls {
                        calls.push(new_tool_call);
                    } else {
                        tool_calls = Some(vec![new_tool_call]);
                    }
                }
            }

            "reasoning" => {
                if let Some(content_array) = &output.content {
                    for content_item in content_array {
                        if content_item.content_type == "output_text" {
                            if let Some(text) = &content_item.text {
                                reasoning_content = Some(text.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Extract reasoning tokens
    let reasoning_tokens = api_response
        .usage
        .output_tokens_details
        .as_ref()
        .map(|d| d.reasoning_tokens)
        .unwrap_or(0);

    let thinking = reasoning_content.map(|rc| ThinkingBlock {
        content: rc,
        tokens: reasoning_tokens,
    });

    // Calculate cost
    let cost = request_body
        .get("model")
        .and_then(|m| m.as_str())
        .and_then(|model| {
            let cached_tokens = api_response
                .usage
                .input_tokens_details
                .as_ref()
                .map(|d| d.cached_tokens)
                .unwrap_or(0);
            if cached_tokens > 0 {
                let regular_input_tokens = api_response
                    .usage
                    .input_tokens
                    .saturating_sub(cached_tokens);
                calculate_cost_with_cache(
                    model,
                    regular_input_tokens,
                    cached_tokens,
                    api_response.usage.output_tokens,
                )
            } else {
                calculate_cost(
                    model,
                    api_response.usage.input_tokens,
                    api_response.usage.output_tokens,
                )
            }
        });

    // OpenAI reports cache_read in input_tokens_details, but NOT cache_write
    // input_tokens from API includes: regular_input + cache_read
    let cache_read_tokens = api_response
        .usage
        .input_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .unwrap_or(0);

    // OpenAI does NOT report cache_write tokens in API response
    let cache_write_tokens = 0_u64;

    // Calculate CLEAN input tokens (no cache)
    let input_tokens_clean = api_response
        .usage
        .input_tokens
        .saturating_sub(cache_read_tokens);

    let usage = TokenUsage {
        input_tokens: input_tokens_clean, // CLEAN input (no cache)
        cache_read_tokens,                // Tokens read from cache
        cache_write_tokens,               // OpenAI doesn't expose this (0)
        output_tokens: api_response.usage.output_tokens,
        reasoning_tokens,
        total_tokens: api_response.usage.total_tokens + reasoning_tokens,
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
                meta: None,
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

    // Try to parse structured output if it was requested
    let structured_output = if content.trim().starts_with('{') || content.trim().starts_with('[') {
        serde_json::from_str(&content).ok()
    } else {
        None
    };

    Ok(ProviderResponse {
        content,
        thinking,
        exchange,
        tool_calls,
        finish_reason: None, // Responses API doesn't have finish_reason
        structured_output,
        id: api_response.id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

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
    fn test_supports_model_gpt5() {
        let provider = OpenAiProvider::new();

        // GPT-5 models should be supported
        assert!(provider.supports_model("gpt-5"));
        assert!(provider.supports_model("gpt-5-2025-08-07"));
        assert!(provider.supports_model("gpt-5-mini"));
        assert!(provider.supports_model("gpt-5-mini-2025-08-07"));
        assert!(provider.supports_model("gpt-5-nano"));
        assert!(provider.supports_model("gpt-5-nano-2025-08-07"));
        assert!(provider.supports_model("gpt-5.2-codex"));
        assert!(provider.supports_model("gpt-5.3-codex"));
        assert!(provider.supports_model("gpt-5.2-chat-latest"));
        assert!(provider.supports_model("codex-mini-latest"));

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
    fn test_supports_model_case_insensitive() {
        let provider = OpenAiProvider::new();

        // Test uppercase
        assert!(provider.supports_model("GPT-5"));
        assert!(provider.supports_model("GPT-4O"));
        assert!(provider.supports_model("GPT-4"));
        // Test mixed case
        assert!(provider.supports_model("Gpt-5"));
        assert!(provider.supports_model("gPT-4o"));
        assert!(provider.supports_model("O1"));
        assert!(provider.supports_model("o3-mini"));
    }

    #[test]
    fn test_get_max_input_tokens_gpt5() {
        let provider = OpenAiProvider::new();

        // GPT-5 models should have 400K context window
        assert_eq!(provider.get_max_input_tokens("gpt-5"), 400_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-2025-08-07"), 400_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-mini"), 400_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5-nano"), 400_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5.2-codex"), 400_000);
        assert_eq!(provider.get_max_input_tokens("gpt-5.3-codex"), 400_000);
        assert_eq!(provider.get_max_input_tokens("codex-mini-latest"), 200_000);

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
        assert!(provider.supports_vision("gpt-5.2-codex"));
        assert!(provider.supports_vision("gpt-5.3-codex"));
        assert!(provider.supports_vision("codex-mini-latest"));
        assert!(provider.supports_vision("gpt-realtime"));

        // Models that should NOT support vision
        assert!(!provider.supports_vision("gpt-3.5-turbo"));
        assert!(!provider.supports_vision("gpt-4"));
        assert!(!provider.supports_vision("o1-preview"));
        assert!(!provider.supports_vision("o1-mini"));
        assert!(!provider.supports_vision("text-davinci-003"));
    }

    #[test]
    #[serial]
    fn test_oauth_token_priority() {
        let provider = OpenAiProvider::new();

        // Set OAuth tokens
        env::set_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV, "test-oauth-token");
        env::set_var(OPENAI_OAUTH_ACCOUNT_ID_ENV, "test-account-id");

        // get_api_key should return error when OAuth is set
        let result = provider.get_api_key();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("OAuth authentication"));

        // Clean up
        env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
        env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);
    }

    #[test]
    #[serial]
    fn test_api_key_fallback() {
        let provider = OpenAiProvider::new();

        // Remove OAuth tokens if set
        env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
        env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);

        // Set API key
        env::set_var(OPENAI_API_KEY_ENV, "test-api-key");

        // get_api_key should return the API key
        let result = provider.get_api_key();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test-api-key");

        // Clean up
        env::remove_var(OPENAI_API_KEY_ENV);
    }

    #[test]
    #[serial]
    fn test_no_auth_error() {
        let provider = OpenAiProvider::new();

        // Remove all auth env vars
        env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
        env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);
        env::remove_var(OPENAI_API_KEY_ENV);

        // get_api_key should return error
        let result = provider.get_api_key();
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("OPENAI_API_KEY") || error_msg.contains("OPENAI_OAUTH"));
    }

    #[test]
    fn test_messages_to_input() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: None,
            },
            Message {
                role: "user".to_string(),
                content: "Hello!".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: None,
            },
        ];

        let input = messages_to_input(&messages, false);
        assert_eq!(input.len(), 2);

        // First message - content is plain string
        let first = &input[0];
        assert_eq!(first["role"], "system");
        assert_eq!(first["content"], "You are a helpful assistant.");

        // Second message - content is plain string
        let second = &input[1];
        assert_eq!(second["role"], "user");
        assert_eq!(second["content"], "Hello!");
    }

    #[test]
    fn test_messages_to_input_with_tool_response() {
        // Scenario: Assistant made a tool call, we're sending the tool result back
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "What is the weather?".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: Some(serde_json::json!([{
                    "id": "call_12345",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{}"
                    }
                }])),
                tool_call_id: None,
                name: None,
                thinking: None,
                id: Some("resp_abc123".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: "{\"temperature\": \"22C\", \"condition\": \"sunny\"}".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: Some("call_12345".to_string()),
                name: Some("get_weather".to_string()),
                thinking: None,
                id: None,
            },
        ];

        // When there are NEW tool responses after assistant, send only those tool outputs
        let input = messages_to_input(&messages, true);
        assert_eq!(input.len(), 1); // Only the NEW tool response

        // Tool response uses function_call_output format
        let tool_output = &input[0];
        assert_eq!(tool_output["type"], "function_call_output");
        assert_eq!(tool_output["call_id"], "call_12345");
        assert_eq!(
            tool_output["output"],
            "{\"temperature\": \"22C\", \"condition\": \"sunny\"}"
        );
    }

    #[test]
    fn test_messages_to_input_continuation_without_tools() {
        // Scenario: Continuing conversation without tool calls (like "what else you can do?")
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "run date in shell".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: Some(serde_json::json!([{
                    "id": "call_old",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": "{\"command\": \"date\"}"
                    }
                }])),
                tool_call_id: None,
                name: None,
                thinking: None,
                id: Some("resp_first".to_string()),
            },
            Message {
                role: "tool".to_string(),
                content: "Mon Jan 19 22:12:18 +07 2026".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: Some("call_old".to_string()),
                name: Some("shell".to_string()),
                thinking: None,
                id: None,
            },
            Message {
                role: "assistant".to_string(),
                content: "The current date is Mon Jan 19 22:12:18 +07 2026".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: Some("resp_second".to_string()),
            },
            Message {
                role: "user".to_string(),
                content: "what else you can do?".to_string(),
                timestamp: 0,
                images: None,
                cached: false,
                tool_calls: None,
                tool_call_id: None,
                name: None,
                thinking: None,
                id: None,
            },
        ];

        // Should send only the NEW user message, NOT the old tool result
        let input = messages_to_input(&messages, true);
        assert_eq!(input.len(), 1);

        // Should be the new user message
        let user_msg = &input[0];
        assert_eq!(user_msg["role"], "user");
        assert_eq!(user_msg["content"], "what else you can do?");
    }

    #[test]
    fn test_codex_pricing() {
        // Test that codex models have pricing defined
        let cost = calculate_cost("gpt-5-codex", 1000, 500);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();
        // Expected: (1000/1M * 1.25) + (500/1M * 10.0) = 0.00125 + 0.005 = 0.00625
        assert!((cost_value - 0.00625).abs() < 0.0000001);

        // Verify gpt-5.2-codex pricing path exists
        let cost_52 = calculate_cost("gpt-5.2-codex", 1000, 500);
        assert!(cost_52.is_some());
        let cost_52_value = cost_52.unwrap();
        // Expected: (1000/1M * 1.75) + (500/1M * 14.0) = 0.00175 + 0.007 = 0.00875
        assert!((cost_52_value - 0.00875).abs() < 0.0000001);

        // Verify gpt-5.3-codex pricing path exists
        let cost_53 = calculate_cost("gpt-5.3-codex", 1000, 500);
        assert!(cost_53.is_some());
        let cost_53_value = cost_53.unwrap();
        assert!((cost_53_value - 0.00875).abs() < 0.0000001);
    }

    #[test]
    fn test_cache_pricing_for_gpt_5_2_codex() {
        // (regular 1000 * 1.75 + cached 1000 * 0.175 + output 500 * 14) / 1M
        let cost = calculate_cost_with_cache("gpt-5.2-codex", 1000, 1000, 500).unwrap();
        assert!((cost - 0.008925).abs() < 0.0000001);
    }
}
