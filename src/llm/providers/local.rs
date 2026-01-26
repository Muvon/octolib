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

//! Local provider for local models (Ollama, LM Studio, LocalAI, etc.)
//!
//! This provider uses the standard OpenAI `/v1/chat/completions` endpoint format,
//! making it compatible with local model servers that implement the OpenAI API.
//!
//! ## Supported Local Servers
//! - **Ollama**: Default `http://localhost:11434/v1/chat/completions`
//! - **LM Studio**: `http://localhost:1234/v1/chat/completions`
//! - **LocalAI**: `http://localhost:8080/v1/chat/completions`
//! - Any other Local server
//!
//! ## Configuration
//! - `LOCAL_API_URL`: API endpoint (default: Ollama)
//! - `LOCAL_API_KEY`: Optional API key (empty by default for local servers)
//!
//! ## Usage
//! ```rust,no_run
//! use octolib::llm::ProviderFactory;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // With Ollama (default)
//! let (provider, model) = ProviderFactory::get_provider_for_model("local:llama3.2")?;
//!
//! // With LM Studio
//! std::env::set_var("LOCAL_API_URL", "http://localhost:1234/v1/chat/completions");
//! let (provider, model) = ProviderFactory::get_provider_for_model("local:mistral-7b")?;
//!
//! // With custom server
//! std::env::set_var("LOCAL_API_URL", "http://192.168.1.100:8080/v1/chat/completions");
//! let (provider, model) = ProviderFactory::get_provider_for_model("local:custom-model")?;
//! # Ok(())
//! # }
//! ```

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

/// Local provider for local models
#[derive(Debug, Clone)]
pub struct LocalProvider;

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalProvider {
    pub fn new() -> Self {
        Self
    }
}

const LOCAL_API_KEY_ENV: &str = "LOCAL_API_KEY";
const LOCAL_API_URL_ENV: &str = "LOCAL_API_URL";
const LOCAL_API_URL: &str = "http://localhost:11434/v1/chat/completions"; // Ollama default

#[async_trait::async_trait]
impl AiProvider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Accept any non-empty model name - local servers can run any model
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // API key is optional for local servers - return empty string if not set
        Ok(env::var(LOCAL_API_KEY_ENV).unwrap_or_default())
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false // Local models typically don't support caching
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false // Conservative default - can be enhanced later
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        // Conservative default for local models
        8_192
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true // Most local models support JSON mode
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to Local format
        let messages = convert_messages(&params.messages);

        // Create the request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
        });

        // Add max_tokens if specified
        if params.max_tokens > 0 {
            request_body["max_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add tools if available
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
                request_body["tool_choice"] = serde_json::json!("auto");
            }
        }

        // Add structured output format if specified
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
        let api_url = env::var(LOCAL_API_URL_ENV).unwrap_or_else(|_| LOCAL_API_URL.to_string());

        let response = execute_request(
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

// Reuse OpenRouter structures since they're Local
#[derive(Serialize, Deserialize, Debug)]
struct LocalMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<LocalToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: LocalFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct LocalFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct LocalResponse {
    id: String,
    choices: Vec<LocalChoice>,
    usage: LocalUsage,
}

#[derive(Deserialize, Debug)]
struct LocalChoice {
    message: LocalResponseMessage,
    finish_reason: String,
}

#[derive(Deserialize, Debug)]
struct LocalResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<LocalToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_details: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct LocalUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

// Convert messages to OpenAI format
fn convert_messages(messages: &[Message]) -> Vec<LocalMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                // Tool messages - MUST include tool_call_id
                result.push(LocalMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: message.tool_call_id.clone(),
                });
            }
            "assistant" if message.tool_calls.is_some() => {
                // Assistant message with tool calls - convert from unified GenericToolCall format
                let content = if !message.content.trim().is_empty() {
                    Some(serde_json::json!(message.content))
                } else {
                    None
                };

                // Convert unified GenericToolCall format to OpenAI format
                let tool_calls = if let Ok(generic_calls) =
                    serde_json::from_value::<Vec<crate::llm::tool_calls::GenericToolCall>>(
                        message.tool_calls.clone().unwrap(),
                    ) {
                    Some(
                        generic_calls
                            .iter()
                            .map(|tc| LocalToolCall {
                                id: tc.id.clone(),
                                tool_type: "function".to_string(),
                                function: LocalFunction {
                                    name: tc.name.clone(),
                                    arguments: serde_json::to_string(&tc.arguments)
                                        .unwrap_or_default(),
                                },
                            })
                            .collect(),
                    )
                } else {
                    None
                };

                result.push(LocalMessage {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                    tool_call_id: None,
                });
            }
            "user" | "assistant" | "system" => {
                // Regular text message
                result.push(LocalMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            _ => {
                // Unknown role - skip
                tracing::warn!("Unknown message role: {}", message.role);
            }
        }
    }

    result
}

// Execute HTTP request
async fn execute_request(
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

            Box::pin(async move {
                let mut request = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .json(&request_body);

                // Add Authorization header only if API key is not empty
                if !api_key.is_empty() {
                    request = request.header("Authorization", format!("Bearer {}", api_key));
                }

                request.send().await
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
            "Local API error {}: {}",
            status,
            error_text
        ));
    }

    let response_text = response.text().await?;
    let api_response: LocalResponse = serde_json::from_str(&response_text)?;

    let choice = api_response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

    let content = choice.message.content.unwrap_or_default();

    // Extract reasoning_details as thinking if present
    let reasoning_details = &choice.message.reasoning_details;
    let thinking = match reasoning_details.as_ref() {
        Some(rd) => {
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

    let reasoning_tokens = thinking.as_ref().map(|t| t.tokens).unwrap_or(0);

    let usage = TokenUsage {
        prompt_tokens: api_response.usage.prompt_tokens,
        output_tokens: api_response.usage.completion_tokens,
        reasoning_tokens,
        total_tokens: api_response.usage.total_tokens,
        cached_tokens: 0,
        cost: Some(0.0), // Local models are free
        request_time_ms: Some(request_time_ms),
    };

    let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;

    // Store tool_calls in unified format
    if let Some(ref tc) = tool_calls {
        let reasoning_details = choice.message.reasoning_details.clone();

        let generic_calls: Vec<crate::llm::tool_calls::GenericToolCall> = tc
            .iter()
            .map(|call| {
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

    let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "local");

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
        finish_reason: Some(choice.finish_reason),
        structured_output,
        id: Some(api_response.id),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = LocalProvider::new();

        // Should accept any non-empty model name
        assert!(provider.supports_model("llama3.2"));
        assert!(provider.supports_model("mistral-7b"));
        assert!(provider.supports_model("gpt4all-j"));
        assert!(provider.supports_model("any-model-name"));

        // Should reject empty model name
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_get_api_key_optional() {
        let provider = LocalProvider::new();

        // Should return empty string if not set (no error)
        let result = provider.get_api_key();
        assert!(result.is_ok());
    }

    #[test]
    fn test_supports_structured_output() {
        let provider = LocalProvider::new();
        assert!(provider.supports_structured_output("any-model"));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = LocalProvider::new();

        assert_eq!(provider.name(), "local");
        assert!(!provider.supports_caching("any-model"));
        assert!(!provider.supports_vision("any-model"));
        assert_eq!(provider.get_max_input_tokens("any-model"), 8_192);
    }
}
