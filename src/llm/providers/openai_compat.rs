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

use super::shared;
use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
    ToolCall,
};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub(crate) struct OpenAiCompatConfig {
    pub provider_name: &'static str,
    pub usage_fallback_cost: Option<f64>,
    pub use_response_cost: bool,
}

pub(crate) fn get_optional_api_key(env_name: &str) -> String {
    std::env::var(env_name).unwrap_or_default()
}

pub(crate) fn get_api_url(env_name: &str, default_url: &str) -> String {
    std::env::var(env_name).unwrap_or_else(|_| default_url.to_string())
}

pub(crate) async fn chat_completion(
    config: OpenAiCompatConfig,
    api_key: String,
    api_url: String,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let messages = convert_messages(&params.messages);

    let mut request_body = serde_json::json!({
        "model": params.model,
        "messages": messages,
        "temperature": params.temperature,
        "top_p": params.top_p,
    });

    if config.provider_name.eq_ignore_ascii_case("ollama") {
        request_body["stream"] = serde_json::json!(false);
    }

    if params.max_tokens > 0 {
        request_body["max_tokens"] = serde_json::json!(params.max_tokens);
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
            request_body["tool_choice"] = serde_json::json!("auto");
        }
    }

    if let Some(response_format) = &params.response_format {
        // Ollama and local servers use a top-level "format" key instead of "response_format".
        // "format": "json" for json_object mode, "format": <schema> for structured output.
        let is_ollama_like = config.provider_name.eq_ignore_ascii_case("ollama")
            || config.provider_name.eq_ignore_ascii_case("local");

        match &response_format.format {
            crate::llm::types::OutputFormat::Json => {
                if is_ollama_like {
                    request_body["format"] = serde_json::json!("json");
                } else {
                    request_body["response_format"] = serde_json::json!({
                        "type": "json_object"
                    });
                }
            }
            crate::llm::types::OutputFormat::JsonSchema => {
                if is_ollama_like {
                    // Ollama accepts the JSON schema directly as the "format" value
                    if let Some(schema) = &response_format.schema {
                        request_body["format"] = schema.clone();
                    } else {
                        // No schema provided — fall back to plain JSON mode
                        request_body["format"] = serde_json::json!("json");
                    }
                } else if let Some(schema) = &response_format.schema {
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

    execute_request(config, api_key, api_url, request_body, params).await
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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct OpenAiCompatToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAiCompatFunction,
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

fn convert_messages(messages: &[Message]) -> Vec<OpenAiCompatMessage> {
    let mut result = Vec::new();

    for message in messages {
        match message.role.as_str() {
            "tool" => {
                result.push(OpenAiCompatMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: message.tool_call_id.clone(),
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
                });
            }
            "user" | "assistant" | "system" => {
                result.push(OpenAiCompatMessage {
                    role: message.role.clone(),
                    content: Some(serde_json::json!(message.content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            _ => {
                tracing::warn!("Unknown message role: {}", message.role);
            }
        }
    }

    result
}

async fn execute_request(
    config: OpenAiCompatConfig,
    api_key: String,
    api_url: String,
    request_body: serde_json::Value,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let client = Client::new();
    let start_time = std::time::Instant::now();

    let response = retry::retry_with_exponential_backoff(
        || {
            let client = client.clone();
            let api_key = api_key.clone();
            let api_url = api_url.clone();
            let request_body = request_body.clone();
            let provider_name = config.provider_name.to_string();

            Box::pin(async move {
                let mut request = client
                    .post(&api_url)
                    .header("Content-Type", "application/json")
                    .json(&request_body);

                if !api_key.is_empty() {
                    request = request.header("Authorization", format!("Bearer {}", api_key));
                }

                let response = request.send().await.map_err(anyhow::Error::from)?;

                // Return Err for retryable HTTP errors so the retry loop catches them
                if retry::is_retryable_status(response.status().as_u16()) {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!(
                        "{} API error {}: {}",
                        provider_name,
                        status,
                        error_text
                    ));
                }

                Ok(response)
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
    )
    .await?;

    let request_time_ms = start_time.elapsed().as_millis() as u64;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = retry::cancellable(
            async { response.text().await.map_err(anyhow::Error::from) },
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
        )
        .await?;
        return Err(anyhow::anyhow!(
            "{} API error {}: {}",
            config.provider_name,
            status,
            error_text
        ));
    }

    let response_text = retry::cancellable(
        async { response.text().await.map_err(anyhow::Error::from) },
        params.cancellation_token.as_ref(),
        || ProviderError::Cancelled.into(),
    )
    .await?;
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
