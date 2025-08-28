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

use crate::retry;
use crate::traits::AiProvider;
use crate::types::{
	ChatCompletionParams, Message, ProviderExchange, ProviderResponse, TokenUsage, ToolCall,
};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;

/// OpenAI pricing constants (per 1M tokens in USD)
const PRICING: &[(&str, f64, f64)] = &[
	// Model, Input price per 1M tokens, Output price per 1M tokens
	("gpt-4o", 2.50, 10.00),
	("gpt-4o-mini", 0.15, 0.60),
	("gpt-4-turbo", 10.00, 30.00),
	("gpt-4", 30.00, 60.00),
	("gpt-3.5-turbo", 0.50, 1.50),
];

/// Calculate cost for OpenAI models
fn calculate_openai_cost(model: &str, prompt_tokens: u64, completion_tokens: u64) -> Option<f64> {
	for (pricing_model, input_price, output_price) in PRICING {
		if model.contains(pricing_model) {
			let input_cost = (prompt_tokens as f64 / 1_000_000.0) * input_price;
			let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;
			return Some(input_cost + output_cost);
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
		model.starts_with("gpt-") || model.contains("gpt")
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

	fn supports_caching(&self, _model: &str) -> bool {
		false // OpenAI doesn't support caching yet
	}

	fn supports_vision(&self, model: &str) -> bool {
		model.contains("gpt-4o") || model.contains("gpt-4-turbo") || model.contains("gpt-4-vision")
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		if model.contains("gpt-4o") || model.contains("gpt-4-turbo") {
			128_000
		} else if model.contains("gpt-4") {
			8_192
		} else if model.contains("gpt-3.5-turbo") {
			16_385
		} else {
			4_096
		}
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = self.get_api_key()?;

		// Convert messages to OpenAI format
		let openai_messages = convert_messages(&params.messages);

		// Create the request body
		let mut request_body = serde_json::json!({
			"model": params.model,
			"messages": openai_messages,
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
				// Assistant message with tool calls - preserve original tool_calls
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

				// Extract original tool_calls from stored data
				let tool_calls = message.tool_calls.clone();

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
						if let crate::types::ImageData::Base64(data) = &image.data {
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
	let tool_calls = choice.message.tool_calls.map(|calls| {
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

	// Calculate cost
	let cost = calculate_openai_cost(
		request_body["model"].as_str().unwrap_or(""),
		openai_response.usage.prompt_tokens,
		openai_response.usage.completion_tokens,
	);

	let usage = TokenUsage {
		prompt_tokens: openai_response.usage.prompt_tokens,
		output_tokens: openai_response.usage.completion_tokens,
		total_tokens: openai_response.usage.total_tokens,
		cached_tokens: 0, // OpenAI doesn't support caching
		cost,
		request_time_ms: Some(request_time_ms),
	};

	let exchange = if rate_limit_headers.is_empty() {
		ProviderExchange::new(
			request_body,
			serde_json::from_str(&response_text)?,
			Some(usage),
			"openai",
		)
	} else {
		ProviderExchange::with_rate_limit_headers(
			request_body,
			serde_json::from_str(&response_text)?,
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
