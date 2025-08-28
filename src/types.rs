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

//! Core types for the AI provider library

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Message in a conversation
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
	pub role: String,
	pub content: String,
	pub timestamp: u64,
	#[serde(default = "default_cache_marker")]
	pub cached: bool, // Marks if this message is a cache breakpoint
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_call_id: Option<String>, // For tool messages: the ID of the tool call
	#[serde(skip_serializing_if = "Option::is_none")]
	pub name: Option<String>, // For tool messages: the name of the tool
	#[serde(skip_serializing_if = "Option::is_none")]
	pub tool_calls: Option<serde_json::Value>, // For assistant messages: original tool calls from API response
	#[serde(skip_serializing_if = "Option::is_none")]
	pub images: Option<Vec<ImageAttachment>>, // For messages with image attachments
}

fn default_cache_marker() -> bool {
	false
}

impl Message {
	/// Create a new user message
	pub fn user(content: &str) -> Self {
		Self {
			role: "user".to_string(),
			content: content.to_string(),
			timestamp: current_timestamp(),
			cached: false,
			tool_call_id: None,
			name: None,
			tool_calls: None,
			images: None,
		}
	}

	/// Create a new assistant message
	pub fn assistant(content: &str) -> Self {
		Self {
			role: "assistant".to_string(),
			content: content.to_string(),
			timestamp: current_timestamp(),
			cached: false,
			tool_call_id: None,
			name: None,
			tool_calls: None,
			images: None,
		}
	}

	/// Create a new system message
	pub fn system(content: &str) -> Self {
		Self {
			role: "system".to_string(),
			content: content.to_string(),
			timestamp: current_timestamp(),
			cached: false,
			tool_call_id: None,
			name: None,
			tool_calls: None,
			images: None,
		}
	}

	/// Create a new tool message
	pub fn tool(content: &str, tool_call_id: &str, name: &str) -> Self {
		Self {
			role: "tool".to_string(),
			content: content.to_string(),
			timestamp: current_timestamp(),
			cached: false,
			tool_call_id: Some(tool_call_id.to_string()),
			name: Some(name.to_string()),
			tool_calls: None,
			images: None,
		}
	}

	/// Add image attachment to message
	pub fn with_images(mut self, images: Vec<ImageAttachment>) -> Self {
		self.images = Some(images);
		self
	}

	/// Mark message as cached
	pub fn with_cache_marker(mut self) -> Self {
		self.cached = true;
		self
	}
}

fn current_timestamp() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs()
}

/// Image attachment for messages
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageAttachment {
	pub data: ImageData,
	pub media_type: String,
	pub source_type: SourceType,
	pub dimensions: Option<(u32, u32)>,
	pub size_bytes: Option<u64>,
}

/// Image data storage format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ImageData {
	Base64(String),
	Url(String),
}

/// Source of the image
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum SourceType {
	File(PathBuf),
	Clipboard,
	Url,
}

/// Common token usage structure across all providers
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenUsage {
	pub prompt_tokens: u64, // ALL input tokens (user messages, system prompts, tool definitions, tool responses)
	pub output_tokens: u64, // AI-generated response tokens only
	pub total_tokens: u64,  // prompt_tokens + output_tokens
	pub cached_tokens: u64, // Subset of prompt_tokens that came from cache (discounted)
	#[serde(default)]
	pub cost: Option<f64>, // Pre-calculated total cost (provider handles cache pricing)
	// Time tracking
	#[serde(default)]
	pub request_time_ms: Option<u64>, // Time spent on this API request
}

/// Common exchange record for logging across all providers
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProviderExchange {
	pub request: serde_json::Value,
	pub response: serde_json::Value,
	pub timestamp: u64,
	pub usage: Option<TokenUsage>,
	pub provider: String, // Which provider was used
}

impl ProviderExchange {
	pub fn new(
		request: serde_json::Value,
		response: serde_json::Value,
		usage: Option<TokenUsage>,
		provider: &str,
	) -> Self {
		Self {
			request,
			response,
			timestamp: current_timestamp(),
			usage,
			provider: provider.to_string(),
		}
	}
}

/// Generic tool call structure (independent of MCP)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
	pub id: String,
	pub name: String,
	pub arguments: serde_json::Value,
}

/// Function definition for tool calling
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefinition {
	pub name: String,
	pub description: String,
	pub parameters: serde_json::Value,
	/// Cache control marker for Anthropic (optional)
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cache_control: Option<serde_json::Value>,
}

/// Provider response containing the AI completion
#[derive(Debug, Clone)]
pub struct ProviderResponse {
	pub content: String,
	pub exchange: ProviderExchange,
	pub tool_calls: Option<Vec<ToolCall>>,
	pub finish_reason: Option<String>,
}

/// Parameters for chat completion requests
///
/// This struct groups all parameters needed for AI provider chat completion calls,
/// following best practices for parameter passing and future extensibility.
#[derive(Clone)]
pub struct ChatCompletionParams {
	/// Array of conversation messages
	pub messages: Vec<Message>,
	/// Model identifier (e.g., "claude-3-5-sonnet", "gpt-4")
	pub model: String,
	/// Sampling temperature (0.0 to 2.0)
	pub temperature: f32,
	/// Top-p nucleus sampling (0.0 to 1.0)
	pub top_p: f32,
	/// Top-k sampling (1 to infinity)
	pub top_k: u32,
	/// Maximum tokens to generate (0 = no limit)
	pub max_tokens: u32,
	/// Maximum retry attempts on failure
	pub max_retries: u32,
	/// Base timeout for exponential backoff retry logic
	pub retry_timeout: std::time::Duration,
	/// Cancellation token for request abortion
	pub cancellation_token: Option<tokio::sync::watch::Receiver<bool>>,
	/// Available tools for function calling
	pub tools: Option<Vec<FunctionDefinition>>,
}

impl ChatCompletionParams {
	/// Create new chat completion parameters
	pub fn new(
		messages: &[Message],
		model: &str,
		temperature: f32,
		top_p: f32,
		top_k: u32,
		max_tokens: u32,
	) -> Self {
		Self {
			messages: messages.to_vec(),
			model: model.to_string(),
			temperature,
			top_p,
			top_k,
			max_tokens,
			max_retries: 3,                                   // Default retry attempts
			retry_timeout: std::time::Duration::from_secs(1), // Default 1 second base timeout
			cancellation_token: None,
			tools: None,
		}
	}

	/// Set maximum retry attempts
	pub fn with_max_retries(mut self, max_retries: u32) -> Self {
		self.max_retries = max_retries;
		self
	}

	/// Set retry timeout
	pub fn with_retry_timeout(mut self, timeout: std::time::Duration) -> Self {
		self.retry_timeout = timeout;
		self
	}

	/// Set cancellation token
	pub fn with_cancellation_token(mut self, token: tokio::sync::watch::Receiver<bool>) -> Self {
		self.cancellation_token = Some(token);
		self
	}

	/// Set available tools
	pub fn with_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
		self.tools = Some(tools);
		self
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_message_constructors() {
		let user_msg = Message::user("Hello");
		assert_eq!(user_msg.role, "user");
		assert_eq!(user_msg.content, "Hello");
		assert!(!user_msg.cached);
		assert!(user_msg.tool_call_id.is_none());
		assert!(user_msg.images.is_none());

		let assistant_msg = Message::assistant("Hi there");
		assert_eq!(assistant_msg.role, "assistant");
		assert_eq!(assistant_msg.content, "Hi there");

		let system_msg = Message::system("You are helpful");
		assert_eq!(system_msg.role, "system");
		assert_eq!(system_msg.content, "You are helpful");

		let tool_msg = Message::tool("Result", "call_123", "test_tool");
		assert_eq!(tool_msg.role, "tool");
		assert_eq!(tool_msg.content, "Result");
		assert_eq!(tool_msg.tool_call_id, Some("call_123".to_string()));
		assert_eq!(tool_msg.name, Some("test_tool".to_string()));
	}

	#[test]
	fn test_message_with_cache_marker() {
		let msg = Message::user("Test").with_cache_marker();
		assert!(msg.cached);
	}

	#[test]
	fn test_chat_completion_params() {
		let messages = vec![Message::user("Hello")];
		let params = ChatCompletionParams::new(&messages, "openai:gpt-4o", 0.7, 1.0, 50, 1000);

		assert_eq!(params.model, "openai:gpt-4o");
		assert_eq!(params.temperature, 0.7);
		assert_eq!(params.top_p, 1.0);
		assert_eq!(params.top_k, 50);
		assert_eq!(params.max_tokens, 1000);
		assert_eq!(params.max_retries, 3); // Default
		assert!(params.cancellation_token.is_none());
		assert!(params.tools.is_none()); // Default

		let params_with_retries = params.with_max_retries(5);
		assert_eq!(params_with_retries.max_retries, 5);

		// Test with tools
		let tools = vec![FunctionDefinition {
			name: "test_function".to_string(),
			description: "A test function".to_string(),
			parameters: serde_json::json!({"type": "object", "properties": {}}),
			cache_control: None,
		}];
		let params_with_tools = params_with_retries.with_tools(tools.clone());
		assert!(params_with_tools.tools.is_some());
		assert_eq!(params_with_tools.tools.unwrap().len(), 1);
	}

	#[test]
	fn test_token_usage() {
		let usage = TokenUsage {
			prompt_tokens: 100,
			output_tokens: 50,
			total_tokens: 150,
			cached_tokens: 20,
			cost: Some(0.01),
			request_time_ms: Some(1500),
		};

		assert_eq!(usage.prompt_tokens, 100);
		assert_eq!(usage.output_tokens, 50);
		assert_eq!(usage.total_tokens, 150);
		assert_eq!(usage.cached_tokens, 20);
		assert_eq!(usage.cost, Some(0.01));
		assert_eq!(usage.request_time_ms, Some(1500));
	}

	#[test]
	fn test_provider_exchange() {
		let request = serde_json::json!({"model": "test", "messages": []});
		let response = serde_json::json!({"choices": []});
		let usage = TokenUsage {
			prompt_tokens: 10,
			output_tokens: 5,
			total_tokens: 15,
			cached_tokens: 0,
			cost: None,
			request_time_ms: None,
		};

		let exchange = ProviderExchange::new(
			request.clone(),
			response.clone(),
			Some(usage.clone()),
			"test_provider",
		);

		assert_eq!(exchange.request, request);
		assert_eq!(exchange.response, response);
		assert_eq!(exchange.provider, "test_provider");
		assert!(exchange.usage.is_some());
		assert!(exchange.timestamp > 0);
	}

	#[test]
	fn test_tool_call() {
		let tool_call = ToolCall {
			id: "call_123".to_string(),
			name: "test_function".to_string(),
			arguments: serde_json::json!({"param": "value"}),
		};

		assert_eq!(tool_call.id, "call_123");
		assert_eq!(tool_call.name, "test_function");
		assert_eq!(tool_call.arguments["param"], "value");
	}

	#[test]
	fn test_function_definition() {
		let func_def = FunctionDefinition {
			name: "test_function".to_string(),
			description: "A test function for demonstration".to_string(),
			parameters: serde_json::json!({
				"type": "object",
				"properties": {
					"param1": {"type": "string", "description": "First parameter"}
				},
				"required": ["param1"]
			}),
			cache_control: None,
		};

		assert_eq!(func_def.name, "test_function");
		assert_eq!(func_def.description, "A test function for demonstration");
		assert_eq!(func_def.parameters["type"], "object");
		assert!(func_def.parameters["properties"]["param1"].is_object());
		assert!(func_def.cache_control.is_none());

		// Test with cache control
		let func_def_with_cache = FunctionDefinition {
			name: "cached_function".to_string(),
			description: "A cached function".to_string(),
			parameters: serde_json::json!({"type": "object"}),
			cache_control: Some(serde_json::json!({
				"type": "ephemeral",
				"ttl": "5m"
			})),
		};

		assert!(func_def_with_cache.cache_control.is_some());
		assert_eq!(
			func_def_with_cache.cache_control.unwrap()["type"],
			"ephemeral"
		);
	}

	#[test]
	fn test_image_attachment() {
		let attachment = ImageAttachment {
			data: ImageData::Base64("base64data".to_string()),
			media_type: "image/png".to_string(),
			source_type: SourceType::File(std::path::PathBuf::from("/path/to/image.png")),
			dimensions: Some((800, 600)),
			size_bytes: Some(1024),
		};

		match &attachment.data {
			ImageData::Base64(data) => assert_eq!(data, "base64data"),
			_ => panic!("Expected Base64 data"),
		}

		assert_eq!(attachment.media_type, "image/png");
		assert_eq!(attachment.dimensions, Some((800, 600)));
		assert_eq!(attachment.size_bytes, Some(1024));

		match &attachment.source_type {
			SourceType::File(path) => {
				assert_eq!(path, &std::path::PathBuf::from("/path/to/image.png"))
			}
			_ => panic!("Expected File source type"),
		}
	}
}
