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
///
/// Messages can contain:
/// - **content**: What was said (text response)
/// - **thinking**: Internal reasoning (separate from content, like tool_calls)
/// - **tool_calls**: Function invocations (separate from content)
/// - **id**: Provider's response ID (for assistant messages, used for conversation continuation)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingBlock>, // Internal reasoning (separate from content)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>, // Provider's response ID (for assistant messages with tool calls)
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
            thinking: None,
            id: None,
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
            thinking: None,
            id: None,
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
            thinking: None,
            id: None,
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
            thinking: None,
            id: None,
        }
    }

    /// Add thinking block to message (for assistant responses with reasoning)
    pub fn with_thinking(mut self, thinking: ThinkingBlock) -> Self {
        self.thinking = Some(thinking);
        self
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

    /// Create a new message builder
    pub fn builder() -> MessageBuilder {
        MessageBuilder::new()
    }
}

/// Builder pattern for creating messages with validation
#[derive(Debug, Default)]
pub struct MessageBuilder {
    role: Option<String>,
    content: Option<String>,
    timestamp: Option<u64>,
    cached: bool,
    tool_call_id: Option<String>,
    name: Option<String>,
    tool_calls: Option<serde_json::Value>,
    images: Option<Vec<ImageAttachment>>,
    thinking: Option<ThinkingBlock>,
    id: Option<String>, // Provider's response ID (for assistant messages)
}

impl MessageBuilder {
    /// Create a new message builder
    pub fn new() -> Self {
        Self {
            timestamp: Some(current_timestamp()),
            ..Default::default()
        }
    }

    /// Set the role
    pub fn role<S: Into<String>>(mut self, role: S) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Set the content
    pub fn content<S: Into<String>>(mut self, content: S) -> Self {
        self.content = Some(content.into());
        self
    }

    /// Set the timestamp
    pub fn timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Mark as cached
    pub fn cached(mut self) -> Self {
        self.cached = true;
        self
    }

    /// Set tool call ID (for tool messages)
    pub fn tool_call_id<S: Into<String>>(mut self, id: S) -> Self {
        self.tool_call_id = Some(id.into());
        self
    }

    /// Set name (for tool messages)
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set tool calls (for assistant messages) using unified GenericToolCall format
    pub fn with_tool_calls(
        mut self,
        tool_calls: Vec<crate::llm::tool_calls::GenericToolCall>,
    ) -> Self {
        // Convert to JSON for storage - providers will convert back to their specific formats
        let tool_calls_json = serde_json::to_value(&tool_calls).unwrap_or_default();
        self.tool_calls = Some(tool_calls_json);
        self
    }

    /// Add images
    pub fn with_images(mut self, images: Vec<ImageAttachment>) -> Self {
        self.images = Some(images);
        self
    }

    /// Add a single image
    pub fn with_image(mut self, image: ImageAttachment) -> Self {
        match self.images {
            Some(ref mut images) => images.push(image),
            None => self.images = Some(vec![image]),
        }
        self
    }

    /// Set thinking block (for assistant messages with reasoning)
    pub fn thinking(mut self, thinking: ThinkingBlock) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set message ID (for assistant messages with tool calls)
    pub fn id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Build the message with validation
    pub fn build(self) -> Result<Message, crate::errors::MessageError> {
        let role = self
            .role
            .ok_or(crate::errors::MessageError::MissingToolField {
                field: "role".to_string(),
            })?;

        let content = self
            .content
            .ok_or(crate::errors::MessageError::MissingContent)?;

        // Validate role
        match role.as_str() {
            "user" | "assistant" | "system" | "tool" => {}
            _ => return Err(crate::errors::MessageError::InvalidRole { role }),
        }

        // Validate tool messages have required fields
        if role == "tool" {
            if self.tool_call_id.is_none() {
                return Err(crate::errors::MessageError::MissingToolField {
                    field: "tool_call_id".to_string(),
                });
            }
            if self.name.is_none() {
                return Err(crate::errors::MessageError::MissingToolField {
                    field: "name".to_string(),
                });
            }
        }

        Ok(Message {
            role,
            content,
            timestamp: self.timestamp.unwrap_or_else(current_timestamp),
            cached: self.cached,
            tool_call_id: self.tool_call_id,
            name: self.name,
            tool_calls: self.tool_calls,
            images: self.images,
            thinking: self.thinking,
            id: self.id,
        })
    }

    /// Convenience method to build a user message
    pub fn user<S: Into<String>>(content: S) -> Self {
        Self::new().role("user").content(content)
    }

    /// Convenience method to build an assistant message
    pub fn assistant<S: Into<String>>(content: S) -> Self {
        Self::new().role("assistant").content(content)
    }

    /// Convenience method to build a system message
    pub fn system<S: Into<String>>(content: S) -> Self {
        Self::new().role("system").content(content)
    }

    /// Convenience method to build a tool message
    pub fn tool<S: Into<String>>(content: S, tool_call_id: S, name: S) -> Self {
        Self::new()
            .role("tool")
            .content(content)
            .tool_call_id(tool_call_id)
            .name(name)
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

/// Thinking/reasoning block from models that support extended reasoning
///
/// Thinking is stored separately from content, similar to how tool_calls are separate.
/// This allows for clean semantic separation between what the model said (content)
/// and how it reasoned (thinking).
///
/// **Example usage:**
/// ```rust
/// use octolib::ThinkingBlock;
///
/// let thinking = ThinkingBlock::new("First, I need to solve for x...");
/// ```
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThinkingBlock {
    /// The thinking/reasoning text content
    pub content: String,
    /// Token count for cost tracking (may not be available from all providers)
    #[serde(default)]
    pub tokens: u64,
}

impl ThinkingBlock {
    /// Create a new thinking block with the given content
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            tokens: 0,
        }
    }

    /// Create a thinking block with token count
    pub fn with_tokens(content: &str, tokens: u64) -> Self {
        Self {
            content: content.to_string(),
            tokens,
        }
    }
}

/// Common token usage structure across all providers
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u64, // ALL input tokens (user messages, system prompts, tool definitions, tool responses)
    pub output_tokens: u64, // AI-generated response tokens only (excludes thinking tokens)
    pub reasoning_tokens: u64, // Tokens used for thinking/reasoning (separate from output)
    pub total_tokens: u64,  // prompt_tokens + output_tokens + reasoning_tokens
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
    pub rate_limit_headers: Option<std::collections::HashMap<String, String>>, // Rate limit headers from API response
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
            rate_limit_headers: None,
        }
    }

    /// Create a new ProviderExchange with rate limit headers
    pub fn with_rate_limit_headers(
        request: serde_json::Value,
        response: serde_json::Value,
        usage: Option<TokenUsage>,
        provider: &str,
        rate_limit_headers: std::collections::HashMap<String, String>,
    ) -> Self {
        Self {
            request,
            response,
            timestamp: current_timestamp(),
            usage,
            provider: provider.to_string(),
            rate_limit_headers: Some(rate_limit_headers),
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

/// Output format for structured responses
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum OutputFormat {
    /// Standard JSON output
    Json,
    /// JSON with schema validation
    JsonSchema,
}

/// Response mode for structured output
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ResponseMode {
    /// Automatic mode (provider decides)
    Auto,
    /// Strict schema adherence
    Strict,
}

/// Structured output request configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StructuredOutputRequest {
    /// Output format type
    pub format: OutputFormat,
    /// Response mode
    pub mode: ResponseMode,
    /// JSON schema for validation (when using JsonSchema format)
    pub schema: Option<serde_json::Value>,
}

impl StructuredOutputRequest {
    /// Create a new structured output request with JSON format
    pub fn json() -> Self {
        Self {
            format: OutputFormat::Json,
            mode: ResponseMode::Auto,
            schema: None,
        }
    }

    /// Create a new structured output request with JSON schema
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self {
            format: OutputFormat::JsonSchema,
            mode: ResponseMode::Auto,
            schema: Some(schema),
        }
    }

    /// Set response mode to strict
    pub fn with_strict_mode(mut self) -> Self {
        self.mode = ResponseMode::Strict;
        self
    }
}

/// Provider response containing the AI completion
///
/// Response contains:
/// - **content**: The final text response
/// - **thinking**: Internal reasoning (if available from provider, separate from content)
/// - **tool_calls**: Any function calls made
#[derive(Debug, Clone)]
pub struct ProviderResponse {
    pub content: String,
    /// Thinking/reasoning content extracted from provider response
    /// This is separate from content, similar to how tool_calls are separate
    pub thinking: Option<ThinkingBlock>,
    pub exchange: ProviderExchange,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub finish_reason: Option<String>,
    /// Parsed structured output (if requested)
    pub structured_output: Option<serde_json::Value>,
    /// Response ID from provider (required for multi-turn conversations with OpenAI Responses API)
    pub id: Option<String>,
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
    /// Structured output configuration
    pub response_format: Option<StructuredOutputRequest>,
    /// Previous response ID for multi-turn conversations (OpenAI Responses API)
    pub previous_id: Option<String>,
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
            response_format: None,
            previous_id: None,
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

    /// Set structured output format
    pub fn with_structured_output(mut self, response_format: StructuredOutputRequest) -> Self {
        self.response_format = Some(response_format);
        self
    }

    /// Set previous response ID for multi-turn conversations (OpenAI Responses API)
    pub fn with_previous_id(mut self, id: &str) -> Self {
        self.previous_id = Some(id.to_string());
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
            reasoning_tokens: 30,
            total_tokens: 180,
            cached_tokens: 20,
            cost: Some(0.01),
            request_time_ms: Some(1500),
        };

        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.reasoning_tokens, 30);
        assert_eq!(usage.total_tokens, 180);
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
            reasoning_tokens: 3,
            total_tokens: 18,
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

    #[test]
    fn test_thinking_block() {
        let thinking = ThinkingBlock::new("Let me solve this step by step...");
        assert_eq!(thinking.content, "Let me solve this step by step...");
        assert_eq!(thinking.tokens, 0);

        let thinking_with_tokens = ThinkingBlock::with_tokens("Reasoning...", 42);
        assert_eq!(thinking_with_tokens.content, "Reasoning...");
        assert_eq!(thinking_with_tokens.tokens, 42);
    }

    #[test]
    fn test_message_with_thinking() {
        let thinking = ThinkingBlock::with_tokens("Let me solve this step by step...", 50);
        let msg = Message::assistant("The answer is 42.").with_thinking(thinking);

        assert!(msg.thinking.is_some());
        assert_eq!(
            msg.thinking.as_ref().unwrap().content,
            "Let me solve this step by step..."
        );
        assert_eq!(msg.thinking.as_ref().unwrap().tokens, 50);
        assert_eq!(msg.content, "The answer is 42.");
    }

    #[test]
    fn test_message_builder_with_thinking() {
        let thinking = ThinkingBlock::new("First, I'll analyze the problem...");
        let msg = Message::builder()
            .role("assistant")
            .content("The answer is 42.")
            .thinking(thinking)
            .build()
            .unwrap();

        assert!(msg.thinking.is_some());
        assert_eq!(
            msg.thinking.unwrap().content,
            "First, I'll analyze the problem..."
        );
    }
}
