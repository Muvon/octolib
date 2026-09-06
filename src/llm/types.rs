// Copyright 2026 Muvon Un Limited
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_ttl: Option<String>, // Cache TTL override (e.g. "1h") — only Anthropic supports this
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>, // For tool messages: the ID of the tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>, // For tool messages: the name of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>, // For assistant messages: original tool calls from API response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ImageAttachment>>, // For messages with image attachments
    #[serde(skip_serializing_if = "Option::is_none")]
    pub videos: Option<Vec<VideoAttachment>>, // For messages with video attachments
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
            cache_ttl: None,
            tool_call_id: None,
            name: None,
            tool_calls: None,
            images: None,
            videos: None,
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
            cache_ttl: None,
            tool_call_id: None,
            name: None,
            tool_calls: None,
            images: None,
            videos: None,
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
            cache_ttl: None,
            tool_call_id: None,
            name: None,
            tool_calls: None,
            images: None,
            videos: None,
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
            cache_ttl: None,
            tool_call_id: Some(tool_call_id.to_string()),
            name: Some(name.to_string()),
            tool_calls: None,
            images: None,
            videos: None,
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

    /// Add video attachment to message
    pub fn with_videos(mut self, videos: Vec<VideoAttachment>) -> Self {
        self.videos = Some(videos);
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
    cache_ttl: Option<String>,
    tool_call_id: Option<String>,
    name: Option<String>,
    tool_calls: Option<serde_json::Value>,
    images: Option<Vec<ImageAttachment>>,
    videos: Option<Vec<VideoAttachment>>,
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

    /// Set cache TTL (e.g. "1h" for long-lived, only Anthropic supports this)
    pub fn cache_ttl<S: Into<String>>(mut self, ttl: S) -> Self {
        self.cache_ttl = Some(ttl.into());
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

    /// Add videos
    pub fn with_videos(mut self, videos: Vec<VideoAttachment>) -> Self {
        self.videos = Some(videos);
        self
    }

    /// Add a single video
    pub fn with_video(mut self, video: VideoAttachment) -> Self {
        match self.videos {
            Some(ref mut videos) => videos.push(video),
            None => self.videos = Some(vec![video]),
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
            cache_ttl: self.cache_ttl,
            tool_call_id: self.tool_call_id,
            name: self.name,
            tool_calls: self.tool_calls,
            images: self.images,
            videos: self.videos,
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

/// Video attachment for messages
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VideoAttachment {
    pub data: VideoData,
    pub media_type: String,
    pub source_type: SourceType,
    pub dimensions: Option<(u32, u32)>,
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
}

/// Video data storage format
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum VideoData {
    Base64(String),
    Url(String),
}

/// Source of the image or video
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
///
/// # Token Categories
/// - `input_tokens`: CLEAN input tokens (never includes cache tokens) - user messages, system prompts, tool definitions, tool responses
/// - `cache_read_tokens`: Tokens read from cache (cheaper rate, already cached from previous request)
/// - `cache_write_tokens`: Tokens written to cache (premium rate, happens once per cache entry)
/// - `output_tokens`: AI-generated response tokens (completion)
/// - `reasoning_tokens`: Tokens used for thinking/reasoning (separate from output, DeepSeek R1, Claude thinking, etc.)
///
/// # Total Calculation
/// `total_tokens` equals: input_tokens + cache_read_tokens + cache_write_tokens
/// + output_tokens + reasoning_tokens
///
/// Reasoning is its own term because `output_tokens` excludes it — see
/// [`TokenUsage::split_output`]. Providers must split before constructing, or
/// every consumer that sums the parts bills thinking twice.
///
/// # Provider-Specific Notes
/// - **Anthropic**: Reports cache_read and cache_creation (write) separately; input_tokens includes everything
/// - **OpenAI**: GPT-5.6 reports cache reads and writes in input token details;
///   input_tokens includes regular + cache_read + cache_write
/// - **DeepSeek**: Reports cache_hit (read) and cache_miss; input_tokens includes everything
/// - **OpenRouter**: NO cache info provided
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TokenUsage {
    /// CLEAN input tokens - NEVER includes any cache tokens (read or write)
    /// These are "fresh" tokens being sent to the API for the first time
    pub input_tokens: u64,

    /// Tokens read from cache (cache hit) - cheaper pricing tier
    /// These were previously cached and are being retrieved
    pub cache_read_tokens: u64,

    /// Tokens written to cache (cache creation/write) - premium pricing tier
    /// These are being stored in cache for future use
    pub cache_write_tokens: u64,

    /// AI-generated response tokens (completion/output)
    pub output_tokens: u64,

    /// Tokens used for thinking/reasoning (DeepSeek R1, Claude thinking, etc.)
    /// These are separate from output_tokens — construct via
    /// [`TokenUsage::split_output`], never straight from the API counter.
    pub reasoning_tokens: u64,

    /// Total tokens as reported by provider (should equal input + cache_read + cache_write + output + reasoning)
    pub total_tokens: u64,

    /// Pre-calculated total cost in USD (provider handles cache pricing)
    #[serde(default)]
    pub cost: Option<f64>,

    /// Time spent on this API request in milliseconds
    #[serde(default)]
    pub request_time_ms: Option<u64>,
}

impl TokenUsage {
    /// Split a provider's completion counter into visible output and reasoning.
    ///
    /// Every OpenAI-shaped API counts thinking INSIDE its completion counter and
    /// reports the reasoning slice as a detail of that same number
    /// (`completion_tokens_details.reasoning_tokens`, `output_tokens_details`);
    /// Anthropic-shaped APIs do the same with `output_tokens`. Providers that
    /// expose no reasoning field at all (z.ai, Together, MiniMax) estimate it
    /// from the emitted thinking text, which the provider also billed inside the
    /// completion counter. Either way the two overlap, while `reasoning_tokens`
    /// is documented as separate, so a consumer summing the parts pays for
    /// thinking twice. Subtract once here instead of in every provider.
    ///
    /// Reasoning is clamped to the completion counter: an estimate can exceed
    /// the tokens it was cut from, and reporting more reasoning than the
    /// provider generated would recreate the overcount this exists to remove.
    ///
    /// Returns `(visible_output, reasoning)`.
    pub fn split_output(completion_tokens: u64, reasoning_tokens: u64) -> (u64, u64) {
        let reasoning = reasoning_tokens.min(completion_tokens);
        (completion_tokens - reasoning, reasoning)
    }

    /// The output counter a provider bills — the inverse of [`Self::split_output`].
    ///
    /// Thinking is charged at the output rate, so pricing must run on the
    /// completion counter the API reported, not on the visible remainder left
    /// after the split. Pass this to any cost calculation instead of
    /// `output_tokens`; a reasoning-heavy call is otherwise billed at a
    /// fraction of its real cost.
    pub fn billable_output_tokens(&self) -> u64 {
        self.output_tokens + self.reasoning_tokens
    }
}

#[cfg(test)]
#[path = "types_token_usage_split_tests.rs"]
mod token_usage_split_tests;

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

/// Provider-agnostic tool selection policy.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Require at least one tool call.
    Required,
    /// Prevent tool calls.
    None,
    /// Require a specific named function.
    Function(String),
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
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ResponseMode {
    /// Automatic mode (provider decides)
    Auto,
    /// Strict schema adherence
    Strict,
}

/// Pricing information for a model
/// All prices are per 1M tokens in USD
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Regular input price per 1M tokens (USD) - for uncached input
    pub input_price_per_1m: f64,
    /// Output price per 1M tokens (USD)
    pub output_price_per_1m: f64,
    /// Cache write price per 1M tokens (USD) - cost to write to cache
    /// For providers without cache write differentiation, same as input_price_per_1m
    pub cache_write_price_per_1m: f64,
    /// Cache read price per 1M tokens (USD) - cost to read from cache
    /// For providers without caching, same as input_price_per_1m
    pub cache_read_price_per_1m: f64,
}

impl ModelPricing {
    /// Create new pricing with explicit cache prices
    pub fn new(
        input_price_per_1m: f64,
        output_price_per_1m: f64,
        cache_write_price_per_1m: f64,
        cache_read_price_per_1m: f64,
    ) -> Self {
        Self {
            input_price_per_1m,
            output_price_per_1m,
            cache_write_price_per_1m,
            cache_read_price_per_1m,
        }
    }

    /// Create pricing without cache support (all cache prices = input price)
    pub fn without_cache(input_price_per_1m: f64, output_price_per_1m: f64) -> Self {
        Self {
            input_price_per_1m,
            output_price_per_1m,
            cache_write_price_per_1m: input_price_per_1m,
            cache_read_price_per_1m: input_price_per_1m,
        }
    }

    /// Calculate cost for given token counts
    /// Returns cost in USD
    pub fn calculate_cost(
        &self,
        regular_input_tokens: u64,
        cache_write_tokens: u64,
        cache_read_tokens: u64,
        output_tokens: u64,
    ) -> f64 {
        let regular_input_cost =
            (regular_input_tokens as f64 / 1_000_000.0) * self.input_price_per_1m;
        let cache_write_cost =
            (cache_write_tokens as f64 / 1_000_000.0) * self.cache_write_price_per_1m;
        let cache_read_cost =
            (cache_read_tokens as f64 / 1_000_000.0) * self.cache_read_price_per_1m;
        let output_cost = (output_tokens as f64 / 1_000_000.0) * self.output_price_per_1m;

        regular_input_cost + cache_write_cost + cache_read_cost + output_cost
    }
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
/// Declares which sampling parameters a model supports.
///
/// Providers return this from `supported_sampling_params()` to declare what a model accepts.
/// Each field is a simple boolean — `true` means the parameter is supported, `false` means
/// it must be omitted from API requests.
///
/// This is intentionally separate from the values themselves: support is a property of the
/// model, while values come from user configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingSupport {
    /// Whether the model accepts the temperature parameter.
    pub temperature: bool,
    /// Whether the model accepts the top_p parameter.
    pub top_p: bool,
    /// Whether the model accepts the top_k parameter.
    pub top_k: bool,
}

impl SamplingSupport {
    /// All sampling parameters supported.
    pub const ALL: Self = Self {
        temperature: true,
        top_p: true,
        top_k: true,
    };

    /// No sampling parameters supported (e.g., reasoning models).
    pub const NONE: Self = Self {
        temperature: false,
        top_p: false,
        top_k: false,
    };

    /// Only temperature supported (no top_p, no top_k).
    pub const TEMPERATURE_ONLY: Self = Self {
        temperature: true,
        top_p: false,
        top_k: false,
    };

    /// Temperature and top_p supported, but not top_k (common for OpenAI-compatible APIs).
    pub const TEMPERATURE_AND_TOP_P: Self = Self {
        temperature: true,
        top_p: true,
        top_k: false,
    };

    /// Merge user-requested values with this support mask.
    ///
    /// Returns `EffectiveSamplingParams` where supported parameters carry the user's value
    /// and unsupported parameters are `None` (to be omitted from API requests).
    pub fn effective(self, temperature: f32, top_p: f32, top_k: u32) -> EffectiveSamplingParams {
        EffectiveSamplingParams {
            temperature: self.temperature.then_some(temperature),
            top_p: self.top_p.then_some(top_p),
            top_k: self.top_k.then_some(top_k),
        }
    }
}

impl Default for SamplingSupport {
    /// Default: all parameters supported.
    fn default() -> Self {
        Self::ALL
    }
}

/// The result of merging user-requested sampling values with model support.
///
/// Each field is `Option` — `Some(value)` means the parameter should be sent with that value,
/// `None` means it must be omitted from the API request entirely.
///
/// Construct this via `SamplingSupport::effective()` or `AiProvider::effective_sampling_params()`.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectiveSamplingParams {
    /// Temperature to send, or None to omit.
    pub temperature: Option<f32>,
    /// Top-p to send, or None to omit.
    pub top_p: Option<f32>,
    /// Top-k to send, or None to omit.
    pub top_k: Option<u32>,
}

/// Generic reasoning effort level for thinking-capable models.
///
/// This enum is intentionally provider-agnostic. Each provider owns its own
/// mapping from these levels to its native knob (effort string, budget tokens,
/// thinking flag, etc.) inside its `chat_completion()` implementation.
///
/// To leave thinking at provider default, keep `ChatCompletionParams::reasoning_effort` as `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

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
    /// Per-request HTTP timeout. `None` = no timeout (LLM may take minutes);
    /// `Some(d)` = abort the HTTP request if it exceeds `d`.
    pub request_timeout: Option<std::time::Duration>,
    /// Cancellation token for request abortion
    pub cancellation_token: Option<tokio::sync::watch::Receiver<bool>>,
    /// Available tools for function calling
    pub tools: Option<Vec<FunctionDefinition>>,
    /// Structured output configuration
    pub response_format: Option<StructuredOutputRequest>,
    /// Previous response ID for multi-turn conversations (OpenAI Responses API)
    pub previous_id: Option<String>,
    /// Enable long-lived cache (provider-specific: pre-GPT-5.6 OpenAI "24h"
    /// retention, Anthropic 1h TTL). GPT-5.6 has only a fixed 30m minimum TTL.
    pub use_long_cache: bool,
    /// Explicit prompt-cache routing key (OpenAI `prompt_cache_key`). Optional hint
    /// that pins requests sharing a long common prefix to the same cache, improving
    /// hit rates. `None` = rely on automatic prefix-hash routing. OpenAI-only.
    pub prompt_cache_key: Option<String>,
    /// Reasoning effort hint for thinking-capable models. `None` = provider default
    /// (most providers omit the field; hybrid models stay non-thinking).
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Extra HTTP headers for the upstream request, applied LAST with upsert
    /// semantics: a passed header replaces the provider's value for that name,
    /// everything else the provider set stays. The library attaches no meaning
    /// to them — callers use this to talk to proxies (e.g. octohub's
    /// `X-Model-Purpose`). `None`/empty = no change to the request.
    pub extra_headers: Option<std::collections::HashMap<String, String>>,
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
            request_timeout: None,                            // No per-request timeout by default
            cancellation_token: None,
            tools: None,
            response_format: None,
            previous_id: None,
            use_long_cache: false,
            prompt_cache_key: None,
            reasoning_effort: None,
            extra_headers: None,
        }
    }

    /// Enable long-lived cache for this request
    pub fn with_long_cache(mut self, enabled: bool) -> Self {
        self.use_long_cache = enabled;
        self
    }

    /// Set an explicit prompt-cache routing key (OpenAI `prompt_cache_key`).
    pub fn with_prompt_cache_key(mut self, key: impl Into<String>) -> Self {
        self.prompt_cache_key = Some(key.into());
        self
    }

    /// Set extra HTTP headers for the upstream request (upsert semantics —
    /// see [`ChatCompletionParams::extra_headers`]).
    pub fn with_extra_headers(
        mut self,
        headers: std::collections::HashMap<String, String>,
    ) -> Self {
        self.extra_headers = Some(headers);
        self
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

    /// Set per-request HTTP timeout. `None` disables (LLM may take minutes);
    /// `Some(d)` aborts the HTTP request if it exceeds `d`.
    pub fn with_request_timeout(mut self, timeout: Option<std::time::Duration>) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set cancellation token
    pub fn with_cancellation_token(mut self, token: tokio::sync::watch::Receiver<bool>) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Set available tools
    /// Set the tool definitions for this request.
    ///
    /// Parameter schemas are normalized to scalar `"type"` keywords first — see
    /// [`crate::llm::utils::normalize_tool_schema`] for why every provider
    /// needs this.
    pub fn with_tools(mut self, mut tools: Vec<FunctionDefinition>) -> Self {
        for tool in tools.iter_mut() {
            crate::llm::utils::normalize_tool_schema(&mut tool.parameters);
        }
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

    /// Set reasoning effort. Has no effect on models without thinking support.
    pub fn with_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning_effort = Some(effort);
        self
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
