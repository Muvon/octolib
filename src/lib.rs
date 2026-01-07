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

//! # Octolib - Self-sufficient AI Provider Library
//!
//! A comprehensive library for interacting with multiple AI providers through a unified interface.
//!
//! ## Features
//!
//! - **Multi-provider support**: OpenAI, Anthropic, OpenRouter, Google Vertex AI, Amazon Bedrock, Cloudflare Workers AI, DeepSeek, Z.ai
//! - **Unified interface**: Single trait for all providers with consistent API
//! - **Model validation**: Strict `provider:model` format validation
//! - **Structured output**: JSON and JSON Schema support for OpenAI, OpenRouter, DeepSeek, and Z.ai
//! - **Cost tracking**: Automatic token usage and cost calculation
//! - **Vision support**: Image attachment support for compatible models
//! - **Caching support**: Automatic detection of caching-capable models
//! - **Retry logic**: Exponential backoff with smart rate limit handling
//! - **Self-sufficient**: No external dependencies on application-specific types
//!
//! ## Usage
//!
//! ### Basic Chat Completion
//!
//! ```rust,no_run
//! use octolib::llm::{ProviderFactory, ChatCompletionParams, Message};
//!
//! // This example shows basic usage but requires API keys to run
//! async fn example() -> anyhow::Result<()> {
//!     // Parse model and get provider
//!     let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;
//!
//!     // Create messages
//!     let messages = vec![
//!         Message::user("Hello, how are you?"),
//!     ];
//!
//!     // Create completion parameters
//!     let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);
//!
//!     // Get completion (requires OPENAI_API_KEY environment variable)
//!     let response = provider.chat_completion(params).await?;
//!     println!("Response: {}", response.content);
//!
//!     Ok(())
//! }
//! ```
//!
//! ### Structured Output
//!
//! ```rust,no_run
//! use octolib::llm::{ProviderFactory, ChatCompletionParams, Message, StructuredOutputRequest};
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Debug)]
//! struct PersonInfo {
//!     name: String,
//!     age: u32,
//!     skills: Vec<String>,
//! }
//!
//! async fn structured_example() -> anyhow::Result<()> {
//!     // Works with OpenAI, OpenRouter, and DeepSeek
//!     let (provider, model) = ProviderFactory::get_provider_for_model("deepseek:deepseek-chat")?;
//!
//!     // Check if provider supports structured output
//!     if !provider.supports_structured_output(&model) {
//!         return Err(anyhow::anyhow!("Provider does not support structured output"));
//!     }
//!
//!     let messages = vec![
//!         Message::user("Tell me about a software engineer in JSON format"),
//!     ];
//!
//!     // Request structured JSON output
//!     let structured_request = StructuredOutputRequest::json();
//!     let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
//!         .with_structured_output(structured_request);
//!
//!     let response = provider.chat_completion(params).await?;
//!
//!     if let Some(structured) = response.structured_output {
//!         let person: PersonInfo = serde_json::from_value(structured)?;
//!         println!("Person: {:?}", person);
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod embedding;
pub mod errors;
pub mod llm;
pub mod storage;

// Re-export main types and traits for easy access (backward compatibility)
pub use embedding::{
    count_tokens, create_embedding_provider_from_parts, generate_embeddings,
    generate_embeddings_batch, split_texts_into_token_limited_batches, truncate_output,
    EmbeddingProvider, EmbeddingProviderType, InputType,
};
pub use errors::{
    ConfigError, ConfigResult, MessageError, MessageResult, ProviderError, ProviderResult,
    StructuredOutputError, StructuredOutputResult, ToolCallError, ToolCallResult,
};
pub use llm::{
    AiProvider, AmazonBedrockProvider, AnthropicProvider, CacheConfig, CacheTTL, CacheType,
    ChatCompletionParams, CloudflareWorkersAiProvider, DeepSeekProvider, FunctionDefinition,
    GenericToolCall, GoogleVertexProvider, ImageAttachment, ImageData, Message, MessageBuilder,
    MinimaxProvider, ModelLimits, OpenAiProvider, OpenRouterProvider, OutputFormat,
    ProviderExchange, ProviderFactory, ProviderResponse, ProviderStrategy, ProviderToolCalls,
    ResponseMode, SourceType, StrategyFactory, StructuredOutputRequest, ThinkingBlock, TokenUsage,
    ToolCall, ToolResult, ZaiProvider,
};
