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
//! - **Multi-provider support**: OpenAI, Anthropic, OpenRouter, Google Vertex AI, Amazon Bedrock, Cloudflare Workers AI, DeepSeek
//! - **Unified interface**: Single trait for all providers with consistent API
//! - **Model validation**: Strict `provider:model` format validation
//! - **Cost tracking**: Automatic token usage and cost calculation
//! - **Vision support**: Image attachment support for compatible models
//! - **Caching support**: Automatic detection of caching-capable models
//! - **Retry logic**: Exponential backoff with smart rate limit handling
//! - **Self-sufficient**: No external dependencies on application-specific types
//!
//! ## Usage
//!
//! ```rust,no_run
//! use octolib::{ProviderFactory, ChatCompletionParams, Message};
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

pub mod config;
pub mod errors;
pub mod factory;
pub mod providers;
pub mod retry;
pub mod tool_calls;
pub mod traits;
pub mod types;

pub mod strategies;

// Re-export main types and traits for easy access
pub use config::{CacheConfig, CacheTTL, CacheType};
pub use errors::{
    ConfigError, ConfigResult, MessageError, MessageResult, ProviderError, ProviderResult,
    ToolCallError, ToolCallResult,
};
pub use factory::ProviderFactory;
pub use strategies::{ModelLimits, ProviderStrategy, StrategyFactory, ToolResult};
pub use tool_calls::ProviderToolCalls;
pub use traits::AiProvider;
pub use types::{
    ChatCompletionParams, FunctionDefinition, ImageAttachment, ImageData, Message, MessageBuilder,
    ProviderExchange, ProviderResponse, SourceType, TokenUsage, ToolCall,
};

// Re-export all provider implementations
pub use providers::{
    AmazonBedrockProvider, AnthropicProvider, CloudflareWorkersAiProvider, DeepSeekProvider,
    GoogleVertexProvider, OpenAiProvider, OpenRouterProvider,
};
