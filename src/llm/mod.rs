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

//! Large Language Model (LLM) functionality
//!
//! This module contains all LLM-related functionality including providers,
//! chat completion, tool calls, and configuration.

pub mod config;
pub mod factory;
pub mod providers;
pub mod retry;
pub mod strategies;
pub mod tool_calls;
pub mod traits;
pub mod types;
pub mod utils;

// Re-export main types and traits for easy access
pub use config::{CacheConfig, CacheTTL, CacheType};
pub use factory::ProviderFactory;
pub use strategies::{ModelLimits, ProviderStrategy, StrategyFactory, ToolResult};
pub use tool_calls::{GenericToolCall, ProviderToolCalls};
pub use traits::AiProvider;
pub use types::{
    ChatCompletionParams, FunctionDefinition, ImageAttachment, ImageData, Message, MessageBuilder,
    OutputFormat, ProviderExchange, ProviderResponse, ResponseMode, SourceType,
    StructuredOutputRequest, ThinkingBlock, TokenUsage, ToolCall,
};

// Re-export all provider implementations
pub use providers::{
    AmazonBedrockProvider, AnthropicProvider, CloudflareWorkersAiProvider, DeepSeekProvider,
    GoogleVertexProvider, MinimaxProvider, OpenAiProvider, OpenRouterProvider, ZaiProvider,
};
