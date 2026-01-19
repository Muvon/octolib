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

//! Example: Using MiniMax provider
//!
//! This example demonstrates how to use the MiniMax provider with octolib.
//! MiniMax uses an Anthropic-compatible API with support for thinking blocks,
//! tool use, and prompt caching.
//!
//! **Setup:**
//! 1. Get your API key from https://platform.minimax.io
//! 2. Set the environment variable:
//!    ```bash
//!    export MINIMAX_API_KEY="your-api-key-here"
//!    ```
//!
//! **Supported Models:**
//! - MiniMax-M2.1: Powerful multi-language programming capabilities ($0.3/$1.2 per 1M tokens)
//! - MiniMax-M2.1-lightning: Faster and more agile ($0.3/$2.4 per 1M tokens)
//! - MiniMax-M2: Agentic capabilities, advanced reasoning ($0.3/$1.2 per 1M tokens)
//!
//! **Features:**
//! - Thinking blocks: MiniMax can show its reasoning process
//! - Tool use: Function calling support
//! - Prompt caching: Reduce costs for repeated prompts
//! - Temperature range: (0.0, 1.0] (must be > 0.0)
//!
//! Run with:
//! ```bash
//! export MINIMAX_API_KEY="your-api-key-here"
//! cargo run --example minimax_chat
//! ```

use octolib::llm::factory::ProviderFactory;
use octolib::llm::types::{ChatCompletionParams, Message};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🚀 MiniMax Provider Example\n");

    // Check for API key
    match env::var("MINIMAX_API_KEY") {
        Ok(key) => {
            println!("✅ API key found: {}...", &key[..key.len().min(20)]);
        }
        Err(_) => {
            eprintln!("❌ Error: MINIMAX_API_KEY not set");
            eprintln!("\nTo use MiniMax:");
            eprintln!("1. Get your API key from https://platform.minimax.io");
            eprintln!("2. Set the environment variable:");
            eprintln!("   export MINIMAX_API_KEY=\"your-api-key-here\"");
            eprintln!("\nSupported models:");
            eprintln!("   - minimax:MiniMax-M2.1 (recommended)");
            eprintln!("   - minimax:MiniMax-M2.1-lightning (faster)");
            eprintln!("   - minimax:MiniMax-M2");
            std::process::exit(1);
        }
    }

    // Create provider with MiniMax-M2.1 model
    let (provider, model) = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2.1")?;

    println!("\n📝 Provider: {}", provider.name());
    println!("🤖 Model: {}", model);
    println!("💾 Caching support: {}", provider.supports_caching(&model));
    println!("👁️  Vision support: {}", provider.supports_vision(&model));
    println!(
        "📏 Max input tokens: {}\n",
        provider.get_max_input_tokens(&model)
    );

    // Example 1: Simple chat
    println!("=== Example 1: Simple Chat ===\n");

    let params = ChatCompletionParams {
        model: model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Explain what makes MiniMax unique in 2-3 sentences.".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            images: None,
            cached: false,
            timestamp: 0,
            thinking: None,
            id: None,
        }],
        temperature: 0.7,
        max_tokens: 200,
        top_p: 1.0,
        top_k: 0,
        tools: None,
        response_format: None,
        max_retries: 3,
        retry_timeout: std::time::Duration::from_secs(10),
        cancellation_token: None,
        previous_response_id: None,
    };

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("✅ Response:\n");
            println!("{}\n", response.content);

            if let Some(usage) = response.exchange.usage {
                println!("📊 Token usage:");
                println!("   Input tokens:  {}", usage.prompt_tokens);
                println!("   Output tokens: {}", usage.output_tokens);
                println!("   Total tokens:  {}", usage.total_tokens);
                if usage.cached_tokens > 0 {
                    println!("   Cached tokens: {} (saved cost!)", usage.cached_tokens);
                }
                if let Some(cost) = usage.cost {
                    println!("   Cost: ${:.6}", cost);
                }
                if let Some(time_ms) = usage.request_time_ms {
                    println!("   Request time: {}ms", time_ms);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            eprintln!("\nPossible issues:");
            eprintln!("- Invalid API key");
            eprintln!("- Network connectivity issues");
            eprintln!("- Model not available");
            std::process::exit(1);
        }
    }

    // Example 2: With thinking blocks
    println!("\n\n=== Example 2: Problem Solving (with thinking) ===\n");

    let params = ChatCompletionParams {
        model: model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "If I have 3 apples and buy 2 more, then give away 1, how many do I have? Show your reasoning.".to_string(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            images: None,
            cached: false,
            timestamp: 0,
            thinking: None,
        id: None,
            }],
        temperature: 0.5,
        max_tokens: 300,
        top_p: 1.0,
        top_k: 0,
        tools: None,
        response_format: None,
        max_retries: 3,
        retry_timeout: std::time::Duration::from_secs(10),
        cancellation_token: None,
        previous_response_id: None,
    };

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("✅ Response:\n");
            println!("{}\n", response.content);

            if let Some(usage) = response.exchange.usage {
                println!("📊 Token usage:");
                println!("   Input tokens:  {}", usage.prompt_tokens);
                println!("   Output tokens: {}", usage.output_tokens);
                if let Some(cost) = usage.cost {
                    println!("   Cost: ${:.6}", cost);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    println!("\n✨ MiniMax example completed successfully!");
    println!("\n💡 Tips:");
    println!("   - MiniMax shows thinking process in [Thinking] blocks");
    println!("   - Use MiniMax-M2.1-lightning for faster responses");
    println!("   - Temperature must be in range (0.0, 1.0]");
    println!("   - Supports prompt caching to reduce costs");
    println!("   - 1M token context window for long conversations");

    Ok(())
}
