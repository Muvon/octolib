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

//! Example: Using Anthropic provider with OAuth authentication
//!
//! This example demonstrates how to use OAuth tokens with the Anthropic provider.
//! The library automatically detects OAuth tokens and uses them instead of API keys.
//!
//! **Authentication Priority:**
//! 1. ANTHROPIC_OAUTH_TOKEN (if set, uses Bearer token auth)
//! 2. ANTHROPIC_API_KEY (fallback, uses x-api-key header)
//!
//! **Your Responsibility:**
//! - Obtain OAuth token through your own OAuth flow
//! - Handle token refresh before expiration
//! - Store tokens securely
//! - Set environment variables before using the library
//!
//! **Library Responsibility:**
//! - Detect which auth method to use
//! - Send requests with appropriate headers
//! - Handle API communication
//!
//! Run with:
//! ```bash
//! export ANTHROPIC_OAUTH_TOKEN="your-oauth-token-here"
//! cargo run --example anthropic_oauth
//! ```

use octolib::llm::factory::ProviderFactory;
use octolib::llm::types::{ChatCompletionParams, Message};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔐 Anthropic OAuth Authentication Example\n");

    // Check for OAuth token
    match env::var("ANTHROPIC_OAUTH_TOKEN") {
        Ok(token) => {
            println!("✅ OAuth token found: {}...", &token[..token.len().min(20)]);
            println!("   Using OAuth Bearer token authentication\n");
        }
        Err(_) => {
            eprintln!("❌ Error: ANTHROPIC_OAUTH_TOKEN not set");
            eprintln!("\nTo use OAuth authentication:");
            eprintln!("1. Obtain an OAuth token through your OAuth flow");
            eprintln!("2. Set the environment variable:");
            eprintln!("   export ANTHROPIC_OAUTH_TOKEN=\"your-token-here\"");
            eprintln!(
                "\nNote: The library does NOT handle OAuth flow, token refresh, or token storage."
            );
            eprintln!("      Your application must manage the complete token lifecycle.");
            eprintln!("\nAlternatively, you can use API key authentication:");
            eprintln!("   export ANTHROPIC_API_KEY=\"your-api-key-here\"");
            std::process::exit(1);
        }
    }

    // Create provider - it will automatically use OAuth token
    let (provider, model) = ProviderFactory::get_provider_for_model("anthropic:claude-3-5-sonnet")?;

    println!("📝 Sending test message to Claude...\n");

    // Create a simple test message
    let params = ChatCompletionParams {
        model: model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Say 'Hello from OAuth!' in a creative way".to_string(),
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
        max_tokens: 100,
        top_p: 1.0,
        top_k: 0,
        tools: None,
        response_format: None,
        max_retries: 3,
        retry_timeout: std::time::Duration::from_secs(10),
        cancellation_token: None,
        previous_response_id: None,
    };

    // Make the API call
    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("✅ Response received:\n");
            println!("{}", response.content);
            println!("\n📊 Token usage:");
            if let Some(usage) = response.exchange.usage {
                println!("   Input tokens:  {}", usage.prompt_tokens);
                println!("   Output tokens: {}", usage.output_tokens);
                println!("   Total tokens:  {}", usage.total_tokens);
                if let Some(cost) = usage.cost {
                    println!("   Cost: ${:.6}", cost);
                }
            }
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            eprintln!("\nPossible issues:");
            eprintln!("- OAuth token is invalid or expired");
            eprintln!("- Token doesn't have required permissions");
            eprintln!("- Network connectivity issues");
            std::process::exit(1);
        }
    }

    println!("\n✨ OAuth authentication successful!");
    println!("\n💡 Remember:");
    println!("   - Library automatically detects OAuth token");
    println!("   - Falls back to API key if OAuth token not set");
    println!("   - You must handle token refresh and storage");

    Ok(())
}
