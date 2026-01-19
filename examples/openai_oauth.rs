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

//! Example: Using OpenAI provider with OAuth authentication (ChatGPT subscriptions)
//!
//! This example demonstrates how to use OAuth tokens with the OpenAI provider
//! for ChatGPT subscription-based access (Plus/Pro/Team/Enterprise).
//! The library automatically detects OAuth tokens and uses them instead of API keys.
//!
//! **Authentication Priority:**
//! 1. OPENAI_OAUTH_ACCESS_TOKEN + OPENAI_OAUTH_ACCOUNT_ID (if both set, uses OAuth)
//! 2. OPENAI_API_KEY (fallback, uses standard API key)
//!
//! **Your Responsibility:**
//! - Implement OAuth flow (browser login, PKCE, callback server)
//! - Obtain access_token and account_id from OAuth response
//! - Handle token refresh before expiration (typically every 8 days)
//! - Store tokens securely
//! - Set environment variables before using the library
//!
//! **Library Responsibility:**
//! - Detect which auth method to use
//! - Send requests with appropriate headers (Bearer + ChatGPT-Account-ID)
//! - Handle API communication
//!
//! Run with:
//! ```bash
//! export OPENAI_OAUTH_ACCESS_TOKEN="your-access-token-here"
//! export OPENAI_OAUTH_ACCOUNT_ID="your-account-id-here"
//! cargo run --example openai_oauth
//! ```

use octolib::llm::factory::ProviderFactory;
use octolib::llm::types::{ChatCompletionParams, Message};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔐 OpenAI OAuth Authentication Example (ChatGPT Subscriptions)\n");

    // Check for OAuth tokens
    let access_token = env::var("OPENAI_OAUTH_ACCESS_TOKEN");
    let account_id = env::var("OPENAI_OAUTH_ACCOUNT_ID");

    match (access_token, account_id) {
        (Ok(token), Ok(id)) => {
            println!(
                "✅ OAuth access token found: {}...",
                &token[..token.len().min(20)]
            );
            println!("✅ Account ID found: {}", id);
            println!("   Using OAuth Bearer token + ChatGPT-Account-ID authentication\n");
        }
        (Ok(_), Err(_)) => {
            eprintln!(
                "❌ Error: OPENAI_OAUTH_ACCESS_TOKEN is set but OPENAI_OAUTH_ACCOUNT_ID is missing"
            );
            eprintln!("\nBoth environment variables are required for OAuth:");
            eprintln!("   export OPENAI_OAUTH_ACCESS_TOKEN=\"your-access-token\"");
            eprintln!("   export OPENAI_OAUTH_ACCOUNT_ID=\"your-account-id\"");
            std::process::exit(1);
        }
        (Err(_), Ok(_)) => {
            eprintln!(
                "❌ Error: OPENAI_OAUTH_ACCOUNT_ID is set but OPENAI_OAUTH_ACCESS_TOKEN is missing"
            );
            eprintln!("\nBoth environment variables are required for OAuth:");
            eprintln!("   export OPENAI_OAUTH_ACCESS_TOKEN=\"your-access-token\"");
            eprintln!("   export OPENAI_OAUTH_ACCOUNT_ID=\"your-account-id\"");
            std::process::exit(1);
        }
        (Err(_), Err(_)) => {
            eprintln!("❌ Error: OAuth tokens not set");
            eprintln!("\nTo use OAuth authentication (ChatGPT subscriptions):");
            eprintln!("1. Implement OAuth flow in your application:");
            eprintln!("   - Start local callback server (e.g., localhost:1455)");
            eprintln!("   - Generate PKCE codes (code_verifier + code_challenge)");
            eprintln!("   - Open browser to: https://auth.openai.com/oauth/authorize");
            eprintln!("   - Exchange authorization code for tokens");
            eprintln!("2. Extract access_token and account_id from OAuth response");
            eprintln!("3. Set environment variables:");
            eprintln!("   export OPENAI_OAUTH_ACCESS_TOKEN=\"your-access-token\"");
            eprintln!("   export OPENAI_OAUTH_ACCOUNT_ID=\"your-account-id\"");
            eprintln!("\nNote: The library does NOT handle:");
            eprintln!("      - OAuth flow implementation");
            eprintln!("      - Token refresh (refresh every ~8 days)");
            eprintln!("      - Token storage");
            eprintln!("      Your application must manage the complete token lifecycle.");
            eprintln!("\nAlternatively, use standard API key authentication:");
            eprintln!("   export OPENAI_API_KEY=\"sk-proj-...\"");
            std::process::exit(1);
        }
    }

    // Create provider - it will automatically use OAuth tokens
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    println!("📝 Sending test message to GPT-4o...\n");

    // Create a simple test message
    let params = ChatCompletionParams {
        model: model.clone(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "Say 'Hello from ChatGPT OAuth!' in a creative way.".to_string(),
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
            eprintln!("- OAuth access token is invalid or expired");
            eprintln!("- Account ID is incorrect");
            eprintln!("- Token doesn't have required permissions");
            eprintln!("- ChatGPT subscription is not active");
            eprintln!("- Network connectivity issues");
            eprintln!("\nToken refresh needed?");
            eprintln!("- Access tokens typically expire after 8 days");
            eprintln!("- Use refresh_token to get new access_token");
            std::process::exit(1);
        }
    }

    println!("\n✨ OAuth authentication successful!");
    println!("\n💡 Remember:");
    println!("   - Library automatically detects OAuth tokens");
    println!("   - Falls back to API key if OAuth tokens not set");
    println!("   - You must handle token refresh (~8 days)");
    println!("   - Both access_token AND account_id are required");
    println!("\n📚 OAuth Flow Reference:");
    println!("   - Authorization: https://auth.openai.com/oauth/authorize");
    println!("   - Token Exchange: https://auth.openai.com/oauth/token");
    println!("   - Client ID: app_EMoamEEZ73f0CkXaXp7hrann");
    println!("   - Scopes: openid profile email offline_access");

    Ok(())
}
