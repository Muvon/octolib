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

//! Cloudflare Workers AI provider example
//!
//! This example demonstrates how to use Cloudflare Workers AI provider with Octolib.
//!
//! # Prerequisites
//!
//! 1. Set `CLOUDFLARE_API_TOKEN` environment variable with your Cloudflare API token
//! 2. Set `CLOUDFLARE_ACCOUNT_ID` environment variable with your Cloudflare account ID
//!
//! # Running the example
//!
//! ```bash
//! export CLOUDFLARE_API_TOKEN="your_api_token_here"
//! export CLOUDFLARE_ACCOUNT_ID="your_account_id_here"
//! cargo run --example cloudflare_workers_ai
//! ```

use octolib::{ChatCompletionParams, Message, ProviderFactory};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check for API token
    let api_token = env::var("CLOUDFLARE_API_TOKEN");
    if api_token.is_err() {
        eprintln!("Error: CLOUDFLARE_API_TOKEN environment variable not set");
        eprintln!("Get your API token from: https://dash.cloudflare.com/profile/api-tokens");
        std::process::exit(1);
    }

    // Check for account ID
    let account_id = env::var("CLOUDFLARE_ACCOUNT_ID");
    if account_id.is_err() {
        eprintln!("Error: CLOUDFLARE_ACCOUNT_ID environment variable not set");
        eprintln!("Get your account ID from: https://dash.cloudflare.com");
        std::process::exit(1);
    }

    println!("🚀 Cloudflare Workers AI Example");
    println!("====================");

    // Example 1: Basic chat completion with Llama 3.1
    println!("\n📝 Example 1: Llama 3.1 70B Instruct");
    let (provider, model) =
        ProviderFactory::get_provider_for_model("cloudflare:@cf/meta/llama-3.1-70b-instruct")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("What is the capital of France?")];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Response: {}", response.content);
            if let Some(usage) = &response.exchange.usage {
                println!("Input tokens: {}", usage.input_tokens);
                println!("Output tokens: {}", usage.output_tokens);
                println!("Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 2: Chat completion with Mistral
    println!("\n📝 Example 2: Mistral 7B Instruct");
    let (provider, model) =
        ProviderFactory::get_provider_for_model("cloudflare:mistral-7b-instruct-v0.1")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("Explain quantum computing in simple terms")];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Response: {}", response.content);
            if let Some(usage) = &response.exchange.usage {
                println!("Input tokens: {}", usage.input_tokens);
                println!("Output tokens: {}", usage.output_tokens);
                println!("Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 3: Chat completion with Gemma
    println!("\n📝 Example 3: Gemma 2 27B IT");
    let (provider, model) = ProviderFactory::get_provider_for_model("cloudflare:gemma-2-27b-it")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("Write a short poem about technology")];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Response: {}", response.content);
            if let Some(usage) = &response.exchange.usage {
                println!("Input tokens: {}", usage.input_tokens);
                println!("Output tokens: {}", usage.output_tokens);
                println!("Total tokens: {}", usage.total_tokens);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    println!("\n====================");
    println!("✅ Examples completed!");

    Ok(())
}
