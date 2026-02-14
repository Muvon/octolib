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

//! Amazon Bedrock provider example
//!
//! This example demonstrates how to use the Amazon Bedrock provider with Octolib.
//!
//! # Prerequisites
//!
//! 1. Set the `AWS_BEARER_TOKEN_BEDROCK` environment variable with your Amazon Bedrock API key
//! 2. Optionally set `AWS_BEDROCK_REGION` (defaults to `us-east-1`)
//!
//! # Running the example
//!
//! ```bash
//! export AWS_BEARER_TOKEN_BEDROCK="your_api_key_here"
//! cargo run --example amazon_bedrock
//! ```

use octolib::{ChatCompletionParams, Message, ProviderFactory};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check for API key
    let api_key = env::var("AWS_BEARER_TOKEN_BEDROCK");
    if api_key.is_err() {
        eprintln!("Error: AWS_BEARER_TOKEN_BEDROCK environment variable not set");
        eprintln!("Get your API key from: https://console.aws.amazon.com/bedrock/home");
        std::process::exit(1);
    }

    println!("🚀 Amazon Bedrock Example");
    println!("====================");

    // Example 1: Basic chat completion with Anthropic Claude
    println!("\n📝 Example 1: Anthropic Claude 3.5 Sonnet");
    let (provider, model) =
        ProviderFactory::get_provider_for_model("amazon:anthropic.claude-3-5-sonnet")?;
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
                if let Some(cost) = usage.cost {
                    println!("Cost: ${:.6}", cost);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 2: Chat completion with Meta Llama
    println!("\n📝 Example 2: Meta Llama 3.1 70B");
    let (provider, model) = ProviderFactory::get_provider_for_model("amazon:meta.llama-3.1-70b")?;
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
                if let Some(cost) = usage.cost {
                    println!("Cost: ${:.6}", cost);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }

    // Example 3: Chat completion with Amazon Nova
    println!("\n📝 Example 3: Amazon Nova Pro");
    let (provider, model) = ProviderFactory::get_provider_for_model("amazon:amazon.nova-pro")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("Write a haiku about technology")];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Response: {}", response.content);
            if let Some(usage) = &response.exchange.usage {
                println!("Input tokens: {}", usage.input_tokens);
                println!("Output tokens: {}", usage.output_tokens);
                println!("Total tokens: {}", usage.total_tokens);
                if let Some(cost) = usage.cost {
                    println!("Cost: ${:.6}", cost);
                }
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
