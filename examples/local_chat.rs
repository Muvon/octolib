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

//! Example: Using local provider with local AI models
//!
//! This example demonstrates how to use the local provider
//! with local model servers like Ollama, LM Studio, LocalAI, Jan, vLLM, etc.
//!
//! ## Setup
//!
//! ### Ollama (default)
//! ```bash
//! # Install Ollama: https://ollama.ai
//! ollama pull llama3.2
//! cargo run --example local_chat
//! ```
//!
//! ### LM Studio
//! ```bash
//! # Start LM Studio server on port 1234
//! export LOCAL_API_URL="http://localhost:1234/v1/chat/completions"
//! cargo run --example local_chat
//! ```
//!
//! ### LocalAI
//! ```bash
//! # Start LocalAI server on port 8080
//! export LOCAL_API_URL="http://localhost:8080/v1/chat/completions"
//! cargo run --example local_chat
//! ```
//!
//! ### Jan
//! ```bash
//! # Start Jan server on port 1337
//! export LOCAL_API_URL="http://localhost:1337/v1/chat/completions"
//! cargo run --example local_chat
//! ```

use octolib::llm::{
    factory::ProviderFactory,
    types::{ChatCompletionParams, Message},
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🦙 Local Provider Example\n");

    // Get API URL from environment or use default (Ollama)
    let api_url = std::env::var("LOCAL_API_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".to_string());

    println!("📡 API URL: {}", api_url);
    println!("💡 Tip: Set LOCAL_API_URL to use different servers\n");

    // Model name - can be any model your local server supports
    // For Ollama: llama3.2, mistral, codellama, etc.
    // For LM Studio: whatever model you loaded
    let model_name = std::env::var("MODEL_NAME").unwrap_or_else(|_| "llama3.2".to_string());

    println!("🤖 Model: {}", model_name);
    println!("💡 Tip: Set MODEL_NAME environment variable to use different models\n");

    // Create provider
    let model_string = format!("local:{}", model_name);
    let (provider, model) = ProviderFactory::get_provider_for_model(&model_string)?;

    println!("✅ Provider: {}", provider.name());
    println!("✅ Model: {}\n", model);

    // Create a simple conversation
    let messages = vec![
        Message::system("You are a helpful assistant running locally."),
        Message::user("What are the benefits of running AI models locally?"),
    ];

    println!("💬 Sending message to local model...\n");

    // Create chat completion parameters
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 500);

    // Get response
    let response = provider.chat_completion(params).await?;

    println!("🤖 Response:\n{}\n", response.content);

    // Display token usage
    if let Some(usage) = response.exchange.usage {
        println!("📊 Token Usage:");
        println!("  - Prompt tokens: {}", usage.prompt_tokens);
        println!("  - Output tokens: {}", usage.output_tokens);
        println!("  - Total tokens: {}", usage.total_tokens);
        if let Some(time) = usage.request_time_ms {
            println!("  - Request time: {}ms", time);
        }
        if let Some(cost) = usage.cost {
            println!("  - Cost: ${:.6} (local models are free!)", cost);
        }
    }

    println!("\n✅ Example completed successfully!");
    println!("\n💡 Try different models:");
    println!("   MODEL_NAME=mistral cargo run --example local_chat");
    println!("   MODEL_NAME=codellama cargo run --example local_chat");
    println!("\n💡 Try different servers:");
    println!(
        "   LOCAL_API_URL=http://localhost:1234/v1/chat/completions cargo run --example local_chat"
    );

    Ok(())
}
