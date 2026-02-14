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

//! Google Vertex AI provider example
//!
//! This example demonstrates how to use Google Vertex AI provider with Octolib.
//!
//! # Prerequisites
//!
//! 1. Set `GOOGLE_CREDENTIAL_FILE` to your downloaded Google service-account JSON file.
//!    (`GOOGLE_APPLICATION_CREDENTIALS` is also accepted.)
//! 2. Optionally set:
//!    - `GOOGLE_CLOUD_PROJECT_ID` (otherwise read from credential file)
//!    - `GOOGLE_CLOUD_LOCATION` (default: `us-central1`)
//!
//! # Running the example
//!
//! ```bash
//! export GOOGLE_CREDENTIAL_FILE="/path/to/service-account.json"
//! export GOOGLE_CLOUD_LOCATION="us-central1"
//! cargo run --example google_vertex_ai
//! ```

use octolib::{ChatCompletionParams, Message, ProviderFactory};
use std::env;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Check for credential file
    let credential_file = env::var("GOOGLE_CREDENTIAL_FILE");
    let application_credentials = env::var("GOOGLE_APPLICATION_CREDENTIALS");

    if credential_file.is_err() && application_credentials.is_err() {
        eprintln!(
            "Error: Set GOOGLE_CREDENTIAL_FILE or GOOGLE_APPLICATION_CREDENTIALS to your service-account JSON file"
        );
        std::process::exit(1);
    }

    if env::var("GOOGLE_CLOUD_PROJECT_ID").is_err() {
        println!("Info: GOOGLE_CLOUD_PROJECT_ID not set, provider will read project_id from credential file");
    }

    if env::var("GOOGLE_CLOUD_LOCATION").is_err() {
        println!("Info: GOOGLE_CLOUD_LOCATION not set, defaulting to us-central1");
    }

    if env::var("GOOGLE_API_URL").is_ok() {
        println!("Info: using GOOGLE_API_URL override");
    }

    println!("🚀 Google Vertex AI Example");
    println!("====================");

    // Example 1: Basic chat completion with Gemini 2.5 Flash
    println!("\n📝 Example 1: Gemini 2.5 Flash");
    let (provider, model) = ProviderFactory::get_provider_for_model("google:gemini-2.5-flash")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("What is the capital of Japan?")];

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

    // Example 2: Chat completion with Gemini 1.5 Pro
    println!("\n📝 Example 2: Gemini 1.5 Pro");
    let (provider, model) = ProviderFactory::get_provider_for_model("google:gemini-1.5-pro")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("Explain machine learning in simple terms")];

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

    // Example 3: Chat completion with Gemini 2.0 Flash
    println!("\n📝 Example 3: Gemini 2.0 Flash");
    let (provider, model) = ProviderFactory::get_provider_for_model("google:gemini-2.0-flash")?;
    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("Write a short story about space exploration")];

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
