// Copyright 2026 Muvon Un Limited
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

//! Fireworks AI chat completion example.
//!
//! Demonstrates a basic chat completion against Fireworks's OpenAI-compatible
//! endpoint, plus a second turn that exercises automatic prefix caching.
//!
//! Usage:
//! ```bash
//! export FIREWORKS_API_KEY="your_key"
//! cargo run --example fireworks_chat
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    // Fireworks model IDs use the `accounts/fireworks/models/<name>` namespace.
    let model_spec = "fireworks:accounts/fireworks/models/kimi-k2-instruct-0905";
    let (provider, model) = ProviderFactory::get_provider_for_model(model_spec)?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", model);
    println!("Caching supported: {}", provider.supports_caching(&model));
    println!(
        "Structured output: {}",
        provider.supports_structured_output(&model)
    );
    println!(
        "Max input tokens: {}",
        provider.get_max_input_tokens(&model)
    );
    if let Some(p) = provider.get_model_pricing(&model) {
        println!(
            "Pricing per 1M: input ${:.4}, output ${:.4}, cache_read ${:.4}",
            p.input_price_per_1m, p.output_price_per_1m, p.cache_read_price_per_1m
        );
    }

    // ---- First turn -------------------------------------------------------
    let messages = vec![Message::user(
        "In one short sentence, what is the Fireworks AI inference platform?",
    )];
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 256);

    println!("\n=== Turn 1 ===");
    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Tokens: input={} cache_read={} output={} total={}",
            usage.input_tokens, usage.cache_read_tokens, usage.output_tokens, usage.total_tokens
        );
        if let Some(cost) = usage.cost {
            println!("Cost: ${:.6}", cost);
        }
    }

    // ---- Second turn (same prefix → exercises automatic cache) -----------
    let messages = vec![
        Message::user("In one short sentence, what is the Fireworks AI inference platform?"),
        Message::assistant(&response.content),
        Message::user("Now name two open-weight model families it serves."),
    ];
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 256);

    println!("\n=== Turn 2 (cache should warm up) ===");
    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Tokens: input={} cache_read={} output={} total={}",
            usage.input_tokens, usage.cache_read_tokens, usage.output_tokens, usage.total_tokens
        );
        if let Some(cost) = usage.cost {
            println!("Cost: ${:.6}", cost);
        }
    }

    Ok(())
}
