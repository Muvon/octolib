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

//! Inception Labs chat completion example.
//!
//! Usage:
//! ```bash
//! export INCEPTION_API_KEY="your_key"
//! cargo run --example inception_chat
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_spec = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "inception:mercury-2.5".to_string());
    let (provider, model) = ProviderFactory::get_provider_for_model(&model_spec)?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user(
        "In one short sentence, what is a diffusion language model?",
    )];
    // Mercury reasons by default, so leave headroom for reasoning tokens
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1024);

    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Tokens: input={} output={} reasoning={} cached={} total={} cost={:?}",
            usage.input_tokens,
            usage.output_tokens,
            usage.reasoning_tokens,
            usage.cache_read_tokens,
            usage.total_tokens,
            usage.cost
        );
    }

    Ok(())
}
