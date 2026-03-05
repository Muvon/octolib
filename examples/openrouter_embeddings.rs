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

//! OpenRouter embedding example using qwen/qwen3-embedding-8b
//!
//! # Setup
//!
//! ```bash
//! export OPENROUTER_API_KEY="your_api_key_here"
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run --example openrouter_embeddings
//! ```

use octolib::embedding::types::InputType;
use octolib::embedding::{generate_embeddings, generate_embeddings_batch};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let provider = "openrouter";
    let model = "qwen/qwen3-embedding-8b";

    println!("=== OpenRouter Embeddings Example ===");
    println!("Provider: {}", provider);
    println!("Model:    {}\n", model);

    // Single embedding
    let text = "Rust is a systems programming language focused on safety and performance.";
    println!("Input: \"{}\"\n", text);

    let embedding = generate_embeddings(text, provider, model).await?;
    println!("Embedding dimension: {}", embedding.len());
    println!("First 8 values: {:?}\n", &embedding[..8]);

    // Batch embeddings
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Machine learning models learn patterns from data.".to_string(),
        "OpenRouter provides a unified API for many AI providers.".to_string(),
    ];

    println!("=== Batch Embeddings ({} texts) ===", texts.len());
    let batch =
        generate_embeddings_batch(texts.clone(), provider, model, InputType::None, 10, 8192)
            .await?;

    for (text, emb) in texts.iter().zip(batch.iter()) {
        println!("Text:      \"{}\"", text);
        println!("Dimension: {}", emb.len());
        println!("First 4:   {:?}\n", &emb[..4]);
    }

    Ok(())
}
