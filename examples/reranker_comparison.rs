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

//! Reranker comparison example
//!
//! This example demonstrates how to use different reranker providers
//! to rerank documents based on query relevance.
//!
//! # API-Based Providers (require API keys):
//! - Voyage: Set VOYAGE_API_KEY
//! - Cohere: Set COHERE_API_KEY
//! - Jina: Set JINA_API_KEY
//!
//! # Local Providers (no API keys):
//! - FastEmbed: Requires `fastembed` feature, runs locally
//!
//! Run with:
//! ```bash
//! # API providers
//! export VOYAGE_API_KEY="your_key"
//! export COHERE_API_KEY="your_key"
//! export JINA_API_KEY="your_key"
//! cargo run --example reranker_comparison
//!
//! # Local provider
//! cargo run --example reranker_comparison --features fastembed
//! ```

use octolib::reranker::rerank;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Sample query and documents
    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.".to_string(),
        "The best pasta recipe includes fresh tomatoes and basil.".to_string(),
        "Deep learning uses neural networks with multiple layers.".to_string(),
        "Paris is the capital city of France.".to_string(),
        "Supervised learning requires labeled training data.".to_string(),
    ];

    println!("Query: {}\n", query);
    println!("Documents to rerank:");
    for (i, doc) in documents.iter().enumerate() {
        println!("  [{}] {}", i, doc);
    }
    println!("\n{}\n", "=".repeat(80));

    // Try Voyage AI
    if std::env::var("VOYAGE_API_KEY").is_ok() {
        println!("🚢 Voyage AI Reranker (rerank-2.5)");
        match rerank(query, documents.clone(), "voyage", "rerank-2.5", Some(3)).await {
            Ok(response) => {
                println!("Top 3 results:");
                for (i, result) in response.results.iter().enumerate() {
                    println!(
                        "  {}. [{}] Score: {:.4} - {}",
                        i + 1,
                        result.index,
                        result.relevance_score,
                        result.document
                    );
                }
                println!("Total tokens: {}\n", response.total_tokens);
            }
            Err(e) => println!("Error: {}\n", e),
        }
    } else {
        println!("⏭️  Skipping Voyage AI (VOYAGE_API_KEY not set)\n");
    }

    // Try Cohere
    if std::env::var("COHERE_API_KEY").is_ok() {
        println!("🔷 Cohere Reranker (rerank-english-v3.0)");
        match rerank(
            query,
            documents.clone(),
            "cohere",
            "rerank-english-v3.0",
            Some(3),
        )
        .await
        {
            Ok(response) => {
                println!("Top 3 results:");
                for (i, result) in response.results.iter().enumerate() {
                    println!(
                        "  {}. [{}] Score: {:.4} - {}",
                        i + 1,
                        result.index,
                        result.relevance_score,
                        result.document
                    );
                }
                println!();
            }
            Err(e) => println!("Error: {}\n", e),
        }
    } else {
        println!("⏭️  Skipping Cohere (COHERE_API_KEY not set)\n");
    }

    // Try Jina AI
    if std::env::var("JINA_API_KEY").is_ok() {
        println!("🔮 Jina AI Reranker (jina-reranker-v3)");
        match rerank(
            query,
            documents.clone(),
            "jina",
            "jina-reranker-v3",
            Some(3),
        )
        .await
        {
            Ok(response) => {
                println!("Top 3 results:");
                for (i, result) in response.results.iter().enumerate() {
                    println!(
                        "  {}. [{}] Score: {:.4} - {}",
                        i + 1,
                        result.index,
                        result.relevance_score,
                        result.document
                    );
                }
                println!("Total tokens: {}\n", response.total_tokens);
            }
            Err(e) => println!("Error: {}\n", e),
        }
    } else {
        println!("⏭️  Skipping Jina AI (JINA_API_KEY not set)\n");
    }

    // Try FastEmbed (local, no API key needed)
    #[cfg(feature = "fastembed")]
    {
        println!("⚡ FastEmbed Local Reranker (bge-reranker-base)");
        match rerank(
            query,
            documents.clone(),
            "fastembed",
            "bge-reranker-base",
            Some(3),
        )
        .await
        {
            Ok(response) => {
                println!("Top 3 results:");
                for (i, result) in response.results.iter().enumerate() {
                    println!(
                        "  {}. [{}] Score: {:.4} - {}",
                        i + 1,
                        result.index,
                        result.relevance_score,
                        result.document
                    );
                }
                println!("(Local inference, no API key needed)\n");
            }
            Err(e) => println!("Error: {}\n", e),
        }
    }

    #[cfg(not(feature = "fastembed"))]
    {
        println!("⏭️  Skipping FastEmbed (compile with --features fastembed)\n");
    }

    println!("{}", "=".repeat(80));
    println!("\n✅ Comparison complete!");
    println!("\nNotes:");
    println!("- API providers require environment variables (VOYAGE_API_KEY, COHERE_API_KEY, JINA_API_KEY)");
    println!("- FastEmbed runs locally without API keys (requires fastembed feature)");
    println!("- Scores are normalized relevance scores (higher = more relevant)");
    println!("- All providers return results sorted by relevance (descending)");

    Ok(())
}
