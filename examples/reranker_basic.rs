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

//! Basic reranker example using Voyage AI
//!
//! This example demonstrates how to use the reranker to improve search results
//! by scoring document relevance to a query.
//!
//! # Setup
//!
//! Set your Voyage API key:
//! ```bash
//! export VOYAGE_API_KEY="your_api_key_here"
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run --example reranker_basic
//! ```

use octolib::reranker::rerank;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt::init();

    println!("=== Octolib Reranker Example ===\n");

    // Example query and documents
    let query = "When is Apple's conference call scheduled?";
    let documents = vec![
        "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.".to_string(),
        "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.".to_string(),
        "20th-century innovations, from radios to smartphones, centered on electronic advancements.".to_string(),
        "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.".to_string(),
        "Apple's conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.".to_string(),
        "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature.".to_string(),
    ];

    println!("Query: {}\n", query);
    println!("Documents to rerank: {}\n", documents.len());

    // Rerank with Voyage AI - get top 3 results
    println!("Reranking with Voyage AI (rerank-2.5)...\n");
    let response = rerank(query, documents.clone(), "voyage", "rerank-2.5", Some(3)).await?;

    println!("Top {} results:", response.results.len());
    println!("Total tokens used: {}\n", response.total_tokens);

    for (rank, result) in response.results.iter().enumerate() {
        println!("Rank {}: (Score: {:.4})", rank + 1, result.relevance_score);
        println!("  Document: {}", result.document);
        println!("  Original index: {}\n", result.index);
    }

    // Example 2: Rerank all documents without top_k limit
    println!("\n=== Reranking all documents ===\n");
    let response_all = rerank(query, documents, "voyage", "rerank-2.5", None).await?;

    println!("All {} results ranked:", response_all.results.len());
    for (rank, result) in response_all.results.iter().enumerate() {
        println!(
            "{}. Score: {:.4} - {}",
            rank + 1,
            result.relevance_score,
            &result.document[..60.min(result.document.len())]
        );
    }

    Ok(())
}
