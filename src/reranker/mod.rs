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

//! Reranker module for document relevance scoring
//!
//! This module provides reranking functionality to refine search results by scoring
//! document relevance to a query. Rerankers use cross-encoder models that jointly
//! process query-document pairs for more accurate relevance prediction than embeddings.
//!
//! # Supported Providers
//!
//! - **Voyage AI**: Latest reranker models with multilingual support
//!
//! # Usage
//!
//! ```rust,no_run
//! use octolib::reranker::rerank;
//!
//! async fn example() -> anyhow::Result<()> {
//!     let query = "What is the capital of France?";
//!     let documents = vec![
//!         "Paris is the capital of France.".to_string(),
//!         "London is the capital of England.".to_string(),
//!         "Berlin is the capital of Germany.".to_string(),
//!     ];
//!
//!     // Rerank with Voyage AI (requires VOYAGE_API_KEY)
//!     let response = rerank(query, documents, "voyage", "rerank-2.5", Some(2)).await?;
//!
//!     for result in response.results {
//!         println!("Score: {:.4} - {}", result.relevance_score, result.document);
//!     }
//!
//!     Ok(())
//! }
//! ```

use anyhow::Result;

pub mod provider;
pub mod types;

pub use provider::{create_rerank_provider_from_parts, RerankProvider};
pub use types::{parse_provider_model, RerankProviderType, RerankResponse, RerankResult};

/// Rerank documents based on query relevance
///
/// # Arguments
///
/// * `query` - The search query
/// * `documents` - List of documents to rerank
/// * `provider` - Provider name (e.g., "voyage")
/// * `model` - Model name (e.g., "rerank-2.5")
/// * `top_k` - Optional number of top results to return (None = all)
///
/// # Returns
///
/// `RerankResponse` containing sorted results with relevance scores
///
/// # Example
///
/// ```rust,no_run
/// use octolib::reranker::rerank;
///
/// # async fn example() -> anyhow::Result<()> {
/// let response = rerank(
///     "machine learning",
///     vec!["AI tutorial".to_string(), "Cooking recipes".to_string()],
///     "voyage",
///     "rerank-2.5",
///     Some(1)
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn rerank(
    query: &str,
    documents: Vec<String>,
    provider: &str,
    model: &str,
    top_k: Option<usize>,
) -> Result<RerankResponse> {
    let (provider_type, _) = parse_provider_model(&format!("{}:{}", provider, model));
    let provider_impl = create_rerank_provider_from_parts(&provider_type, model).await?;
    provider_impl.rerank(query, documents, top_k, true).await
}

/// Rerank documents with custom truncation setting
///
/// # Arguments
///
/// * `query` - The search query
/// * `documents` - List of documents to rerank
/// * `provider` - Provider name (e.g., "voyage")
/// * `model` - Model name (e.g., "rerank-2.5")
/// * `top_k` - Optional number of top results to return
/// * `truncation` - Whether to truncate long inputs (true = truncate, false = error on overflow)
///
/// # Returns
///
/// `RerankResponse` containing sorted results with relevance scores
pub async fn rerank_with_truncation(
    query: &str,
    documents: Vec<String>,
    provider: &str,
    model: &str,
    top_k: Option<usize>,
    truncation: bool,
) -> Result<RerankResponse> {
    let (provider_type, _) = parse_provider_model(&format!("{}:{}", provider, model));
    let provider_impl = create_rerank_provider_from_parts(&provider_type, model).await?;
    provider_impl
        .rerank(query, documents, top_k, truncation)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider_model() {
        let (provider, model) = parse_provider_model("voyage:rerank-2.5");
        assert_eq!(provider, RerankProviderType::Voyage);
        assert_eq!(model, "rerank-2.5");
    }

    #[tokio::test]
    async fn test_create_provider() {
        let result =
            create_rerank_provider_from_parts(&RerankProviderType::Voyage, "rerank-2.5").await;
        assert!(result.is_ok());
    }
}
