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
//! ## API-Based Providers (require API keys)
//!
//! - **Voyage AI**: Latest reranker models with multilingual support
//!   - Models: `rerank-2.5`, `rerank-2.5-lite`, `rerank-2`, `rerank-2-lite`
//!   - Requires: `VOYAGE_API_KEY` environment variable
//!
//! - **Cohere**: Enterprise-grade reranking with multiple language support
//!   - Models: `rerank-english-v3.0`, `rerank-multilingual-v3.0`, `rerank-english-v2.0`, `rerank-multilingual-v2.0`
//!   - Requires: `COHERE_API_KEY` environment variable
//!
//! - **Jina AI**: Multilingual reranking with automatic chunking for long documents
//!   - Models: `jina-reranker-v3`, `jina-reranker-v2-base-multilingual`, `jina-reranker-v1-base-en`, `jina-colbert-v2`
//!   - Requires: `JINA_API_KEY` environment variable
//!
//! ## Local Providers (no API keys, requires features)
//!
//! - **FastEmbed**: Fast local ONNX-based reranking (requires `fastembed` feature)
//!   - Models: `bge-reranker-base`, `bge-reranker-large`, `jina-reranker-v1-turbo-en`, `jina-reranker-v2-base-multilingual`
//!   - No API key needed, runs locally with CPU
//!
//! # Usage Examples
//!
//! ## Voyage AI
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
//!
//! ## Cohere
//! ```rust,no_run
//! use octolib::reranker::rerank;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let response = rerank(
//!     "machine learning tutorial",
//!     vec!["AI guide".to_string(), "Cooking recipes".to_string()],
//!     "cohere",
//!     "rerank-english-v3.0",
//!     Some(1)
//! ).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Jina AI
//! ```rust,no_run
//! use octolib::reranker::rerank;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let response = rerank(
//!     "artificial intelligence",
//!     vec!["ML basics".to_string(), "Cooking tips".to_string()],
//!     "jina",
//!     "jina-reranker-v3",
//!     Some(1)
//! ).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## FastEmbed (Local)
//! ```rust,no_run
//! use octolib::reranker::rerank;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // No API key needed - runs locally
//! let response = rerank(
//!     "deep learning",
//!     vec!["Neural networks".to_string(), "Pasta recipes".to_string()],
//!     "fastembed",
//!     "bge-reranker-base",
//!     Some(1)
//! ).await?;
//! # Ok(())
//! # }
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

        let (provider, model) = parse_provider_model("cohere:rerank-english-v3.0");
        assert_eq!(provider, RerankProviderType::Cohere);
        assert_eq!(model, "rerank-english-v3.0");

        let (provider, model) = parse_provider_model("jina:jina-reranker-v3");
        assert_eq!(provider, RerankProviderType::Jina);
        assert_eq!(model, "jina-reranker-v3");

        #[cfg(feature = "fastembed")]
        {
            let (provider, model) = parse_provider_model("fastembed:bge-reranker-base");
            assert_eq!(provider, RerankProviderType::FastEmbed);
            assert_eq!(model, "bge-reranker-base");
        }
    }

    #[tokio::test]
    async fn test_create_provider() {
        // API-based providers - require API keys
        let voyage =
            create_rerank_provider_from_parts(&RerankProviderType::Voyage, "rerank-2.5").await;
        match voyage {
            Ok(_) => {}
            Err(e) => {
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("VOYAGE_API_KEY"),
                    "Expected API key error, got: {}",
                    e
                );
            }
        }

        let cohere =
            create_rerank_provider_from_parts(&RerankProviderType::Cohere, "rerank-english-v3.0")
                .await;
        match cohere {
            Ok(_) => {}
            Err(e) => {
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("COHERE_API_KEY"),
                    "Expected API key error, got: {}",
                    e
                );
            }
        }

        let jina =
            create_rerank_provider_from_parts(&RerankProviderType::Jina, "jina-reranker-v3").await;
        match jina {
            Ok(_) => {}
            Err(e) => {
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("JINA_API_KEY"),
                    "Expected API key error, got: {}",
                    e
                );
            }
        }

        // FastEmbed - local provider, may require model download
        #[cfg(feature = "fastembed")]
        {
            let fastembed = create_rerank_provider_from_parts(
                &RerankProviderType::FastEmbed,
                "bge-reranker-base",
            )
            .await;
            match fastembed {
                Ok(provider) => {
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Model download may be needed - graceful handling for CI
                    println!(
                        "FastEmbed provider creation skipped (model download needed): {}",
                        e
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_models() {
        let result =
            create_rerank_provider_from_parts(&RerankProviderType::Voyage, "invalid-model").await;
        assert!(result.is_err());

        let result =
            create_rerank_provider_from_parts(&RerankProviderType::Cohere, "invalid-model").await;
        assert!(result.is_err());

        let result =
            create_rerank_provider_from_parts(&RerankProviderType::Jina, "invalid-model").await;
        assert!(result.is_err());

        #[cfg(feature = "fastembed")]
        {
            let result =
                create_rerank_provider_from_parts(&RerankProviderType::FastEmbed, "invalid-model")
                    .await;
            assert!(result.is_err());
        }
    }
}
