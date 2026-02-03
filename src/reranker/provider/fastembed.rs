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

//! FastEmbed reranker provider implementation

#[cfg(feature = "fastembed")]
use anyhow::{Context, Result};
#[cfg(feature = "fastembed")]
use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
#[cfg(feature = "fastembed")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "fastembed")]
use super::super::types::{RerankResponse, RerankResult};
#[cfg(feature = "fastembed")]
use super::RerankProvider;

#[cfg(feature = "fastembed")]
/// FastEmbed provider implementation
pub struct FastEmbedProvider {
    model: Arc<Mutex<TextRerank>>,
    model_name: String,
}

#[cfg(feature = "fastembed")]
impl FastEmbedProvider {
    pub fn new(model_name: &str) -> Result<Self> {
        let model_enum = Self::map_model_name(model_name)?;

        let cache_dir = crate::storage::get_fastembed_cache_dir()
            .context("Failed to get FastEmbed cache directory")?;

        let model = TextRerank::try_new(
            RerankInitOptions::new(model_enum)
                .with_show_download_progress(true)
                .with_cache_dir(cache_dir),
        )
        .context("Failed to initialize FastEmbed reranker model")?;

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: model_name.to_string(),
        })
    }

    fn map_model_name(model_name: &str) -> Result<RerankerModel> {
        match model_name {
            "bge-reranker-base" | "BAAI/bge-reranker-base" => Ok(RerankerModel::BGERerankerBase),
            "bge-reranker-large" | "BAAI/bge-reranker-large" => Ok(RerankerModel::BGERerankerLarge),
            "jina-reranker-v1-turbo-en" | "jinaai/jina-reranker-v1-turbo-en" => {
                Ok(RerankerModel::JinaRerankerV1TurboEn)
            }
            "jina-reranker-v2-base-multilingual" | "jinaai/jina-reranker-v2-base-multilingual" => {
                Ok(RerankerModel::JinaRerankerV2BaseMultilingual)
            }
            _ => Err(anyhow::anyhow!(
                "Unsupported FastEmbed reranker model: '{}'. Supported: bge-reranker-base, bge-reranker-large, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual",
                model_name
            )),
        }
    }

    pub fn list_supported_models() -> Vec<String> {
        vec![
            "bge-reranker-base".to_string(),
            "bge-reranker-large".to_string(),
            "jina-reranker-v1-turbo-en".to_string(),
            "jina-reranker-v2-base-multilingual".to_string(),
        ]
    }
}

#[cfg(feature = "fastembed")]
#[async_trait::async_trait]
impl RerankProvider for FastEmbedProvider {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let query = query.to_string();
        let model = self.model.clone();

        let results = tokio::task::spawn_blocking(move || -> Result<Vec<RerankResult>> {
            let mut model = model.lock().unwrap();

            // Convert documents to string slices
            let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();

            // Rerank returns scores for each document
            let scores = model
                .rerank(&query, doc_refs, true, None)
                .context("Failed to rerank documents")?;

            // Create results with original indices
            let mut results: Vec<RerankResult> = scores
                .into_iter()
                .enumerate()
                .map(|(idx, score)| RerankResult {
                    index: idx,
                    document: documents[idx].clone(),
                    relevance_score: score.score as f64,
                })
                .collect();

            // Sort by relevance score descending
            results.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Apply top_k if specified
            if let Some(k) = top_k {
                results.truncate(k);
            }

            Ok(results)
        })
        .await
        .context("Reranking task failed")??;

        Ok(RerankResponse {
            results,
            total_tokens: 0, // FastEmbed doesn't provide token count
        })
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "bge-reranker-base"
                | "BAAI/bge-reranker-base"
                | "bge-reranker-large"
                | "BAAI/bge-reranker-large"
                | "jina-reranker-v1-turbo-en"
                | "jinaai/jina-reranker-v1-turbo-en"
                | "jina-reranker-v2-base-multilingual"
                | "jinaai/jina-reranker-v2-base-multilingual"
        )
    }
}

#[cfg(feature = "fastembed")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fastembed_provider_creation() {
        assert!(FastEmbedProvider::new("bge-reranker-base").is_ok());
        assert!(FastEmbedProvider::new("BAAI/bge-reranker-base").is_ok());
        assert!(FastEmbedProvider::new("bge-reranker-large").is_ok());
        assert!(FastEmbedProvider::new("jina-reranker-v1-turbo-en").is_ok());
        assert!(FastEmbedProvider::new("jina-reranker-v2-base-multilingual").is_ok());
        assert!(FastEmbedProvider::new("invalid-model").is_err());
    }

    #[test]
    fn test_fastembed_model_validation() {
        let models = [
            "bge-reranker-base",
            "bge-reranker-large",
            "jina-reranker-v1-turbo-en",
            "jina-reranker-v2-base-multilingual",
        ];
        for model in models {
            let provider = FastEmbedProvider::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }

    #[test]
    fn test_list_supported_models() {
        let models = FastEmbedProvider::list_supported_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"bge-reranker-base".to_string()));
    }
}
