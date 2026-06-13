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

//! Local reranker provider for OpenAI-compatible servers with rerank support.
//!
//! Works with llama.cpp server, vLLM, and HuggingFace TEI (Text Embeddings Inference)
//! that expose a Jina-compatible `/v1/rerank` endpoint.
//!
//! Note: Ollama does NOT support reranking — it has no `/v1/rerank` endpoint.
//!
//! ## Configuration
//! - `LOCAL_RERANK_API_URL`: Full rerank endpoint URL (default: `http://localhost:8012/v1/rerank`, targets llama.cpp server)
//! - `LOCAL_RERANK_API_KEY`: Optional API key

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::{RerankResponse, RerankResult};
use super::{RerankProvider, HTTP_CLIENT};

const LOCAL_RERANK_API_KEY_ENV: &str = "LOCAL_RERANK_API_KEY";
const LOCAL_RERANK_API_URL_ENV: &str = "LOCAL_RERANK_API_URL";
const LOCAL_RERANK_API_URL: &str = "http://localhost:8012/v1/rerank";

/// Local reranker provider for OpenAI-compatible servers.
pub struct LocalRerankerProvider {
    model_name: String,
}

impl LocalRerankerProvider {
    pub fn new(model: &str) -> Result<Self> {
        if model.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }
        Ok(Self {
            model_name: model.to_string(),
        })
    }

    fn api_url() -> String {
        std::env::var(LOCAL_RERANK_API_URL_ENV)
            .unwrap_or_else(|_| LOCAL_RERANK_API_URL.to_string())
    }

    fn api_key() -> Option<String> {
        std::env::var(LOCAL_RERANK_API_KEY_ENV).ok()
    }
}

#[derive(Debug, Deserialize)]
struct LocalRerankResponse {
    results: Vec<LocalRerankResult>,
    #[serde(default)]
    usage: Option<LocalRerankUsage>,
}

#[derive(Debug, Deserialize)]
struct LocalRerankResult {
    index: usize,
    #[serde(default)]
    relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct LocalRerankUsage {
    #[serde(default)]
    total_tokens: usize,
}

#[async_trait::async_trait]
impl RerankProvider for LocalRerankerProvider {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let url = Self::api_url();

        let mut body = json!({
            "model": self.model_name,
            "query": query,
            "documents": documents,
        });

        if let Some(k) = top_k {
            body["top_n"] = json!(k);
        }

        let mut req = HTTP_CLIENT
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(key) = Self::api_key() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response = req
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Failed to connect to local rerank server at {}", url))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Local rerank API error ({}): {}",
                status,
                error_text
            ));
        }

        let result: LocalRerankResponse = response
            .json()
            .await
            .context("Failed to parse local rerank response")?;

        let mut results: Vec<RerankResult> = result
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: documents.get(r.index).cloned().unwrap_or_default(),
                relevance_score: r.relevance_score,
            })
            .collect();

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_tokens = result.usage.map(|u| u.total_tokens).unwrap_or(0);

        Ok(RerankResponse {
            results,
            total_tokens,
        })
    }

    fn is_model_supported(&self) -> bool {
        !self.model_name.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        assert!(LocalRerankerProvider::new("bge-reranker-v2-m3").is_ok());
        assert!(LocalRerankerProvider::new("any-model-name").is_ok());
        assert!(LocalRerankerProvider::new("").is_err());
    }

    #[test]
    fn test_api_url_default() {
        std::env::remove_var(LOCAL_RERANK_API_URL_ENV);
        assert_eq!(
            LocalRerankerProvider::api_url(),
            "http://localhost:8012/v1/rerank"
        );
    }

    #[test]
    fn test_is_model_supported() {
        let provider = LocalRerankerProvider::new("bge-reranker-v2-m3").unwrap();
        assert!(provider.is_model_supported());
    }
}
