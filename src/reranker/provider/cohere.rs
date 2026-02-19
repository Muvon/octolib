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

//! Cohere reranker provider implementation

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::{RerankResponse, RerankResult};
use super::{RerankProvider, HTTP_CLIENT};

/// Cohere provider implementation
pub struct CohereProvider {
    model_name: String,
}

impl CohereProvider {
    pub fn new(model: &str) -> Result<Self> {
        let supported_models = [
            // v4 models (Dec 2025) - use /v2/rerank endpoint
            "rerank-v4.0-pro",
            "rerank-v4.0-fast",
            // v3 models - use /v1/rerank endpoint
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            // Legacy alias still accepted by Cohere
            "rerank-v3.5",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Cohere reranker model: '{}'. Supported models: {:?}",
                model,
                supported_models
            ));
        }

        Ok(Self {
            model_name: model.to_string(),
        })
    }
    /// Returns true for v4 models that use the /v2/rerank endpoint
    fn is_v4_model(&self) -> bool {
        self.model_name.starts_with("rerank-v4.")
    }
}

#[derive(Debug, Deserialize)]
struct CohereRerankResult {
    index: usize,
    #[serde(rename = "relevance_score")]
    relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct CohereRerankResponse {
    results: Vec<CohereRerankResult>,
}

#[async_trait::async_trait]
impl RerankProvider for CohereProvider {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let cohere_api_key = std::env::var("COHERE_API_KEY")
            .context("COHERE_API_KEY environment variable not set")?;

        // v4 models use the v2 API endpoint; v3 and earlier use v1
        let endpoint = if self.is_v4_model() {
            "https://api.cohere.com/v2/rerank"
        } else {
            "https://api.cohere.com/v1/rerank"
        };

        let mut request_body = json!({
            "query": query,
            "documents": documents,
            "model": self.model_name,
        });

        if let Some(k) = top_k {
            request_body["top_n"] = json!(k);
        }

        let response = HTTP_CLIENT
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", cohere_api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Cohere API error: {}", error_text));
        }

        let cohere_response: CohereRerankResponse = response.json().await?;

        // Cohere doesn't return documents in response, so we map them back by index
        let results = cohere_response
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: documents[r.index].clone(),
                relevance_score: r.relevance_score,
            })
            .collect();

        Ok(RerankResponse {
            results,
            total_tokens: 0, // Cohere doesn't provide token count in response
        })
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "rerank-v4.0-pro"
                | "rerank-v4.0-fast"
                | "rerank-english-v3.0"
                | "rerank-multilingual-v3.0"
                | "rerank-v3.5"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_provider_creation() {
        assert!(CohereProvider::new("rerank-v4.0-pro").is_ok());
        assert!(CohereProvider::new("rerank-v4.0-fast").is_ok());
        assert!(CohereProvider::new("rerank-english-v3.0").is_ok());
        assert!(CohereProvider::new("rerank-multilingual-v3.0").is_ok());
        assert!(CohereProvider::new("rerank-v3.5").is_ok());
        assert!(CohereProvider::new("invalid-model").is_err());
        // Removed deprecated v2 models
        assert!(CohereProvider::new("rerank-english-v2.0").is_err());
        assert!(CohereProvider::new("rerank-multilingual-v2.0").is_err());
    }

    #[test]
    fn test_cohere_v4_endpoint_routing() {
        let v4_pro = CohereProvider::new("rerank-v4.0-pro").unwrap();
        assert!(v4_pro.is_v4_model());

        let v4_fast = CohereProvider::new("rerank-v4.0-fast").unwrap();
        assert!(v4_fast.is_v4_model());

        let v3 = CohereProvider::new("rerank-english-v3.0").unwrap();
        assert!(!v3.is_v4_model());
    }

    #[test]
    fn test_cohere_model_validation() {
        let models = [
            "rerank-v4.0-pro",
            "rerank-v4.0-fast",
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-v3.5",
        ];
        for model in models {
            let provider = CohereProvider::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
