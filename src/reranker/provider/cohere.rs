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
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-english-v2.0",
            "rerank-multilingual-v2.0",
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

        let mut request_body = json!({
            "query": query,
            "documents": documents,
            "model": self.model_name,
        });

        if let Some(k) = top_k {
            request_body["top_n"] = json!(k);
        }

        let response = HTTP_CLIENT
            .post("https://api.cohere.com/v1/rerank")
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

        // Cohere doesn't return documents in response, so we need to map them back
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
            "rerank-english-v3.0"
                | "rerank-multilingual-v3.0"
                | "rerank-english-v2.0"
                | "rerank-multilingual-v2.0"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cohere_provider_creation() {
        assert!(CohereProvider::new("rerank-english-v3.0").is_ok());
        assert!(CohereProvider::new("rerank-multilingual-v3.0").is_ok());
        assert!(CohereProvider::new("rerank-english-v2.0").is_ok());
        assert!(CohereProvider::new("rerank-multilingual-v2.0").is_ok());
        assert!(CohereProvider::new("invalid-model").is_err());
    }

    #[test]
    fn test_cohere_model_validation() {
        let models = [
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-english-v2.0",
            "rerank-multilingual-v2.0",
        ];
        for model in models {
            let provider = CohereProvider::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
