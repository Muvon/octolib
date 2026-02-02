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

//! Voyage AI reranker provider implementation

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::{RerankResponse, RerankResult};
use super::{RerankProvider, HTTP_CLIENT};

/// Voyage provider implementation for trait
pub struct VoyageProviderImpl {
    model_name: String,
}

impl VoyageProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        // Validate model first - fail fast if unsupported
        let supported_models = [
            "rerank-2.5",
            "rerank-2.5-lite",
            "rerank-2",
            "rerank-2-lite",
            "rerank-1",
            "rerank-lite-1",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Voyage reranker model: '{}'. Supported models: {:?}",
                model,
                supported_models
            ));
        }

        Ok(Self {
            model_name: model.to_string(),
        })
    }
}

#[async_trait::async_trait]
impl RerankProvider for VoyageProviderImpl {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        truncation: bool,
    ) -> Result<RerankResponse> {
        VoyageProvider::rerank(query, documents, &self.model_name, top_k, truncation).await
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "rerank-2.5"
                | "rerank-2.5-lite"
                | "rerank-2"
                | "rerank-2-lite"
                | "rerank-1"
                | "rerank-lite-1"
        )
    }
}

/// Voyage AI provider implementation
pub struct VoyageProvider;

#[derive(Debug, Deserialize)]
struct VoyageRerankResult {
    index: usize,
    document: String,
    relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct VoyageRerankResponse {
    results: Vec<VoyageRerankResult>,
    total_tokens: usize,
}

impl VoyageProvider {
    pub async fn rerank(
        query: &str,
        documents: Vec<String>,
        model: &str,
        top_k: Option<usize>,
        truncation: bool,
    ) -> Result<RerankResponse> {
        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .context("VOYAGE_API_KEY environment variable not set")?;

        // Build request body
        let mut request_body = json!({
            "query": query,
            "documents": documents,
            "model": model,
            "truncation": truncation,
        });

        // Add top_k if specified
        if let Some(k) = top_k {
            request_body["top_k"] = json!(k);
        }

        let response = HTTP_CLIENT
            .post("https://api.voyageai.com/v1/rerank")
            .header("Authorization", format!("Bearer {}", voyage_api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Voyage API error: {}", error_text));
        }

        let voyage_response: VoyageRerankResponse = response.json().await?;

        // Convert to our response format
        let results = voyage_response
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: r.document,
                relevance_score: r.relevance_score,
            })
            .collect();

        Ok(RerankResponse {
            results,
            total_tokens: voyage_response.total_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voyage_provider_creation() {
        // Test valid models
        assert!(VoyageProviderImpl::new("rerank-2.5").is_ok());
        assert!(VoyageProviderImpl::new("rerank-2.5-lite").is_ok());
        assert!(VoyageProviderImpl::new("rerank-2").is_ok());
        assert!(VoyageProviderImpl::new("rerank-2-lite").is_ok());
        assert!(VoyageProviderImpl::new("rerank-1").is_ok());
        assert!(VoyageProviderImpl::new("rerank-lite-1").is_ok());

        // Test invalid model
        assert!(VoyageProviderImpl::new("invalid-model").is_err());
    }

    #[test]
    fn test_voyage_model_validation() {
        let models = [
            "rerank-2.5",
            "rerank-2.5-lite",
            "rerank-2",
            "rerank-2-lite",
            "rerank-1",
            "rerank-lite-1",
        ];
        for model in models {
            let provider = VoyageProviderImpl::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
