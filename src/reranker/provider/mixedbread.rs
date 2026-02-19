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

//! Mixedbread AI reranker provider implementation

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::{RerankResponse, RerankResult};
use super::{RerankProvider, HTTP_CLIENT};

/// Mixedbread AI provider implementation
pub struct MixedbreadProvider {
    model_name: String,
}

impl MixedbreadProvider {
    pub fn new(model: &str) -> Result<Self> {
        let supported_models = [
            // v2 models (2025) - RL-trained, 100+ languages, 8K context
            "mxbai-rerank-large-v2",
            "mxbai-rerank-base-v2",
            // v1 models - open-source, Apache 2.0
            "mxbai-rerank-large-v1",
            "mxbai-rerank-base-v1",
            "mxbai-rerank-xsmall-v1",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Mixedbread reranker model: '{}'. Supported models: {:?}",
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
struct MixedbreadRerankItem {
    index: usize,
    score: f64,
}

#[derive(Debug, Deserialize)]
struct MixedbreadUsage {
    #[serde(default)]
    total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct MixedbreadRerankResponse {
    data: Vec<MixedbreadRerankItem>,
    #[serde(default)]
    usage: Option<MixedbreadUsage>,
}

#[async_trait::async_trait]
impl RerankProvider for MixedbreadProvider {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let api_key =
            std::env::var("MXBAI_API_KEY").context("MXBAI_API_KEY environment variable not set")?;

        let mut request_body = json!({
            "model": self.model_name,
            "query": query,
            "input": documents,
        });

        if let Some(k) = top_k {
            request_body["top_k"] = json!(k);
        }

        let response = HTTP_CLIENT
            .post("https://api.mixedbread.com/v1/reranking")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Mixedbread API error: {}", error_text));
        }

        let mxbai_response: MixedbreadRerankResponse = response.json().await?;

        let results = mxbai_response
            .data
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: documents[r.index].clone(),
                relevance_score: r.score,
            })
            .collect();

        let total_tokens = mxbai_response.usage.map(|u| u.total_tokens).unwrap_or(0);

        Ok(RerankResponse {
            results,
            total_tokens,
        })
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "mxbai-rerank-large-v2"
                | "mxbai-rerank-base-v2"
                | "mxbai-rerank-large-v1"
                | "mxbai-rerank-base-v1"
                | "mxbai-rerank-xsmall-v1"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixedbread_provider_creation() {
        assert!(MixedbreadProvider::new("mxbai-rerank-large-v2").is_ok());
        assert!(MixedbreadProvider::new("mxbai-rerank-base-v2").is_ok());
        assert!(MixedbreadProvider::new("mxbai-rerank-large-v1").is_ok());
        assert!(MixedbreadProvider::new("mxbai-rerank-base-v1").is_ok());
        assert!(MixedbreadProvider::new("mxbai-rerank-xsmall-v1").is_ok());
        assert!(MixedbreadProvider::new("invalid-model").is_err());
    }

    #[test]
    fn test_mixedbread_model_validation() {
        let models = [
            "mxbai-rerank-large-v2",
            "mxbai-rerank-base-v2",
            "mxbai-rerank-large-v1",
            "mxbai-rerank-base-v1",
            "mxbai-rerank-xsmall-v1",
        ];
        for model in models {
            let provider = MixedbreadProvider::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
