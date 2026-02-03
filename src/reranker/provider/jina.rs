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

//! Jina AI reranker provider implementation

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::{RerankResponse, RerankResult};
use super::{RerankProvider, HTTP_CLIENT};

/// Jina AI provider implementation
pub struct JinaProvider {
    model_name: String,
}

impl JinaProvider {
    pub fn new(model: &str) -> Result<Self> {
        let supported_models = [
            "jina-reranker-v3",
            "jina-reranker-v2-base-multilingual",
            "jina-reranker-v1-base-en",
            "jina-colbert-v2",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Jina reranker model: '{}'. Supported models: {:?}",
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
struct JinaRerankResult {
    index: usize,
    document: Option<String>,
    #[serde(rename = "relevance_score")]
    relevance_score: f64,
}

#[derive(Debug, Deserialize)]
struct JinaRerankResponse {
    results: Vec<JinaRerankResult>,
    #[serde(default)]
    usage: JinaUsage,
}

#[derive(Debug, Deserialize, Default)]
struct JinaUsage {
    #[serde(default)]
    total_tokens: usize,
}

#[async_trait::async_trait]
impl RerankProvider for JinaProvider {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let jina_api_key =
            std::env::var("JINA_API_KEY").context("JINA_API_KEY environment variable not set")?;

        let mut request_body = json!({
            "query": query,
            "documents": documents,
            "model": self.model_name,
        });

        if let Some(k) = top_k {
            request_body["top_n"] = json!(k);
        }

        let response = HTTP_CLIENT
            .post("https://api.jina.ai/v1/rerank")
            .header("Authorization", format!("Bearer {}", jina_api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Jina API error: {}", error_text));
        }

        let jina_response: JinaRerankResponse = response.json().await?;

        let results = jina_response
            .results
            .into_iter()
            .map(|r| RerankResult {
                index: r.index,
                document: r.document.unwrap_or_else(|| documents[r.index].clone()),
                relevance_score: r.relevance_score,
            })
            .collect();

        Ok(RerankResponse {
            results,
            total_tokens: jina_response.usage.total_tokens,
        })
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "jina-reranker-v3"
                | "jina-reranker-v2-base-multilingual"
                | "jina-reranker-v1-base-en"
                | "jina-colbert-v2"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jina_provider_creation() {
        assert!(JinaProvider::new("jina-reranker-v3").is_ok());
        assert!(JinaProvider::new("jina-reranker-v2-base-multilingual").is_ok());
        assert!(JinaProvider::new("jina-reranker-v1-base-en").is_ok());
        assert!(JinaProvider::new("jina-colbert-v2").is_ok());
        assert!(JinaProvider::new("invalid-model").is_err());
    }

    #[test]
    fn test_jina_model_validation() {
        let models = [
            "jina-reranker-v3",
            "jina-reranker-v2-base-multilingual",
            "jina-reranker-v1-base-en",
            "jina-colbert-v2",
        ];
        for model in models {
            let provider = JinaProvider::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
