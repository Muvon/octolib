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

//! Local embedding provider for OpenAI-compatible local servers.
//!
//! Works with Ollama, llama.cpp server, LM Studio, vLLM, LocalAI, and any
//! server exposing the OpenAI-compatible `/v1/embeddings` endpoint.
//!
//! ## Configuration
//! - `LOCAL_EMBED_API_URL`: Full embedding endpoint URL (default: `http://localhost:11434/v1/embeddings`)
//! - `LOCAL_EMBED_API_KEY`: Optional API key

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

const LOCAL_EMBED_API_KEY_ENV: &str = "LOCAL_EMBED_API_KEY";
const LOCAL_EMBED_API_URL_ENV: &str = "LOCAL_EMBED_API_URL";
const LOCAL_EMBED_API_URL: &str = "http://localhost:11434/v1/embeddings";

/// Local embedding provider for OpenAI-compatible local servers.
pub struct LocalEmbeddingProvider {
    model_name: String,
}

impl LocalEmbeddingProvider {
    pub fn new(model: &str) -> Result<Self> {
        if model.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }
        Ok(Self {
            model_name: model.to_string(),
        })
    }

    fn api_url() -> String {
        std::env::var(LOCAL_EMBED_API_URL_ENV).unwrap_or_else(|_| LOCAL_EMBED_API_URL.to_string())
    }

    fn api_key() -> Option<String> {
        std::env::var(LOCAL_EMBED_API_KEY_ENV).ok()
    }
}

#[derive(Debug, Deserialize)]
struct LocalEmbeddingResponse {
    data: Vec<LocalEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct LocalEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[async_trait::async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let texts = vec![text.to_string()];
        let batch = self
            .generate_embeddings_batch(texts, InputType::None)
            .await?;
        batch
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();

        let url = Self::api_url();

        let body = json!({
            "model": self.model_name,
            "input": processed_texts,
        });

        let mut req = HTTP_CLIENT
            .post(&url)
            .header("Content-Type", "application/json");

        if let Some(key) = Self::api_key() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let response =
            req.json(&body).send().await.with_context(|| {
                format!("Failed to connect to local embedding server at {}", url)
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "Local embedding API error ({}): {}",
                status,
                error_text
            ));
        }

        let result: LocalEmbeddingResponse = response
            .json()
            .await
            .context("Failed to parse local embedding response")?;

        let mut embeddings: Vec<(usize, Vec<f32>)> = result
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(i, _)| *i);

        Ok(embeddings.into_iter().map(|(_, e)| e).collect())
    }

    fn get_dimension(&self) -> usize {
        0
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
        assert!(LocalEmbeddingProvider::new("nomic-embed-text").is_ok());
        assert!(LocalEmbeddingProvider::new("any-model-name").is_ok());
        assert!(LocalEmbeddingProvider::new("").is_err());
    }

    #[test]
    fn test_api_url_default() {
        std::env::remove_var(LOCAL_EMBED_API_URL_ENV);
        assert_eq!(
            LocalEmbeddingProvider::api_url(),
            "http://localhost:11434/v1/embeddings"
        );
    }

    #[test]
    fn test_is_model_supported() {
        let provider = LocalEmbeddingProvider::new("nomic-embed-text").unwrap();
        assert!(provider.is_model_supported());
        assert_eq!(provider.get_dimension(), 0);
    }
}
