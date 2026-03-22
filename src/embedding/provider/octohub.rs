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

//! OctoHub embedding provider implementation.
//!
//! Proxies embedding requests through an OctoHub server which handles
//! model routing, logging, and multi-provider support.
//!
//! Configuration:
//! - `OCTOHUB_API_KEY`: Optional API key for OctoHub server authentication
//! - `OCTOHUB_API_URL`: OctoHub server base URL (default: http://127.0.0.1:8080)

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

const OCTOHUB_API_KEY_ENV: &str = "OCTOHUB_API_KEY";
const OCTOHUB_API_URL_ENV: &str = "OCTOHUB_API_URL";
const OCTOHUB_DEFAULT_URL: &str = "http://127.0.0.1:8080";

/// OctoHub embedding provider - routes through OctoHub proxy server
pub struct OctoHubEmbeddingProvider {
    model_name: String,
}

impl OctoHubEmbeddingProvider {
    pub fn new(model: &str) -> Result<Self> {
        if model.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }
        Ok(Self {
            model_name: model.to_string(),
        })
    }

    fn api_url() -> String {
        let base =
            std::env::var(OCTOHUB_API_URL_ENV).unwrap_or_else(|_| OCTOHUB_DEFAULT_URL.to_string());
        format!("{}/v1/embeddings", base.trim_end_matches('/'))
    }

    fn api_key() -> Option<String> {
        std::env::var(OCTOHUB_API_KEY_ENV).ok()
    }

    async fn call_api(&self, input: Value) -> Result<Value> {
        let url = Self::api_url();

        let body = json!({
            "model": self.model_name,
            "input": input,
        });

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
            .with_context(|| format!("Failed to connect to OctoHub at {}", url))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "OctoHub embedding API error ({}): {}",
                status,
                error_text
            ));
        }

        response
            .json()
            .await
            .context("Failed to parse OctoHub embedding response")
    }

    fn extract_embeddings(response: &Value) -> Result<Vec<Vec<f32>>> {
        let data = response["data"]
            .as_array()
            .context("Missing 'data' array in response")?;

        data.iter()
            .map(|item| {
                let embedding = item["embedding"]
                    .as_array()
                    .context("Missing 'embedding' in data item")?;
                Ok(embedding
                    .iter()
                    .map(|v| v.as_f64().unwrap_or_default() as f32)
                    .collect())
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OctoHubEmbeddingProvider {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let response = self.call_api(json!(text)).await?;
        let embeddings = Self::extract_embeddings(&response)?;
        embeddings
            .into_iter()
            .next()
            .context("No embeddings returned from OctoHub")
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        _input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let response = self.call_api(json!(texts)).await?;
        Self::extract_embeddings(&response)
    }

    /// Dimension is unknown until the underlying provider responds;
    /// callers should not rely on this for OctoHub.
    fn get_dimension(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        assert!(OctoHubEmbeddingProvider::new("voyage-3.5").is_ok());
        assert!(OctoHubEmbeddingProvider::new("any-model").is_ok());
        assert!(OctoHubEmbeddingProvider::new("").is_err());
    }

    #[test]
    fn test_api_url_default() {
        // Clear env to test default
        std::env::remove_var(OCTOHUB_API_URL_ENV);
        assert_eq!(
            OctoHubEmbeddingProvider::api_url(),
            "http://127.0.0.1:8080/v1/embeddings"
        );
    }

    #[test]
    fn test_extract_embeddings() {
        let response = json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ]
        });
        let result = OctoHubEmbeddingProvider::extract_embeddings(&response).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![0.1_f32, 0.2, 0.3]);
        assert_eq!(result[1], vec![0.4_f32, 0.5, 0.6]);
    }
}
