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
use super::super::EmbeddingUsage;
use super::{http_client, EmbeddingProvider};

const OCTOHUB_API_KEY_ENV: &str = "OCTOHUB_API_KEY";
const OCTOHUB_API_URL_ENV: &str = "OCTOHUB_API_URL";
const OCTOHUB_DEFAULT_URL: &str = "https://hub.octomind.run";

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

        let mut req = http_client()
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
            let status = response.status().as_u16();
            let error_text = response.text().await.unwrap_or_default();
            // A rejected credential is the one failure the user can act on, so
            // say how: `octomind login` both obtains the key and stores it.
            if status == 401 || status == 403 {
                let hint = match Self::api_key() {
                    Some(k) if !k.trim().is_empty() => "the stored OctoHub key was rejected (revoked, or replaced by a newer login) — run `octomind login` to sign in again",
                    _ => "no OctoHub key is set — run `octomind login` to sign in",
                };
                return Err(anyhow::anyhow!(
                    "OctoHub embedding API error ({status}): {hint}. Server said: {error_text}"
                ));
            }
            return Err(anyhow::anyhow!(
                "OctoHub embedding API error ({status}): {error_text}"
            ));
        }

        response
            .json()
            .await
            .context("Failed to parse OctoHub embedding response")
    }

    fn parse_single(response: &Value) -> Result<Vec<f32>> {
        response
            .as_array()
            .context("Expected a JSON array for single embedding response")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|f| f as f32)
                    .context("Embedding value is not a number")
            })
            .collect()
    }

    fn parse_batch(response: &Value) -> Result<Vec<Vec<f32>>> {
        response
            .as_array()
            .context("Expected a JSON array for batch embedding response")?
            .iter()
            .map(|row| {
                row.as_array()
                    .context("Expected inner array in batch embedding response")?
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .context("Embedding value is not a number")
                    })
                    .collect()
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OctoHubEmbeddingProvider {
    async fn generate_embedding(&self, text: &str) -> Result<(Vec<f32>, EmbeddingUsage)> {
        let response = self.call_api(json!(text)).await?;
        let vector = Self::parse_single(&response)?;
        // OctoHub's embedding response is a bare vector array (no usage envelope),
        // so estimate; the underlying provider bills the real tokens upstream.
        let usage = EmbeddingUsage::estimate(&self.model_name, &[text.to_string()]);
        Ok((vector, usage))
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        _input_type: InputType,
    ) -> Result<(Vec<Vec<f32>>, EmbeddingUsage)> {
        if texts.is_empty() {
            return Ok((Vec::new(), EmbeddingUsage::from_tokens(&self.model_name, 0)));
        }
        let usage = EmbeddingUsage::estimate(&self.model_name, &texts);
        let response = self.call_api(json!(texts)).await?;
        let vectors = Self::parse_batch(&response)?;
        Ok((vectors, usage))
    }

    /// Dimension is unknown until the underlying provider responds;
    /// callers should not rely on this for OctoHub.
    fn get_dimension(&self) -> usize {
        0
    }
}

#[cfg(test)]
#[path = "octohub_tests.rs"]
mod tests;
