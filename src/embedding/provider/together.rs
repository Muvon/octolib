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

//! Together.ai embedding provider implementation
//!
//! Supports embedding models hosted on Together.ai infrastructure.

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use std::env;

use super::{EmbeddingProvider, HTTP_CLIENT};
use crate::embedding::types::InputType;

const TOGETHER_API_KEY_ENV: &str = "TOGETHER_API_KEY";
const TOGETHER_EMBEDDING_URL: &str = "https://api.together.xyz/v1/embeddings";

// Supported embedding models with their dimensions
const EMBEDDING_MODELS: &[(&str, usize)] = &[("intfloat/multilingual-e5-large-instruct", 1024)];

/// Together.ai embedding provider implementation
pub struct TogetherProviderImpl {
    model_name: String,
    dimension: usize,
}

impl TogetherProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        // Validate model is supported
        let dimension = Self::get_model_dimension(model);
        if dimension == 0 {
            return Err(anyhow::anyhow!(
                "Unsupported Together.ai embedding model: '{}'. Supported models: {}",
                model,
                EMBEDDING_MODELS
                    .iter()
                    .map(|(m, _)| *m)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        Ok(Self {
            model_name: model.to_string(),
            dimension,
        })
    }

    fn get_model_dimension(model: &str) -> usize {
        EMBEDDING_MODELS
            .iter()
            .find(|(m, _)| m == &model)
            .map(|(_, dim)| *dim)
            .unwrap_or(0)
    }
}

#[async_trait]
impl EmbeddingProvider for TogetherProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        TogetherProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        _input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        TogetherProvider::generate_embeddings_batch(texts, &self.model_name).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        Self::get_model_dimension(&self.model_name) > 0
    }
}

/// Together.ai embedding provider (static methods)
pub struct TogetherProvider;

impl TogetherProvider {
    pub async fn generate_embeddings(contents: &str, model: &str) -> Result<Vec<f32>> {
        let api_key = env::var(TOGETHER_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Together.ai API key not found. Set {} environment variable.",
                TOGETHER_API_KEY_ENV
            )
        })?;

        let response = HTTP_CLIENT
            .post(TOGETHER_EMBEDDING_URL)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": model,
                "input": contents
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Together.ai embedding API error: {}",
                error_text
            ));
        }

        let result: TogetherEmbeddingResponse = response.json().await?;

        result
            .data
            .first()
            .map(|d| d.embedding.clone())
            .ok_or_else(|| anyhow::anyhow!("No embeddings found in response"))
    }

    pub async fn generate_embeddings_batch(
        texts: Vec<String>,
        model: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let api_key = env::var(TOGETHER_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Together.ai API key not found. Set {} environment variable.",
                TOGETHER_API_KEY_ENV
            )
        })?;

        let response = HTTP_CLIENT
            .post(TOGETHER_EMBEDDING_URL)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": model,
                "input": texts
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!(
                "Together.ai embedding API error: {}",
                error_text
            ));
        }

        let result: TogetherEmbeddingResponse = response.json().await?;

        // Sort by index to ensure correct order
        let mut embeddings: Vec<(usize, Vec<f32>)> = result
            .data
            .into_iter()
            .map(|d| (d.index, d.embedding))
            .collect();
        embeddings.sort_by_key(|(i, _)| *i);

        Ok(embeddings.into_iter().map(|(_, e)| e).collect())
    }
}

#[derive(Debug, Deserialize)]
struct TogetherEmbeddingResponse {
    data: Vec<TogetherEmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct TogetherEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        // Test valid models
        let provider = TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct");
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().get_dimension(), 1024);

        // Test invalid model
        let invalid = TogetherProviderImpl::new("invalid-model");
        assert!(invalid.is_err());
    }

    #[test]
    fn test_model_dimensions() {
        let provider =
            TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct").unwrap();
        assert_eq!(provider.get_dimension(), 1024);
    }

    #[test]
    fn test_model_validation() {
        let provider_valid =
            TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct").unwrap();
        assert!(provider_valid.is_model_supported());

        let provider_invalid = TogetherProviderImpl::new("unknown-model");
        assert!(provider_invalid.is_err());
    }
}
