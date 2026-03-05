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

//! OpenRouter embedding provider implementation

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// Known OpenRouter embedding models with their dimensions.
/// OpenRouter proxies many providers - dimensions are model-specific.
const KNOWN_MODELS: &[(&str, usize)] = &[
    ("qwen/qwen3-embedding-8b", 4096),
    ("qwen/qwen3-embedding-4b", 2560),
    ("qwen/qwen3-embedding-0.6b", 1024),
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 3072),
    ("text-embedding-ada-002", 1536),
];

/// Fallback dimension when model is not in the known list.
/// OpenRouter supports arbitrary models, so we allow unknown ones with a sensible default.
const FALLBACK_DIMENSION: usize = 1536;

/// OpenRouter provider implementation for trait
pub struct OpenRouterProviderImpl {
    model_name: String,
    dimension: usize,
}

impl OpenRouterProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        let dimension = KNOWN_MODELS
            .iter()
            .find(|(name, _)| *name == model)
            .map(|(_, dim)| *dim)
            .unwrap_or(FALLBACK_DIMENSION);

        Ok(Self {
            model_name: model.to_string(),
            dimension,
        })
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenRouterProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        OpenRouterProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        OpenRouterProvider::generate_embeddings_batch(texts, &self.model_name, input_type).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        // OpenRouter proxies many providers - we allow any non-empty model name
        !self.model_name.is_empty()
    }
}

/// OpenRouter provider implementation
pub struct OpenRouterProvider;

impl OpenRouterProvider {
    pub async fn generate_embeddings(contents: &str, model: &str) -> Result<Vec<f32>> {
        let result =
            Self::generate_embeddings_batch(vec![contents.to_string()], model, InputType::None)
                .await?;
        result
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No embeddings returned"))
    }

    pub async fn generate_embeddings_batch(
        texts: Vec<String>,
        model: &str,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .context("OPENROUTER_API_KEY environment variable not set")?;

        // Apply input type prefixes - OpenRouter does not have a native input_type field
        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();

        let request_body = json!({
            "model": model,
            "input": processed_texts,
            "encoding_format": "float"
        });

        let response = HTTP_CLIENT
            .post("https://openrouter.ai/api/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenRouter API error: {}", error_text));
        }

        let response_json: Value = response.json().await?;

        let embeddings = response_json["data"]
            .as_array()
            .context("Failed to get embeddings array from OpenRouter response")?
            .iter()
            .map(|data| {
                data["embedding"]
                    .as_array()
                    .unwrap_or(&Vec::new())
                    .iter()
                    .map(|v| v.as_f64().unwrap_or_default() as f32)
                    .collect()
            })
            .collect();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openrouter_provider_creation() {
        // Known models should resolve correct dimensions
        let provider = OpenRouterProviderImpl::new("qwen/qwen3-embedding-8b").unwrap();
        assert_eq!(provider.get_dimension(), 4096);

        let provider = OpenRouterProviderImpl::new("text-embedding-3-small").unwrap();
        assert_eq!(provider.get_dimension(), 1536);

        let provider = OpenRouterProviderImpl::new("text-embedding-3-large").unwrap();
        assert_eq!(provider.get_dimension(), 3072);
    }

    #[test]
    fn test_unknown_model_fallback_dimension() {
        // Unknown models should fall back to FALLBACK_DIMENSION, not error
        let provider = OpenRouterProviderImpl::new("some/future-model").unwrap();
        assert_eq!(provider.get_dimension(), FALLBACK_DIMENSION);
    }

    #[test]
    fn test_model_supported() {
        let provider = OpenRouterProviderImpl::new("qwen/qwen3-embedding-8b").unwrap();
        assert!(provider.is_model_supported());

        let provider = OpenRouterProviderImpl::new("any/arbitrary-model").unwrap();
        assert!(provider.is_model_supported());
    }
}
