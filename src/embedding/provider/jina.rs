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

//! Jina AI embedding provider implementation

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// Jina provider implementation for trait
pub struct JinaProviderImpl {
    model_name: String,
    dimension: usize,
}

impl JinaProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        // Validate model first - fail fast if unsupported
        let supported_models = [
            "jina-embeddings-v4",
            "jina-clip-v2",
            "jina-embeddings-v3",
            "jina-clip-v1",
            "jina-embeddings-v2-base-es",
            "jina-embeddings-v2-base-code",
            "jina-embeddings-v2-base-de",
            "jina-embeddings-v2-base-zh",
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-small-en",
            "jina-colbert-v2",
            "jina-colbert-v2-96",
            "jina-colbert-v2-64",
            "jina-code-embeddings-0.5b",
            "jina-code-embeddings-1.5b",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Jina model: '{}'. Supported models: {:?}",
                model,
                supported_models
            ));
        }

        let dimension = Self::get_model_dimension(model);
        Ok(Self {
            model_name: model.to_string(),
            dimension,
        })
    }

    fn get_model_dimension(model: &str) -> usize {
        match model {
            "jina-embeddings-v4" => 2048,
            "jina-clip-v2" => 1024,
            "jina-embeddings-v3" => 1024,
            "jina-clip-v1" => 768,
            "jina-embeddings-v2-base-es" => 768,
            "jina-embeddings-v2-base-code" => 768,
            "jina-embeddings-v2-base-de" => 768,
            "jina-embeddings-v2-base-zh" => 768,
            "jina-embeddings-v2-base-en" => 768,
            "jina-embeddings-v2-small-en" => 512,
            "jina-colbert-v2" => 128,
            "jina-colbert-v2-96" => 96,
            "jina-colbert-v2-64" => 64,
            "jina-code-embeddings-0.5b" => 1024,
            "jina-code-embeddings-1.5b" => 1024,
            _ => unreachable!("Invalid Jina model '{}' passed to get_model_dimension - this is a bug as model should be validated in new()", model),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for JinaProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        JinaProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        // Apply prefix manually for Jina (doesn't support input_type API)
        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();
        JinaProvider::generate_embeddings_batch(processed_texts, &self.model_name).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        // REAL validation - only support actual Jina models
        matches!(
            self.model_name.as_str(),
            "jina-embeddings-v4"
                | "jina-clip-v2"
                | "jina-embeddings-v3"
                | "jina-clip-v1"
                | "jina-embeddings-v2-base-es"
                | "jina-embeddings-v2-base-code"
                | "jina-embeddings-v2-base-de"
                | "jina-embeddings-v2-base-zh"
                | "jina-embeddings-v2-base-en"
                | "jina-embeddings-v2-small-en"
                | "jina-colbert-v2"
                | "jina-colbert-v2-96"
                | "jina-colbert-v2-64"
                | "jina-code-embeddings-0.5b"
                | "jina-code-embeddings-1.5b"
        )
    }
}

/// Jina provider implementation
pub struct JinaProvider;

impl JinaProvider {
    pub async fn generate_embeddings(contents: &str, model: &str) -> Result<Vec<f32>> {
        let result = Self::generate_embeddings_batch(vec![contents.to_string()], model).await?;
        result
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No embeddings found"))
    }

    pub async fn generate_embeddings_batch(
        texts: Vec<String>,
        model: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let jina_api_key =
            std::env::var("JINA_API_KEY").context("JINA_API_KEY environment variable not set")?;

        let response = HTTP_CLIENT
            .post("https://api.jina.ai/v1/embeddings")
            .header("Authorization", format!("Bearer {}", jina_api_key))
            .json(&json!({
                "input": texts,
                "model": model,
            }))
            .send()
            .await?;

        let response_json: Value = response.json().await?;

        let embeddings = response_json["data"]
            .as_array()
            .context("Failed to get embeddings array")?
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
    fn test_jina_provider_creation() {
        // Test valid models
        assert!(JinaProviderImpl::new("jina-embeddings-v4").is_ok());
        assert!(JinaProviderImpl::new("jina-embeddings-v3").is_ok());
        assert!(JinaProviderImpl::new("jina-clip-v2").is_ok());
        assert!(JinaProviderImpl::new("jina-colbert-v2").is_ok());
        assert!(JinaProviderImpl::new("jina-code-embeddings-0.5b").is_ok());

        // Test invalid model
        assert!(JinaProviderImpl::new("invalid-model").is_err());
    }

    #[test]
    fn test_jina_model_dimensions() {
        assert_eq!(
            JinaProviderImpl::new("jina-embeddings-v4")
                .unwrap()
                .get_dimension(),
            2048
        );
        assert_eq!(
            JinaProviderImpl::new("jina-embeddings-v3")
                .unwrap()
                .get_dimension(),
            1024
        );
        assert_eq!(
            JinaProviderImpl::new("jina-clip-v2")
                .unwrap()
                .get_dimension(),
            1024
        );
        assert_eq!(
            JinaProviderImpl::new("jina-embeddings-v2-small-en")
                .unwrap()
                .get_dimension(),
            512
        );
        assert_eq!(
            JinaProviderImpl::new("jina-colbert-v2")
                .unwrap()
                .get_dimension(),
            128
        );
        assert_eq!(
            JinaProviderImpl::new("jina-colbert-v2-96")
                .unwrap()
                .get_dimension(),
            96
        );
        assert_eq!(
            JinaProviderImpl::new("jina-colbert-v2-64")
                .unwrap()
                .get_dimension(),
            64
        );
        assert_eq!(
            JinaProviderImpl::new("jina-code-embeddings-0.5b")
                .unwrap()
                .get_dimension(),
            1024
        );
        assert_eq!(
            JinaProviderImpl::new("jina-code-embeddings-1.5b")
                .unwrap()
                .get_dimension(),
            1024
        );
    }

    #[test]
    fn test_jina_model_validation() {
        let models = [
            "jina-embeddings-v4",
            "jina-embeddings-v3",
            "jina-clip-v2",
            "jina-clip-v1",
            "jina-embeddings-v2-small-en",
            "jina-colbert-v2",
            "jina-colbert-v2-96",
            "jina-colbert-v2-64",
            "jina-code-embeddings-0.5b",
            "jina-code-embeddings-1.5b",
        ];
        for model in models {
            let provider = JinaProviderImpl::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
