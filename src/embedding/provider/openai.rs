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

//! OpenAI embedding provider implementation

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// OpenAI provider implementation for trait
pub struct OpenAIProviderImpl {
    model_name: String,
    dimension: usize,
}

impl OpenAIProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        // Validate model first - fail fast if unsupported
        let supported_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported OpenAI model: '{}'. Supported models: {:?}",
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
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => unreachable!("Invalid OpenAI model '{}' passed to get_model_dimension - this is a bug as model should be validated in new()", model),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAIProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        OpenAIProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        OpenAIProvider::generate_embeddings_batch(texts, &self.model_name, input_type).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        // REAL validation - only support actual OpenAI models, NO HALLUCINATIONS
        matches!(
            self.model_name.as_str(),
            "text-embedding-3-small" | "text-embedding-3-large" | "text-embedding-ada-002"
        )
    }
}

/// OpenAI provider implementation
pub struct OpenAIProvider;

impl OpenAIProvider {
    pub async fn generate_embeddings(contents: &str, model: &str) -> Result<Vec<f32>> {
        let result =
            Self::generate_embeddings_batch(vec![contents.to_string()], model, InputType::None)
                .await?;
        result
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("No embeddings found"))
    }

    pub async fn generate_embeddings_batch(
        texts: Vec<String>,
        model: &str,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        let openai_api_key = std::env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY environment variable not set")?;

        // Apply input type prefixes since OpenAI doesn't have native input_type support
        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();

        // Build request body
        let request_body = json!({
            "input": processed_texts,
            "model": model,
            "encoding_format": "float"
        });

        let response = HTTP_CLIENT
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", openai_api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error: {}", error_text));
        }

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
    fn test_openai_provider_creation() {
        // Test valid models
        assert!(OpenAIProviderImpl::new("text-embedding-3-small").is_ok());
        assert!(OpenAIProviderImpl::new("text-embedding-3-large").is_ok());
        assert!(OpenAIProviderImpl::new("text-embedding-ada-002").is_ok());

        // Test invalid model
        assert!(OpenAIProviderImpl::new("invalid-model").is_err());
    }

    #[test]
    fn test_model_dimensions() {
        let provider_small = OpenAIProviderImpl::new("text-embedding-3-small").unwrap();
        assert_eq!(provider_small.get_dimension(), 1536);

        let provider_large = OpenAIProviderImpl::new("text-embedding-3-large").unwrap();
        assert_eq!(provider_large.get_dimension(), 3072);

        let provider_ada = OpenAIProviderImpl::new("text-embedding-ada-002").unwrap();
        assert_eq!(provider_ada.get_dimension(), 1536);
    }

    #[test]
    fn test_model_validation() {
        let provider_valid = OpenAIProviderImpl::new("text-embedding-3-small").unwrap();
        assert!(provider_valid.is_model_supported());

        // This would panic if we tried to create an invalid model, so we test indirectly
        let supported_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ];
        for model in supported_models {
            let provider = OpenAIProviderImpl::new(model).unwrap();
            assert!(provider.is_model_supported());
        }
    }
}
