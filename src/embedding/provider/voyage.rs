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

//! Voyage AI embedding provider implementation

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// Voyage provider implementation for trait
pub struct VoyageProviderImpl {
    model_name: String,
    dimension: usize,
}

impl VoyageProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        // Validate model first - fail fast if unsupported
        let supported_models = [
            "voyage-3.5",
            "voyage-3.5-lite",
            "voyage-3-large",
            "voyage-code-2",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
            "voyage-2",
        ];

        if !supported_models.contains(&model) {
            return Err(anyhow::anyhow!(
                "Unsupported Voyage model: '{}'. Supported models: {:?}",
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
            "voyage-3.5" => 1024,
            "voyage-3.5-lite" => 1024,
            "voyage-3-large" => 1024,
            "voyage-code-2" => 1536,
            "voyage-code-3" => 1024,
            "voyage-finance-2" => 1024,
            "voyage-law-2" => 1024,
            "voyage-2" => 1024,
            _ => {
                // This should never be reached due to validation in new()
                panic!(
                    "Invalid Voyage model '{}' passed to get_model_dimension",
                    model
                );
            }
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for VoyageProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        VoyageProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        VoyageProvider::generate_embeddings_batch(texts, &self.model_name, input_type).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        // REAL validation - only support actual Voyage models, NO HALLUCINATIONS
        matches!(
            self.model_name.as_str(),
            "voyage-3.5"
                | "voyage-3.5-lite"
                | "voyage-3-large"
                | "voyage-code-2"
                | "voyage-code-3"
                | "voyage-finance-2"
                | "voyage-law-2"
                | "voyage-2"
        )
    }
}

/// Voyage AI provider implementation
pub struct VoyageProvider;

impl VoyageProvider {
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
        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .context("VOYAGE_API_KEY environment variable not set")?;

        // Build request body with optional input_type
        let mut request_body = json!({
            "input": texts,
            "model": model,
        });

        // Add input_type if specified (Voyage API native support)
        if let Some(input_type_str) = input_type.as_api_str() {
            request_body["input_type"] = json!(input_type_str);
        }

        let response = HTTP_CLIENT
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", voyage_api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("Voyage API error: {}", error_text));
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
