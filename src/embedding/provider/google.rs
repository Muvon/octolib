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

//! Google AI embedding provider implementation

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// Google provider implementation for trait
pub struct GoogleProviderImpl {
    model_name: String,
    dimension: usize,
}

impl GoogleProviderImpl {
    pub fn new(model: &str) -> Result<Self> {
        let dimension = Self::get_model_dimension(model)?;
        Ok(Self {
            model_name: model.to_string(),
            dimension,
        })
    }

    fn get_model_dimension(model: &str) -> Result<usize> {
        match model {
			"gemini-embedding-001" => Ok(3072),  // Up to 3072 dimensions, state-of-the-art performance
			"text-embedding-005" => Ok(768),     // Specialized in English and code tasks
			"text-multilingual-embedding-002" => Ok(768), // Specialized in multilingual tasks
			_ => Err(anyhow::anyhow!(
				"Unsupported Google model: '{}'. Supported models: gemini-embedding-001 (3072d), text-embedding-005 (768d), text-multilingual-embedding-002 (768d)",
				model
			)),
		}
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for GoogleProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        GoogleProvider::generate_embeddings(text, &self.model_name).await
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<Vec<Vec<f32>>> {
        // Apply prefix manually for Google (doesn't support input_type API)
        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();
        GoogleProvider::generate_embeddings_batch(processed_texts, &self.model_name).await
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    fn is_model_supported(&self) -> bool {
        matches!(
            self.model_name.as_str(),
            "gemini-embedding-001" | "text-embedding-005" | "text-multilingual-embedding-002"
        )
    }
}

/// Google provider implementation
pub struct GoogleProvider;

impl GoogleProvider {
    /// Get list of supported models for dynamic discovery
    pub fn get_supported_models() -> Vec<&'static str> {
        vec![
            "gemini-embedding-001",
            "text-embedding-005",
            "text-multilingual-embedding-002",
        ]
    }
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
        let google_api_key = std::env::var("GOOGLE_API_KEY")
            .context("GOOGLE_API_KEY environment variable not set")?;

        // For batch processing, we'll need to send individual requests as Google's API structure is different
        let mut all_embeddings = Vec::new();

        for text in texts {
            let response = HTTP_CLIENT
				.post(format!("https://generativelanguage.googleapis.com/v1beta/models/{}:embedContent?key={}", model, google_api_key))
				.header("Content-Type", "application/json")
				.json(&json!({
					"content": {
						"parts": [{
							"text": text
						}]
					}
				}))
				.send()
			.await?;

            if !response.status().is_success() {
                let error_text = response.text().await?;
                return Err(anyhow::anyhow!("Google API error: {}", error_text));
            }

            let response_json: Value = response.json().await?;

            let embedding = response_json["embedding"]["values"]
                .as_array()
                .context("Failed to get embedding values")?
                .iter()
                .map(|v| v.as_f64().unwrap_or_default() as f32)
                .collect();

            all_embeddings.push(embedding);
        }

        Ok(all_embeddings)
    }
}
