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
use std::sync::OnceLock;

use super::super::types::InputType;
use super::{EmbeddingProvider, HTTP_CLIENT};

/// Cached set of valid embedding model IDs fetched from the OpenRouter API.
/// Populated once on first use; subsequent calls reuse the cached value.
static SUPPORTED_MODELS: OnceLock<Vec<String>> = OnceLock::new();

/// Fetch the list of supported embedding model IDs from OpenRouter.
/// Returns an empty vec on failure so callers degrade gracefully.
async fn fetch_supported_models() -> Vec<String> {
    let Ok(api_key) = std::env::var("OPENROUTER_API_KEY") else {
        return Vec::new();
    };

    let Ok(response) = HTTP_CLIENT
        .get("https://openrouter.ai/api/v1/embeddings/models")
        .header("Authorization", format!("Bearer {}", api_key))
        .send()
        .await
    else {
        return Vec::new();
    };

    let Ok(json) = response.json::<Value>().await else {
        return Vec::new();
    };

    json["data"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Returns the cached model list, fetching it on first call.
async fn supported_models() -> &'static Vec<String> {
    // OnceLock only supports sync init, so we check and set manually for async.
    if let Some(models) = SUPPORTED_MODELS.get() {
        return models;
    }
    let models = fetch_supported_models().await;
    // Another task may have raced us — ignore the error if already set.
    let _ = SUPPORTED_MODELS.set(models);
    SUPPORTED_MODELS.get().unwrap()
}

/// OpenRouter provider implementation for trait.
/// The supported model list is fetched once from the API and cached.
/// Dimension is probed at construction time since the API does not expose it.
pub struct OpenRouterProviderImpl {
    model_name: String,
    dimension: usize,
}

impl OpenRouterProviderImpl {
    pub async fn new(model: &str) -> Result<Self> {
        // Populate the model cache and validate in one shot
        let models = supported_models().await;
        // Only reject if the cache is non-empty and the model isn't in it.
        // Empty cache means the API was unreachable — allow through to let the
        // actual embedding call surface the real error.
        if !models.is_empty() && !models.iter().any(|m| m == model) {
            return Err(anyhow::anyhow!(
                "Unsupported OpenRouter embedding model '{}'. Run `curl https://openrouter.ai/api/v1/embeddings/models` to see available models.",
                model
            ));
        }

        // Probe the actual dimension — the API has no dimension field.
        let probe = OpenRouterProvider::generate_embeddings("probe", model).await?;
        let dimension = probe.len();
        if dimension == 0 {
            return Err(anyhow::anyhow!(
                "OpenRouter model '{}' returned zero-length embedding",
                model
            ));
        }

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
        // Checked at construction time against the live model list
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

    #[tokio::test]
    async fn test_openrouter_provider_dimension_probe() {
        if std::env::var("OPENROUTER_API_KEY").is_err() {
            return;
        }
        let provider = OpenRouterProviderImpl::new("qwen/qwen3-embedding-8b")
            .await
            .unwrap();
        assert_eq!(provider.get_dimension(), 4096);
        assert!(provider.is_model_supported());
    }

    #[tokio::test]
    async fn test_openrouter_invalid_model_rejected() {
        if std::env::var("OPENROUTER_API_KEY").is_err() {
            return;
        }
        let result = OpenRouterProviderImpl::new("not/a-real-model-xyz").await;
        assert!(result.is_err());
    }
}
