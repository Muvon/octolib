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

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Input type for embedding generation
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum InputType {
    /// Default - no input_type (existing behavior)
    #[default]
    None,
    /// For search operations
    Query,
    /// For indexing operations
    Document,
}

impl InputType {
    /// Convert to API string for providers that support it (like Voyage)
    pub fn as_api_str(&self) -> Option<&'static str> {
        match self {
            InputType::None => None,
            InputType::Query => Some("query"),
            InputType::Document => Some("document"),
        }
    }

    /// Get prefix for manual injection (for providers that don't support input_type API)
    pub fn get_prefix(&self) -> Option<&'static str> {
        match self {
            InputType::None => None,
            InputType::Query => Some(super::constants::QUERY_PREFIX),
            InputType::Document => Some(super::constants::DOCUMENT_PREFIX),
        }
    }

    /// Apply prefix to text for manual injection
    pub fn apply_prefix(&self, text: &str) -> String {
        match self.get_prefix() {
            Some(prefix) => format!("{}{}", prefix, text),
            None => text.to_string(),
        }
    }
}

/// Supported embedding providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProviderType {
    FastEmbed,
    Jina,
    Voyage,
    Google,
    HuggingFace,
    OpenAI,
    OpenRouter,
    OctoHub,
    Local,
    Together,
}

#[allow(clippy::derivable_impls)]
impl Default for EmbeddingProviderType {
    fn default() -> Self {
        #[cfg(feature = "fastembed")]
        {
            Self::FastEmbed
        }
        #[cfg(not(feature = "fastembed"))]
        {
            Self::Voyage
        }
    }
}

/// Configuration for embedding models (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Code embedding model (format: "provider:model")
    pub code_model: String,

    /// Text embedding model (format: "provider:model")
    pub text_model: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        // Use FastEmbed models if available, otherwise fall back to Voyage
        #[cfg(feature = "fastembed")]
        {
            Self {
                code_model: "fastembed:jinaai/jina-embeddings-v2-base-code".to_string(),
                text_model: "fastembed:sentence-transformers/all-MiniLM-L6-v2-quantized"
                    .to_string(),
            }
        }
        #[cfg(not(feature = "fastembed"))]
        {
            Self {
                code_model: "voyage:voyage-code-4".to_string(),
                text_model: "voyage:voyage-3.5-lite".to_string(),
            }
        }
    }
}

/// Parse provider and model from a string in format "provider:model"
pub fn parse_provider_model(input: &str) -> Result<(EmbeddingProviderType, String)> {
    let input = input.trim();
    let (provider_str, model) = input.split_once(':').ok_or_else(|| {
        anyhow::anyhow!("Model format must be 'provider:model' (e.g., 'jina:jina-embeddings-v4')")
    })?;
    let provider_str = provider_str.trim();
    let model = model.trim();

    if provider_str.is_empty() || model.is_empty() {
        return Err(anyhow::anyhow!(
            "Model format must be 'provider:model' with non-empty provider and model"
        ));
    }

    let provider = match provider_str.to_lowercase().as_str() {
        "fastembed" => EmbeddingProviderType::FastEmbed,
        "jinaai" | "jina" => EmbeddingProviderType::Jina,
        "voyageai" | "voyage" => EmbeddingProviderType::Voyage,
        "google" => EmbeddingProviderType::Google,
        "huggingface" | "hf" => EmbeddingProviderType::HuggingFace,
        "openai" => EmbeddingProviderType::OpenAI,
        "openrouter" => EmbeddingProviderType::OpenRouter,
        "octohub" => EmbeddingProviderType::OctoHub,
        "local" => EmbeddingProviderType::Local,
        "together" => EmbeddingProviderType::Together,
        unknown => {
            return Err(anyhow::anyhow!(
                "Unknown embedding provider '{}'. Supported: fastembed, jina, voyage, google, huggingface, openai, openrouter, octohub, local, together. \
                 This is a programming error - the provider should be validated before calling parse_provider_model.",
                unknown
            ));
        }
    };

    Ok((provider, model.to_string()))
}

impl EmbeddingConfig {
    /// Get the currently active provider based on the code model
    pub fn get_active_provider(&self) -> Result<EmbeddingProviderType> {
        let (provider, _) = parse_provider_model(&self.code_model)?;
        Ok(provider)
    }
    /// Get API key for a specific provider (from environment variables only)
    pub fn get_api_key(&self, provider: &EmbeddingProviderType) -> Option<String> {
        match provider {
            EmbeddingProviderType::Jina => std::env::var("JINA_API_KEY").ok(),
            EmbeddingProviderType::Voyage => std::env::var("VOYAGE_API_KEY").ok(),
            EmbeddingProviderType::Google => std::env::var("GOOGLE_API_KEY").ok(),
            EmbeddingProviderType::Together => std::env::var("TOGETHER_API_KEY").ok(),
            EmbeddingProviderType::Local => std::env::var("LOCAL_EMBED_API_KEY").ok(),
            _ => None, // FastEmbed, HuggingFace, OctoHub, OpenAI, OpenRouter don't use this path
        }
    }

    /// Get vector dimension by creating a provider instance
    pub async fn get_vector_dimension(
        &self,
        provider: &EmbeddingProviderType,
        model: &str,
    ) -> Result<usize> {
        // Try to create provider and get dimension
        let provider_impl =
            super::provider::create_embedding_provider_from_parts(provider, model).await?;
        Ok(provider_impl.get_dimension())
    }

    /// Validate model by trying to create provider
    pub async fn validate_model(
        &self,
        provider: &EmbeddingProviderType,
        model: &str,
    ) -> Result<()> {
        let provider_impl =
            super::provider::create_embedding_provider_from_parts(provider, model).await?;
        if !provider_impl.is_model_supported() {
            return Err(anyhow::anyhow!(
                "Model {} is not supported by provider {:?}",
                model,
                provider
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
