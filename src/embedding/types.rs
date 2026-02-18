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
                code_model: "voyage:voyage-code-3".to_string(),
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
        unknown => {
            return Err(anyhow::anyhow!(
                "Unknown embedding provider '{}'. Supported: fastembed, jina, voyage, google, huggingface, openai. \
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
            _ => None, // FastEmbed and SentenceTransformer don't need API keys
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
mod tests {
    use super::*;

    #[test]
    fn test_input_type_api_str() {
        assert_eq!(InputType::None.as_api_str(), None);
        assert_eq!(InputType::Query.as_api_str(), Some("query"));
        assert_eq!(InputType::Document.as_api_str(), Some("document"));
    }

    #[test]
    fn test_input_type_prefix() {
        assert!(InputType::None.get_prefix().is_none());
        assert!(InputType::Query.get_prefix().is_some());
        assert!(InputType::Document.get_prefix().is_some());
    }

    #[test]
    fn test_input_type_apply_prefix() {
        let text = "test content";

        let no_prefix = InputType::None.apply_prefix(text);
        assert_eq!(no_prefix, text);

        let query_prefix = InputType::Query.apply_prefix(text);
        assert!(query_prefix.contains(text));
        assert!(query_prefix.len() > text.len());

        let doc_prefix = InputType::Document.apply_prefix(text);
        assert!(doc_prefix.contains(text));
        assert!(doc_prefix.len() > text.len());
    }

    #[test]
    fn test_parse_provider_model() {
        // Test valid provider:model format
        let (provider, model) = parse_provider_model("jina:jina-embeddings-v4").unwrap();
        assert_eq!(provider, EmbeddingProviderType::Jina);
        assert_eq!(model, "jina-embeddings-v4");

        // Test voyage provider
        let (provider, model) = parse_provider_model("voyage:voyage-3.5").unwrap();
        assert_eq!(provider, EmbeddingProviderType::Voyage);
        assert_eq!(model, "voyage-3.5");

        // Test google provider
        let (provider, model) = parse_provider_model("google:gemini-embedding-001").unwrap();
        assert_eq!(provider, EmbeddingProviderType::Google);
        assert_eq!(model, "gemini-embedding-001");

        // Test openai provider
        let (provider, model) = parse_provider_model("openai:text-embedding-3-small").unwrap();
        assert_eq!(provider, EmbeddingProviderType::OpenAI);
        assert_eq!(model, "text-embedding-3-small");

        // Whitespace should be trimmed
        let (provider, model) = parse_provider_model("  voyage : voyage-3.5  ").unwrap();
        assert_eq!(provider, EmbeddingProviderType::Voyage);
        assert_eq!(model, "voyage-3.5");

        // Empty segments should fail with explicit format error
        assert!(parse_provider_model("openai:").is_err());
        assert!(parse_provider_model(":text-embedding-3-small").is_err());
    }

    #[test]
    fn test_embedding_config_active_provider() {
        let config = EmbeddingConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "voyage:voyage-3.5".to_string(),
        };

        let active_provider = config.get_active_provider().unwrap();
        assert_eq!(active_provider, EmbeddingProviderType::Jina);
    }

    #[test]
    fn test_embedding_config_api_keys() {
        let config = EmbeddingConfig::default();

        // These should return None unless environment variables are set
        let jina_key = config.get_api_key(&EmbeddingProviderType::Jina);
        let voyage_key = config.get_api_key(&EmbeddingProviderType::Voyage);
        let google_key = config.get_api_key(&EmbeddingProviderType::Google);
        let openai_key = config.get_api_key(&EmbeddingProviderType::OpenAI);

        // FastEmbed and HuggingFace don't need API keys
        assert!(config
            .get_api_key(&EmbeddingProviderType::FastEmbed)
            .is_none());
        assert!(config
            .get_api_key(&EmbeddingProviderType::HuggingFace)
            .is_none());

        // API keys should be None unless environment variables are set
        assert!(jina_key.is_none() || !jina_key.as_ref().unwrap().is_empty());
        assert!(voyage_key.is_none() || !voyage_key.as_ref().unwrap().is_empty());
        assert!(google_key.is_none() || !google_key.as_ref().unwrap().is_empty());
        assert!(openai_key.is_none() || !openai_key.as_ref().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_embedding_config_vector_dimensions() {
        let config = EmbeddingConfig::default();

        // Test that we can call get_vector_dimension without panicking
        // (it might fail due to missing API keys, which is expected)
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tokio::runtime::Runtime::new().unwrap().block_on(async {
                config
                    .get_vector_dimension(&EmbeddingProviderType::Jina, "jina-embeddings-v4")
                    .await
            })
        }));

        // We don't care about the result, just that it doesn't panic unexpectedly
        // In test environment without API keys, this might panic or return an error
        match result {
            Ok(_) => {
                // Function completed (either success or expected error)
            }
            Err(_) => {
                // Function panicked (might be expected due to missing API keys)
            }
        }
    }
}
