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

//! Reranker types and configurations

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Result of a single document reranking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Original index of the document in the input list
    pub index: usize,
    /// The document text
    pub document: String,
    /// Relevance score (higher = more relevant)
    pub relevance_score: f64,
}

/// Response from a reranking operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Reranked results sorted by relevance score (descending)
    pub results: Vec<RerankResult>,
    /// Total tokens used in the reranking operation
    pub total_tokens: usize,
}

/// Supported reranker provider types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RerankProviderType {
    Voyage,
    Cohere,
    Jina,
    MixedBread,
    #[cfg(feature = "fastembed")]
    FastEmbed,
    #[cfg(feature = "huggingface")]
    HuggingFace,
}

impl FromStr for RerankProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "voyage" => Ok(Self::Voyage),
            "cohere" => Ok(Self::Cohere),
            "jina" => Ok(Self::Jina),
            "mixedbread" | "mxbai" => Ok(Self::MixedBread),
            #[cfg(feature = "fastembed")]
            "fastembed" => Ok(Self::FastEmbed),
            #[cfg(feature = "huggingface")]
            "huggingface" | "hf" => Ok(Self::HuggingFace),
            _ => Err(format!("Unknown reranker provider: {}", s)),
        }
    }
}

impl RerankProviderType {
    /// Get provider name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Voyage => "voyage",
            Self::Cohere => "cohere",
            Self::Jina => "jina",
            Self::MixedBread => "mixedbread",
            #[cfg(feature = "fastembed")]
            Self::FastEmbed => "fastembed",
            #[cfg(feature = "huggingface")]
            Self::HuggingFace => "huggingface",
        }
    }
}

/// Parse provider and model from a string in format "provider:model"
pub fn parse_provider_model(input: &str) -> Result<(RerankProviderType, String)> {
    let input = input.trim();
    let (provider_str, model) = if let Some((provider, model)) = input.split_once(':') {
        (provider.trim(), model.trim())
    } else {
        ("voyage", input)
    };

    if provider_str.is_empty() || model.is_empty() {
        return Err(anyhow::anyhow!(
            "Model format must be 'provider:model' or just 'model' (defaults to voyage)"
        ));
    }

    let provider = provider_str
        .parse()
        .map_err(|_| anyhow::anyhow!("Unknown reranker provider: {}", provider_str))?;

    Ok((provider, model.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider_model() {
        let (provider, model) = parse_provider_model("voyage:rerank-2.5").unwrap();
        assert_eq!(provider, RerankProviderType::Voyage);
        assert_eq!(model, "rerank-2.5");

        let (provider, model) = parse_provider_model("cohere:rerank-english-v3.0").unwrap();
        assert_eq!(provider, RerankProviderType::Cohere);
        assert_eq!(model, "rerank-english-v3.0");

        let (provider, model) = parse_provider_model("jina:jina-reranker-v3").unwrap();
        assert_eq!(provider, RerankProviderType::Jina);
        assert_eq!(model, "jina-reranker-v3");

        let (provider, model) = parse_provider_model("mixedbread:mxbai-rerank-large-v2").unwrap();
        assert_eq!(provider, RerankProviderType::MixedBread);
        assert_eq!(model, "mxbai-rerank-large-v2");

        // "mxbai" is an accepted alias for mixedbread
        let (provider, _) = parse_provider_model("mxbai:mxbai-rerank-base-v2").unwrap();
        assert_eq!(provider, RerankProviderType::MixedBread);

        #[cfg(feature = "huggingface")]
        {
            let (provider, model) =
                parse_provider_model("huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2").unwrap();
            assert_eq!(provider, RerankProviderType::HuggingFace);
            assert_eq!(model, "cross-encoder/ms-marco-MiniLM-L-6-v2");

            // "hf" is an accepted alias
            let (provider, _) = parse_provider_model("hf:BAAI/bge-reranker-base").unwrap();
            assert_eq!(provider, RerankProviderType::HuggingFace);
        }
        #[cfg(feature = "fastembed")]
        {
            let (provider, model) = parse_provider_model("fastembed:bge-reranker-base").unwrap();
            assert_eq!(provider, RerankProviderType::FastEmbed);
            assert_eq!(model, "bge-reranker-base");
        }

        // Default to voyage if no provider specified
        let (provider, model) = parse_provider_model("rerank-2").unwrap();
        assert_eq!(provider, RerankProviderType::Voyage);
        assert_eq!(model, "rerank-2");

        // Trims surrounding whitespace
        let (provider, model) = parse_provider_model("  cohere : rerank-english-v3.0  ").unwrap();
        assert_eq!(provider, RerankProviderType::Cohere);
        assert_eq!(model, "rerank-english-v3.0");

        // Explicit unknown provider should error instead of silently falling back
        assert!(parse_provider_model("unknown:rerank-2").is_err());
        assert!(parse_provider_model(":rerank-2").is_err());
        assert!(parse_provider_model("voyage:").is_err());
    }

    #[test]
    fn test_provider_type_conversions() {
        assert_eq!(
            "voyage".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::Voyage
        );
        assert_eq!(
            "Voyage".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::Voyage
        );
        assert_eq!(
            "cohere".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::Cohere
        );
        assert_eq!(
            "jina".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::Jina
        );
        assert_eq!(
            "mixedbread".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::MixedBread
        );
        assert_eq!(
            "mxbai".parse::<RerankProviderType>().unwrap(),
            RerankProviderType::MixedBread
        );
        #[cfg(feature = "huggingface")]
        {
            assert_eq!(
                "huggingface".parse::<RerankProviderType>().unwrap(),
                RerankProviderType::HuggingFace
            );
            assert_eq!(
                "hf".parse::<RerankProviderType>().unwrap(),
                RerankProviderType::HuggingFace
            );
        }
        #[cfg(feature = "fastembed")]
        {
            assert_eq!(
                "fastembed".parse::<RerankProviderType>().unwrap(),
                RerankProviderType::FastEmbed
            );
        }
        assert!("unknown".parse::<RerankProviderType>().is_err());

        assert_eq!(RerankProviderType::Voyage.as_str(), "voyage");
        assert_eq!(RerankProviderType::Cohere.as_str(), "cohere");
        assert_eq!(RerankProviderType::Jina.as_str(), "jina");
        assert_eq!(RerankProviderType::MixedBread.as_str(), "mixedbread");
        #[cfg(feature = "huggingface")]
        {
            assert_eq!(RerankProviderType::HuggingFace.as_str(), "huggingface");
        }
        #[cfg(feature = "fastembed")]
        {
            assert_eq!(RerankProviderType::FastEmbed.as_str(), "fastembed");
        }
    }
}
