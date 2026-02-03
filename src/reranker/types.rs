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
    #[cfg(feature = "fastembed")]
    FastEmbed,
}

impl FromStr for RerankProviderType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "voyage" => Ok(Self::Voyage),
            "cohere" => Ok(Self::Cohere),
            "jina" => Ok(Self::Jina),
            #[cfg(feature = "fastembed")]
            "fastembed" => Ok(Self::FastEmbed),
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
            #[cfg(feature = "fastembed")]
            Self::FastEmbed => "fastembed",
        }
    }
}

/// Parse provider and model from a string in format "provider:model"
pub fn parse_provider_model(input: &str) -> (RerankProviderType, String) {
    let (provider_str, model) = input.split_once(':').unwrap_or(("voyage", input));

    let provider = provider_str.parse().unwrap_or(RerankProviderType::Voyage);

    (provider, model.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider_model() {
        let (provider, model) = parse_provider_model("voyage:rerank-2.5");
        assert_eq!(provider, RerankProviderType::Voyage);
        assert_eq!(model, "rerank-2.5");

        let (provider, model) = parse_provider_model("cohere:rerank-english-v3.0");
        assert_eq!(provider, RerankProviderType::Cohere);
        assert_eq!(model, "rerank-english-v3.0");

        let (provider, model) = parse_provider_model("jina:jina-reranker-v3");
        assert_eq!(provider, RerankProviderType::Jina);
        assert_eq!(model, "jina-reranker-v3");

        #[cfg(feature = "fastembed")]
        {
            let (provider, model) = parse_provider_model("fastembed:bge-reranker-base");
            assert_eq!(provider, RerankProviderType::FastEmbed);
            assert_eq!(model, "bge-reranker-base");
        }

        // Default to voyage if no provider specified
        let (provider, model) = parse_provider_model("rerank-2");
        assert_eq!(provider, RerankProviderType::Voyage);
        assert_eq!(model, "rerank-2");
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
        #[cfg(feature = "fastembed")]
        {
            assert_eq!(RerankProviderType::FastEmbed.as_str(), "fastembed");
        }
    }
}
