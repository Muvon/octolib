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
    Local,
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
            "ollama" => Err("Ollama does not support reranking. Use local provider with llama.cpp server, vLLM, or TEI for local reranking, or cloud providers (voyage, cohere, jina).".to_string()),
            "local" => Ok(Self::Local),
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
            Self::Local => "local",
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
        .parse::<RerankProviderType>()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok((provider, model.to_string()))
}

#[cfg(test)]
#[path = "types_tests.rs"]
mod tests;
