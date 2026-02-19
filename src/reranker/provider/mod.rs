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

//! Reranker providers module

use anyhow::Result;
use reqwest::Client;
use std::sync::LazyLock;
use std::time::Duration;

use super::types::{RerankProviderType, RerankResponse};

// Shared HTTP client with connection pooling for optimal performance
static HTTP_CLIENT: LazyLock<Client> = LazyLock::new(|| {
    Client::builder()
        .pool_max_idle_per_host(10)
        .pool_idle_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(120))
        .connect_timeout(Duration::from_secs(10))
        .build()
        .expect("Failed to create HTTP client")
});

pub mod cohere;
pub mod jina;
pub mod mixedbread;
pub mod voyage;

#[cfg(feature = "fastembed")]
pub mod fastembed;

#[cfg(feature = "huggingface")]
pub mod huggingface;

pub use cohere::CohereProvider;
pub use jina::JinaProvider;
pub use mixedbread::MixedbreadProvider;
pub use voyage::{VoyageProvider, VoyageProviderImpl};

#[cfg(feature = "fastembed")]
pub use self::fastembed::FastEmbedProvider;

#[cfg(feature = "huggingface")]
pub use self::huggingface::HuggingFaceReranker;

/// Trait for reranker providers
#[async_trait::async_trait]
pub trait RerankProvider: Send + Sync {
    /// Rerank documents based on query relevance
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        truncation: bool,
    ) -> Result<RerankResponse>;

    /// Validate if the model is supported
    fn is_model_supported(&self) -> bool {
        true
    }
}

/// Create a reranker provider from provider type and model
pub async fn create_rerank_provider_from_parts(
    provider: &RerankProviderType,
    model: &str,
) -> Result<Box<dyn RerankProvider>> {
    match provider {
        RerankProviderType::Voyage => Ok(Box::new(VoyageProviderImpl::new(model)?)),
        RerankProviderType::Cohere => Ok(Box::new(CohereProvider::new(model)?)),
        RerankProviderType::Jina => Ok(Box::new(JinaProvider::new(model)?)),
        RerankProviderType::MixedBread => Ok(Box::new(MixedbreadProvider::new(model)?)),
        #[cfg(feature = "fastembed")]
        RerankProviderType::FastEmbed => Ok(Box::new(FastEmbedProvider::new(model)?)),
        #[cfg(feature = "huggingface")]
        RerankProviderType::HuggingFace => Ok(Box::new(HuggingFaceReranker::new(model)?)),
    }
}
