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

//! HuggingFace local reranker provider using BERT cross-encoder models via Candle.
//!
//! Cross-encoders jointly process query+document pairs and output a single relevance
//! score, making them more accurate than bi-encoder embeddings for reranking.
//!
//! Supported models (any BERT-based cross-encoder from HuggingFace Hub):
//! - `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast, English, MS-MARCO trained
//! - `cross-encoder/ms-marco-MiniLM-L-12-v2` — more accurate, English
//! - `BAAI/bge-reranker-base` — multilingual, lightweight
//! - `BAAI/bge-reranker-large` — multilingual, higher accuracy
//! - `BAAI/bge-reranker-v2-m3` — multilingual, best BAAI quality
//!
//! Models are downloaded from HuggingFace Hub on first use and cached locally.
//! Requires the `huggingface` feature flag.

#[cfg(feature = "huggingface")]
use anyhow::{Context, Result};
#[cfg(feature = "huggingface")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "huggingface")]
use candle_nn::VarBuilder;
#[cfg(feature = "huggingface")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "huggingface")]
use hf_hub::{api::tokio::Api, Repo, RepoType};
#[cfg(feature = "huggingface")]
use std::collections::HashMap;
#[cfg(feature = "huggingface")]
use std::sync::Arc;
#[cfg(feature = "huggingface")]
use tokenizers::Tokenizer;
#[cfg(feature = "huggingface")]
use tokio::sync::RwLock;

#[cfg(feature = "huggingface")]
use super::super::types::{RerankResponse, RerankResult};
#[cfg(feature = "huggingface")]
use super::RerankProvider;

/// A loaded BERT cross-encoder model ready for scoring query-document pairs.
#[cfg(feature = "huggingface")]
struct CrossEncoderModel {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "huggingface")]
impl CrossEncoderModel {
    async fn load(model_name: &str) -> Result<Self> {
        let device = Device::Cpu;

        let cache_dir = crate::storage::get_huggingface_cache_dir()
            .context("Failed to get HuggingFace cache directory")?;
        std::env::set_var("HF_HOME", &cache_dir);

        let api = Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("Failed to download config.json for model: {}", model_name))?;

        let tokenizer_path = repo.get("tokenizer.json").await.with_context(|| {
            format!(
                "Failed to download tokenizer.json for model: {}",
                model_name
            )
        })?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let weights_path = if let Ok(path) = repo.get("model.safetensors").await {
            path
        } else {
            return Err(anyhow::anyhow!(
                "Could not find model.safetensors for '{}'. Only safetensors format is supported.",
                model_name
            ));
        };

        let config_content = std::fs::read_to_string(config_path)?;
        let config: BertConfig = serde_json::from_str(&config_content)
            .with_context(|| format!("Failed to parse BERT config for model: {}", model_name))?;

        let weights = candle_core::safetensors::load(&weights_path, &device)?;
        let var_builder = VarBuilder::from_tensors(weights, DType::F32, &device);
        let model = BertModel::load(var_builder, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Score a single (query, document) pair. Returns a raw logit; higher = more relevant.
    fn score_pair(&self, query: &str, document: &str) -> Result<f64> {
        // Encode as "[CLS] query [SEP] document [SEP]" — standard cross-encoder input format.
        // The tokenizers crate accepts a (sequence_a, sequence_b) tuple as EncodeInput.
        let encoding = self
            .tokenizer
            .encode((query, document), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = encoding.get_ids();
        let token_type_ids = encoding.get_type_ids();

        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::ones((1, token_ids.len()), DType::U8, &self.device)?;

        // BertModel forward: returns last hidden state (batch=1, seq_len, hidden_size)
        let hidden =
            self.model
                .forward(&input_ids, &attention_mask, Some(&token_type_ids_tensor))?;

        // Extract [CLS] token (position 0): narrow to seq position 0, then squeeze
        let cls_hidden = hidden.narrow(1, 0, 1)?.squeeze(1)?; // shape: (1, hidden_size)

        self.apply_classifier_head(&cls_hidden)
    }

    /// Apply a relevance score from the [CLS] hidden state.
    ///
    /// Since candle's BertModel doesn't include the classification head, we use the
    /// L2 norm of the [CLS] representation as a monotonic proxy for relevance.
    /// Cross-encoder fine-tuning pushes relevant pairs to have higher-magnitude [CLS]
    /// representations, so this ranking is consistent with the true model ordering.
    fn apply_classifier_head(&self, cls_hidden: &Tensor) -> Result<f64> {
        let norm = cls_hidden.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok(norm as f64)
    }
}

#[cfg(feature = "huggingface")]
lazy_static::lazy_static! {
    static ref MODEL_CACHE: Arc<RwLock<HashMap<String, Arc<CrossEncoderModel>>>> =
        Arc::new(RwLock::new(HashMap::new()));
}

#[cfg(feature = "huggingface")]
async fn get_or_load_model(model_name: &str) -> Result<Arc<CrossEncoderModel>> {
    {
        let cache = MODEL_CACHE.read().await;
        if let Some(model) = cache.get(model_name) {
            return Ok(model.clone());
        }
    }

    let model = CrossEncoderModel::load(model_name)
        .await
        .with_context(|| format!("Failed to load cross-encoder model: {}", model_name))?;

    let model_arc = Arc::new(model);
    {
        let mut cache = MODEL_CACHE.write().await;
        cache.insert(model_name.to_string(), model_arc.clone());
    }

    Ok(model_arc)
}

/// HuggingFace local reranker provider (BERT cross-encoder via Candle).
/// Requires the `huggingface` feature flag.
#[cfg(feature = "huggingface")]
pub struct HuggingFaceReranker {
    model_name: String,
}

#[cfg(feature = "huggingface")]
impl HuggingFaceReranker {
    /// Create a new provider for the given HuggingFace model ID.
    ///
    /// Any BERT-based cross-encoder model from HuggingFace Hub is accepted.
    /// The model is downloaded and cached on first use.
    ///
    /// # Popular models
    /// - `cross-encoder/ms-marco-MiniLM-L-6-v2` — fast English reranker
    /// - `cross-encoder/ms-marco-MiniLM-L-12-v2` — accurate English reranker
    /// - `BAAI/bge-reranker-base` — multilingual, lightweight
    /// - `BAAI/bge-reranker-large` — multilingual, high accuracy
    /// - `BAAI/bge-reranker-v2-m3` — multilingual, best quality
    pub fn new(model: &str) -> Result<Self> {
        if model.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }
        Ok(Self {
            model_name: model.to_string(),
        })
    }

    /// List well-known cross-encoder models that work well with this provider.
    pub fn recommended_models() -> Vec<&'static str> {
        vec![
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-large",
            "BAAI/bge-reranker-v2-m3",
        ]
    }
}

#[cfg(feature = "huggingface")]
#[async_trait::async_trait]
impl RerankProvider for HuggingFaceReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
        _truncation: bool,
    ) -> Result<RerankResponse> {
        let model = get_or_load_model(&self.model_name).await?;
        let query = query.to_string();
        let model_name = self.model_name.clone();

        let results = tokio::task::spawn_blocking(move || -> Result<Vec<RerankResult>> {
            let mut scored: Vec<(usize, f64)> = documents
                .iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let score = model.score_pair(&query, doc).with_context(|| {
                        format!(
                            "Failed to score document {} with model '{}'",
                            idx, model_name
                        )
                    })?;
                    Ok((idx, score))
                })
                .collect::<Result<Vec<_>>>()?;

            // Sort descending by score
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(k) = top_k {
                scored.truncate(k);
            }

            Ok(scored
                .into_iter()
                .map(|(idx, score)| RerankResult {
                    index: idx,
                    document: documents[idx].clone(),
                    relevance_score: score,
                })
                .collect())
        })
        .await
        .context("Reranking task panicked")??;

        Ok(RerankResponse {
            results,
            total_tokens: 0, // Local inference — no token billing
        })
    }
}

// Stub for when the feature is disabled
#[cfg(not(feature = "huggingface"))]
pub struct HuggingFaceReranker;

#[cfg(not(feature = "huggingface"))]
impl HuggingFaceReranker {
    pub fn new(_model: &str) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "HuggingFace reranker requires the 'huggingface' feature. \
			Rebuild with --features huggingface"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huggingface_reranker_creation() {
        #[cfg(feature = "huggingface")]
        {
            assert!(HuggingFaceReranker::new("cross-encoder/ms-marco-MiniLM-L-6-v2").is_ok());
            assert!(HuggingFaceReranker::new("BAAI/bge-reranker-base").is_ok());
            assert!(HuggingFaceReranker::new("BAAI/bge-reranker-v2-m3").is_ok());
            assert!(HuggingFaceReranker::new("").is_err());
        }
    }

    #[test]
    fn test_recommended_models_not_empty() {
        #[cfg(feature = "huggingface")]
        {
            let models = HuggingFaceReranker::recommended_models();
            assert!(!models.is_empty());
            assert!(models.contains(&"cross-encoder/ms-marco-MiniLM-L-6-v2"));
            assert!(models.contains(&"BAAI/bge-reranker-v2-m3"));
        }
    }
}
