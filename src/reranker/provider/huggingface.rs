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

//! HuggingFace local reranker provider — cross-encoders via Candle.
//!
//! Supports two transformer families, detected from `config.json` at load time:
//!
//! - **BERT** (`BertModel` + `BertForSequenceClassification`):
//!   `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`,
//!   and other standard BERT cross-encoders.
//!
//! - **XLM-RoBERTa** (`XLMRobertaForSequenceClassification`):
//!   `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-v2-m3`,
//!   `jinaai/jina-reranker-v2-base-multilingual`, etc.
//!
//! Both paths apply the **real** classification head from the model's safetensors
//! — pooler+dropout+classifier for BERT, dense+tanh+out_proj for XLM-RoBERTa.
//! Fine-tuned classifier weights are honored at runtime.
//!
//! Models are downloaded from HuggingFace Hub on first use and cached locally.
//! Requires the `huggingface` feature flag.

#[cfg(feature = "huggingface")]
use anyhow::{Context, Result};
#[cfg(feature = "huggingface")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "huggingface")]
use candle_nn::{linear, Linear, Module, VarBuilder};
#[cfg(feature = "huggingface")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "huggingface")]
use candle_transformers::models::xlm_roberta::{
    Config as XLMRobertaConfig, XLMRobertaForSequenceClassification,
};
#[cfg(feature = "huggingface")]
use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
#[cfg(feature = "huggingface")]
use serde::Deserialize;
#[cfg(feature = "huggingface")]
use std::collections::HashMap;
#[cfg(feature = "huggingface")]
use std::sync::{Arc, LazyLock};
#[cfg(feature = "huggingface")]
use tokenizers::Tokenizer;
#[cfg(feature = "huggingface")]
use tokio::sync::RwLock;

#[cfg(feature = "huggingface")]
use super::super::types::{RerankResponse, RerankResult};
#[cfg(feature = "huggingface")]
use super::RerankProvider;

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

#[cfg(feature = "huggingface")]
#[derive(Debug, Clone, Copy)]
enum RerankerArch {
    Bert,
    XLMRoberta,
}

/// Minimal config.json fields we read directly — just enough to pick the
/// architecture and the classifier output dim. The full BERT/RoBERTa configs
/// are deserialized separately once we know which architecture we're loading.
#[cfg(feature = "huggingface")]
#[derive(Debug, Deserialize)]
struct ConfigHead {
    architectures: Option<Vec<String>>,
    #[serde(default = "default_num_labels")]
    num_labels: usize,
}

#[cfg(feature = "huggingface")]
fn default_num_labels() -> usize {
    1
}

#[cfg(feature = "huggingface")]
fn detect_arch(head: &ConfigHead) -> Result<RerankerArch> {
    let archs = head
        .architectures
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("config.json missing 'architectures' field"))?;
    let first = archs
        .first()
        .ok_or_else(|| anyhow::anyhow!("config.json has empty 'architectures' array"))?;
    match first.as_str() {
        "BertModel"
        | "BertForSequenceClassification"
        | "BertForMaskedLM"
        | "BertForTokenClassification" => Ok(RerankerArch::Bert),
        "XLMRobertaModel"
        | "XLMRobertaForSequenceClassification"
        | "RobertaModel"
        | "RobertaForSequenceClassification" => Ok(RerankerArch::XLMRoberta),
        other => anyhow::bail!(
            "Unsupported reranker architecture '{other}'. \
			 Supported: BERT family, XLM-RoBERTa / RoBERTa family.",
        ),
    }
}

// ---------------------------------------------------------------------------
// Loaded model — enum dispatch over architecture
// ---------------------------------------------------------------------------

#[cfg(feature = "huggingface")]
struct BertHead {
    model: BertModel,
    pooler: Linear,
    classifier: Linear,
}

#[cfg(feature = "huggingface")]
enum Backend {
    Bert(BertHead),
    XLMRoberta(XLMRobertaForSequenceClassification),
}

#[cfg(feature = "huggingface")]
struct CrossEncoderModel {
    backend: Backend,
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

        // Disable hf-hub progress bars; octomind owns terminal UI.
        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("Failed to initialize HuggingFace API")?;
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
        let weights_path = repo
            .get("model.safetensors")
            .await
            .with_context(|| format!("Failed to download model.safetensors for: {}", model_name))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let config_content = std::fs::read_to_string(&config_path)?;
        let head: ConfigHead = serde_json::from_str(&config_content)
            .context("Failed to parse config.json head fields")?;
        let arch = detect_arch(&head)?;

        let weights = candle_core::safetensors::load(&weights_path, &device)?;
        let var_builder = VarBuilder::from_tensors(weights, DType::F32, &device);

        let backend = match arch {
            RerankerArch::Bert => {
                let cfg: BertConfig =
                    serde_json::from_str(&config_content).context("Failed to parse BERT config")?;
                let hidden_size = cfg.hidden_size;
                let num_labels = if head.num_labels > 0 {
                    head.num_labels
                } else {
                    1
                };

                // `BertModel` lives under the `bert.` prefix in cross-encoder
                // checkpoints (BertForSequenceClassification serializes its
                // internal `bert` field with that name). The pooler is part
                // of the BERT block at `bert.pooler.dense.*`. The final
                // classifier sits at `classifier.*` (root-level).
                let model = BertModel::load(var_builder.pp("bert"), &cfg)
                    .context("Failed to load BertModel weights")?;
                let pooler = linear(
                    hidden_size,
                    hidden_size,
                    var_builder.pp("bert.pooler.dense"),
                )
                .context("Failed to load BERT pooler weights")?;
                let classifier = linear(hidden_size, num_labels, var_builder.pp("classifier"))
                    .context("Failed to load BERT classifier head")?;

                Backend::Bert(BertHead {
                    model,
                    pooler,
                    classifier,
                })
            }

            RerankerArch::XLMRoberta => {
                let cfg: XLMRobertaConfig = serde_json::from_str(&config_content)
                    .context("Failed to parse XLM-RoBERTa config")?;
                let num_labels = if head.num_labels > 0 {
                    head.num_labels
                } else {
                    1
                };

                // candle's `XLMRobertaForSequenceClassification` wraps the
                // `roberta` model and the `classifier` head (dense + out_proj),
                // matching how HF transformers serializes the weights.
                let model = XLMRobertaForSequenceClassification::new(num_labels, &cfg, var_builder)
                    .context("Failed to load XLM-RoBERTa weights")?;
                Backend::XLMRoberta(model)
            }
        };

        Ok(Self {
            backend,
            tokenizer,
            device,
        })
    }

    /// Score a single (query, document) pair. Returns the raw classifier
    /// logit for the single-label output (num_labels=1 case, which is the
    /// standard cross-encoder configuration). Higher = more relevant.
    fn score_pair(&self, query: &str, document: &str) -> Result<f64> {
        // Cross-encoder input: "[CLS] query [SEP] document [SEP]" — pass as
        // a (sequence_a, sequence_b) tuple so the tokenizer applies the
        // model's native pair-encoding template (token_type_ids, special
        // tokens, etc.).
        let encoding = self
            .tokenizer
            .encode((query, document), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = encoding.get_ids();
        let token_type_ids = encoding.get_type_ids();
        let attention_mask_vec: Vec<u32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as u32)
            .collect();

        let input_ids = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(token_type_ids, &self.device)?.unsqueeze(0)?;

        let logits = match &self.backend {
            Backend::Bert(b) => {
                // candle's BertModel uses an u8 attention mask (1 = attend).
                let attn_u8 = Tensor::ones((1, token_ids.len()), DType::U8, &self.device)?;
                let hidden = b
                    .model
                    .forward(&input_ids, &attn_u8, Some(&token_type_ids_tensor))?;
                // Pooler: take [CLS] hidden, project + tanh.
                let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?; // (1, hidden_size)
                let pooled = b.pooler.forward(&cls)?.tanh()?; // (1, hidden_size)
                let logits = b.classifier.forward(&pooled)?; // (1, num_labels)
                logits
            }

            Backend::XLMRoberta(m) => {
                // candle's XLM-RoBERTa expects f32 attention mask (1.0 = attend).
                let attn = Tensor::from_vec(
                    attention_mask_vec.iter().map(|&x| x as f32).collect(),
                    (1, token_ids.len()),
                    &self.device,
                )?;
                m.forward(&input_ids, &attn, &token_type_ids_tensor)?
            }
        };

        // Single-label classifier — take the scalar from logits[0, 0] and
        // apply sigmoid so the output is a relevance probability in [0, 1].
        // This matches what Cohere / Voyage / Jina rerank APIs return and
        // makes thresholds/margins interpretable in human terms (0.5 = uncertain,
        // 0.9+ = confident match) regardless of how the model was fine-tuned.
        let scalar = logits.flatten_all()?.to_vec1::<f32>()?;
        let logit = *scalar
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty classifier output"))?;
        let prob = 1.0_f32 / (1.0 + (-logit).exp());
        Ok(prob as f64)
    }
}

// ---------------------------------------------------------------------------
// Process-wide model cache + provider impl
// ---------------------------------------------------------------------------

#[cfg(feature = "huggingface")]
#[allow(clippy::type_complexity)]
static MODEL_CACHE: LazyLock<Arc<RwLock<HashMap<String, Arc<CrossEncoderModel>>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(HashMap::new())));

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

/// HuggingFace local reranker provider. Loads BERT or XLM-RoBERTa
/// cross-encoders via candle. Requires the `huggingface` feature flag.
#[cfg(feature = "huggingface")]
pub struct HuggingFaceReranker {
    model_name: String,
}

#[cfg(feature = "huggingface")]
impl HuggingFaceReranker {
    /// Create a new provider for the given HuggingFace model ID.
    ///
    /// # Supported architectures
    /// - **BERT** family: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, …
    /// - **XLM-RoBERTa** family: `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-v2-m3`, `jinaai/jina-reranker-v2-base-multilingual`, …
    ///
    /// Architecture is detected from `config.json` at load time. Models are
    /// downloaded on first use and cached locally.
    pub fn new(model: &str) -> Result<Self> {
        if model.is_empty() {
            return Err(anyhow::anyhow!("Model name cannot be empty"));
        }
        Ok(Self {
            model_name: model.to_string(),
        })
    }

    /// List well-known cross-encoder models that work with this provider.
    pub fn recommended_models() -> Vec<&'static str> {
        vec![
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "BAAI/bge-reranker-base",
            "BAAI/bge-reranker-large",
            "BAAI/bge-reranker-v2-m3",
            "jinaai/jina-reranker-v2-base-multilingual",
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
            total_tokens: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Feature-disabled stub
// ---------------------------------------------------------------------------

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
            assert!(HuggingFaceReranker::new("jinaai/jina-reranker-v2-base-multilingual").is_ok());
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
            assert!(models.contains(&"jinaai/jina-reranker-v2-base-multilingual"));
        }
    }
}
