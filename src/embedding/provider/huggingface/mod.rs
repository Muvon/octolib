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

/*!
 * HuggingFace Provider Implementation
 *
 * This module provides local embedding generation using HuggingFace models via the Candle library.
 * It supports multiple model architectures with safetensors format from the HuggingFace Hub.
 *
 * Supported architectures:
 * - BERT: Standard BERT models (bert-base-uncased, sentence-transformers/all-MiniLM-L6-v2)
 * - RoBERTa: RoBERTa/XLM-RoBERTa models (microsoft/codebert-base, xlm-roberta-base)
 * - MPNet: MPNet models with relative position bias (sentence-transformers/all-mpnet-base-v2)
 * - JinaBert: Jina embedding models with ALiBi position embeddings (jinaai/jina-embeddings-v2-base-*)
 * - Qwen2: Qwen2 decoder models for embeddings (jinaai/jina-code-embeddings-1.5b)
 *
 * Key features:
 * - Automatic architecture detection from config.json
 * - Automatic model downloading and caching
 * - Local CPU-based inference (GPU support can be added)
 * - Thread-safe model cache for efficient reuse
 * - Mean pooling and L2 normalization for sentence embeddings
 * - Full compatibility with provider:model syntax
 *
 * Usage:
 * - Set provider: `octocode config --embedding-provider huggingface`
 * - Set models: `octocode config --code-embedding-model "huggingface:jinaai/jina-embeddings-v2-base-code"`
 * - Popular models: jinaai/jina-embeddings-v2-base-code, jinaai/jina-code-embeddings-1.5b
 *
 * Models are automatically downloaded to the system cache directory and reused across sessions.
 */

mod jina_code_bert;
mod mpnet;

#[cfg(feature = "huggingface")]
use anyhow::{Context, Result};
#[cfg(feature = "huggingface")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "huggingface")]
use candle_nn::Module;
#[cfg(feature = "huggingface")]
use candle_nn::VarBuilder;
#[cfg(feature = "huggingface")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
#[cfg(feature = "huggingface")]
use candle_transformers::models::jina_bert::{
    BertModel as JinaBertModel, Config as JinaBertConfig,
};
#[cfg(feature = "huggingface")]
use candle_transformers::models::qwen2::{Config as Qwen2Config, Model as Qwen2Model};
#[cfg(feature = "huggingface")]
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};
#[cfg(feature = "huggingface")]
use candle_transformers::models::xlm_roberta::{Config as XLMRobertaConfig, XLMRobertaModel};
#[cfg(feature = "huggingface")]
use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
#[cfg(feature = "huggingface")]
use jina_code_bert::JinaCodeBertModel;
#[cfg(feature = "huggingface")]
use mpnet::{MPNetConfig, MPNetModel};
#[cfg(feature = "huggingface")]
use serde::Deserialize;
#[cfg(feature = "huggingface")]
use std::collections::HashMap;
#[cfg(feature = "huggingface")]
use std::sync::{Arc, LazyLock, Weak};
#[cfg(feature = "huggingface")]
use tokenizers::Tokenizer;
#[cfg(feature = "huggingface")]
use tokio::sync::{Mutex as AsyncMutex, RwLock};

/// Select Metal when this build enables it; otherwise preserve CPU portability.
#[cfg(feature = "huggingface")]
pub(crate) fn embedding_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        Device::metal_if_available(0).context("Failed to initialize Metal embedding device")
    }
    #[cfg(not(feature = "metal"))]
    {
        Ok(Device::Cpu)
    }
}

/// Model architecture types supported by this provider
#[cfg(feature = "huggingface")]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ModelArchitecture {
    /// Standard BERT models
    Bert,
    /// RoBERTa / XLM-RoBERTa models
    Roberta,
    /// Jina BERT models with ALiBi position embeddings
    JinaBert,
    /// Jina BERT QK-post-norm models (e.g. jina-embeddings-v2-base-code)
    JinaCodeBert,
    /// Qwen2 decoder models
    Qwen2,
    /// Qwen3 decoder models
    Qwen3,
    /// MPNet models with relative position bias
    MPNet,
}

/// Configuration parsed from HuggingFace config.json
#[cfg(feature = "huggingface")]
#[derive(Debug, Deserialize)]
pub(crate) struct ModelConfig {
    pub(crate) architectures: Option<Vec<String>>,
    pub(crate) position_embedding_type: Option<String>,
    #[serde(rename = "_name_or_path")]
    pub(crate) name_or_path: Option<String>,
}

#[cfg(feature = "huggingface")]
impl ModelArchitecture {
    /// Detect architecture from config.json architectures field
    pub(crate) fn from_config(config: &ModelConfig) -> Result<Self> {
        let architectures = config
            .architectures
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No 'architectures' field in config.json"))?;

        if architectures.is_empty() {
            anyhow::bail!("Empty 'architectures' field in config.json");
        }

        let arch = &architectures[0];
        match arch.as_str() {
            // BERT variants — check position_embedding_type to distinguish Jina (ALiBi) from standard BERT
            "BertModel"
            | "BertForMaskedLM"
            | "BertForSequenceClassification"
            | "BertForTokenClassification" => {
                if config
                    .position_embedding_type
                    .as_deref()
                    .map(|t| t == "alibi")
                    .unwrap_or(false)
                {
                    // Distinguish QK-post-norm variant (e.g. jina-embeddings-v2-base-code)
                    if config
                        .name_or_path
                        .as_deref()
                        .map(|p| p.contains("qk-post-norm"))
                        .unwrap_or(false)
                    {
                        Ok(Self::JinaCodeBert)
                    } else {
                        Ok(Self::JinaBert)
                    }
                } else {
                    Ok(Self::Bert)
                }
            }

            // RoBERTa / XLM-RoBERTa variants
            "RobertaModel"
            | "RobertaForMaskedLM"
            | "RobertaForSequenceClassification"
            | "XLMRobertaModel"
            | "XLMRobertaForMaskedLM"
            | "XLMRobertaForSequenceClassification" => Ok(Self::Roberta),

            // Jina BERT variants (explicit Jina architecture names)
            "JinaBertModel"
            | "JinaBertForMaskedLM"
            | "JinaBertForSequenceClassification" => {
                // Explicit Jina architecture — also check for QK-post-norm
                if config
                    .name_or_path
                    .as_deref()
                    .map(|p| p.contains("qk-post-norm"))
                    .unwrap_or(false)
                {
                    Ok(Self::JinaCodeBert)
                } else {
                    Ok(Self::JinaBert)
                }
            }

            // Qwen2 variants
            "Qwen2ForCausalLM" | "Qwen2Model" | "Qwen2ForSequenceClassification" => Ok(Self::Qwen2),

            // Qwen3 variants
            "Qwen3ForCausalLM" | "Qwen3Model" | "Qwen3ForSequenceClassification" => Ok(Self::Qwen3),

            // MPNet variants
            "MPNetModel"
            | "MPNetForMaskedLM"
            | "MPNetForSequenceClassification"
            | "MPNetForTokenClassification" => Ok(Self::MPNet),

            _ => Err(anyhow::anyhow!(
                "Unsupported model architecture: '{}'. Supported: BertModel, RobertaModel, XLMRobertaModel, JinaBertModel, MPNetModel, Qwen2ForCausalLM, Qwen3ForCausalLM",
                arch
            )),
        }
    }
}

/// Architecture-specific network. Tokenizer and device are shared by every
/// architecture and live on [`HuggingFaceModel`].
#[cfg(feature = "huggingface")]
enum Backend {
    Bert(BertModel),
    Roberta(XLMRobertaModel),
    JinaBert(JinaBertModel),
    // Qwen2Model::forward takes &mut self, so wrap in Mutex for shared access
    Qwen2(std::sync::Mutex<Qwen2Model>),
    // Qwen3 maintains a KV cache during inference. Clone its empty base per input.
    Qwen3(Qwen3Model),
    JinaCodeBert(JinaCodeBertModel),
    MPNet(MPNetModel),
}

#[cfg(feature = "huggingface")]
/// HuggingFace model instance supporting multiple architectures
pub struct HuggingFaceModel {
    backend: Backend,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    /// HF commit sha of the snapshot the weights were loaded from.
    revision: String,
}

#[cfg(feature = "huggingface")]
impl HuggingFaceModel {
    /// Load a SentenceTransformer model from HuggingFace Hub
    pub async fn load(model_name: &str) -> Result<Self> {
        let load_lock = model_load_lock(model_name).await;
        let _load_guard = load_lock.lock().await;
        Self::load_inner(model_name).await
    }

    async fn load_inner(model_name: &str) -> Result<Self> {
        let device = embedding_device()?;

        // Use our custom cache directory for consistency with FastEmbed
        // Set HF_HOME environment variable to control where models are downloaded
        let cache_dir = crate::storage::get_model_cache_dir()
            .context("Failed to get HuggingFace cache directory")?;

        // Set the HuggingFace cache directory via environment variable
        std::env::set_var("HF_HOME", &cache_dir);

        // Download model files from HuggingFace Hub with proper error handling.
        // Disable progress bars — callers wrap downloads in their own UI
        // (octomind shows a generic "Working …" spinner; we don't want
        // hf-hub's indicatif bar fighting with it for the terminal).
        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("Failed to initialize HuggingFace API")?;
        let repo = api.repo(Repo::new(model_name.to_string(), RepoType::Model));

        // Download required files with enhanced error handling
        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("Failed to download config.json for model: {}", model_name))?;

        // hf_hub materializes every file under `snapshots/<commit sha>/`, so the
        // parent dir of any fetched file names the exact revision being loaded.
        let revision = config_path
            .parent()
            .and_then(|dir| dir.file_name())
            .and_then(|sha| sha.to_str())
            .map(str::to_owned)
            .with_context(|| {
                format!(
                    "Unexpected hf_hub cache layout for model {}: {}",
                    model_name,
                    config_path.display()
                )
            })?;

        // Load tokenizer - try different formats
        let tokenizer = if let Ok(tokenizer_json_path) = repo.get("tokenizer.json").await {
            // Direct tokenizer.json file (most models)
            Tokenizer::from_file(tokenizer_json_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?
        } else {
            // Try to build tokenizer from components (for models like microsoft/codebert-base)
            // Check for RoBERTa-style tokenizer (vocab.json + merges.txt)
            if let (Ok(vocab_path), Ok(merges_path)) =
                (repo.get("vocab.json").await, repo.get("merges.txt").await)
            {
                // Build RoBERTa/GPT2-style BPE tokenizer using BPE::from_file
                use tokenizers::{
                    models::bpe::BPE, normalizers, pre_tokenizers::byte_level::ByteLevel,
                    processors::roberta::RobertaProcessing,
                };

                // Use BPE::from_file which handles the vocab and merges loading
                let bpe = BPE::from_file(
                    vocab_path
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid vocab path"))?,
                    merges_path
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("Invalid merges path"))?,
                )
                .unk_token("<unk>".to_string())
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build BPE tokenizer: {:?}", e))?;

                let mut tokenizer = Tokenizer::new(bpe);

                // Add ByteLevel pre-tokenizer (for RoBERTa)
                tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));

                // Add RoBERTa post-processing
                let post_processor = RobertaProcessing::new(
                    ("</s>".to_string(), 2), // SEP token
                    ("<s>".to_string(), 0),  // CLS token
                )
                .trim_offsets(false)
                .add_prefix_space(true);
                tokenizer.with_post_processor(Some(post_processor));

                // Add normalizer
                let normalizer =
                    normalizers::Sequence::new(vec![normalizers::Strip::new(true, true).into()]);
                tokenizer.with_normalizer(Some(normalizer));

                tokenizer
            } else {
                return Err(anyhow::anyhow!(
                    "Could not find tokenizer files for model: {}. \
                    Expected either tokenizer.json or (vocab.json + merges.txt). \
                    This model may not be compatible.",
                    model_name
                ));
            }
        };

        // Try different weight file formats
        let weights_path = if let Ok(path) = repo.get("model.safetensors").await {
            path
        } else if let Ok(path) = repo.get("pytorch_model.bin").await {
            path
        } else {
            return Err(anyhow::anyhow!(
                "Could not find model weights in safetensors or pytorch format"
            ));
        };

        // Load configuration and detect architecture
        let config_content = std::fs::read_to_string(&config_path)?;
        let model_config: ModelConfig = serde_json::from_str(&config_content)
            .with_context(|| "Failed to parse config.json for architecture detection")?;

        let architecture = ModelArchitecture::from_config(&model_config)?;

        // Load model weights - only support safetensors for now
        let mut weights = if weights_path.to_string_lossy().ends_with(".safetensors") {
            candle_core::safetensors::load(&weights_path, &device)?
        } else {
            return Err(anyhow::anyhow!("PyTorch .bin format not supported in this implementation. Please use a model with safetensors format."));
        };

        // Qwen decoder models may omit Candle's required `model.` prefix.
        if matches!(
            architecture,
            ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3
        ) {
            let needs_prefix = weights
                .keys()
                .any(|k| k.starts_with("embed_tokens") || k.starts_with("layers."));
            if needs_prefix {
                let mut prefixed_weights = HashMap::new();
                for (key, value) in weights.into_iter() {
                    prefixed_weights.insert(format!("model.{}", key), value);
                }
                weights = prefixed_weights;
            }
        }

        // For MPNet models, strip "mpnet." prefix if present
        // sentence-transformers models store weights as "mpnet.embeddings.*", "mpnet.encoder.*"
        // but our MPNetModel expects "embeddings.*", "encoder.*"
        if matches!(architecture, ModelArchitecture::MPNet) {
            let has_prefix = weights.keys().any(|k| k.starts_with("mpnet."));
            if has_prefix {
                let mut stripped_weights = HashMap::new();
                for (key, value) in weights.into_iter() {
                    let new_key = key
                        .strip_prefix("mpnet.")
                        .map(|s| s.to_string())
                        .unwrap_or(key);
                    stripped_weights.insert(new_key, value);
                }
                weights = stripped_weights;
            }
        }

        let var_builder = VarBuilder::from_tensors(weights, DType::F32, &device);

        // Create model based on detected architecture
        let backend = match architecture {
            ModelArchitecture::Bert => {
                let config: BertConfig = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as BERT config")?;
                let model = BertModel::load(var_builder, &config)
                    .with_context(|| "Failed to load BERT model")?;
                Backend::Bert(model)
            }
            ModelArchitecture::Roberta => {
                let config: XLMRobertaConfig = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as RoBERTa config")?;
                let model = XLMRobertaModel::new(&config, var_builder)
                    .with_context(|| "Failed to load RoBERTa model")?;
                Backend::Roberta(model)
            }
            ModelArchitecture::JinaBert => {
                let config: JinaBertConfig = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as JinaBert config")?;
                let model = JinaBertModel::new(var_builder, &config)
                    .with_context(|| "Failed to load JinaBert model")?;
                Backend::JinaBert(model)
            }
            ModelArchitecture::JinaCodeBert => {
                let config: JinaBertConfig = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as JinaBert config")?;
                let model = JinaCodeBertModel::new(var_builder, &config)
                    .with_context(|| "Failed to load JinaCodeBert (QK-post-norm) model")?;
                Backend::JinaCodeBert(model)
            }
            ModelArchitecture::Qwen2 => {
                let config: Qwen2Config = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as Qwen2 config")?;
                let model = Qwen2Model::new(&config, var_builder)
                    .with_context(|| "Failed to load Qwen2 model")?;
                Backend::Qwen2(std::sync::Mutex::new(model))
            }
            ModelArchitecture::Qwen3 => {
                let config: Qwen3Config = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as Qwen3 config")?;
                let model = Qwen3Model::new(&config, var_builder)
                    .with_context(|| "Failed to load Qwen3 model")?;
                Backend::Qwen3(model)
            }
            ModelArchitecture::MPNet => {
                let config: MPNetConfig = serde_json::from_str(&config_content)
                    .with_context(|| "Failed to parse config.json as MPNet config")?;
                let model = MPNetModel::new(var_builder, &config)
                    .with_context(|| "Failed to load MPNet model")?;
                Backend::MPNet(model)
            }
        };

        Ok(Self {
            backend,
            tokenizer: Arc::new(tokenizer),
            device,
            revision,
        })
    }

    /// HF commit sha of the loaded weights.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// The tokenizer the model was loaded with.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    /// Generate embeddings for a single text
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        self.encode_batch(&[text.to_string()])
            .map(|embeddings| embeddings.into_iter().next().unwrap_or_default())
    }

    /// Generate embeddings for multiple texts
    pub fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::new();
        let tokenizer = &*self.tokenizer;
        let device = &self.device;

        for text in texts {
            let embedding = match &self.backend {
                Backend::Bert(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    // BertModel.forward: (input_ids, token_type_ids, attention_mask)
                    let token_type_ids = Tensor::zeros_like(&token_ids)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    let output =
                        model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
                    Self::mean_pool_and_normalize(&output, &attention_mask)?
                }
                Backend::Roberta(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    let token_type_ids = Tensor::zeros_like(&token_ids)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::F32, device)?;
                    // XLMRobertaModel.forward: (input_ids, attention_mask, token_type_ids, ...)
                    let output = model.forward(
                        &token_ids,
                        &attention_mask,
                        &token_type_ids,
                        None,
                        None,
                        None,
                    )?;
                    let attention_mask_u8 = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    Self::mean_pool_and_normalize(&output, &attention_mask_u8)?
                }
                Backend::JinaBert(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    // JinaBertModel.forward: (input_ids) only
                    let output = model.forward(&token_ids)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    Self::mean_pool_and_normalize(&output, &attention_mask)?
                }
                Backend::JinaCodeBert(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    // JinaCodeBertModel.forward: (input_ids) only — same interface as JinaBert
                    let output = model.forward(&token_ids)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    Self::mean_pool_and_normalize(&output, &attention_mask)?
                }
                Backend::Qwen2(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    // Qwen2Model.forward takes &mut self, so lock the mutex
                    let output = model
                        .lock()
                        .map_err(|e| anyhow::anyhow!("Qwen2 model mutex poisoned: {}", e))?
                        .forward(&token_ids, 0, None)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    Self::mean_pool_and_normalize(&output, &attention_mask)?
                }
                Backend::Qwen3(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    let output = model.clone().forward(&token_ids, 0)?;
                    Self::last_token_pool_and_normalize(&output)?
                }
                Backend::MPNet(model) => {
                    let encoding = tokenizer
                        .encode(text.as_str(), true)
                        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                    let tokens = encoding.get_ids();
                    let token_ids = Tensor::new(tokens, device)?.unsqueeze(0)?;
                    // MPNetModel.forward: (input_ids) only — position bias computed internally
                    let output = model.forward(&token_ids)?;
                    let attention_mask = Tensor::ones((1, tokens.len()), DType::U8, device)?;
                    Self::mean_pool_and_normalize(&output, &attention_mask)?
                }
            };
            all_embeddings.push(embedding);
        }

        Ok(all_embeddings)
    }

    /// Mean pooling + L2 normalization to produce a sentence embedding
    fn mean_pool_and_normalize(
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Vec<f32>> {
        // Convert attention mask to f32.
        // candle's mul requires exact shape match (no implicit broadcasting), so we expand the mask
        // to (batch, seq_len, hidden_size) before multiplying with hidden_states.
        // We keep the unexpanded mask to compute the token count for mean pooling.
        let mask_f32 = attention_mask.to_dtype(DType::F32)?; // (batch, seq_len)
        let mask_expanded = mask_f32
            .unsqueeze(2)? // (batch, seq_len, 1)
            .expand(hidden_states.shape())?; // (batch, seq_len, hidden_size)
        let masked = hidden_states.mul(&mask_expanded)?;
        let sum_hidden = masked.sum(1)?; // (batch, hidden_size)
                                         // sum_mask is (batch, 1) — use broadcast_div since candle's Div requires exact shape match
        let sum_mask = mask_f32.sum_keepdim(1)?; // (batch, 1)
        let mean_pooled = sum_hidden.broadcast_div(&sum_mask)?; // (batch, hidden_size)

        // L2 normalize — norm is (batch, 1), broadcast_div handles the shape mismatch
        let norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean_pooled.broadcast_div(&norm)?;

        Ok(normalized.squeeze(0)?.to_vec1::<f32>()?)
    }

    /// Qwen3 Embedding uses the final hidden state, followed by L2 normalization.
    fn last_token_pool_and_normalize(hidden_states: &Tensor) -> Result<Vec<f32>> {
        let (_, sequence_length, _) = hidden_states.dims3()?;
        let last_token = hidden_states.narrow(1, sequence_length.saturating_sub(1), 1)?;
        let norm = last_token.sqr()?.sum_keepdim(2)?.sqrt()?;
        Ok(last_token
            .broadcast_div(&norm)?
            .squeeze(1)?
            .squeeze(0)?
            .to_vec1::<f32>()?)
    }
}
#[cfg(feature = "huggingface")]
#[allow(clippy::type_complexity)]
static MODEL_CACHE: LazyLock<Arc<RwLock<HashMap<String, Arc<HuggingFaceModel>>>>> =
    LazyLock::new(|| Arc::new(RwLock::new(HashMap::new())));

#[cfg(feature = "huggingface")]
static MODEL_LOAD_LOCKS: LazyLock<AsyncMutex<HashMap<String, Weak<AsyncMutex<()>>>>> =
    LazyLock::new(|| AsyncMutex::new(HashMap::new()));

#[cfg(feature = "huggingface")]
async fn model_load_lock(model_name: &str) -> Arc<AsyncMutex<()>> {
    let mut locks = MODEL_LOAD_LOCKS.lock().await;
    locks.retain(|_, lock| lock.strong_count() > 0);

    if let Some(lock) = locks.get(model_name).and_then(Weak::upgrade) {
        return lock;
    }

    let lock = Arc::new(AsyncMutex::new(()));
    locks.insert(model_name.to_string(), Arc::downgrade(&lock));
    lock
}

#[cfg(feature = "huggingface")]
/// HuggingFace provider implementation
pub struct HuggingFaceProvider;

#[cfg(feature = "huggingface")]
impl HuggingFaceProvider {
    /// Get or load a model from cache
    async fn get_model(model_name: &str) -> Result<Arc<HuggingFaceModel>> {
        {
            let cache = MODEL_CACHE.read().await;
            if let Some(model) = cache.get(model_name) {
                return Ok(model.clone());
            }
        }

        // Coordinate the first load per model. Different models can still load in parallel.
        let load_lock = model_load_lock(model_name).await;
        let _load_guard = load_lock.lock().await;

        // Another request may have populated the cache while this one was waiting.
        {
            let cache = MODEL_CACHE.read().await;
            if let Some(model) = cache.get(model_name) {
                return Ok(model.clone());
            }
        }

        let model = HuggingFaceModel::load_inner(model_name)
            .await
            .with_context(|| format!("Failed to load HuggingFace model: {}", model_name))?;

        let model_arc = Arc::new(model);

        // Add to cache
        {
            let mut cache = MODEL_CACHE.write().await;
            cache.insert(model_name.to_string(), model_arc.clone());
        }

        Ok(model_arc)
    }

    /// Generate embeddings for a single text
    pub async fn generate_embeddings(contents: &str, model: &str) -> Result<Vec<f32>> {
        let model_instance = Self::get_model(model).await?;

        // Run encoding in a blocking task to avoid blocking async runtime
        let contents = contents.to_string();
        let result =
            tokio::task::spawn_blocking(move || model_instance.encode(&contents)).await??;

        Ok(result)
    }

    /// Generate batch embeddings for multiple texts
    pub async fn generate_embeddings_batch(
        texts: Vec<String>,
        model: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let model_instance = Self::get_model(model).await?;

        // Run encoding in a blocking task to avoid blocking async runtime
        let result =
            tokio::task::spawn_blocking(move || model_instance.encode_batch(&texts)).await??;

        Ok(result)
    }
}

// Stubs for when huggingface feature is disabled
#[cfg(not(feature = "huggingface"))]
use anyhow::Result;

#[cfg(not(feature = "huggingface"))]
pub struct HuggingFaceProvider;

#[cfg(not(feature = "huggingface"))]
impl HuggingFaceProvider {
    pub async fn generate_embeddings(_contents: &str, _model: &str) -> Result<Vec<f32>> {
        Err(anyhow::anyhow!(
            "HuggingFace support is not compiled in. Please rebuild with --features huggingface"
        ))
    }

    pub async fn generate_embeddings_batch(
        _texts: Vec<String>,
        _model: &str,
    ) -> Result<Vec<Vec<f32>>> {
        Err(anyhow::anyhow!(
            "HuggingFace support is not compiled in. Please rebuild with --features huggingface"
        ))
    }
}
use super::super::types::InputType;
use super::super::EmbeddingUsage;
use super::EmbeddingProvider;

/// HuggingFace provider implementation for trait
#[cfg(feature = "huggingface")]
pub struct HuggingFaceProviderImpl {
    model_name: String,
    dimension: usize,
}

#[cfg(feature = "huggingface")]
impl HuggingFaceProviderImpl {
    pub async fn new(model: &str) -> Result<Self> {
        #[cfg(not(feature = "huggingface"))]
        {
            Err(anyhow::anyhow!("HuggingFace provider requires 'huggingface' feature to be enabled. Cannot validate model '{}' without Hub API access.", model))
        }

        #[cfg(feature = "huggingface")]
        {
            let dimension = Self::get_model_dimension(model).await?;
            Ok(Self {
                model_name: model.to_string(),
                dimension,
            })
        }
    }

    #[cfg(feature = "huggingface")]
    async fn get_model_dimension(model: &str) -> Result<usize> {
        Self::get_dimension_from_config(model).await
    }

    /// Get model dimension using Candle config structs (like examples)
    #[cfg(feature = "huggingface")]
    async fn get_dimension_from_config(model_name: &str) -> Result<usize> {
        // Download config.json
        let config_json = Self::download_config_direct(model_name).await?;

        // Try different Candle config types - JinaBert first, then standard Bert
        if let Ok(config) = Self::parse_as_jina_bert_config(&config_json) {
            return Ok(config.hidden_size);
        }

        if let Ok(config) = Self::parse_as_bert_config(&config_json) {
            return Ok(config.hidden_size);
        }

        // Fallback to JSON parsing
        Self::parse_hidden_size_from_json(&config_json, model_name)
    }

    /// Try to parse config as JinaBert config (for Jina models)
    #[cfg(feature = "huggingface")]
    fn parse_as_jina_bert_config(config_json: &str) -> Result<JinaBertConfig> {
        serde_json::from_str::<JinaBertConfig>(config_json)
            .map_err(|e| anyhow::anyhow!("Failed to parse as JinaBertConfig: {}", e))
    }

    /// Try to parse config as standard Candle BertConfig
    #[cfg(feature = "huggingface")]
    fn parse_as_bert_config(
        config_json: &str,
    ) -> Result<candle_transformers::models::bert::Config> {
        use candle_transformers::models::bert::Config as BertConfig;
        serde_json::from_str::<BertConfig>(config_json)
            .map_err(|e| anyhow::anyhow!("Failed to parse as BertConfig: {}", e))
    }

    /// Parse hidden_size from JSON config flexibly
    #[cfg(feature = "huggingface")]
    fn parse_hidden_size_from_json(config_json: &str, model_name: &str) -> Result<usize> {
        use serde_json::Value;

        let config: Value = serde_json::from_str(config_json).with_context(|| {
            format!(
                "Failed to parse config.json as JSON for model: {}",
                model_name
            )
        })?;

        // Try different field names that contain embedding dimensions
        let dimension_fields = ["hidden_size", "d_model", "embedding_size", "dim"];

        for field in &dimension_fields {
            if let Some(dim) = config.get(field).and_then(|v| v.as_u64()) {
                tracing::debug!(
                    "Found dimension {} for model {} from config.json field '{}'",
                    dim,
                    model_name,
                    field
                );
                return Ok(dim as usize);
            }
        }

        Err(anyhow::anyhow!(
            "No dimension field found in config.json for model '{}'. \
			Searched for fields: {:?}. Available fields: {:?}",
            model_name,
            dimension_fields,
            config
                .as_object()
                .map(|obj| obj.keys().collect::<Vec<_>>())
                .unwrap_or_default()
        ))
    }

    /// Download config.json directly from HuggingFace Hub using HTTP
    #[cfg(feature = "huggingface")]
    async fn download_config_direct(model_name: &str) -> Result<String> {
        use reqwest;

        // Construct direct URL to config.json
        let config_url = format!("https://huggingface.co/{}/raw/main/config.json", model_name);

        tracing::debug!("Downloading config from: {}", config_url);

        // Use reqwest for direct HTTP download
        let client = reqwest::Client::new();
        let response = client
            .get(&config_url)
            .header("User-Agent", "octocode/0.7.1")
            .send()
            .await
            .with_context(|| format!("Failed to download config.json from {}", config_url))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download config.json for model '{}'. HTTP status: {}. \
				This could be due to:\n\
				1. Model doesn't exist on HuggingFace Hub\n\
				2. Network connectivity issues\n\
				3. Model is private and requires authentication\n\
				4. Model doesn't have a config.json file",
                model_name,
                response.status()
            ));
        }

        let config_text = response.text().await.with_context(|| {
            format!(
                "Failed to read config.json response for model: {}",
                model_name
            )
        })?;

        Ok(config_text)
    }
}

#[cfg(feature = "huggingface")]
#[async_trait::async_trait]
impl EmbeddingProvider for HuggingFaceProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<(Vec<f32>, EmbeddingUsage)> {
        // In-process, local, always unpriced → cost None; tokens estimated (tiktoken).
        let input_tokens = super::super::count_tokens(text) as u64;
        let vector = HuggingFaceProvider::generate_embeddings(text, &self.model_name).await?;
        Ok((
            vector,
            EmbeddingUsage {
                input_tokens,
                cost: None,
            },
        ))
    }

    async fn generate_embeddings_batch(
        &self,
        texts: Vec<String>,
        input_type: InputType,
    ) -> Result<(Vec<Vec<f32>>, EmbeddingUsage)> {
        // Apply prefix manually for HuggingFace (doesn't support input_type API)
        let processed_texts: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();
        let input_tokens: u64 = processed_texts
            .iter()
            .map(|t| super::super::count_tokens(t) as u64)
            .sum();
        let vectors =
            HuggingFaceProvider::generate_embeddings_batch(processed_texts, &self.model_name)
                .await?;
        Ok((
            vectors,
            EmbeddingUsage {
                input_tokens,
                cost: None,
            },
        ))
    }

    fn get_dimension(&self) -> usize {
        self.dimension
    }

    async fn model_revision(&self) -> Result<Option<String>> {
        let model = HuggingFaceProvider::get_model(&self.model_name).await?;
        Ok(Some(model.revision().to_owned()))
    }

    async fn tokenizer(&self) -> Result<Option<Arc<Tokenizer>>> {
        let model = HuggingFaceProvider::get_model(&self.model_name).await?;
        Ok(Some(model.tokenizer().clone()))
    }

    fn is_model_supported(&self) -> bool {
        // For HuggingFace, we support many models, so return true for most cases
        // The actual validation happens when trying to load the model
        true
    }
}

#[cfg(all(test, feature = "huggingface"))]
#[path = "mod_tests.rs"]
mod tests;
