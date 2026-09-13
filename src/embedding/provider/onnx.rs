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
 * ONNX Runtime Provider Implementation
 *
 * Runs a HuggingFace-hosted sentence-transformer through ONNX Runtime
 * (via fastembed's "user defined model" path) instead of candle.
 *
 * Why this exists: the `huggingface` provider loads `model.safetensors`
 * through candle and always mean-pools. That is fine, but it cannot use a
 * quantized graph, and it ignores the repo's declared pooling mode. This
 * provider loads the ONNX graph from the same repo, honours
 * `1_Pooling/config.json`, and gets ORT's fused int8 kernels — typically
 * 2-4x faster on CPU at ~1/4 the on-disk size.
 *
 * Model string format:
 *
 *   onnx:<org>/<repo>                     → auto-pick the ONNX graph
 *   onnx:<org>/<repo>#onnx/model.onnx     → pin an exact file in the repo
 *
 * Auto-pick order (first hit wins):
 *   onnx/model_quantized.onnx, onnx/model_int8.onnx, onnx/model.onnx,
 *   model_quantized.onnx, model.onnx
 *
 * Quantization note: export STATIC (QDQ) int8, not dynamic. Dynamic
 * quantization recomputes activation scales per batch, which makes vectors
 * depend on batch composition; fastembed rejects batching for such graphs.
 * This provider always declares `QuantizationMode::None` so batching stays
 * available, which is correct for fp32, fp16 and static-int8 graphs.
 */

#[cfg(feature = "onnx")]
use anyhow::{Context, Result};
#[cfg(feature = "onnx")]
use fastembed::{
    InitOptionsUserDefined, Pooling, QuantizationMode, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
#[cfg(feature = "onnx")]
use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
#[cfg(feature = "onnx")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

#[cfg(feature = "onnx")]
use super::super::{types::InputType, EmbeddingProvider, EmbeddingUsage};

/// Candidate graph paths inside the HF repo, most-preferred first.
#[cfg(feature = "onnx")]
const GRAPH_CANDIDATES: &[&str] = &[
    "onnx/model_quantized.onnx",
    "onnx/model_int8.onnx",
    "onnx/model.onnx",
    "model_quantized.onnx",
    "model.onnx",
];

/// Token window used when building the ORT session. The caller is expected to
/// respect the model's own trained window; this is only the hard ceiling.
#[cfg(feature = "onnx")]
const MAX_SEQ_LEN: usize = 512;

/// Parse `<repo>` or `<repo>#<file>` out of the model string.
#[cfg(feature = "onnx")]
fn split_repo_and_file(model: &str) -> (String, Option<String>) {
    match model.split_once('#') {
        Some((repo, file)) => (repo.trim().to_string(), Some(file.trim().to_string())),
        None => (model.trim().to_string(), None),
    }
}

/// ONNX Runtime provider backed by fastembed's user-defined model path.
#[cfg(feature = "onnx")]
pub struct OnnxProviderImpl {
    /// `TextEmbedding::embed` takes `&mut self`, so shared access is serialized.
    model: Arc<Mutex<TextEmbedding>>,
    tokenizer: Arc<Tokenizer>,
    dimension: usize,
    revision: String,
}

#[cfg(feature = "onnx")]
impl OnnxProviderImpl {
    pub async fn new(model: &str) -> Result<Self> {
        let (repo_id, pinned_file) = split_repo_and_file(model);
        if repo_id.is_empty() {
            anyhow::bail!(
                "onnx provider needs a HuggingFace repo id, e.g. onnx:muvon/octomind-embed"
            );
        }

        // Share the model cache with the fastembed / huggingface providers so a
        // machine downloads each repo once regardless of backend.
        let cache_dir =
            crate::storage::get_model_cache_dir().context("Failed to get model cache directory")?;
        std::env::set_var("HF_HOME", &cache_dir);

        let api = ApiBuilder::new()
            .with_progress(false)
            .build()
            .context("Failed to initialize HuggingFace API")?;
        let repo = api.repo(Repo::new(repo_id.clone(), RepoType::Model));

        // config.json is fetched first: it is the cheapest file that is always
        // present, and its snapshot dir names the exact revision in use.
        let config_path = repo.get("config.json").await.with_context(|| {
            format!("Failed to download config.json for ONNX model: {}", repo_id)
        })?;
        let revision = config_path
            .parent()
            .and_then(|dir| dir.file_name())
            .and_then(|sha| sha.to_str())
            .map(str::to_owned)
            .with_context(|| {
                format!(
                    "Unexpected hf_hub cache layout for model {}: {}",
                    repo_id,
                    config_path.display()
                )
            })?;

        // Locate the graph. A pinned file must exist; otherwise probe candidates.
        let mut graph_path = None;
        let candidates: Vec<String> = match &pinned_file {
            Some(f) => vec![f.clone()],
            None => GRAPH_CANDIDATES.iter().map(|s| s.to_string()).collect(),
        };
        for candidate in &candidates {
            if let Ok(path) = repo.get(candidate).await {
                graph_path = Some((candidate.clone(), path));
                break;
            }
        }
        let (graph_name, graph_path) = graph_path.ok_or_else(|| {
            anyhow::anyhow!(
                "No ONNX graph found in '{}'. Looked for: {}. \
				 Publish an `onnx/` export in the repo, or pin one with \
				 `onnx:{}#<path-to.onnx>`.",
                repo_id,
                candidates.join(", "),
                repo_id
            )
        })?;
        let onnx_file = std::fs::read(&graph_path)
            .with_context(|| format!("Failed to read ONNX graph {}", graph_path.display()))?;

        // Graphs above ~2 GB keep their weights in a sibling `*_data` file. Small
        // encoders never do, so a miss here is normal and not an error.
        let external_data_name = format!("{}_data", graph_name);
        let external_data = match repo.get(&external_data_name).await {
            Ok(path) => Some((
                std::path::Path::new(&external_data_name)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&external_data_name)
                    .to_string(),
                std::fs::read(&path).with_context(|| {
                    format!("Failed to read external initializer {}", path.display())
                })?,
            )),
            Err(_) => None,
        };

        let tokenizer_files = Self::fetch_tokenizer_files(&repo, &repo_id).await?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .await
            .with_context(|| format!("Failed to download tokenizer.json for: {}", repo_id))?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer for {}: {}", repo_id, e))?;

        // Pooling is the single most dangerous default here: getting it wrong
        // produces plausible-looking but semantically wrong vectors, with no
        // error. Read the repo's own declaration; fall back to mean, which is
        // what the candle provider has always applied.
        let (pooling, pooling_dim) = Self::fetch_pooling(&repo).await;
        let dimension = match pooling_dim {
            Some(dim) => dim,
            None => Self::dimension_from_config(&config_path)?,
        };

        let mut user_model = UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files)
            .with_pooling(pooling)
            // Always None: fp32/fp16/static-int8 graphs are batch-invariant, and
            // declaring Dynamic would disable batching entirely.
            .with_quantization(QuantizationMode::None);
        if let Some((name, buffer)) = external_data {
            user_model = user_model.with_external_initializer(name, buffer);
        }

        let options = InitOptionsUserDefined::new().with_max_length(MAX_SEQ_LEN);
        let embedding = TextEmbedding::try_new_from_user_defined(user_model, options)
            .with_context(|| format!("Failed to initialize ONNX session for {}", repo_id))?;

        tracing::debug!(
            "onnx provider ready: {} graph={} dim={} revision={}",
            repo_id,
            graph_name,
            dimension,
            revision
        );

        Ok(Self {
            model: Arc::new(Mutex::new(embedding)),
            tokenizer: Arc::new(tokenizer),
            dimension,
            revision,
        })
    }

    /// The four tokenizer files fastembed requires, as raw bytes.
    async fn fetch_tokenizer_files(
        repo: &hf_hub::api::tokio::ApiRepo,
        repo_id: &str,
    ) -> Result<TokenizerFiles> {
        async fn read(
            repo: &hf_hub::api::tokio::ApiRepo,
            name: &str,
            repo_id: &str,
        ) -> Result<Vec<u8>> {
            let path = repo
                .get(name)
                .await
                .with_context(|| format!("Failed to download {} for: {}", name, repo_id))?;
            std::fs::read(&path).with_context(|| format!("Failed to read {}", path.display()))
        }

        Ok(TokenizerFiles {
            tokenizer_file: read(repo, "tokenizer.json", repo_id).await?,
            config_file: read(repo, "config.json", repo_id).await?,
            special_tokens_map_file: read(repo, "special_tokens_map.json", repo_id).await?,
            tokenizer_config_file: read(repo, "tokenizer_config.json", repo_id).await?,
        })
    }

    /// Read `1_Pooling/config.json`. Returns the declared pooling mode and, when
    /// present, the embedding dimension it records.
    async fn fetch_pooling(repo: &hf_hub::api::tokio::ApiRepo) -> (Pooling, Option<usize>) {
        let Ok(path) = repo.get("1_Pooling/config.json").await else {
            tracing::debug!("onnx provider: no 1_Pooling/config.json — assuming mean pooling");
            return (Pooling::Mean, None);
        };
        let Ok(raw) = std::fs::read_to_string(&path) else {
            return (Pooling::Mean, None);
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&raw) else {
            return (Pooling::Mean, None);
        };

        let dim = value
            .get("embedding_dimension")
            .and_then(|v| v.as_u64())
            .map(|d| d as usize);

        // sentence-transformers writes either the `pooling_mode` string or a set
        // of `pooling_mode_*` booleans, depending on the version that saved it.
        let mode = value.get("pooling_mode").and_then(|v| v.as_str());
        let pooling = match mode {
            Some(m) if m.eq_ignore_ascii_case("cls") => Pooling::Cls,
            Some(m) if m.eq_ignore_ascii_case("mean") => Pooling::Mean,
            _ => {
                if value
                    .get("pooling_mode_cls_token")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    Pooling::Cls
                } else {
                    Pooling::Mean
                }
            }
        };
        (pooling, dim)
    }

    /// `hidden_size` from config.json — the fallback when 1_Pooling is absent.
    fn dimension_from_config(config_path: &std::path::Path) -> Result<usize> {
        let raw = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let value: serde_json::Value =
            serde_json::from_str(&raw).context("Failed to parse config.json")?;
        for field in ["hidden_size", "d_model", "embedding_size", "dim"] {
            if let Some(dim) = value.get(field).and_then(|v| v.as_u64()) {
                return Ok(dim as usize);
            }
        }
        anyhow::bail!(
            "No embedding dimension found in config.json ({})",
            config_path.display()
        )
    }
}

#[cfg(feature = "onnx")]
#[async_trait::async_trait]
impl EmbeddingProvider for OnnxProviderImpl {
    async fn generate_embedding(&self, text: &str) -> Result<(Vec<f32>, EmbeddingUsage)> {
        // In-process, local, always unpriced → cost None; tokens estimated (tiktoken).
        let input_tokens = super::super::count_tokens(text) as u64;
        let (mut vectors, _) = self
            .generate_embeddings_batch(vec![text.to_string()], InputType::None)
            .await?;
        let vector = if vectors.is_empty() {
            anyhow::bail!("ONNX session returned no embeddings")
        } else {
            vectors.remove(0)
        };
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
        // Prefixes are applied manually — ONNX graphs carry no input_type API.
        let processed: Vec<String> = texts
            .into_iter()
            .map(|text| input_type.apply_prefix(&text))
            .collect();
        let input_tokens: u64 = processed
            .iter()
            .map(|t| super::super::count_tokens(t) as u64)
            .sum();

        let model = self.model.clone();
        let vectors = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<f32>>> {
            let mut model = model
                .lock()
                .map_err(|e| anyhow::anyhow!("ONNX model mutex poisoned: {}", e))?;
            model.embed(processed, None)
        })
        .await??;

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
        Ok(Some(self.revision.clone()))
    }

    async fn tokenizer(&self) -> Result<Option<Arc<Tokenizer>>> {
        Ok(Some(self.tokenizer.clone()))
    }

    fn is_model_supported(&self) -> bool {
        // Construction downloads and opens the graph, so an existing instance is
        // by definition supported.
        true
    }
}

// Stub for builds without the `onnx` feature.
#[cfg(not(feature = "onnx"))]
use anyhow::Result;

#[cfg(not(feature = "onnx"))]
pub struct OnnxProviderImpl;

#[cfg(not(feature = "onnx"))]
impl OnnxProviderImpl {
    pub async fn new(_model: &str) -> Result<Self> {
        Err(anyhow::anyhow!(
            "ONNX support is not compiled in. Please rebuild with --features onnx"
        ))
    }
}

// The stub still implements the trait so `create_embedding_provider_from_parts`
// compiles feature-free; every method reports the missing feature. Unreachable
// in practice — `new` above fails before an instance can exist.
#[cfg(not(feature = "onnx"))]
#[async_trait::async_trait]
impl super::EmbeddingProvider for OnnxProviderImpl {
    async fn generate_embedding(
        &self,
        _text: &str,
    ) -> Result<(Vec<f32>, super::super::EmbeddingUsage)> {
        Err(anyhow::anyhow!(
            "ONNX support is not compiled in. Please rebuild with --features onnx"
        ))
    }

    async fn generate_embeddings_batch(
        &self,
        _texts: Vec<String>,
        _input_type: super::super::types::InputType,
    ) -> Result<(Vec<Vec<f32>>, super::super::EmbeddingUsage)> {
        Err(anyhow::anyhow!(
            "ONNX support is not compiled in. Please rebuild with --features onnx"
        ))
    }

    fn get_dimension(&self) -> usize {
        0
    }

    fn is_model_supported(&self) -> bool {
        false
    }
}

#[cfg(all(test, feature = "onnx"))]
#[path = "onnx_tests.rs"]
mod tests;
