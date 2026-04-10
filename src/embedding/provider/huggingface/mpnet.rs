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

//! MPNet model implementation for the HuggingFace provider.
//!
//! MPNet (Masked and Permuted Pre-training for Language Understanding) is a
//! transformer encoder similar to BERT but with two key differences:
//!
//! - **Relative position bias**: Uses bucketed relative position embeddings
//!   (similar to T5) instead of absolute position embeddings. The encoder has
//!   a shared `relative_attention_bias` embedding table that maps relative
//!   position buckets to per-head bias values.
//! - **Position embeddings in embeddings layer**: Despite using relative
//!   position bias in attention, MPNet also has absolute position embeddings
//!   in the embedding layer (with padding_idx=1).
//!
//! Attention naming differs from BERT: uses `q`, `k`, `v`, `o` instead of
//! `query`, `key`, `value`, `dense` in the self-attention layer.
//!
//! Reference: <https://arxiv.org/abs/2004.09297>
//! Weight format: `sentence-transformers/all-mpnet-base-v2` (safetensors)

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct MPNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    #[serde(default = "default_hidden_dropout")]
    pub hidden_dropout_prob: f64,
    #[serde(default = "default_attention_dropout")]
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_relative_attention_num_buckets")]
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
}

fn default_hidden_dropout() -> f64 {
    0.1
}
fn default_attention_dropout() -> f64 {
    0.1
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_relative_attention_num_buckets() -> usize {
    32
}
fn default_pad_token_id() -> usize {
    1
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    padding_idx: usize,
}

impl MPNetEmbeddings {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let word_embeddings = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            padding_idx: cfg.pad_token_id,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Validate input is 2D (batch, seq_len)
        input_ids.dims2()?;

        let input_embeddings = self.word_embeddings.forward(input_ids)?;

        // Create position ids from input_ids, accounting for padding
        let position_ids = self.create_position_ids_from_input_ids(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        let embeddings = (input_embeddings + position_embeddings)?;
        self.layer_norm.forward(&embeddings)
    }

    /// Create position IDs from input IDs, respecting padding.
    /// Non-padding tokens get incremental positions starting from padding_idx + 1.
    fn create_position_ids_from_input_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Create a mask: 1 for non-padding, 0 for padding
        let padding_idx_tensor = Tensor::new(&[self.padding_idx as u32], input_ids.device())?;
        let mask = input_ids
            .ne(&padding_idx_tensor.broadcast_as(input_ids.shape())?)?
            .to_dtype(DType::U32)?;

        // Cumulative sum along seq dimension to get incremental positions
        let cumsum = cumsum_u32(&mask)?;

        // Add padding_idx offset: positions start at padding_idx + 1
        let offset = Tensor::new(&[self.padding_idx as u32], input_ids.device())?
            .broadcast_as(cumsum.shape())?;
        let position_ids = (cumsum + offset)?;

        // Zero out padding positions (multiply by mask)
        position_ids.broadcast_mul(&mask)
    }
}

/// Simple cumulative sum for u32 tensors along the last dimension.
fn cumsum_u32(tensor: &Tensor) -> Result<Tensor> {
    let (batch, seq_len) = tensor.dims2()?;
    let data = tensor.to_vec2::<u32>()?;
    let mut result = vec![vec![0u32; seq_len]; batch];
    for b in 0..batch {
        let mut acc = 0u32;
        for s in 0..seq_len {
            acc += data[b][s];
            result[b][s] = acc;
        }
    }
    Tensor::new(result, tensor.device())
}

// ---------------------------------------------------------------------------
// Self-Attention
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetSelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl MPNetSelfAttention {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;

        let q = linear(cfg.hidden_size, all_head_size, vb.pp("q"))?;
        let k = linear(cfg.hidden_size, all_head_size, vb.pp("k"))?;
        let v = linear(cfg.hidden_size, all_head_size, vb.pp("v"))?;
        let o = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("o"))?;

        Ok(Self {
            q,
            k,
            v,
            o,
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_shape = xs.dims().to_vec();
        new_shape.pop();
        new_shape.push(self.num_attention_heads);
        new_shape.push(self.attention_head_size);
        xs.reshape(new_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let query_layer = self.transpose_for_scores(&self.q.forward(hidden_states)?)?;
        let key_layer = self.transpose_for_scores(&self.k.forward(hidden_states)?)?;
        let value_layer = self.transpose_for_scores(&self.v.forward(hidden_states)?)?;

        // Scaled dot-product attention
        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;

        // Add relative position bias if provided
        let attention_scores = if let Some(bias) = position_bias {
            attention_scores.broadcast_add(bias)?
        } else {
            attention_scores
        };

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;

        self.o.forward(&context_layer)
    }
}

// ---------------------------------------------------------------------------
// Attention block (self-attention + LayerNorm + residual)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetAttention {
    attn: MPNetSelfAttention,
    layer_norm: LayerNorm,
}

impl MPNetAttention {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let attn = MPNetSelfAttention::new(vb.pp("attn"), cfg)?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { attn, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let self_output = self.attn.forward(hidden_states, position_bias)?;
        // Residual + LayerNorm (post-norm)
        self.layer_norm.forward(&(self_output + hidden_states)?)
    }
}

// ---------------------------------------------------------------------------
// Intermediate (FFN first half)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetIntermediate {
    dense: Linear,
}

impl MPNetIntermediate {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("dense"))?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        hidden_states.gelu()
    }
}

// ---------------------------------------------------------------------------
// Output (FFN second half + LayerNorm + residual)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl MPNetOutput {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let dense = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        // Residual + LayerNorm
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer layer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetLayer {
    attention: MPNetAttention,
    intermediate: MPNetIntermediate,
    output: MPNetOutput,
}

impl MPNetLayer {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let attention = MPNetAttention::new(vb.pp("attention"), cfg)?;
        let intermediate = MPNetIntermediate::new(vb.pp("intermediate"), cfg)?;
        let output = MPNetOutput::new(vb.pp("output"), cfg)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, position_bias: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, position_bias)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct MPNetEncoder {
    layers: Vec<MPNetLayer>,
    relative_attention_bias: Embedding,
    num_attention_heads: usize,
}

impl MPNetEncoder {
    fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| MPNetLayer::new(vb.pp(format!("layer.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;

        let relative_attention_bias = embedding(
            cfg.relative_attention_num_buckets,
            cfg.num_attention_heads,
            vb.pp("relative_attention_bias"),
        )?;

        Ok(Self {
            layers,
            relative_attention_bias,
            num_attention_heads: cfg.num_attention_heads,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let position_bias = self.compute_position_bias(hidden_states)?;
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, Some(&position_bias))?;
        }
        Ok(hidden_states)
    }

    /// Compute relative position bias using bucketed relative positions.
    ///
    /// This mirrors the T5-style relative position bias used by MPNet:
    /// 1. Compute relative positions between all (query, key) pairs
    /// 2. Map to buckets using logarithmic bucketing
    /// 3. Look up bias values from the embedding table
    /// 4. Reshape to (batch, num_heads, seq_len, seq_len)
    fn compute_position_bias(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let device = x.device();
        let num_buckets = self.relative_attention_bias.embeddings().dim(0)?;

        // Compute relative position matrix: memory_position - context_position
        let context_position = Tensor::arange(0i64, seq_len as i64, device)?.unsqueeze(1)?; // (seq_len, 1)
        let memory_position = Tensor::arange(0i64, seq_len as i64, device)?.unsqueeze(0)?; // (1, seq_len)
        let relative_position = memory_position.broadcast_sub(&context_position)?; // (seq_len, seq_len)

        // Map to buckets
        let rp_bucket = self.relative_position_bucket(
            &relative_position,
            num_buckets,
            128, // max_distance
        )?;

        // Look up bias values: (seq_len, seq_len) -> (seq_len, seq_len, num_heads)
        let values = self.relative_attention_bias.forward(&rp_bucket)?;

        // Permute to (num_heads, seq_len, seq_len) then expand to (batch, num_heads, seq_len, seq_len)
        let values = values.permute((2, 0, 1))?.unsqueeze(0)?;
        values.expand((bsz, self.num_attention_heads, seq_len, seq_len))
    }

    /// Map relative positions to buckets using logarithmic bucketing.
    ///
    /// Half the buckets are for exact positions (small distances),
    /// the other half are for logarithmically larger distances.
    fn relative_position_bucket(
        &self,
        relative_position: &Tensor,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Tensor> {
        let device = relative_position.device();
        let rp = relative_position.to_vec2::<i64>()?;
        let half_buckets = num_buckets / 2;
        let max_exact = half_buckets / 2;

        let (rows, cols) = (rp.len(), rp[0].len());
        let mut result = vec![vec![0u32; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                let rel = rp[i][j];
                let mut ret = 0u32;

                let n = if rel < 0 {
                    ret += half_buckets as u32;
                    (-rel) as usize
                } else {
                    rel as usize
                };

                if n < max_exact {
                    ret += n as u32;
                } else {
                    let val = max_exact as f64
                        + ((n as f64 / max_exact as f64).ln()
                            / (max_distance as f64 / max_exact as f64).ln()
                            * (half_buckets - max_exact) as f64);
                    let val = val.min((num_buckets - 1) as f64) as u32;
                    ret += val;
                }

                result[i][j] = ret;
            }
        }

        Tensor::new(result, device)
    }
}

// ---------------------------------------------------------------------------
// Full model
// ---------------------------------------------------------------------------

/// MPNet model for sentence embeddings.
///
/// Used by `sentence-transformers/all-mpnet-base-v2` and similar models.
/// Architecture: `MPNetForMaskedLM` / `MPNetModel`.
#[derive(Clone, Debug)]
pub struct MPNetModel {
    embeddings: MPNetEmbeddings,
    encoder: MPNetEncoder,
    pub device: Device,
}

impl MPNetModel {
    pub fn new(vb: VarBuilder, cfg: &MPNetConfig) -> Result<Self> {
        let embeddings = MPNetEmbeddings::new(vb.pp("embeddings"), cfg)?;
        let encoder = MPNetEncoder::new(vb.pp("encoder"), cfg)?;
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }
}

impl Module for MPNetModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        self.encoder.forward(&embedding_output)
    }
}
