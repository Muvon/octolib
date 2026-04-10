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

//! JinaBERT QK-post-norm model implementation for the HuggingFace provider.
//!
//! This implements the QK-post-norm variant of JinaBERT used by models like
//! `jinaai/jina-embeddings-v2-base-code`. The key architectural differences from
//! standard JinaBERT (candle's `jina_bert.rs`) are:
//!
//! - **Attention**: LayerNorm applied to Q and K projections before computing
//!   attention scores (`layer_norm_q`, `layer_norm_k` tensors)
//! - **MLP**: Uses `up_gated_layer` + `down_layer` instead of `gated_layers` + `wo`
//! - **Block norms**: Pre-norm with `layer_norm_1` and `layer_norm_2` instead of
//!   post-norm `mlp.layernorm`
//!
//! Based on HuggingFace TEI's `jina_code.rs`, adapted for candle 0.9.2 primitives.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, Embedding, LayerNorm, Module, VarBuilder};
use candle_transformers::models::jina_bert::Config;
use candle_transformers::models::with_tracing::{linear, Linear};

/// Build ALiBi position bias tensor.
///
/// Replicates candle's `jina_bert::build_alibi_bias` which is not public.
fn build_alibi_bias(cfg: &Config) -> Result<Tensor> {
    let n_heads = cfg.num_attention_heads;
    let seq_len = cfg.max_position_embeddings;
    let alibi_bias = Tensor::arange(0, seq_len as i64, &Device::Cpu)?.to_dtype(DType::F32)?;
    let alibi_bias = {
        let a1 = alibi_bias.reshape((1, seq_len))?;
        let a2 = alibi_bias.reshape((seq_len, 1))?;
        a1.broadcast_sub(&a2)?.abs()?.broadcast_left(n_heads)?
    };
    let mut n_heads2 = 1;
    while n_heads2 < n_heads {
        n_heads2 *= 2;
    }
    let slopes = (1..=n_heads2)
        .map(|v| -1f32 / 2f32.powf((v * 8) as f32 / n_heads2 as f32))
        .collect::<Vec<_>>();
    let slopes = if n_heads2 == n_heads {
        slopes
    } else {
        slopes
            .iter()
            .skip(1)
            .step_by(2)
            .chain(slopes.iter().step_by(2))
            .take(n_heads)
            .cloned()
            .collect::<Vec<f32>>()
    };
    let slopes = Tensor::new(slopes, &Device::Cpu)?.reshape((1, (), 1, 1))?;
    alibi_bias.to_dtype(DType::F32)?.broadcast_mul(&slopes)
}

// ---------------------------------------------------------------------------
// Embeddings — identical to standard JinaBERT but candle's is private
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct JinaCodeEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl JinaCodeEmbeddings {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let word_embeddings = Embedding::new(
            vb.pp("word_embeddings")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?,
            cfg.hidden_size,
        );
        let token_type_embeddings = Embedding::new(
            vb.pp("token_type_embeddings")
                .get((cfg.type_vocab_size, cfg.hidden_size), "weight")?,
            cfg.hidden_size,
        );
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }
}

impl Module for JinaCodeEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_ids =
            Tensor::zeros(seq_len, DType::U32, input_ids.device())?.broadcast_left(b_size)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        self.layer_norm.forward(&embeddings)
    }
}

// ---------------------------------------------------------------------------
// Self-attention with QK post-normalization
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct JinaCodeAttention {
    qkv_linear: Linear,
    dense: Linear,
    layer_norm_q: LayerNorm,
    layer_norm_k: LayerNorm,
    layer_norm_out: LayerNorm,
    num_attention_heads: usize,
    attention_head_size: usize,
    softmax_scale: f64,
    span: tracing::Span,
}

impl JinaCodeAttention {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        let hidden_size = cfg.hidden_size;

        // Load Q/K/V weights directly then fuse into single linear
        let query_weight = vb
            .pp("self.query")
            .get((all_head_size, hidden_size), "weight")?;
        let query_bias = vb.pp("self.query").get(all_head_size, "bias")?;

        let key_weight = vb
            .pp("self.key")
            .get((all_head_size, hidden_size), "weight")?;
        let key_bias = vb.pp("self.key").get(all_head_size, "bias")?;

        let value_weight = vb
            .pp("self.value")
            .get((all_head_size, hidden_size), "weight")?;
        let value_bias = vb.pp("self.value").get(all_head_size, "bias")?;

        let qkv_weight = Tensor::cat(&[&query_weight, &key_weight, &value_weight], 0)?;
        let qkv_bias = Tensor::cat(&[&query_bias, &key_bias, &value_bias], 0)?;
        let qkv_linear = Linear::from_weights(qkv_weight, Some(qkv_bias));

        let layer_norm_q = layer_norm(hidden_size, cfg.layer_norm_eps, vb.pp("self.layer_norm_q"))?;
        let layer_norm_k = layer_norm(hidden_size, cfg.layer_norm_eps, vb.pp("self.layer_norm_k"))?;

        let dense = linear(hidden_size, hidden_size, vb.pp("output.dense"))?;
        let layer_norm_out =
            layer_norm(hidden_size, cfg.layer_norm_eps, vb.pp("output.LayerNorm"))?;

        Ok(Self {
            qkv_linear,
            dense,
            layer_norm_q,
            layer_norm_k,
            layer_norm_out,
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            softmax_scale: 1.0 / (attention_head_size as f64).sqrt(),
            span: tracing::span!(tracing::Level::TRACE, "jina-code-attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = hidden_states.clone();

        let qkv = self.qkv_linear.forward(hidden_states)?;

        // Reshape to (batch, seq_len, 3 * num_heads, head_size) then split
        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?;
        let qkv = qkv.chunk(3, 2)?;

        // Flatten last dims for layer norm: (batch, seq_len, hidden_size)
        let query_layer = qkv[0].flatten_from(D::Minus2)?;
        let key_layer = qkv[1].flatten_from(D::Minus2)?;

        // QK post-norm: LayerNorm on Q and K before attention scores
        let query_layer = self.layer_norm_q.forward(&query_layer)?;
        let key_layer = self.layer_norm_k.forward(&key_layer)?;

        // Reshape to multi-head: (batch, num_heads, seq_len, head_size)
        let mut new_qk_shape = query_layer.dims().to_vec();
        new_qk_shape.pop();
        new_qk_shape.push(self.num_attention_heads);
        new_qk_shape.push(self.attention_head_size);

        let query_layer = query_layer
            .reshape(new_qk_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()?;
        let key_layer = key_layer
            .reshape(new_qk_shape.as_slice())?
            .transpose(1, 2)?
            .contiguous()?;
        let value_layer = qkv[2].transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention with ALiBi bias
        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores * self.softmax_scale)?;
        let attention_scores = attention_scores.broadcast_add(attention_bias)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Reshape back: (batch, seq_len, hidden_size)
        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        // Output projection + residual + LayerNorm
        let hidden_states = self.dense.forward(&context_layer)?;
        self.layer_norm_out.forward(&(hidden_states + residual)?)
    }
}

// ---------------------------------------------------------------------------
// Transformer layer: attention + gated MLP with pre/post norms
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct JinaCodeBertLayer {
    attention: JinaCodeAttention,
    up_gated_layer: Linear,
    down_layer: Linear,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
    act: candle_nn::Activation,
    intermediate_size: usize,
    span: tracing::Span,
}

impl JinaCodeBertLayer {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attention = JinaCodeAttention::new(vb.pp("attention"), cfg)?;

        // up_gated_layer: hidden_size -> intermediate_size * 2 (no bias)
        let up_gated_weight = vb
            .pp("mlp.up_gated_layer")
            .get((cfg.intermediate_size * 2, cfg.hidden_size), "weight")?;
        let up_gated_layer = Linear::from_weights(up_gated_weight, None);

        // down_layer: intermediate_size -> hidden_size (with bias)
        let down_layer = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp.down_layer"),
        )?;

        let layer_norm_1 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_1"))?;
        let layer_norm_2 = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm_2"))?;

        Ok(Self {
            attention,
            up_gated_layer,
            down_layer,
            layer_norm_1,
            layer_norm_2,
            act: candle_nn::Activation::Gelu,
            intermediate_size: cfg.intermediate_size,
            span: tracing::span!(tracing::Level::TRACE, "jina-code-layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        // Self-attention (includes its own residual + LayerNorm)
        let hidden_states = self.attention.forward(hidden_states, attention_bias)?;

        // Pre-MLP LayerNorm
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm_1.forward(&hidden_states)?;

        // Gated MLP: split into non-gated and gated halves
        let hidden_states = self.up_gated_layer.forward(&hidden_states)?;
        let non_gated = hidden_states.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gated =
            hidden_states.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let hidden_states = (non_gated * gated.apply(&self.act))?;
        let hidden_states = self.down_layer.forward(&hidden_states)?;

        // Post-MLP LayerNorm with residual
        self.layer_norm_2.forward(&(hidden_states + residual)?)
    }
}

// ---------------------------------------------------------------------------
// Encoder + full model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct JinaCodeBertEncoder {
    alibi: Tensor,
    layers: Vec<JinaCodeBertLayer>,
    span: tracing::Span,
}

impl JinaCodeBertEncoder {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| JinaCodeBertLayer::new(vb.pp(format!("layer.{index}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let alibi = build_alibi_bias(cfg)?.to_device(vb.device())?;
        Ok(Self {
            alibi,
            layers,
            span: tracing::span!(tracing::Level::TRACE, "jina-code-encoder"),
        })
    }
}

impl Module for JinaCodeBertEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let seq_len = hidden_states.dim(1)?;
        let alibi_bias = self.alibi.i((.., .., ..seq_len, ..seq_len))?;
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &alibi_bias)?;
        }
        Ok(hidden_states)
    }
}

/// QK-post-norm JinaBERT model.
///
/// Used by `jinaai/jina-embeddings-v2-base-code` and similar models whose
/// `_name_or_path` contains `qk-post-norm`. Shares the same config format
/// as standard JinaBERT but has a different encoder architecture.
#[derive(Clone, Debug)]
pub struct JinaCodeBertModel {
    embeddings: JinaCodeEmbeddings,
    encoder: JinaCodeBertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl JinaCodeBertModel {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embeddings = JinaCodeEmbeddings::new(vb.pp("embeddings"), cfg)?;
        let encoder = JinaCodeBertEncoder::new(vb.pp("encoder"), cfg)?;
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "jina-code-bert"),
        })
    }
}

impl Module for JinaCodeBertModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids)?;
        self.encoder.forward(&embedding_output)
    }
}
