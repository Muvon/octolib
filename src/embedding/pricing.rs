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

//! Embedding pricing — the reference rate table for cloud embedding models,
//! mirroring the LLM `PricingTuple` tables (`crate::llm::utils`). Embeddings are
//! input-only, so a single per-1M-token rate per model.
//!
//! A model absent from the table returns `None` (unpriced): local / self-hosted
//! models (fastembed, local, huggingface) are billed as compute, not tokens, so
//! they are intentionally not listed. Multimodal (`*-clip-*`, `voyage-multimodal-*`)
//! and late-interaction (`*-colbert-*`) models are also omitted — their pricing
//! is not a flat text-token rate, so we return `None` rather than guess.

use serde::{Deserialize, Serialize};

/// Usage for one embedding call: the input-token count and the computed cost
/// (`None` when the model is unpriced — local/self-hosted). The embedding
/// analogue of the LLM `Usage`: the provider builds it and returns it from the
/// generation call, so callers (octohub, octomind) read one struct instead of
/// recomputing tokens + cost themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub input_tokens: u64,
    pub cost: Option<f64>,
}

impl EmbeddingUsage {
    /// From the provider's own reported token count (the accurate path — the
    /// provider's tokenizer, not a tiktoken guess). Cost from the reference table.
    pub fn from_tokens(model: &str, input_tokens: u64) -> Self {
        Self {
            input_tokens,
            cost: calculate_embedding_cost(model, input_tokens),
        }
    }

    /// Fallback for providers that report no token count (local: fastembed,
    /// in-process huggingface) — estimate with tiktoken. These models are
    /// unpriced, so `cost` is `None` and the estimate is informational only.
    pub fn estimate(model: &str, texts: &[String]) -> Self {
        let input_tokens: u64 = texts.iter().map(|t| super::count_tokens(t) as u64).sum();
        Self::from_tokens(model, input_tokens)
    }
}

/// `(model, USD per 1,000,000 input tokens)`.
pub type EmbeddingPricingTuple = (&'static str, f64);

/// Reference embedding pricing, USD per 1M input tokens.
///
/// Voyage rates verified from docs.voyageai.com (2026-07); OpenAI and Jina v3
/// are stable published rates. Lines tagged `estimate` are best-effort and must
/// be confirmed against the provider before launch (same convention as the LLM
/// tables and `spec/pricing.md`). These are the raw *provider list* prices; any
/// margin is applied downstream by the billing layer.
pub const EMBEDDING_PRICING: &[EmbeddingPricingTuple] = &[
    // ── Voyage (verified: docs.voyageai.com, 2026-07) ──
    ("voyage-4-large", 0.12),
    ("voyage-4", 0.06),
    ("voyage-4-lite", 0.02),
    ("voyage-3-large", 0.18),
    ("voyage-3.5", 0.06),
    ("voyage-3.5-lite", 0.02),
    ("voyage-code-3", 0.18),
    ("voyage-code-2", 0.12),
    ("voyage-context-3", 0.18),
    ("voyage-finance-2", 0.12),
    ("voyage-law-2", 0.12),
    ("voyage-2", 0.10),
    // ── OpenAI (verified: openai.com/pricing) ──
    ("text-embedding-3-small", 0.02),
    ("text-embedding-3-large", 0.13),
    ("text-embedding-ada-002", 0.10),
    // ── Jina (v3 verified: jina.ai; others estimate at Jina's flat $0.02/M) ──
    ("jina-embeddings-v3", 0.02),
    ("jina-embeddings-v4", 0.02),           // estimate
    ("jina-embeddings-v2-base-code", 0.02), // estimate
    ("jina-embeddings-v2-base-en", 0.02),   // estimate
    ("jina-code-embeddings-1.5b", 0.02),    // estimate
    ("jina-code-embeddings-0.5b", 0.02),    // estimate
    // ── Google (estimate — verify before launch) ──
    ("gemini-embedding-001", 0.15), // estimate
    ("text-embedding-005", 0.10),   // estimate
    // ── Together (verified: together.ai model page, 2026-07; flat serverless rate) ──
    ("intfloat/multilingual-e5-large-instruct", 0.02),
];

/// Cost in USD for `input_tokens` of an embedding `model`, or `None` when the
/// model has no reference rate (unpriced — local/self-hosted, or a model we
/// don't offer). Matches the resolved model name case-insensitively.
pub fn calculate_embedding_cost(model: &str, input_tokens: u64) -> Option<f64> {
    let m = model.trim();
    EMBEDDING_PRICING
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case(m))
        .map(|(_, per_mtok)| (input_tokens as f64 / 1_000_000.0) * per_mtok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(got: Option<f64>, want: f64) -> bool {
        got.is_some_and(|x| (x - want).abs() < 1e-9)
    }

    #[test]
    fn prices_known_models() {
        // 1M tokens of voyage-3.5 at $0.06/M = $0.06
        assert!(approx(
            calculate_embedding_cost("voyage-3.5", 1_000_000),
            0.06
        ));
        // resolved names match case-insensitively
        assert!(approx(
            calculate_embedding_cost("VOYAGE-3.5", 1_000_000),
            0.06
        ));
        // 500k tokens of text-embedding-3-small at $0.02/M = $0.01
        assert!(approx(
            calculate_embedding_cost("text-embedding-3-small", 500_000),
            0.01
        ));
        // zero tokens = zero cost, still priced (Some)
        assert!(approx(calculate_embedding_cost("voyage-code-3", 0), 0.0));
    }

    #[test]
    fn unknown_or_local_model_is_unpriced() {
        assert_eq!(
            calculate_embedding_cost("nomic-embed-text", 1_000_000),
            None
        );
        assert_eq!(
            calculate_embedding_cost("sentence-transformers/all-MiniLM-L6-v2", 1_000_000),
            None
        );
        // multimodal / colbert intentionally omitted → unpriced
        assert_eq!(calculate_embedding_cost("jina-clip-v2", 1_000_000), None);
        assert_eq!(
            calculate_embedding_cost("voyage-multimodal-3.5", 1_000_000),
            None
        );
    }
}
