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

//! Reference pricing for well-known models across providers.
//!
//! When a provider (Ollama, Together.ai, Amazon Bedrock, etc.) doesn't expose
//! its own token pricing, this module provides baseline "cloud equivalent" costs
//! derived from the cheapest major provider offering each model.
//!
//! These are approximate — actual costs vary by provider and billing tier.

use crate::llm::types::ModelPricing;
use crate::llm::utils::{normalize_model_name, sanitize_model_name};

/// Reference pricing entry: (pattern, input, output, cache_write, cache_read)
/// Patterns are matched case-insensitively via substring against the model name.
/// More specific patterns must come before less specific ones (longest match first).
type RefPricingTuple = (&'static str, f64, f64, f64, f64);

/// Baseline pricing for well-known open/open-weight models (per 1M tokens, USD).
///
/// Sources: cheapest major provider for each model family (Apr 2026).
/// These are NOT authoritative — they're best-effort baselines for cost estimation
/// when the actual provider doesn't report pricing.
const REFERENCE_PRICING: &[RefPricingTuple] = &[
    // --- OpenAI open-weight ---
    ("gpt-oss-120b", 0.35, 0.75, 0.35, 0.35),
    ("gpt-oss-20b", 0.03, 0.10, 0.03, 0.03),
    // --- Meta Llama 4 ---
    ("llama-4-maverick", 0.17, 0.60, 0.17, 0.17),
    ("llama-4-scout", 0.08, 0.30, 0.08, 0.08),
    // --- Meta Llama 3.3 ---
    ("llama-3.3-70b", 0.60, 0.60, 0.60, 0.60),
    // --- Meta Llama 3.1 ---
    ("llama-3.1-405b", 3.00, 3.00, 3.00, 3.00),
    ("llama-3.1-70b", 0.60, 0.60, 0.60, 0.60),
    ("llama-3.1-8b", 0.10, 0.10, 0.10, 0.10),
    // --- Meta Llama 3 ---
    ("llama-3-70b", 0.60, 0.60, 0.60, 0.60),
    ("llama-3-8b", 0.10, 0.10, 0.10, 0.10),
    // --- Qwen 3 ---
    ("qwen-3-coder-480b", 2.00, 2.00, 2.00, 2.00),
    ("qwen-3-235b", 0.60, 1.20, 0.60, 0.60),
    ("qwen-3-32b", 0.10, 0.10, 0.10, 0.10),
    ("qwen-3-8b", 0.05, 0.05, 0.05, 0.05),
    // --- Qwen 2.5 ---
    ("qwen-2.5-72b", 0.60, 0.60, 0.60, 0.60),
    ("qwen-2.5-32b", 0.10, 0.10, 0.10, 0.10),
    ("qwen-2.5-coder-32b", 0.10, 0.10, 0.10, 0.10),
    ("qwen-2.5-7b", 0.05, 0.05, 0.05, 0.05),
    // --- DeepSeek ---
    ("deepseek-v4-pro", 1.74, 3.48, 1.74, 0.145),
    ("deepseek-v4-flash", 0.14, 0.28, 0.14, 0.028),
    ("deepseek-v4", 0.14, 0.28, 0.14, 0.028),
    ("deepseek-v3", 0.28, 0.42, 0.28, 0.028),
    ("deepseek-r1", 0.28, 0.42, 0.28, 0.028),
    ("deepseek-v2", 0.14, 0.28, 0.14, 0.014),
    // --- Mistral ---
    ("mistral-large-3", 0.50, 1.50, 0.50, 0.125),
    ("mistral-large", 2.00, 6.00, 2.00, 2.00),
    ("mistral-medium-3", 0.40, 2.00, 0.40, 0.10),
    ("mistral-medium", 2.70, 8.10, 2.70, 2.70),
    ("mistral-small", 0.10, 0.30, 0.10, 0.10),
    ("mixtral-8x22b", 0.90, 0.90, 0.90, 0.90),
    ("mixtral-8x7b", 0.24, 0.24, 0.24, 0.24),
    ("mistral-7b", 0.05, 0.05, 0.05, 0.05),
    ("codestral", 0.30, 0.90, 0.30, 0.30),
    // --- xAI Grok ---
    ("grok-4.1-fast", 0.20, 0.50, 0.20, 0.05),
    ("grok-4-fast", 0.20, 0.50, 0.20, 0.05),
    ("grok-4", 3.00, 15.00, 3.00, 0.75),
    ("grok-3", 3.00, 15.00, 3.00, 0.75),
    // --- Google Gemma ---
    ("gemma-4-31b", 0.20, 0.20, 0.20, 0.20),
    ("gemma-4-26b", 0.20, 0.20, 0.20, 0.20),
    ("gemma-4-e4b", 0.05, 0.05, 0.05, 0.05),
    ("gemma-4-e2b", 0.02, 0.02, 0.02, 0.02),
    ("gemma-3n-e4b", 0.02, 0.04, 0.02, 0.02),
    ("gemma-3-27b", 0.20, 0.20, 0.20, 0.20),
    ("gemma-3-12b", 0.10, 0.10, 0.10, 0.10),
    ("gemma-3-4b", 0.05, 0.05, 0.05, 0.05),
    ("gemma-2-27b", 0.20, 0.20, 0.20, 0.20),
    ("gemma-2-9b", 0.05, 0.05, 0.05, 0.05),
    // --- Google Gemini ---
    ("gemini-3.1-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3.1-flash-lite", 0.25, 1.50, 0.25, 0.025),
    ("gemini-3-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3-flash", 0.50, 3.00, 0.50, 0.05),
    ("gemini-2.5-flash-lite", 0.10, 0.40, 0.10, 0.01),
    ("gemini-2.5-flash", 0.30, 2.50, 0.30, 0.03),
    ("gemini-2.5-pro", 1.25, 10.00, 1.25, 0.125),
    ("gemini-2.0-flash", 0.10, 0.40, 0.10, 0.025),
    // --- Zhipu GLM ---
    ("glm-5v-turbo", 1.20, 4.00, 0.00, 0.24),
    ("glm-5.1-turbo", 1.00, 3.20, 0.00, 0.20),
    ("glm-5.1", 1.00, 3.20, 0.00, 0.20),
    ("glm-5-turbo", 1.20, 4.00, 0.00, 0.24),
    ("glm-5", 1.00, 3.20, 0.00, 0.20),
    ("glm-4.7-flash", 0.00, 0.00, 0.00, 0.00),
    ("glm-4.7", 0.60, 2.20, 0.60, 0.11),
    ("glm-4", 0.60, 2.20, 0.60, 0.06),
    // --- MiniMax ---
    ("minimax-m2.7", 0.30, 1.20, 0.375, 0.06),
    ("minimax-m2.5", 0.30, 1.20, 0.375, 0.03),
    ("minimax-m2", 0.255, 1.00, 0.255, 0.0255),
    // --- Microsoft Phi ---
    // --- Moonshot Kimi ---
    ("kimi-k2.5", 0.60, 3.00, 0.60, 0.10),
    ("kimi-k2-thinking-turbo", 1.15, 8.00, 1.15, 0.15),
    ("kimi-k2-turbo", 1.15, 8.00, 1.15, 0.15),
    ("kimi-k2-thinking", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2", 0.60, 2.50, 0.60, 0.15),
    ("phi-4", 0.07, 0.14, 0.07, 0.07),
    ("phi-3", 0.05, 0.10, 0.05, 0.05),
    // --- Cohere Command ---
    ("command-r-plus", 2.50, 10.00, 2.50, 2.50),
    ("command-r", 0.15, 0.60, 0.15, 0.15),
    // --- DBRX ---
    ("dbrx", 0.75, 0.75, 0.75, 0.75),
];

/// Look up reference pricing for a model by fuzzy name matching.
///
/// Normalizes the model name and checks if any reference pattern is a substring.
/// Returns the first match (table is ordered by specificity — longer/more specific first).
///
/// Handles naming variations across providers:
/// - `meta-llama/Llama-3.3-70B-Instruct` → matches `llama-3.3-70b`
/// - `llama3.3:70b` (Ollama) → matches after normalization
/// - `Qwen/Qwen2.5-72B-Instruct` → matches `qwen-2.5-72b`
pub fn get_reference_pricing(model: &str) -> Option<ModelPricing> {
    let normalized = sanitize_model_name(&normalize_model_name(model));

    REFERENCE_PRICING
        .iter()
        .find(|(pattern, _, _, _, _)| {
            // Sanitize patterns too so both sides use the same canonical form
            let sanitized_pattern = sanitize_model_name(pattern);
            normalized.contains(&sanitized_pattern)
        })
        .map(|(_, input, output, cache_write, cache_read)| {
            ModelPricing::new(*input, *output, *cache_write, *cache_read)
        })
}

/// Calculate cost using reference pricing.
pub fn calculate_reference_cost(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    let pricing = get_reference_pricing(model)?;
    Some(pricing.calculate_cost(input_tokens, 0, 0, output_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_pricing_exact() {
        let p = get_reference_pricing("llama-3.1-8b").unwrap();
        assert_eq!(p.input_price_per_1m, 0.10);
        assert_eq!(p.output_price_per_1m, 0.10);
    }

    #[test]
    fn test_reference_pricing_ollama_format() {
        // Ollama uses format like "llama3.1:8b"
        assert!(get_reference_pricing("llama3.1:8b").is_some());
    }

    #[test]
    fn test_reference_pricing_together_format() {
        // Together uses HuggingFace-style names
        assert!(get_reference_pricing("meta-llama/Llama-3.3-70B-Instruct").is_some());
    }

    #[test]
    fn test_reference_pricing_deepseek() {
        let p = get_reference_pricing("deepseek-r1-distill-llama-70b").unwrap();
        assert_eq!(p.input_price_per_1m, 0.28);
    }

    #[test]
    fn test_reference_pricing_unknown() {
        assert!(get_reference_pricing("totally-unknown-model-xyz").is_none());
    }

    #[test]
    fn test_calculate_reference_cost() {
        let cost = calculate_reference_cost("llama-3.1-8b", 1_000_000, 500_000).unwrap();
        // 1M input * $0.10/1M + 500K output * $0.10/1M = $0.10 + $0.05 = $0.15
        assert!((cost - 0.15).abs() < 0.001);
    }
}
