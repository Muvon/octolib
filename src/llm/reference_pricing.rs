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
    // --- Anthropic Claude (most specific first) ---
    ("claude-fable-5", 10.00, 50.00, 12.50, 1.00),
    ("claude-opus-4-8", 5.00, 25.00, 6.25, 0.50),
    ("claude-opus-4-7", 5.00, 25.00, 6.25, 0.50),
    ("claude-opus-4-6", 5.00, 25.00, 6.25, 0.50),
    ("claude-sonnet-4-6", 3.00, 15.00, 3.75, 0.30),
    ("claude-opus-4-5", 5.00, 25.00, 6.25, 0.50),
    ("claude-sonnet-4-5", 3.00, 15.00, 3.75, 0.30),
    ("claude-haiku-4-5", 1.00, 5.00, 1.25, 0.10),
    ("claude-opus-4-1", 15.00, 75.00, 18.75, 1.50),
    ("claude-opus-4", 15.00, 75.00, 18.75, 1.50),
    ("claude-sonnet-4", 3.00, 15.00, 3.75, 0.30),
    ("claude-3-7-sonnet", 3.00, 15.00, 3.75, 0.30),
    ("claude-3-5-sonnet", 3.00, 15.00, 3.75, 0.30),
    ("claude-3-5-haiku", 0.80, 4.00, 1.00, 0.08),
    ("claude-3-opus", 15.00, 75.00, 18.75, 1.50),
    ("claude-3-sonnet", 3.00, 15.00, 3.75, 0.30),
    ("claude-3-haiku", 0.25, 1.25, 0.30, 0.03),
    // --- OpenAI proprietary (most specific first) ---
    ("gpt-5.5-pro", 30.00, 180.00, 30.00, 30.00),
    ("gpt-5.5", 5.00, 30.00, 5.00, 0.50),
    ("gpt-5.4-pro", 30.00, 180.00, 30.00, 30.00),
    ("gpt-5.4-mini", 0.75, 4.50, 0.75, 0.075),
    ("gpt-5.4-nano", 0.20, 1.25, 0.20, 0.02),
    ("gpt-5.4", 2.50, 15.00, 2.50, 0.25),
    ("gpt-5.3-instant", 1.75, 14.00, 0.175, 0.175),
    ("gpt-5.3-codex", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.3-chat-latest", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2-pro", 21.00, 168.00, 21.00, 21.00),
    ("gpt-5.2-codex", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2-chat-latest", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.2", 1.75, 14.00, 1.75, 0.175),
    ("gpt-5.1-codex-mini", 0.25, 2.00, 0.25, 0.025),
    ("gpt-5.1-codex-max", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1-codex", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1-chat-latest", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5.1", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5-pro", 15.00, 120.00, 15.00, 15.00),
    ("gpt-5-codex", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5-chat-latest", 1.25, 10.00, 1.25, 0.125),
    ("gpt-5-mini", 0.25, 2.00, 0.25, 0.025),
    ("gpt-5-nano", 0.05, 0.40, 0.05, 0.005),
    ("gpt-5", 1.25, 10.00, 1.25, 0.125),
    ("codex-mini-latest", 1.50, 6.00, 1.50, 0.375),
    ("gpt-4.1-mini", 0.40, 1.60, 0.40, 0.10),
    ("gpt-4.1-nano", 0.10, 0.40, 0.10, 0.025),
    ("gpt-4.1", 2.00, 8.00, 2.00, 0.50),
    ("gpt-realtime-1.5", 4.00, 16.00, 4.00, 0.40),
    ("gpt-realtime-mini", 0.60, 2.40, 0.60, 0.06),
    ("gpt-realtime", 4.00, 16.00, 4.00, 0.40),
    ("gpt-audio-1.5", 2.50, 10.00, 2.50, 0.25),
    ("gpt-audio-mini", 0.15, 0.60, 0.15, 0.015),
    ("gpt-audio", 2.50, 10.00, 2.50, 2.50),
    ("gpt-4o-mini-realtime-preview", 0.60, 2.40, 0.60, 0.30),
    ("gpt-4o-realtime-preview", 5.00, 20.00, 5.00, 2.50),
    ("gpt-4o-mini", 0.15, 0.60, 0.15, 0.075),
    ("gpt-4o", 2.50, 10.00, 2.50, 1.25),
    ("gpt-4.5-preview", 75.00, 150.00, 75.00, 75.00),
    ("o4-mini-deep-research", 1.00, 4.00, 1.00, 0.25),
    ("o4-mini", 1.10, 4.40, 1.10, 0.275),
    ("o3-deep-research", 5.00, 20.00, 5.00, 1.25),
    ("o3-pro", 20.00, 80.00, 20.00, 20.00),
    ("o3-mini", 1.10, 4.40, 1.10, 0.55),
    ("o3", 2.00, 8.00, 2.00, 0.50),
    ("o1-pro", 150.00, 600.00, 150.00, 150.00),
    ("o1-mini", 1.10, 4.40, 1.10, 0.55),
    ("o1", 15.00, 60.00, 15.00, 7.50),
    ("gpt-4-32k", 60.00, 120.00, 60.00, 60.00),
    ("gpt-4-turbo", 10.00, 30.00, 10.00, 10.00),
    ("gpt-4", 30.00, 60.00, 30.00, 30.00),
    ("gpt-3.5-turbo-16k-0613", 3.00, 4.00, 3.00, 3.00),
    ("gpt-3.5-turbo-instruct", 1.50, 2.00, 1.50, 1.50),
    ("gpt-3.5-turbo", 0.50, 1.50, 0.50, 0.50),
    // --- OpenAI open-weight ---
    ("gpt-oss-120b", 0.35, 0.75, 0.35, 0.35),
    ("gpt-oss-20b", 0.03, 0.10, 0.03, 0.03),
    // --- ByteDance Seed ---
    ("seed-2-0-pro", 0.50, 3.00, 0.50, 0.10),
    ("seed-2-0-code", 0.50, 3.00, 0.50, 0.10),
    ("seed-2-0-lite", 0.25, 2.00, 0.25, 0.05),
    ("seed-2-0-mini", 0.10, 0.40, 0.10, 0.02),
    ("seed-1-8", 0.25, 2.00, 0.25, 0.05),
    ("seed-1-6-flash", 0.075, 0.30, 0.075, 0.015),
    ("seed-1-6", 0.25, 2.00, 0.25, 0.05),
    // --- BytePlus aliases (dot-form names) ---
    ("dola-seed-2.0-pro", 0.50, 3.00, 0.50, 0.10),
    ("dola-seed-2.0-lite", 0.25, 2.00, 0.25, 0.05),
    ("dola-seed-2.0-code", 0.50, 3.00, 0.50, 0.10),
    ("bytedance-seed-code", 0.50, 3.00, 0.50, 0.10),
    ("glm-4-7-251222", 0.60, 2.20, 0.60, 0.11),
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
    // --- Qwen 3.7 / 3.5 (Together serverless rates) ---
    ("qwen-3.7-max", 1.25, 3.75, 1.25, 0.13),
    ("qwen-3.5-397b", 0.60, 3.60, 0.60, 0.35),
    ("qwen-3.5-9b", 0.17, 0.25, 0.17, 0.17),
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
    ("gemini-3.5-flash", 1.50, 9.00, 1.50, 0.15),
    ("gemini-3.1-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3.1-flash-lite", 0.25, 1.50, 0.25, 0.025),
    ("gemini-3-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3-flash", 0.50, 3.00, 0.50, 0.05),
    ("gemini-2.5-flash-lite", 0.10, 0.40, 0.10, 0.01),
    ("gemini-2.5-flash", 0.30, 2.50, 0.30, 0.03),
    ("gemini-2.5-pro", 1.25, 10.00, 1.25, 0.125),
    ("gemini-2.0-flash", 0.10, 0.40, 0.10, 0.025),
    // --- Zhipu GLM (most specific first) ---
    ("glm-5v-turbo", 1.20, 4.00, 0.00, 0.24),
    ("glm-5.2", 1.40, 4.40, 0.00, 0.26),
    ("glm-5.1-turbo", 1.40, 4.40, 0.00, 0.26),
    ("glm-5.1", 1.40, 4.40, 0.00, 0.26),
    ("glm-5-turbo", 1.20, 4.00, 0.00, 0.24),
    ("glm-5", 1.00, 3.20, 0.00, 0.20),
    ("glm-4.7-flashx", 0.07, 0.40, 0.00, 0.01),
    ("glm-4.7-flash", 0.00, 0.00, 0.00, 0.00),
    ("glm-4.7", 0.60, 2.20, 0.60, 0.11),
    ("glm-4.6v-flashx", 0.04, 0.40, 0.00, 0.004),
    ("glm-4.6v-flash", 0.00, 0.00, 0.00, 0.00),
    ("glm-4.6v", 0.30, 0.90, 0.00, 0.05),
    ("glm-4.6", 0.60, 2.20, 0.00, 0.11),
    ("glm-4.5-airx", 1.10, 4.50, 0.00, 0.22),
    ("glm-4.5-air", 0.20, 1.10, 0.00, 0.03),
    ("glm-4.5-flash", 0.00, 0.00, 0.00, 0.00),
    ("glm-4.5v", 0.60, 1.80, 0.00, 0.11),
    ("glm-4.5-x", 2.20, 8.90, 0.00, 0.45),
    ("glm-4.5", 0.60, 2.20, 0.00, 0.11),
    ("glm-4-32b-0414-128k", 0.10, 0.10, 0.00, 0.01),
    ("glm-ocr", 0.03, 0.03, 0.00, 0.00),
    ("glm-4", 0.60, 2.20, 0.60, 0.06),
    // --- MiniMax (most specific first) ---
    ("minimax-m3-highspeed", 0.60, 2.40, 0.75, 0.06),
    ("minimax-m3", 0.60, 2.40, 0.75, 0.06),
    ("minimax-m2.7-highspeed", 0.60, 2.40, 0.75, 0.06),
    ("minimax-m2.7", 0.30, 1.20, 0.375, 0.06),
    ("minimax-m2.5-highspeed", 0.60, 2.40, 0.75, 0.03),
    ("minimax-m2.5-lightning", 0.60, 2.40, 0.75, 0.03),
    ("minimax-m2.5", 0.30, 1.20, 0.375, 0.03),
    ("minimax-m2.1-lightning", 0.30, 2.40, 0.30, 0.03),
    ("minimax-m2.1", 0.27, 0.95, 0.27, 0.027),
    ("minimax-m2", 0.255, 1.00, 0.255, 0.0255),
    ("m2-her", 0.30, 1.20, 0.0, 0.0),
    // --- Microsoft Phi ---
    ("phi-4", 0.07, 0.14, 0.07, 0.07),
    ("phi-3", 0.05, 0.10, 0.05, 0.05),
    // --- Moonshot Kimi (most specific first) ---
    ("kimi-k2.7-code-highspeed", 1.90, 8.00, 1.90, 0.38),
    ("kimi-k2.7-code", 0.95, 4.00, 0.95, 0.19),
    ("kimi-k2.6-code-preview", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2.6", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2.5", 0.60, 3.00, 0.60, 0.10),
    ("kimi-k2-thinking-turbo", 1.15, 8.00, 1.15, 0.15),
    ("kimi-k2-turbo", 1.15, 8.00, 1.15, 0.15),
    ("kimi-k2-thinking", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2-0915", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2-0905", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2-0711", 0.60, 2.50, 0.60, 0.15),
    ("kimi-k2", 0.60, 2.50, 0.60, 0.15),
    ("moonshot-v1-128k", 2.00, 5.00, 2.00, 2.00),
    ("moonshot-v1-32k", 1.00, 3.00, 1.00, 1.00),
    ("moonshot-v1-8k", 0.20, 2.00, 0.20, 0.20),
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
    fn test_reference_pricing_qwen_3_5_3_7() {
        // Realistic Together model IDs must resolve via sanitized substring matching.
        let p = get_reference_pricing("Qwen/Qwen3.7-Max").unwrap();
        assert_eq!(p.input_price_per_1m, 1.25);
        assert_eq!(p.cache_read_price_per_1m, 0.13);

        let p = get_reference_pricing("Qwen/Qwen3.5-397B-A17B").unwrap();
        assert_eq!(p.input_price_per_1m, 0.60);
        assert_eq!(p.output_price_per_1m, 3.60);
        assert_eq!(p.cache_read_price_per_1m, 0.35);

        // 9B FP8 must NOT collide with the 397B entry
        let p = get_reference_pricing("Qwen/Qwen3.5-9B-FP8").unwrap();
        assert_eq!(p.input_price_per_1m, 0.17);
        assert_eq!(p.output_price_per_1m, 0.25);
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

    #[test]
    fn test_proprietary_added() {
        // gpt-5.5-pro must NOT match the gpt-5 catch-all
        let p = get_reference_pricing("gpt-5.5-pro").unwrap();
        assert_eq!(p.input_price_per_1m, 30.00);
        let p = get_reference_pricing("claude-opus-4-7").unwrap();
        assert_eq!(p.input_price_per_1m, 5.00);
        let p = get_reference_pricing("claude-fable-5").unwrap();
        assert_eq!(p.input_price_per_1m, 10.00);
        assert_eq!(p.output_price_per_1m, 50.00);
        let p = get_reference_pricing("kimi-k2.6").unwrap();
        assert_eq!(p.input_price_per_1m, 0.60);
        let p = get_reference_pricing("kimi-k2.7-code").unwrap();
        assert_eq!(p.input_price_per_1m, 0.95);
        assert_eq!(p.output_price_per_1m, 4.00);
        // highspeed must NOT fall back to the base kimi-k2.7-code entry
        let p = get_reference_pricing("kimi-k2.7-code-highspeed").unwrap();
        assert_eq!(p.input_price_per_1m, 1.90);
        assert_eq!(p.output_price_per_1m, 8.00);
        let p = get_reference_pricing("gemini-3.5-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 1.50);
        assert_eq!(p.output_price_per_1m, 9.00);
    }
}
