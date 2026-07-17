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

//! Compatibility facade for reference model pricing.
//!
//! The authoritative reference data lives in [`crate::llm::reference_models`]
//! so pricing, capabilities, and proxy schema policy use the same model match.

use crate::llm::types::ModelPricing;

/// Look up baseline cloud-equivalent pricing for a model by fuzzy name matching.
pub fn get_reference_pricing(model: &str) -> Option<ModelPricing> {
    crate::llm::reference_models::get_reference_pricing(model)
}

/// Calculate cost using reference pricing.
pub fn calculate_reference_cost(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    crate::llm::reference_models::calculate_reference_cost(model, input_tokens, output_tokens)
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

        // Together-hosted Plus variants
        let p = get_reference_pricing("Qwen/Qwen3.7-Plus").unwrap();
        assert_eq!(p.input_price_per_1m, 0.32);
        assert_eq!(p.output_price_per_1m, 1.28);
        let p = get_reference_pricing("Qwen/Qwen3.6-Plus").unwrap();
        assert_eq!(p.input_price_per_1m, 0.50);
        assert_eq!(p.output_price_per_1m, 3.00);

        // Alibaba proprietary API names
        let p = get_reference_pricing("qwen3.6-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 0.25);
        let p = get_reference_pricing("qwen3-max-2026-01-25").unwrap();
        assert_eq!(p.input_price_per_1m, 1.20);
        let p = get_reference_pricing("qwen3-coder-plus").unwrap();
        assert_eq!(p.input_price_per_1m, 1.00);
        // unversioned aliases must not shadow versioned entries
        let p = get_reference_pricing("qwen-plus-latest").unwrap();
        assert_eq!(p.input_price_per_1m, 0.40);
        assert_eq!(p.output_price_per_1m, 1.20);
        let p = get_reference_pricing("qwen-turbo").unwrap();
        assert_eq!(p.output_price_per_1m, 0.20);
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
        let p = get_reference_pricing("kimi-k3").unwrap();
        assert_eq!(p.input_price_per_1m, 3.00);
        assert_eq!(p.output_price_per_1m, 15.00);
        assert_eq!(p.cache_read_price_per_1m, 0.30);
        let p = get_reference_pricing("kimi-k2.6").unwrap();
        assert_eq!(p.input_price_per_1m, 0.60);
        let p = get_reference_pricing("kimi-k2.7-code").unwrap();
        assert_eq!(p.input_price_per_1m, 0.95);
        assert_eq!(p.output_price_per_1m, 4.00);
        // "k"-less aliases (gateway/self-hosted deployment names) price identically —
        // the sanitizer bridges qwen3.7→qwen-3.7 but not a real letter difference.
        let p = get_reference_pricing("kimi-2.6").unwrap();
        assert_eq!(p.input_price_per_1m, 0.60);
        let p = get_reference_pricing("kimi-2.7-code").unwrap();
        assert_eq!(p.input_price_per_1m, 0.95);
        let p = get_reference_pricing("qwen3.7-max").unwrap();
        assert_eq!(p.input_price_per_1m, 1.25);
        let p = get_reference_pricing("qwen3.7-plus").unwrap();
        assert_eq!(p.input_price_per_1m, 0.32);
        // highspeed must NOT fall back to the base kimi-k2.7-code entry
        let p = get_reference_pricing("kimi-k2.7-code-highspeed").unwrap();
        assert_eq!(p.input_price_per_1m, 1.90);
        assert_eq!(p.output_price_per_1m, 8.00);
        let p = get_reference_pricing("gemini-3.5-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 1.50);
        assert_eq!(p.output_price_per_1m, 9.00);
    }
}
