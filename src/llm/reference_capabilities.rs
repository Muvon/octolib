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

//! Compatibility facade for reference model capabilities.
//!
//! The authoritative reference data lives in [`crate::llm::reference_models`]
//! so pricing, capabilities, and proxy schema policy use the same model match.

pub use crate::llm::reference_models::ModelCapabilities;

/// Look up reference capabilities for a model by fuzzy name matching.
pub fn get_reference_capabilities(model: &str) -> Option<ModelCapabilities> {
    crate::llm::reference_models::get_reference_capabilities(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_only_model() {
        let caps = get_reference_capabilities("llama-3.1-8b").unwrap();
        assert!(!caps.vision);
        assert!(!caps.video);
        assert!(caps.structured_output);
        assert_eq!(caps.max_input_tokens, 131_072);
    }

    #[test]
    fn test_vision_model() {
        let caps = get_reference_capabilities("llama-3.2-90b-vision").unwrap();
        assert!(caps.vision);
        assert!(!caps.video);
        assert!(caps.structured_output);
    }

    #[test]
    fn test_vl_model_with_video() {
        let caps = get_reference_capabilities("qwen-2.5-vl-72b").unwrap();
        assert!(caps.vision);
        assert!(caps.video);
        assert!(caps.structured_output);
    }

    #[test]
    fn test_ollama_format() {
        // Ollama uses format like "llama3.1:8b"
        let caps = get_reference_capabilities("llama3.1:8b").unwrap();
        assert!(!caps.vision);
        assert_eq!(caps.max_input_tokens, 131_072);
    }

    #[test]
    fn test_ollama_vision_model() {
        let caps = get_reference_capabilities("llava:latest").unwrap();
        assert!(caps.vision);
    }

    #[test]
    fn test_together_format() {
        // Together uses HuggingFace-style names
        let caps = get_reference_capabilities("meta-llama/Llama-3.3-70B-Instruct").unwrap();
        assert!(!caps.vision);
        assert_eq!(caps.max_input_tokens, 131_072);
    }

    #[test]
    fn test_small_context_model() {
        let caps = get_reference_capabilities("mistral-7b").unwrap();
        assert!(!caps.vision);
        assert!(!caps.structured_output);
        assert_eq!(caps.max_input_tokens, 32_768);
    }

    #[test]
    fn test_llama3_old_context() {
        let caps = get_reference_capabilities("llama-3-8b").unwrap();
        assert_eq!(caps.max_input_tokens, 8_192);
    }

    #[test]
    fn test_unknown_model() {
        assert!(get_reference_capabilities("totally-unknown-model-xyz").is_none());
    }

    #[test]
    fn test_specificity_vision_vs_text() {
        // "llama-3.2-11b-vision" should match the vision entry, not "llama-3.2-1b"
        let caps = get_reference_capabilities("llama-3.2-11b-vision-instruct").unwrap();
        assert!(caps.vision);

        // "llama-3.2-3b" should match the text-only entry
        let caps = get_reference_capabilities("llama-3.2-3b-instruct").unwrap();
        assert!(!caps.vision);
    }

    #[test]
    fn test_gemma_multimodal_vs_text() {
        // Gemma 3 is multimodal
        let caps = get_reference_capabilities("gemma-3-27b").unwrap();
        assert!(caps.vision);

        // Gemma 2 is text-only
        let caps = get_reference_capabilities("gemma-2-27b").unwrap();
        assert!(!caps.vision);
    }

    #[test]
    fn test_deepseek_context() {
        let caps = get_reference_capabilities("deepseek-r1-distill-llama-70b").unwrap();
        assert!(!caps.vision);
        assert_eq!(caps.max_input_tokens, 65_536);
    }

    #[test]
    fn test_pixtral_vision() {
        let caps = get_reference_capabilities("pixtral-large").unwrap();
        assert!(caps.vision);
    }

    #[test]
    fn test_phi_multimodal_vs_text() {
        let caps = get_reference_capabilities("phi-4-multimodal").unwrap();
        assert!(caps.vision);

        let caps = get_reference_capabilities("phi-4").unwrap();
        assert!(!caps.vision);
        assert_eq!(caps.max_input_tokens, 16_384);
    }

    #[test]
    fn test_deepseek_v4_pro_ollama_format() {
        // Regression: previously fell through to 8192 default via Ollama
        let caps = get_reference_capabilities("deepseek-v4-pro").unwrap();
        assert_eq!(caps.max_input_tokens, 1_000_000);
        let caps = get_reference_capabilities("deepseek-v4-pro:671b").unwrap();
        assert_eq!(caps.max_input_tokens, 1_000_000);
        let caps = get_reference_capabilities("deepseek-v4-flash").unwrap();
        assert_eq!(caps.max_input_tokens, 1_000_000);
    }

    #[test]
    fn test_openai_gpt5_family() {
        // gpt-5.5-pro must NOT match the gpt-5 catch-all (400K) — specificity ordering proof
        assert_eq!(
            get_reference_capabilities("gpt-5.5-pro")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-5.5")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-5.4")
                .unwrap()
                .max_input_tokens,
            400_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-5")
                .unwrap()
                .max_input_tokens,
            400_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-5.3-instant")
                .unwrap()
                .max_input_tokens,
            128_000
        );
    }

    #[test]
    fn test_openai_o_series_and_gpt4() {
        assert_eq!(
            get_reference_capabilities("o4-mini")
                .unwrap()
                .max_input_tokens,
            200_000
        );
        assert_eq!(
            get_reference_capabilities("o3").unwrap().max_input_tokens,
            200_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-4-32k")
                .unwrap()
                .max_input_tokens,
            32_768
        );
        assert_eq!(
            get_reference_capabilities("gpt-4o")
                .unwrap()
                .max_input_tokens,
            128_000
        );
        assert_eq!(
            get_reference_capabilities("gpt-4.1-mini")
                .unwrap()
                .max_input_tokens,
            1_047_576
        );
    }

    #[test]
    fn test_anthropic_claude() {
        assert_eq!(
            get_reference_capabilities("claude-opus-4-7")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("claude-sonnet-4-6")
                .unwrap()
                .max_input_tokens,
            200_000
        );
        assert_eq!(
            get_reference_capabilities("claude-haiku-4-5-20251001")
                .unwrap()
                .max_input_tokens,
            200_000
        );
        assert!(
            get_reference_capabilities("claude-opus-4-7")
                .unwrap()
                .vision
        );
        // Mythos-class: 1M context, vision
        let fable = get_reference_capabilities("claude-fable-5").unwrap();
        assert_eq!(fable.max_input_tokens, 1_000_000);
        assert!(fable.vision);
    }

    #[test]
    fn test_kimi_k3_1m() {
        let caps = get_reference_capabilities("kimi-k3").unwrap();
        assert!(caps.vision);
        assert!(caps.video);
        assert_eq!(caps.max_input_tokens, 1_048_576);
    }

    #[test]
    fn test_kimi_k2_256k() {
        // All Kimi K2 variants now report 256K (matches MoonshotProvider override)
        assert_eq!(
            get_reference_capabilities("kimi-k2.7-code")
                .unwrap()
                .max_input_tokens,
            256_000
        );
        assert_eq!(
            get_reference_capabilities("kimi-k2.6")
                .unwrap()
                .max_input_tokens,
            256_000
        );
        assert_eq!(
            get_reference_capabilities("kimi-k2-0905")
                .unwrap()
                .max_input_tokens,
            256_000
        );
        assert_eq!(
            get_reference_capabilities("kimi-k2")
                .unwrap()
                .max_input_tokens,
            256_000
        );
    }

    #[test]
    fn test_glm_5_1_200k() {
        assert_eq!(
            get_reference_capabilities("glm-5.1")
                .unwrap()
                .max_input_tokens,
            200_000
        );
        assert_eq!(
            get_reference_capabilities("glm-5.1-turbo")
                .unwrap()
                .max_input_tokens,
            200_000
        );
    }

    #[test]
    fn test_minimax_variants() {
        // M3 is multimodal: image + video
        let m3 = get_reference_capabilities("MiniMax-M3").unwrap();
        assert!(m3.vision);
        assert!(m3.video);
        assert_eq!(m3.max_input_tokens, 1_000_000);
        assert_eq!(
            get_reference_capabilities("MiniMax-M3-highspeed")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("MiniMax-M2.7-highspeed")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("MiniMax-M2.5-lightning")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
        assert_eq!(
            get_reference_capabilities("MiniMax-M2.1")
                .unwrap()
                .max_input_tokens,
            1_000_000
        );
    }

    #[test]
    fn test_byteplus_dot_aliases() {
        // BytePlus dot-form aliases (don't sanitize to "2-0" form)
        assert_eq!(
            get_reference_capabilities("dola-seed-2.0-pro")
                .unwrap()
                .max_input_tokens,
            256_000
        );
        assert_eq!(
            get_reference_capabilities("bytedance-seed-code")
                .unwrap()
                .max_input_tokens,
            256_000
        );
    }
}
