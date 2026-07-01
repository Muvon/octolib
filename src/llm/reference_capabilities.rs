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

//! Reference capabilities for well-known models across providers.
//!
//! When a provider (Ollama, Local, Together.ai, etc.) doesn't know whether a
//! specific model supports vision, structured output, or what its context
//! window size is, this module provides baseline capabilities derived from
//! each model's official documentation.
//!
//! These are approximate — actual capabilities may vary by provider and
//! quantization level.

use crate::llm::utils::{normalize_model_name, sanitize_model_name};

/// Reference capabilities entry: (pattern, vision, video, structured_output, max_input_tokens)
/// Patterns are matched case-insensitively via substring against the model name.
/// More specific patterns must come before less specific ones (longest match first).
type RefCapsTuple = (&'static str, bool, bool, bool, usize);

/// Capabilities for a well-known model, looked up from the reference table.
#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub vision: bool,
    pub video: bool,
    pub structured_output: bool,
    pub max_input_tokens: usize,
}

/// Baseline capabilities for well-known open/open-weight models.
///
/// Sources: official model cards and documentation (Apr 2026).
/// These are NOT authoritative for every provider — they represent the model's
/// native capabilities. Some providers may serve reduced-capability versions.
const REFERENCE_CAPABILITIES: &[RefCapsTuple] = &[
    // --- Vision-Language Models (explicit vision, must precede base patterns) ---
    ("llava", true, false, false, 4_096),
    ("bakllava", true, false, false, 4_096),
    ("moondream", true, false, false, 8_192),
    // --- ByteDance Seed ---
    ("seed-2-0-pro", false, false, true, 256_000),
    ("seed-2-0-code", false, false, true, 256_000),
    ("seed-2-0-lite", false, false, true, 256_000),
    ("seed-2-0-mini", false, false, true, 256_000),
    ("seed-1-8", false, false, true, 131_072),
    ("seed-1-6-flash", false, false, true, 131_072),
    ("seed-1-6", false, false, true, 131_072),
    // --- BytePlus aliases (dot-form names that don't sanitize to "2-0" form) ---
    ("dola-seed-2.0-pro", false, false, true, 256_000),
    ("dola-seed-2.0-lite", false, false, true, 256_000),
    ("dola-seed-2.0-code", false, false, true, 256_000),
    ("bytedance-seed-code", false, false, true, 256_000),
    ("glm-4-7-251222", false, false, true, 200_000),
    // --- Meta Llama 4 (multimodal) ---
    ("llama-4-maverick", true, false, true, 1_048_576),
    ("llama-4-scout", true, false, true, 524_288),
    // --- Meta Llama 3.3 (text-only) ---
    ("llama-3.3-70b", false, false, true, 131_072),
    // --- Meta Llama 3.2 Vision ---
    ("llama-3.2-90b-vision", true, false, true, 131_072),
    ("llama-3.2-11b-vision", true, false, true, 131_072),
    // --- Meta Llama 3.2 (text-only small) ---
    ("llama-3.2-3b", false, false, true, 131_072),
    ("llama-3.2-1b", false, false, true, 131_072),
    // --- Meta Llama 3.1 (text-only) ---
    ("llama-3.1-405b", false, false, true, 131_072),
    ("llama-3.1-70b", false, false, true, 131_072),
    ("llama-3.1-8b", false, false, true, 131_072),
    // --- Meta Llama 3 (text-only) ---
    ("llama-3-70b", false, false, true, 8_192),
    ("llama-3-8b", false, false, true, 8_192),
    // --- Qwen 3.7 / 3.5 (text-only) ---
    ("qwen-3.7-max", false, false, true, 262_144),
    ("qwen-3.5-397b", false, false, true, 262_144),
    ("qwen-3.5-9b", false, false, true, 131_072),
    // --- Qwen 3 (text-only) ---
    ("qwen-3-coder-480b", false, false, true, 262_144),
    ("qwen-3-235b", false, false, true, 131_072),
    ("qwen-3-32b", false, false, true, 131_072),
    ("qwen-3-8b", false, false, true, 131_072),
    // --- Qwen 2.5 VL (vision+video) ---
    ("qwen-2.5-vl-72b", true, true, true, 131_072),
    ("qwen-2.5-vl-7b", true, true, true, 131_072),
    ("qwen-2.5-vl-3b", true, true, true, 131_072),
    ("qwen-2.5-vl", true, true, true, 131_072),
    // --- Qwen 2.5 (text-only) ---
    ("qwen-2.5-coder-32b", false, false, true, 131_072),
    ("qwen-2.5-72b", false, false, true, 131_072),
    ("qwen-2.5-32b", false, false, true, 131_072),
    ("qwen-2.5-7b", false, false, true, 131_072),
    // --- Qwen 2 VL ---
    ("qwen-2-vl", true, true, true, 32_768),
    // --- Qwen 2 (text-only) ---
    ("qwen-2-72b", false, false, true, 131_072),
    // --- DeepSeek ---
    ("deepseek-v4-pro", false, false, true, 1_000_000),
    ("deepseek-v4-flash", false, false, true, 1_000_000),
    ("deepseek-v4", false, false, true, 1_000_000),
    ("deepseek-v3", false, false, true, 65_536),
    ("deepseek-r1", false, false, true, 65_536),
    ("deepseek-v2", false, false, true, 131_072),
    // --- Mistral ---
    ("mistral-large-3", false, false, true, 131_072),
    ("mistral-large", false, false, true, 131_072),
    ("mistral-medium-3", false, false, true, 131_072),
    ("mistral-medium", false, false, true, 32_768),
    ("mistral-small", false, false, true, 131_072),
    ("mixtral-8x22b", false, false, true, 65_536),
    ("mixtral-8x7b", false, false, true, 32_768),
    ("mistral-7b", false, false, false, 32_768),
    ("codestral", false, false, true, 262_144),
    ("pixtral", true, false, true, 131_072),
    // --- xAI Grok ---
    ("grok-4.1-fast", true, false, true, 131_072),
    ("grok-4-fast", true, false, true, 131_072),
    ("grok-4", true, false, true, 131_072),
    ("grok-3", true, false, true, 131_072),
    // --- Google Gemma 4 (multimodal) ---
    ("gemma-4-31b", true, false, true, 131_072),
    ("gemma-4-26b", true, false, true, 131_072),
    ("gemma-4-e4b", true, false, true, 131_072),
    ("gemma-4-e2b", true, false, true, 131_072),
    // --- Google Gemma 3 (multimodal) ---
    ("gemma-3n-e4b", true, false, true, 131_072),
    ("gemma-3-27b", true, false, true, 131_072),
    ("gemma-3-12b", true, false, true, 131_072),
    ("gemma-3-4b", true, false, true, 131_072),
    // --- Google Gemma 2 (text-only) ---
    ("gemma-2-27b", false, false, true, 8_192),
    ("gemma-2-9b", false, false, true, 8_192),
    // --- Google Gemini ---
    ("gemini-3.5-flash", true, true, true, 1_048_576),
    ("gemini-3.1-pro", true, true, true, 1_048_576),
    ("gemini-3.1-flash-lite", true, true, true, 1_048_576),
    ("gemini-3-pro", true, true, true, 1_048_576),
    ("gemini-3-flash", true, true, true, 1_048_576),
    ("gemini-2.5-flash-lite", true, true, true, 1_048_576),
    ("gemini-2.5-flash", true, true, true, 1_048_576),
    ("gemini-2.5-pro", true, true, true, 1_048_576),
    ("gemini-2.0-flash", true, true, true, 1_048_576),
    // --- Zhipu GLM (most specific first) ---
    ("glm-5v-turbo", true, false, true, 128_000),
    ("glm-5v", true, false, true, 128_000),
    ("glm-5.2", false, false, true, 200_000),
    ("glm-5.1-turbo", false, false, true, 200_000),
    ("glm-5.1", false, false, true, 200_000),
    ("glm-5-turbo", false, false, true, 128_000),
    ("glm-5", false, false, true, 128_000),
    ("glm-4.7-flashx", false, false, true, 200_000),
    ("glm-4.7-flash", false, false, true, 200_000),
    ("glm-4.7", false, false, true, 200_000),
    ("glm-4.6v-flashx", true, false, true, 128_000),
    ("glm-4.6v-flash", true, false, true, 128_000),
    ("glm-4.6v", true, false, true, 128_000),
    ("glm-4.6", false, false, true, 128_000),
    ("glm-4.5-airx", false, false, true, 131_072),
    ("glm-4.5-air", false, false, true, 131_072),
    ("glm-4.5-flash", false, false, true, 131_072),
    ("glm-4.5v", true, false, true, 131_072),
    ("glm-4.5-x", false, false, true, 131_072),
    ("glm-4.5", false, false, true, 131_072),
    ("glm-4-32b-0414-128k", false, false, true, 128_000),
    ("glm-ocr", true, false, false, 128_000),
    ("glm-4", false, false, true, 128_000),
    // --- MiniMax (most specific first) ---
    // M3 is natively multimodal (image + video); earlier M2.x models are text-only.
    ("minimax-m3-highspeed", true, true, false, 1_000_000),
    ("minimax-m3", true, true, false, 1_000_000),
    ("minimax-m2.7-highspeed", false, false, false, 1_000_000),
    ("minimax-m2.7", false, false, false, 1_000_000),
    ("minimax-m2.5-highspeed", false, false, false, 1_000_000),
    ("minimax-m2.5-lightning", false, false, false, 1_000_000),
    ("minimax-m2.5", false, false, false, 1_000_000),
    ("minimax-m2.1-lightning", false, false, false, 1_000_000),
    ("minimax-m2.1", false, false, false, 1_000_000),
    ("minimax-m2", false, false, false, 1_000_000),
    ("m2-her", false, false, false, 128_000),
    // --- Microsoft Phi ---
    ("phi-4-multimodal", true, false, true, 131_072),
    ("phi-4", false, false, true, 16_384),
    ("phi-3-vision", true, false, true, 131_072),
    ("phi-3", false, false, true, 131_072),
    // --- Moonshot Kimi (most specific first) ---
    ("kimi-k2.7-code-highspeed", true, true, true, 256_000),
    ("kimi-k2.7-code", true, true, true, 256_000),
    ("kimi-k2.6-code-preview", false, false, true, 256_000),
    ("kimi-k2.6", false, false, true, 256_000),
    ("kimi-k2.5", true, false, true, 256_000),
    ("kimi-k2-thinking-turbo", false, false, true, 256_000),
    ("kimi-k2-turbo", false, false, true, 256_000),
    ("kimi-k2-thinking", false, false, true, 256_000),
    ("kimi-k2-0915", false, false, true, 256_000),
    ("kimi-k2-0905", false, false, true, 256_000),
    ("kimi-k2-0711", false, false, true, 256_000),
    ("kimi-k2", false, false, true, 256_000),
    ("moonshot-v1-128k", false, false, true, 131_072),
    ("moonshot-v1-32k", false, false, true, 32_768),
    ("moonshot-v1-8k", false, false, true, 8_192),
    // --- Cohere Command ---
    ("command-r-plus", false, false, true, 131_072),
    ("command-r", false, false, true, 131_072),
    // --- DBRX ---
    ("dbrx", false, false, true, 32_768),
    // --- Anthropic Claude (most specific first; substring match is order-sensitive) ---
    ("claude-fable-5", true, false, false, 1_000_000),
    ("claude-mythos-5", true, false, false, 1_000_000),
    ("claude-opus-4-7", true, false, false, 1_000_000),
    ("claude-opus-4-6", true, false, false, 200_000),
    ("claude-sonnet-4-6", true, false, false, 200_000),
    ("claude-opus-4-5", true, false, false, 200_000),
    ("claude-sonnet-4-5", true, false, false, 200_000),
    ("claude-haiku-4-5", true, false, false, 200_000),
    ("claude-opus-4-1", true, false, false, 200_000),
    ("claude-opus-4", true, false, false, 200_000),
    ("claude-sonnet-4", true, false, false, 200_000),
    ("claude-haiku-4", true, false, false, 200_000),
    ("claude-3-7-sonnet", true, false, false, 200_000),
    ("claude-3-5-sonnet", true, false, false, 200_000),
    ("claude-3-5-haiku", true, false, false, 200_000),
    ("claude-3-opus", true, false, false, 200_000),
    ("claude-3-sonnet", true, false, false, 200_000),
    ("claude-3-haiku", true, false, false, 200_000),
    // --- OpenAI proprietary (most specific first; substring match is order-sensitive) ---
    ("gpt-5.5-pro", false, false, true, 1_000_000),
    ("gpt-5.5", false, false, true, 1_000_000),
    ("gpt-5.4-pro", false, false, true, 400_000),
    ("gpt-5.4-mini", false, false, true, 400_000),
    ("gpt-5.4-nano", false, false, true, 400_000),
    ("gpt-5.4", false, false, true, 400_000),
    ("gpt-5.3-instant", false, false, true, 128_000),
    ("gpt-5.3-codex", false, false, true, 400_000),
    ("gpt-5.3-chat-latest", false, false, true, 400_000),
    ("gpt-5.3", false, false, true, 400_000),
    ("gpt-5.2-pro", false, false, true, 400_000),
    ("gpt-5.2-codex", false, false, true, 400_000),
    ("gpt-5.2-chat-latest", false, false, true, 400_000),
    ("gpt-5.2", false, false, true, 400_000),
    ("gpt-5.1-codex-mini", false, false, true, 400_000),
    ("gpt-5.1-codex-max", false, false, true, 400_000),
    ("gpt-5.1-codex", false, false, true, 400_000),
    ("gpt-5.1-chat-latest", false, false, true, 400_000),
    ("gpt-5.1", false, false, true, 400_000),
    ("gpt-5-pro", false, false, true, 400_000),
    ("gpt-5-codex", false, false, true, 400_000),
    ("gpt-5-chat-latest", false, false, true, 400_000),
    ("gpt-5-mini", false, false, true, 400_000),
    ("gpt-5-nano", false, false, true, 400_000),
    ("gpt-5", false, false, true, 400_000),
    ("codex-mini-latest", false, false, true, 200_000),
    ("gpt-4.1-mini", false, false, true, 1_047_576),
    ("gpt-4.1-nano", false, false, true, 1_047_576),
    ("gpt-4.1", false, false, true, 1_047_576),
    ("gpt-4o-mini-realtime-preview", false, false, true, 128_000),
    ("gpt-4o-realtime-preview", false, false, true, 128_000),
    ("gpt-4o-mini", true, false, true, 128_000),
    ("gpt-4o", true, false, true, 128_000),
    ("gpt-4.5-preview", false, false, true, 128_000),
    ("gpt-realtime-1.5", false, false, true, 32_000),
    ("gpt-realtime-mini", false, false, true, 32_000),
    ("gpt-realtime", false, false, true, 32_000),
    ("gpt-audio-1.5", false, false, true, 128_000),
    ("gpt-audio-mini", false, false, true, 128_000),
    ("gpt-audio", false, false, true, 128_000),
    ("o4-mini-deep-research", false, false, true, 200_000),
    ("o4-mini", false, false, true, 200_000),
    ("o3-deep-research", false, false, true, 200_000),
    ("o3-pro", false, false, true, 200_000),
    ("o3-mini", false, false, true, 200_000),
    ("o3", false, false, true, 200_000),
    ("o1-pro", false, false, true, 200_000),
    ("o1-mini", false, false, true, 128_000),
    ("o1", false, false, true, 200_000),
    ("gpt-4-32k", false, false, true, 32_768),
    ("gpt-4-turbo", true, false, true, 128_000),
    ("gpt-4", false, false, true, 8_192),
    ("gpt-3.5-turbo-16k-0613", false, false, true, 16_384),
    ("gpt-3.5-turbo-instruct", false, false, true, 4_096),
    ("gpt-3.5-turbo", false, false, true, 16_385),
    // --- OpenAI open-weight ---
    ("gpt-oss-120b", false, false, true, 131_072),
    ("gpt-oss-20b", false, false, true, 131_072),
    // --- Older code models ---
    ("starcoder", false, false, false, 8_192),
    ("codegen", false, false, false, 2_048),
];

/// Look up reference capabilities for a model by fuzzy name matching.
///
/// Normalizes the model name and checks if any reference pattern is a substring.
/// Returns the first match (table is ordered by specificity — longer/more specific first).
///
/// Handles naming variations across providers:
/// - `meta-llama/Llama-3.3-70B-Instruct` → matches `llama-3.3-70b`
/// - `llama3.1:8b` (Ollama) → matches after normalization
/// - `Qwen/Qwen2.5-72B-Instruct` → matches `qwen-2.5-72b`
pub fn get_reference_capabilities(model: &str) -> Option<ModelCapabilities> {
    let normalized = sanitize_model_name(&normalize_model_name(model));

    REFERENCE_CAPABILITIES
        .iter()
        .find(|(pattern, _, _, _, _)| {
            let sanitized_pattern = sanitize_model_name(pattern);
            normalized.contains(&sanitized_pattern)
        })
        .map(
            |(_, vision, video, structured_output, max_input_tokens)| ModelCapabilities {
                vision: *vision,
                video: *video,
                structured_output: *structured_output,
                max_input_tokens: *max_input_tokens,
            },
        )
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
