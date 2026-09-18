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

use super::*;

#[test]
fn test_supports_model() {
    let provider = CloudflareWorkersAiProvider::new();

    // Cloudflare Workers AI accepts any non-empty model identifier
    assert!(provider.supports_model("llama-3.1-70b-instruct"));
    assert!(provider.supports_model("@cf/meta/llama-3.1-70b-instruct"));
    assert!(provider.supports_model("@hf/meta/llama-3.1-8b-instruct"));
    assert!(provider.supports_model("mistral-7b-instruct-v0.1"));
    assert!(provider.supports_model("gemma-2-27b-it"));
    assert!(provider.supports_model("gpt-4"));
    assert!(provider.supports_model("claude-3"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = CloudflareWorkersAiProvider::new();

    // Test uppercase
    assert!(provider.supports_model("LLAMA-3.1-70B-INSTRUCT"));
    assert!(provider.supports_model("MISTRAL-7B-INSTRUCT-V0.1"));
    // Test mixed case
    assert!(provider.supports_model("Llama-3.1-70B-Instruct"));
    assert!(provider.supports_model("GEMMA-2-27B-IT"));
}

#[test]
fn current_workers_ai_models_use_cloudflare_prices() {
    let provider = CloudflareWorkersAiProvider::new();

    let deepseek = provider
        .get_model_pricing("@cf/deepseek-ai/deepseek-v4-flash-0731")
        .unwrap();
    assert_eq!(deepseek.input_price_per_1m, 0.440);
    assert_eq!(deepseek.cache_read_price_per_1m, 0.014);
    assert_eq!(deepseek.output_price_per_1m, 1.320);
    assert!(provider.supports_caching("@cf/deepseek-ai/deepseek-v4-flash-0731"));

    let qwen = provider.get_model_pricing("@cf/qwen/qwen3.8-27b").unwrap();
    assert_eq!(qwen.input_price_per_1m, 0.450);
    assert_eq!(qwen.output_price_per_1m, 3.200);
    assert_eq!(qwen.cache_read_price_per_1m, 0.050);
    assert!(provider.supports_caching("@cf/qwen/qwen3.8-27b"));
}

#[test]
fn legacy_models_use_cloudflare_prices_with_longest_id_first() {
    let provider = CloudflareWorkersAiProvider::new();

    // Variants share a prefix with the base ID, so the longer ID must win.
    let fast = provider
        .get_model_pricing("@cf/meta/llama-3.1-8b-instruct-fp8-fast")
        .unwrap();
    assert_eq!(fast.input_price_per_1m, 0.045);
    let fp8 = provider
        .get_model_pricing("@cf/meta/llama-3.1-8b-instruct-fp8")
        .unwrap();
    assert_eq!(fp8.input_price_per_1m, 0.152);
    let awq = provider
        .get_model_pricing("@cf/meta/llama-3-8b-instruct-awq")
        .unwrap();
    assert_eq!(awq.input_price_per_1m, 0.123);
    let base = provider
        .get_model_pricing("@cf/meta/llama-3-8b-instruct")
        .unwrap();
    assert_eq!(base.input_price_per_1m, 0.282);

    // 5500 neurons per M input tokens is $0.0605, not the rounded $0.060.
    let glm = provider
        .get_model_pricing("@cf/zai-org/glm-4.7-flash")
        .unwrap();
    assert_eq!(glm.input_price_per_1m, 0.0605);
    assert!(!provider.supports_caching("@cf/zai-org/glm-4.7-flash"));
    assert!(!provider.supports_caching("@cf/meta/llama-3.3-70b-instruct-fp8-fast"));
}

#[test]
fn model_pages_drive_context_vision_and_tool_support() {
    let provider = CloudflareWorkersAiProvider::new();

    // Context windows come from the model pages, not the vendor reference.
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-5.3-flash"),
        1_310_720
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-5.3"),
        1_310_720
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/qwen/qwen3.8-27b"),
        262_144
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/nvidia/nemotron-3-120b-a12b"),
        256_000
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-4.7-flash"),
        131_072
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        24_000
    );

    assert!(provider.supports_vision("@cf/moonshotai/kimi-k2.6"));
    assert!(provider.supports_vision("@cf/qwen/qwen3.8-27b"));
    assert!(provider.supports_vision("@cf/google/gemma-4-26b-a4b-it"));
    assert!(!provider.supports_vision("@cf/zai-org/glm-5.3"));

    // Structured output rides on function calling; JSON Mode is best effort.
    assert!(provider.supports_structured_output("@cf/zai-org/glm-5.3"));
    assert!(provider.supports_structured_output("@cf/openai/gpt-oss-120b"));
    assert!(!provider.supports_structured_output("@cf/qwen/qwq-32b"));
    assert!(!provider.enforces_response_schema("@cf/zai-org/glm-5.3"));

    // Only the OpenAI-shaped schemas document tool_choice "required".
    assert!(provider.supports_required_tool_choice("@cf/zai-org/glm-5.3"));
    assert!(provider.supports_required_tool_choice("@cf/moonshotai/kimi-k2.6"));
    assert!(!provider.supports_required_tool_choice("@cf/openai/gpt-oss-120b"));
    assert!(!provider.supports_required_tool_choice("@cf/meta/llama-4-scout-17b-16e-instruct"));
}

#[test]
fn august_2026_additions_use_cloudflare_prices() {
    let provider = CloudflareWorkersAiProvider::new();

    // GLM-5.3-Flash must resolve before the GLM-5.3 substring entry.
    let flash = provider
        .get_model_pricing("@cf/zai-org/glm-5.3-flash")
        .unwrap();
    assert_eq!(flash.input_price_per_1m, 0.150);
    assert_eq!(flash.output_price_per_1m, 0.500);
    assert_eq!(flash.cache_read_price_per_1m, 0.030);
    assert!(provider.supports_caching("@cf/zai-org/glm-5.3-flash"));

    let glm = provider.get_model_pricing("@cf/zai-org/glm-5.3").unwrap();
    assert_eq!(glm.input_price_per_1m, 1.400);
    assert_eq!(glm.output_price_per_1m, 4.400);
    assert_eq!(glm.cache_read_price_per_1m, 0.260);
    assert!(provider.supports_caching("@cf/zai-org/glm-5.3"));

    let kimi = provider
        .get_model_pricing("@cf/moonshotai/kimi-k2.5")
        .unwrap();
    assert_eq!(kimi.input_price_per_1m, 0.600);
    assert_eq!(kimi.output_price_per_1m, 3.000);
    assert_eq!(kimi.cache_read_price_per_1m, 0.100);
    assert!(provider.supports_caching("@cf/moonshotai/kimi-k2.5"));
}
