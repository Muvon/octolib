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
    assert!(!provider.supports_caching("@cf/qwen/qwen3.8-27b"));
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
