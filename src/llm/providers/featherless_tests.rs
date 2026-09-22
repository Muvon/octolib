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
    let provider = FeatherlessProvider::new();
    assert!(provider.supports_model("Qwen/Qwen2.5-7B-Instruct"));
    assert!(provider.supports_model("meta-llama/Meta-Llama-3.1-8B-Instruct"));
    assert!(provider.supports_model("mistralai/Mistral-7B-Instruct-v0.3"));
    assert!(provider.supports_model("any-future-model"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = FeatherlessProvider::new();
    assert_eq!(provider.name(), "featherless");
    assert!(provider.supports_structured_output("any-model"));
    assert!(!provider.supports_caching("Qwen/Qwen2.5-7B-Instruct"));
    assert!(!provider.supports_caching("any-model"));
    assert!(provider.supports_caching("deepseek-ai/DeepSeek-V4-Flash-0731"));
}

#[test]
fn test_current_developer_pricing() {
    let provider = FeatherlessProvider::new();
    let pricing = provider
        .get_model_pricing("deepseek-ai/DeepSeek-V4-Flash-0731")
        .unwrap();
    assert_eq!(pricing.input_price_per_1m, 0.14);
    assert_eq!(pricing.cache_read_price_per_1m, 0.03);
    assert_eq!(pricing.output_price_per_1m, 0.28);

    let kimi = provider.get_model_pricing("moonshotai/Kimi-K3").unwrap();
    assert_eq!(kimi.input_price_per_1m, 2.00);
    assert_eq!(kimi.output_price_per_1m, 10.00);

    let glm = provider.get_model_pricing("zai-org/GLM-5.2").unwrap();
    assert_eq!(glm.input_price_per_1m, 1.40);
    assert_eq!(glm.cache_read_price_per_1m, 0.15);
    assert_eq!(glm.output_price_per_1m, 4.40);

    let gpt_oss = provider.get_model_pricing("openai/gpt-oss-120b").unwrap();
    assert_eq!(gpt_oss.input_price_per_1m, 0.15);
    assert_eq!(gpt_oss.cache_read_price_per_1m, 0.02);
    assert_eq!(gpt_oss.output_price_per_1m, 0.60);

    let deepseek_v3 = provider
        .get_model_pricing("deepseek-ai/DeepSeek-V3.2")
        .unwrap();
    assert_eq!(deepseek_v3.input_price_per_1m, 0.264);
    assert_eq!(deepseek_v3.cache_read_price_per_1m, 0.06);
    assert_eq!(deepseek_v3.output_price_per_1m, 0.41);

    // Unlisted model classes retain the shared reference estimate.
    assert!(provider
        .get_model_pricing("meta-llama/Llama-3.1-8B-Instruct")
        .is_some());
}

#[test]
fn test_newly_listed_developer_pricing() {
    let provider = FeatherlessProvider::new();

    let glm = provider.get_model_pricing("zai-org/GLM-4.7").unwrap();
    assert_eq!(glm.input_price_per_1m, 0.55);
    assert_eq!(glm.cache_read_price_per_1m, 0.11);
    assert_eq!(glm.output_price_per_1m, 2.20);

    let kimi = provider.get_model_pricing("moonshotai/Kimi-K2.5").unwrap();
    assert_eq!(kimi.input_price_per_1m, 0.80);
    assert_eq!(kimi.cache_read_price_per_1m, 0.154);
    assert_eq!(kimi.output_price_per_1m, 3.40);

    let qwen = provider
        .get_model_pricing("Qwen/Qwen3.5-397B-A17B")
        .unwrap();
    assert_eq!(qwen.input_price_per_1m, 0.55);
    assert_eq!(qwen.output_price_per_1m, 3.50);
    assert!(!provider.supports_caching("Qwen/Qwen3.5-397B-A17B"));

    let coder = provider
        .get_model_pricing("Qwen/Qwen3-Coder-480B-A35B-Instruct")
        .unwrap();
    assert_eq!(coder.input_price_per_1m, 0.38);
    assert_eq!(coder.cache_read_price_per_1m, 0.076);
    assert_eq!(coder.output_price_per_1m, 1.55);

    let nemotron = provider
        .get_model_pricing("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
        .unwrap();
    assert_eq!(nemotron.input_price_per_1m, 0.125);
    assert_eq!(nemotron.cache_read_price_per_1m, 0.025);
    assert_eq!(nemotron.output_price_per_1m, 1.15);
    assert!(provider.supports_caching("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"));
}
