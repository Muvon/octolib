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

    // Unlisted model classes retain the shared reference estimate.
    assert!(provider
        .get_model_pricing("meta-llama/Llama-3.1-8B-Instruct")
        .is_some());
}
