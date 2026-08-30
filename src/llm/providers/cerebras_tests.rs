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
    let provider = CerebrasProvider::new();
    assert!(provider.supports_model("gpt-oss-120b"));
    assert!(provider.supports_model("llama-3.1-8b"));
    assert!(provider.supports_model("QWEN-3-235B-A22B-INSTRUCT-2507"));
    assert!(provider.supports_model("zai-glm-4.7"));
    assert!(!provider.supports_model(""));
    assert!(!provider.supports_model("random-model"));
}

#[test]
fn test_default_capabilities() {
    let provider = CerebrasProvider::new();
    assert_eq!(provider.name(), "cerebras");
    assert!(!provider.supports_caching("any-model"));
    assert!(!provider.supports_vision("llama-3.1-8b"));
    assert!(provider.supports_structured_output("any-model"));
    assert_eq!(provider.get_max_input_tokens("llama-3.1-8b"), 131_072);
}

#[test]
fn test_pricing_support_partial() {
    let provider = CerebrasProvider::new();
    assert!(provider.get_model_pricing("llama-3.1-8b").is_some());
    assert!(provider.get_model_pricing("gpt-oss-120b").is_some());
    assert!(provider.get_model_pricing("zai-glm-4.7").is_some());
    assert!(provider
        .get_model_pricing("qwen-3-235b-a22b-instruct-2507")
        .is_some());
    assert!(crate::llm::utils::is_model_in_pricing_table(
        "LLAMA-3.1-8B",
        PRICING
    ));
}
