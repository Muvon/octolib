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
    let provider = HetznerProvider::new();
    assert!(provider.supports_model("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(provider.supports_model("DeepSeek-V4-Flash-0731"));
    assert!(provider.supports_model("GLM-5.2-NVFP4"));
    assert!(provider.supports_model("Kimi-K2.7-Code"));
    assert!(!provider.supports_model("unknown-model"));
    assert!(!provider.supports_model(""));

    // Case-insensitive: the provider canonicalizes before sending
    assert!(provider.supports_model("deepseek-v4-flash-0731"));
    assert!(provider.supports_model("qwen/qwen3.6-35b-a3b-fp8"));
    assert!(provider.supports_model("KIMI-K2.7-CODE"));
    assert_eq!(
        find_model("glm-5.2-nvfp4").map(|(id, _, _)| *id),
        Some("GLM-5.2-NVFP4")
    );
}

#[test]
fn test_model_capabilities() {
    let provider = HetznerProvider::new();
    assert!(provider.supports_vision("Kimi-K2.7-Code"));
    assert!(provider.supports_vision("Qwen/Qwen3.6-35B-A3B-FP8"));
    assert!(!provider.supports_vision("GLM-5.2-NVFP4"));
    assert_eq!(
        provider.get_max_input_tokens("DeepSeek-V4-Flash-0731"),
        512_000
    );
    assert_eq!(provider.get_max_input_tokens("Kimi-K2.7-Code"), 262_144);
}

#[test]
fn test_default_capabilities() {
    let provider = HetznerProvider::new();
    assert_eq!(provider.name(), "hetzner");
    assert!(!provider.supports_caching("any-model"));
    assert!(provider.supports_structured_output("any-model"));
}

#[test]
fn test_free_pricing() {
    let provider = HetznerProvider::new();
    let pricing = provider.get_model_pricing("GLM-5.2-NVFP4").unwrap();
    assert_eq!(pricing.input_price_per_1m, 0.0);
    assert_eq!(pricing.output_price_per_1m, 0.0);
}
