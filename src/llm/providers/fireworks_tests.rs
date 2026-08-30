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
    let provider = FireworksProvider::new();
    assert!(provider.supports_model("accounts/fireworks/models/kimi-k2-instruct-0905"));
    assert!(provider.supports_model("accounts/fireworks/models/deepseek-v3"));
    assert!(provider.supports_model("accounts/fireworks/models/qwen3-coder-480b-a35b-instruct"));
    assert!(provider.supports_model("any-future-model"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = FireworksProvider::new();
    assert_eq!(provider.name(), "fireworks");
    assert!(provider.supports_caching("any-model"));
    assert!(provider.supports_structured_output("any-model"));
}

#[test]
fn test_pricing_reference_fallback() {
    let provider = FireworksProvider::new();
    assert!(provider
        .get_model_pricing("accounts/fireworks/models/deepseek-v3")
        .is_some());
}

#[test]
fn current_serverless_routes_use_fireworks_pricing_and_context() {
    let provider = FireworksProvider::new();

    let qwen = provider
        .get_model_pricing("accounts/fireworks/models/qwen3p8-2p4t-a95b")
        .unwrap();
    assert_eq!(qwen.input_price_per_1m, 2.00);
    assert_eq!(qwen.cache_read_price_per_1m, 0.25);
    assert_eq!(qwen.output_price_per_1m, 6.00);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/qwen3p8-2p4t-a95b"),
        262_144
    );

    let deepseek = provider
        .get_model_pricing("accounts/fireworks/models/deepseek-v4-flash")
        .unwrap();
    assert_eq!(deepseek.input_price_per_1m, 0.22);
    assert_eq!(deepseek.cache_read_price_per_1m, 0.007);
    assert_eq!(deepseek.output_price_per_1m, 0.66);

    let kimi = provider
        .get_model_pricing("accounts/fireworks/models/kimi-k3")
        .unwrap();
    assert_eq!(kimi.input_price_per_1m, 3.00);
    assert_eq!(kimi.cache_read_price_per_1m, 0.30);
    assert_eq!(kimi.output_price_per_1m, 15.00);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/kimi-k3"),
        1_040_000
    );
}
