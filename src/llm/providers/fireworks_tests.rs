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

    let qwen_max = provider
        .get_model_pricing("accounts/fireworks/models/qwen3p8-max")
        .unwrap();
    assert_eq!(qwen_max.input_price_per_1m, 2.00);
    assert_eq!(qwen_max.cache_read_price_per_1m, 0.25);
    assert_eq!(qwen_max.output_price_per_1m, 6.00);

    let deepseek_pro = provider
        .get_model_pricing("accounts/fireworks/models/deepseek-v4-pro-0813")
        .unwrap();
    assert_eq!(deepseek_pro.input_price_per_1m, 1.32);
    assert_eq!(deepseek_pro.cache_read_price_per_1m, 0.044);
    assert_eq!(deepseek_pro.output_price_per_1m, 3.96);
}

#[test]
fn newly_listed_serverless_routes_use_fireworks_pricing_and_context() {
    let provider = FireworksProvider::new();

    let glm = provider
        .get_model_pricing("accounts/fireworks/models/glm-5p3")
        .unwrap();
    assert_eq!(glm.input_price_per_1m, 1.40);
    assert_eq!(glm.cache_read_price_per_1m, 0.26);
    assert_eq!(glm.output_price_per_1m, 4.40);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/glm-5p3"),
        1_040_000
    );

    let glm_flash = provider
        .get_model_pricing("accounts/fireworks/models/glm-5p3-flash")
        .unwrap();
    assert_eq!(glm_flash.input_price_per_1m, 0.15);
    assert_eq!(glm_flash.cache_read_price_per_1m, 0.03);
    assert_eq!(glm_flash.output_price_per_1m, 0.50);

    let kimi = provider
        .get_model_pricing("accounts/fireworks/models/kimi-k2p6")
        .unwrap();
    assert_eq!(kimi.input_price_per_1m, 0.95);
    assert_eq!(kimi.cache_read_price_per_1m, 0.16);
    assert_eq!(kimi.output_price_per_1m, 4.00);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/kimi-k2p6"),
        262_144
    );

    let gpt_oss = provider
        .get_model_pricing("accounts/fireworks/models/gpt-oss-120b")
        .unwrap();
    assert_eq!(gpt_oss.input_price_per_1m, 0.15);
    assert_eq!(gpt_oss.cache_read_price_per_1m, 0.015);
    assert_eq!(gpt_oss.output_price_per_1m, 0.60);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/gpt-oss-120b"),
        131_072
    );

    let nemotron = provider
        .get_model_pricing("accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b")
        .unwrap();
    assert_eq!(nemotron.input_price_per_1m, 0.05);
    assert_eq!(nemotron.cache_read_price_per_1m, 0.01);
    assert_eq!(nemotron.output_price_per_1m, 0.20);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b"),
        262_144
    );

    let muse = provider
        .get_model_pricing("accounts/fireworks/models/muse-glimmer-30b")
        .unwrap();
    assert_eq!(muse.input_price_per_1m, 0.35);
    assert_eq!(muse.cache_read_price_per_1m, 0.04);
    assert_eq!(muse.output_price_per_1m, 1.50);
    assert_eq!(
        provider.get_max_input_tokens("accounts/fireworks/models/muse-glimmer-30b"),
        131_072
    );
}

#[test]
fn retired_routes_leave_fireworks_table() {
    assert!(fireworks_model_pricing("accounts/fireworks/models/qwen3p7-plus").is_none());
}
