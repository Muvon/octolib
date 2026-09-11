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
    let provider = TinkerProvider::new();
    // Known Tinker IDs (colons intact — the factory splits only the first)
    assert!(provider.supports_model("thinkingmachines/Inkling"));
    assert!(provider.supports_model("thinkingmachines/Inkling-Small:peft:262144"));
    assert!(provider.supports_model("thinkingmachines/Inkling:peft:262144:sampling-nvfp4"));
    assert!(provider.supports_model("moonshotai/Kimi-K2.6"));
    // Arbitrary sampler checkpoint paths
    assert!(provider.supports_model(
        "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"
    ));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_pricing_ordering() {
    let provider = TinkerProvider::new();

    // Serverless variant wins over the `:peft:` training ID
    let serverless = provider
        .get_model_pricing("thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4")
        .unwrap();
    assert_eq!(serverless.input_price_per_1m, 0.30);
    assert_eq!(serverless.output_price_per_1m, 1.20);
    assert_eq!(serverless.cache_read_price_per_1m, 0.06);

    // 256K `:peft:` variant wins over the base ID
    let peft = provider
        .get_model_pricing("thinkingmachines/Inkling:peft:262144")
        .unwrap();
    assert_eq!(peft.input_price_per_1m, 3.74);
    assert_eq!(peft.output_price_per_1m, 9.36);

    // Base IDs
    let base = provider
        .get_model_pricing("thinkingmachines/Inkling")
        .unwrap();
    assert_eq!(base.input_price_per_1m, 1.87);
    assert_eq!(base.output_price_per_1m, 4.68);

    // Inkling-Small must not be captured by the Inkling pattern
    let small = provider
        .get_model_pricing("thinkingmachines/Inkling-Small")
        .unwrap();
    assert_eq!(small.input_price_per_1m, 0.58);
    assert_eq!(small.output_price_per_1m, 1.44);

    // `-base` suffix must not be captured by the instruct pattern
    let base_suffix = provider.get_model_pricing("Qwen/Qwen3.5-9B-Base").unwrap();
    assert_eq!(base_suffix.input_price_per_1m, 0.66);

    // Unknown checkpoint paths carry no pricing
    assert!(provider
        .get_model_pricing(
            "tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"
        )
        .is_none());
}

#[test]
fn test_context_windows() {
    let provider = TinkerProvider::new();
    assert_eq!(
        provider.get_max_input_tokens("thinkingmachines/Inkling"),
        65_536
    );
    assert_eq!(
        provider.get_max_input_tokens("thinkingmachines/Inkling:peft:262144:sampling-nvfp4"),
        262_144
    );
    assert_eq!(
        provider.get_max_input_tokens("moonshotai/Kimi-K2.6"),
        32_768
    );
    assert_eq!(
        provider.get_max_input_tokens("moonshotai/Kimi-K2.6:peft:131072"),
        131_072
    );
    // Unknown checkpoints fall back to the library default
    assert_eq!(
        provider.get_max_input_tokens("tinker://some/checkpoint"),
        262_144
    );
}

#[test]
fn test_conservative_capabilities() {
    let provider = TinkerProvider::new();
    assert_eq!(provider.name(), "tinker");
    assert!(!provider.supports_caching("thinkingmachines/Inkling"));
    assert!(!provider.supports_vision("thinkingmachines/Inkling"));
    assert!(!provider.supports_structured_output("thinkingmachines/Inkling"));
    assert!(!provider.enforces_response_schema("thinkingmachines/Inkling"));
}
