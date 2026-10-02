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
    let provider = AmazonBedrockProvider::new();

    // Amazon Bedrock accepts any non-empty model identifier
    assert!(provider.supports_model("anthropic.claude-3-haiku-20240307-v1:0"));
    assert!(provider.supports_model("anthropic.claude-3-5-sonnet-20241022-v2:0"));
    assert!(provider.supports_model("meta.llama3-2-90b-instruct-v1:0"));
    assert!(provider.supports_model("amazon.titan-embed-text-v2:0"));
    assert!(provider.supports_model("gpt-4"));
    assert!(provider.supports_model("deepseek-chat"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = AmazonBedrockProvider::new();

    // Test uppercase
    assert!(provider.supports_model("ANTHROPIC.CLAUDE-3-HAIKU-20240307-V1:0"));
    assert!(provider.supports_model("META.LLAMA3-2-90B-INSTRUCT-V1:0"));
    // Test mixed case
    assert!(provider.supports_model("Anthropic.Claude-3-Haiku"));
    assert!(provider.supports_model("AMAZON.TITAN-EMBED-TEXT-V2:0"));
}

#[test]
fn test_supports_vision_case_insensitive() {
    let provider = AmazonBedrockProvider::new();

    // Test lowercase
    assert!(provider.supports_vision("claude-3-haiku"));
    assert!(provider.supports_vision("claude-3-sonnet"));

    // Test uppercase
    assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
    assert!(provider.supports_vision("CLAUDE-3-SONNET"));
    // Test mixed case
    assert!(provider.supports_vision("Anthropic.Claude-3-Haiku"));
}

#[test]
fn test_nova_vision_and_pricing_resolve_from_reference_table() {
    let provider = AmazonBedrockProvider::new();

    // Multimodal Nova models keep vision through the reference fallback.
    assert!(provider.supports_vision("amazon.nova-2-lite-v1:0"));
    assert!(provider.supports_vision("amazon.nova-pro-v1:0"));
    // Nova Micro is text-only and must not inherit family-wide vision.
    assert!(!provider.supports_vision("amazon.nova-micro-v1:0"));

    let pricing = provider
        .get_model_pricing("amazon.nova-pro-v1:0")
        .expect("nova-pro must resolve to pricing");
    assert_eq!(pricing.input_price_per_1m, 0.80);
    assert_eq!(pricing.output_price_per_1m, 3.20);
    assert_eq!(pricing.cache_read_price_per_1m, 0.20);
}

#[test]
fn test_nova_has_no_native_structured_outputs() {
    let provider = AmazonBedrockProvider::new();

    // Nova model cards list structured outputs as not supported; Claude
    // routes on Bedrock keep them.
    assert!(!provider.supports_structured_output("amazon.nova-pro-v1:0"));
    assert!(!provider.enforces_response_schema("amazon.nova-micro-v1:0"));
    assert!(provider.supports_structured_output("anthropic.claude-sonnet-4-5"));
    assert!(provider.enforces_response_schema("anthropic.claude-sonnet-4-5"));
}

#[test]
fn bedrock_rates_override_maker_reference_rates() {
    let provider = AmazonBedrockProvider::new();

    for (model, input, output) in [
        ("qwen.qwen3-32b-v1:0", 0.15, 0.60),
        ("qwen.qwen3-235b-a22b-2507-v1:0", 0.22, 0.88),
        ("qwen.qwen3-coder-480b-a35b-v1:0", 0.45, 1.80),
        ("qwen.qwen3-coder-next", 0.50, 1.20),
        ("qwen.qwen3-next-80b-a3b-instruct", 0.14, 1.20),
        ("openai.gpt-oss-120b-1:0", 0.15, 0.60),
        ("openai.gpt-oss-20b-1:0", 0.07, 0.30),
        ("deepseek.v3.2", 0.62, 1.85),
        ("deepseek.v3-v1:0", 0.58, 1.68),
        ("deepseek.r1-v1:0", 1.35, 5.40),
        ("zai.glm-4.7-flash", 0.07, 0.40),
        ("minimax.minimax-m2.1", 0.30, 1.20),
        ("google.gemma-4-31b", 0.14, 0.40),
        ("google.gemma-4-26b-a4b", 0.13, 0.40),
        ("google.gemma-4-e2b", 0.04, 0.08),
        ("moonshotai.kimi-k3", 3.30, 16.50),
        ("us.moonshotai.kimi-k3", 3.30, 16.50),
        ("global.moonshotai.kimi-k3", 3.00, 15.00),
        ("xai.grok-4.7", 2.20, 6.60),
        ("global.xai.grok-4.7", 2.00, 6.00),
    ] {
        let pricing = provider
            .get_model_pricing(model)
            .unwrap_or_else(|| panic!("{model} must resolve to pricing"));
        assert_eq!(pricing.input_price_per_1m, input, "{model}");
        assert_eq!(pricing.output_price_per_1m, output, "{model}");
    }

    // The vendor prefix keeps Bedrock rates off the makers' own ids.
    assert!(get_model_pricing("deepseek-v3.2", PRICING).is_none());
    assert!(get_model_pricing("qwen/qwen3-32b", PRICING).is_none());
}
