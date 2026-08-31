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
