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
    let provider = OpenRouterProvider::new();

    // OpenRouter supports many models
    assert!(provider.supports_model("anthropic/claude-3.5-sonnet"));
    assert!(provider.supports_model("openai/gpt-4o"));
    assert!(provider.supports_model("meta/llama-3.1-70b"));
    assert!(provider.supports_model("deepseek-chat"));

    // Should accept any non-empty model string as fallback
    assert!(provider.supports_model("any-model-name"));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = OpenRouterProvider::new();

    // Test uppercase
    assert!(provider.supports_model("ANTHROPIC/CLAUDE-3.5-SONNET"));
    assert!(provider.supports_model("OPENAI/GPT-4O"));
    assert!(provider.supports_model("META/LLAMA-3.1-70B"));
    // Test mixed case
    assert!(provider.supports_model("Anthropic/Claude-3.5-Sonnet"));
    assert!(provider.supports_model("DEEPSEEK-CHAT"));
}

#[test]
fn test_supports_vision_case_insensitive() {
    let provider = OpenRouterProvider::new();

    // Test lowercase
    assert!(provider.supports_vision("gpt-4o"));
    assert!(provider.supports_vision("claude-3-haiku"));

    // Test uppercase
    assert!(provider.supports_vision("GPT-4O"));
    assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
    // Test mixed case
    assert!(provider.supports_vision("Gemini-1.5-Pro"));
}

#[test]
fn test_supports_caching_case_insensitive() {
    let provider = OpenRouterProvider::new();

    // Test lowercase
    assert!(provider.supports_caching("anthropic/claude-3.5-sonnet"));
    assert!(provider.supports_caching("claude-3-haiku"));

    // Test uppercase
    assert!(provider.supports_caching("ANTHROPIC/CLAUDE-3.5-SONNET"));
    assert!(provider.supports_caching("CLAUDE-3-HAIKU"));
}

#[test]
fn test_anthropic_fast_route_doubles_pricing() {
    let provider = OpenRouterProvider::new();
    let base = provider
        .get_model_pricing("anthropic/claude-opus-5")
        .unwrap();
    let fast = provider
        .get_model_pricing("anthropic/claude-opus-5-fast")
        .unwrap();
    assert_eq!(fast.input_price_per_1m, base.input_price_per_1m * 2.0);
    assert_eq!(fast.output_price_per_1m, base.output_price_per_1m * 2.0);
    assert_eq!(
        fast.cache_write_price_per_1m,
        base.cache_write_price_per_1m * 2.0
    );
    assert_eq!(
        fast.cache_read_price_per_1m,
        base.cache_read_price_per_1m * 2.0
    );
}

#[test]
fn test_schema_enforcement_proxy_policy() {
    let provider = OpenRouterProvider::new();
    assert!(provider.enforces_response_schema("deepseek-v4-pro"));
    assert!(provider.enforces_response_schema("openai/gpt-4o"));
    assert!(provider.enforces_response_schema("unknown/provider-model"));
    assert!(!provider.enforces_response_schema("mistral-7b"));
}
