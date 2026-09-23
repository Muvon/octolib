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
    let provider = CloudflareWorkersAiProvider::new();

    // Cloudflare Workers AI accepts any non-empty model identifier
    assert!(provider.supports_model("llama-3.1-70b-instruct"));
    assert!(provider.supports_model("@cf/meta/llama-3.1-70b-instruct"));
    assert!(provider.supports_model("@hf/meta/llama-3.1-8b-instruct"));
    assert!(provider.supports_model("mistral-7b-instruct-v0.1"));
    assert!(provider.supports_model("gemma-2-27b-it"));
    assert!(provider.supports_model("gpt-4"));
    assert!(provider.supports_model("claude-3"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = CloudflareWorkersAiProvider::new();

    // Test uppercase
    assert!(provider.supports_model("LLAMA-3.1-70B-INSTRUCT"));
    assert!(provider.supports_model("MISTRAL-7B-INSTRUCT-V0.1"));
    // Test mixed case
    assert!(provider.supports_model("Llama-3.1-70B-Instruct"));
    assert!(provider.supports_model("GEMMA-2-27B-IT"));
}

#[test]
fn current_workers_ai_models_use_cloudflare_prices() {
    let provider = CloudflareWorkersAiProvider::new();

    let deepseek = provider
        .get_model_pricing("@cf/deepseek-ai/deepseek-v4-flash-0731")
        .unwrap();
    assert_eq!(deepseek.input_price_per_1m, 0.440);
    assert_eq!(deepseek.cache_read_price_per_1m, 0.014);
    assert_eq!(deepseek.output_price_per_1m, 1.320);
    assert!(provider.supports_caching("@cf/deepseek-ai/deepseek-v4-flash-0731"));

    let qwen = provider.get_model_pricing("@cf/qwen/qwen3.8-27b").unwrap();
    assert_eq!(qwen.input_price_per_1m, 0.450);
    assert_eq!(qwen.output_price_per_1m, 3.200);
    assert_eq!(qwen.cache_read_price_per_1m, 0.050);
    assert!(provider.supports_caching("@cf/qwen/qwen3.8-27b"));
}

#[test]
fn legacy_models_use_cloudflare_prices_with_longest_id_first() {
    let provider = CloudflareWorkersAiProvider::new();

    // Variants share a prefix with the base ID, so the longer ID must win.
    let fast = provider
        .get_model_pricing("@cf/meta/llama-3.1-8b-instruct-fp8-fast")
        .unwrap();
    assert_eq!(fast.input_price_per_1m, 0.045);
    let fp8 = provider
        .get_model_pricing("@cf/meta/llama-3.1-8b-instruct-fp8")
        .unwrap();
    assert_eq!(fp8.input_price_per_1m, 0.152);
    let awq = provider
        .get_model_pricing("@cf/meta/llama-3-8b-instruct-awq")
        .unwrap();
    assert_eq!(awq.input_price_per_1m, 0.123);
    let base = provider
        .get_model_pricing("@cf/meta/llama-3-8b-instruct")
        .unwrap();
    assert_eq!(base.input_price_per_1m, 0.282);

    // 5500 neurons per M input tokens is $0.0605, not the rounded $0.060.
    let glm = provider
        .get_model_pricing("@cf/zai-org/glm-4.7-flash")
        .unwrap();
    assert_eq!(glm.input_price_per_1m, 0.0605);
    assert!(!provider.supports_caching("@cf/zai-org/glm-4.7-flash"));
    assert!(!provider.supports_caching("@cf/meta/llama-3.3-70b-instruct-fp8-fast"));
}

#[test]
fn model_pages_drive_context_vision_and_tool_support() {
    let provider = CloudflareWorkersAiProvider::new();

    // Context windows come from the model pages, not the vendor reference.
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-5.3-flash"),
        1_310_720
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-5.3"),
        1_310_720
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/qwen/qwen3.8-27b"),
        262_144
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/nvidia/nemotron-3-120b-a12b"),
        256_000
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/zai-org/glm-4.7-flash"),
        131_072
    );
    assert_eq!(
        provider.get_max_input_tokens("@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
        24_000
    );

    assert!(provider.supports_vision("@cf/moonshotai/kimi-k2.6"));
    assert!(provider.supports_vision("@cf/qwen/qwen3.8-27b"));
    assert!(provider.supports_vision("@cf/google/gemma-4-26b-a4b-it"));
    assert!(!provider.supports_vision("@cf/zai-org/glm-5.3"));

    // OpenAI-shaped models decode against the schema; legacy ones need a
    // forced tool call, so their structured output rides on function calling.
    assert!(provider.supports_structured_output("@cf/zai-org/glm-5.3"));
    assert!(provider.supports_structured_output("@cf/openai/gpt-oss-120b"));
    assert!(!provider.supports_structured_output("@cf/qwen/qwq-32b"));
    assert!(provider.enforces_response_schema("@cf/zai-org/glm-5.3-flash"));
    assert!(provider.enforces_response_schema("@cf/deepseek-ai/deepseek-v4-flash-0731"));
    assert!(!provider.enforces_response_schema("@cf/openai/gpt-oss-120b"));
    assert!(!provider.enforces_response_schema("@cf/meta/llama-3.3-70b-instruct-fp8-fast"));

    // Only the OpenAI-shaped schemas document tool_choice "required".
    assert!(provider.supports_required_tool_choice("@cf/zai-org/glm-5.3"));
    assert!(provider.supports_required_tool_choice("@cf/moonshotai/kimi-k2.6"));
    assert!(!provider.supports_required_tool_choice("@cf/openai/gpt-oss-120b"));
    assert!(!provider.supports_required_tool_choice("@cf/meta/llama-4-scout-17b-16e-instruct"));
}

#[test]
fn august_2026_additions_use_cloudflare_prices() {
    let provider = CloudflareWorkersAiProvider::new();

    // GLM-5.3-Flash must resolve before the GLM-5.3 substring entry.
    let flash = provider
        .get_model_pricing("@cf/zai-org/glm-5.3-flash")
        .unwrap();
    assert_eq!(flash.input_price_per_1m, 0.150);
    assert_eq!(flash.output_price_per_1m, 0.500);
    assert_eq!(flash.cache_read_price_per_1m, 0.030);
    assert!(provider.supports_caching("@cf/zai-org/glm-5.3-flash"));

    let glm = provider.get_model_pricing("@cf/zai-org/glm-5.3").unwrap();
    assert_eq!(glm.input_price_per_1m, 1.400);
    assert_eq!(glm.output_price_per_1m, 4.400);
    assert_eq!(glm.cache_read_price_per_1m, 0.260);
    assert!(provider.supports_caching("@cf/zai-org/glm-5.3"));

    let kimi = provider
        .get_model_pricing("@cf/moonshotai/kimi-k2.5")
        .unwrap();
    assert_eq!(kimi.input_price_per_1m, 0.600);
    assert_eq!(kimi.output_price_per_1m, 3.000);
    assert_eq!(kimi.cache_read_price_per_1m, 0.100);
    assert!(provider.supports_caching("@cf/moonshotai/kimi-k2.5"));
}

fn catalog_fixture() -> Vec<SearchEntry> {
    // Trimmed from the catalog entries the docs site is generated from.
    serde_json::from_value(serde_json::json!([
        {
            "name": "@cf/zai-org/glm-5.3",
            "task": {"name": "Text Generation"},
            "properties": [
                {"property_id": "require_workers_paid", "value": "true"},
                {"property_id": "context_window", "value": "1310720"},
                {"property_id": "function_calling", "value": "true"},
                {"property_id": "reasoning", "value": "true"},
                {"property_id": "reasoning_effort", "value": {
                    "supported_efforts": ["max", "high", "low"],
                    "default_effort": "max",
                    "normalizes_to": {"none": "max", "medium": "max", "null": "max"},
                    "mandatory": true,
                    "default_enabled": true
                }},
                {"property_id": "price", "value": [
                    {"unit": "per M input tokens", "price": 1.4, "currency": "USD"},
                    {"unit": "per M output tokens", "price": 4.4, "currency": "USD"},
                    {"unit": "per M cached input tokens", "price": 0.26, "currency": "USD"}
                ]}
            ]
        },
        {
            "name": "@cf/moonshotai/kimi-k2.6",
            "task": {"name": "Text Generation"},
            "properties": [
                {"property_id": "context_window", "value": "262144"},
                {"property_id": "function_calling", "value": "true"},
                {"property_id": "vision", "value": "true"},
                {"property_id": "reasoning_effort", "value": {
                    "supported_efforts": ["high", "none"],
                    "default_effort": "high",
                    "normalizes_to": {"low": "high", "medium": "high", "max": "high", "null": "high"},
                    "mandatory": false,
                    "default_enabled": true
                }},
                {"property_id": "price", "value": [
                    {"unit": "per M input tokens", "price": 0.95, "currency": "USD"},
                    {"unit": "per M output tokens", "price": 4, "currency": "USD"},
                    {"unit": "per M cached input tokens", "price": 0.16, "currency": "USD"}
                ]}
            ]
        },
        {
            "name": "@cf/meta/llama-3.2-1b-instruct",
            "task": {"name": "Text Generation"},
            "properties": [
                {"property_id": "context_window", "value": "60000"},
                {"property_id": "price", "value": [
                    {"unit": "per M input tokens", "price": 0.027, "currency": "USD"},
                    {"unit": "per M output tokens", "price": 0.201, "currency": "USD"}
                ]}
            ]
        },
        {
            "name": "@cf/openai/whisper",
            "task": {"name": "Automatic Speech Recognition"},
            "properties": [
                {"property_id": "price", "value": [
                    {"unit": "per audio minute", "price": 0.000453, "currency": "USD"}
                ]}
            ]
        },
        {
            "name": "@cf/example/no-task",
            "properties": []
        }
    ]))
    .unwrap()
}

#[test]
fn catalog_keeps_text_generation_entries_with_their_properties() {
    let catalog = parse_catalog(catalog_fixture());
    assert_eq!(catalog.len(), 3);

    let glm = catalog_model(&catalog, "@CF/zai-org/GLM-5.3").unwrap();
    assert_eq!(glm.context_window, Some(1_310_720));
    assert!(glm.function_calling);
    assert!(!glm.vision);
    let pricing = glm.pricing.as_ref().unwrap();
    assert_eq!(pricing.input_price_per_1m, 1.4);
    assert_eq!(pricing.output_price_per_1m, 4.4);
    assert_eq!(pricing.cache_write_price_per_1m, 1.4);
    assert_eq!(pricing.cache_read_price_per_1m, 0.26);

    let kimi = catalog_model(&catalog, "@cf/moonshotai/kimi-k2.6").unwrap();
    assert!(kimi.vision);

    // No cached rate means cache hits bill at the input rate.
    let llama = catalog_model(&catalog, "@cf/meta/llama-3.2-1b-instruct").unwrap();
    assert!(!llama.function_calling);
    let pricing = llama.pricing.as_ref().unwrap();
    assert_eq!(pricing.cache_read_price_per_1m, pricing.input_price_per_1m);

    // Membership is exact, not a substring match.
    assert!(catalog_model(&catalog, "@cf/zai-org/glm-5").is_none());
    assert!(catalog_model(&catalog, "@cf/openai/whisper").is_none());
}

#[test]
fn catalog_entries_without_price_or_context_stay_open() {
    let catalog = parse_catalog(
        serde_json::from_value(serde_json::json!([{
            "name": "@cf/example/bare",
            "task": {"name": "Text Generation"},
            "properties": [{"property_id": "price", "value": "n/a"}]
        }]))
        .unwrap(),
    );
    let bare = catalog_model(&catalog, "@cf/example/bare").unwrap();
    assert!(bare.pricing.is_none());
    assert!(bare.context_window.is_none());
    assert!(!bare.vision);
}

#[test]
fn catalog_effort_prefers_supported_then_documented_normalization_then_floor() {
    use crate::llm::types::ReasoningEffort;
    let catalog = parse_catalog(catalog_fixture());

    // GLM-5.3: max|high|low supported; medium is documented as normalized.
    let glm = catalog_model(&catalog, "@cf/zai-org/glm-5.3").unwrap();
    assert_eq!(glm.supported_efforts, ["max", "high", "low"]);
    assert_eq!(select_effort(glm, ReasoningEffort::Max), Some("max"));
    assert_eq!(select_effort(glm, ReasoningEffort::Low), Some("low"));
    assert_eq!(select_effort(glm, ReasoningEffort::Medium), Some("medium"));
    // xhigh is neither supported nor normalized, so it floors to high.
    assert_eq!(select_effort(glm, ReasoningEffort::XHigh), Some("high"));

    // Kimi K2.6: only high (and none); everything Cloudflare normalizes goes verbatim.
    let kimi = catalog_model(&catalog, "@cf/moonshotai/kimi-k2.6").unwrap();
    assert_eq!(select_effort(kimi, ReasoningEffort::Max), Some("max"));
    assert_eq!(select_effort(kimi, ReasoningEffort::Low), Some("low"));
    // xhigh is undocumented there too; high is the only ladder level supported.
    assert_eq!(select_effort(kimi, ReasoningEffort::XHigh), Some("high"));

    // No effort knob at all: the shared ladder stays in charge.
    let llama = catalog_model(&catalog, "@cf/meta/llama-3.2-1b-instruct").unwrap();
    assert_eq!(select_effort(llama, ReasoningEffort::Max), None);

    // Only high supported and nothing below it: the lowest supported wins.
    let only_high = CatalogModel {
        id: "@cf/moonshotai/kimi-k2.7-code".to_string(),
        context_window: None,
        vision: false,
        function_calling: true,
        pricing: None,
        supported_efforts: vec!["high".to_string()],
        normalized_efforts: Vec::new(),
    };
    assert_eq!(
        select_effort(&only_high, ReasoningEffort::Low),
        Some("high")
    );
    assert_eq!(
        select_effort(&only_high, ReasoningEffort::Max),
        Some("high")
    );
}

#[test]
fn catalog_effort_none_is_sent_only_where_the_model_lists_it() {
    use crate::llm::types::ReasoningEffort;
    let catalog = parse_catalog(catalog_fixture());
    // GLM-5.3 lists max|high|low: "none" is not an effort there, and it must
    // not floor to "low" — no reasoning means no effort field at all.
    let glm = catalog_model(&catalog, "@cf/zai-org/glm-5.3").unwrap();
    assert_eq!(select_effort(glm, ReasoningEffort::None), None);
}
