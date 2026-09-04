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

/// The verifier route the completion gate runs on. The openrouter catalogue
/// advertises structured_outputs=false for this model while the live route
/// honours a strict json_schema, so the entry is measured rather than
/// inherited — and an entry that never matches would silently fall back to
/// the optimistic "unknown models enforce" default.
#[test]
fn qwen_3_7_flash_resolves_for_the_openrouter_route() {
    let caps = get_reference_capabilities("qwen/qwen3.7-flash")
        .expect("qwen3.7-flash must resolve to a reference entry");
    assert!(caps.structured_output);
    assert_eq!(caps.max_input_tokens, 1_000_000);
    assert!(proxy_route_enforces_response_schema("qwen/qwen3.7-flash"));
    let pricing = get_reference_pricing("qwen/qwen3.7-flash")
        .expect("qwen3.7-flash must resolve to reference pricing");
    assert_eq!(pricing.input_price_per_1m, 0.03);
    assert_eq!(pricing.output_price_per_1m, 0.13);
}

fn assert_same_capabilities(left: ModelCapabilities, right: ModelCapabilities) {
    assert_eq!(left.vision, right.vision);
    assert_eq!(left.video, right.video);
    assert_eq!(left.structured_output, right.structured_output);
    assert_eq!(left.max_input_tokens, right.max_input_tokens);
}

fn assert_same_pricing(left: ModelPricing, right: ModelPricing) {
    assert_eq!(left.input_price_per_1m, right.input_price_per_1m);
    assert_eq!(left.output_price_per_1m, right.output_price_per_1m);
    assert_eq!(
        left.cache_write_price_per_1m,
        right.cache_write_price_per_1m
    );
    assert_eq!(left.cache_read_price_per_1m, right.cache_read_price_per_1m);
}

/// The reference table is the pricing fallback for aggregator routes
/// (OpenCode Zen, Ollama, NVIDIA, local). Drift from the first-party
/// provider tables silently misbills those routes, so they must agree.
#[test]
fn reference_pricing_matches_first_party_provider_tables() {
    use crate::llm::providers::{AnthropicProvider, GoogleVertexProvider};
    use crate::llm::traits::AiProvider;

    let anthropic = AnthropicProvider::new();
    for model in [
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-haiku-4-5",
    ] {
        assert_same_pricing(
            get_reference_pricing(model).unwrap(),
            anthropic.get_model_pricing(model).unwrap(),
        );
    }

    let google = GoogleVertexProvider::new();
    for model in [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro",
        "gemini-3.1-flash",
        "gemini-3.1-flash-lite",
        "gemini-3-pro",
        "gemini-3-flash",
    ] {
        assert_same_pricing(
            get_reference_pricing(model).unwrap(),
            google.get_model_pricing(model).unwrap(),
        );
    }
}

#[test]
fn unified_properties_can_return_capabilities_and_pricing() {
    let props = get_reference_model_properties("llama3.1:8b").unwrap();
    assert!(props.capabilities.unwrap().structured_output);
    assert_eq!(props.pricing.unwrap().input_price_per_1m, 0.10);
}

#[test]
fn opus_5_properties_match_anthropic_model_facts() {
    let props = get_reference_model_properties("claude-opus-5").unwrap();
    assert_eq!(props.capability_pattern, Some("claude-opus-5"));
    assert_eq!(props.pricing_pattern, Some("claude-opus-5"));

    let capabilities = props.capabilities.unwrap();
    assert!(capabilities.vision);
    assert!(!capabilities.video);
    assert!(!capabilities.structured_output);
    assert_eq!(capabilities.max_input_tokens, 1_000_000);

    let pricing = props.pricing.unwrap();
    assert_eq!(pricing.input_price_per_1m, 5.0);
    assert_eq!(pricing.output_price_per_1m, 25.0);
    assert_eq!(pricing.cache_write_price_per_1m, 6.25);
    assert_eq!(pricing.cache_read_price_per_1m, 0.50);
}

#[test]
fn pricing_only_entries_do_not_imply_capabilities() {
    let props = get_reference_model_properties("qwen-plus-latest").unwrap();
    assert!(props.pricing.is_some());
    assert_eq!(props.pricing_pattern, Some("qwen-plus"));
    assert_eq!(props.capability_pattern, None);
    assert!(get_reference_capabilities("qwen-plus-latest").is_none());
}

#[test]
fn unified_properties_merge_independent_best_matches() {
    // Capabilities come from the specific variant, pricing from the family entry.
    let props = get_reference_model_properties("phi-4-multimodal").unwrap();
    assert_eq!(props.capability_pattern, Some("phi-4-multimodal"));
    assert_eq!(props.pricing_pattern, Some("phi-4"));
    assert_eq!(
        props.capabilities.unwrap().max_input_tokens,
        get_reference_capabilities("phi-4-multimodal")
            .unwrap()
            .max_input_tokens
    );
    assert_eq!(
        props.pricing.unwrap().input_price_per_1m,
        get_reference_pricing("phi-4-multimodal")
            .unwrap()
            .input_price_per_1m
    );
}

#[test]
fn proxy_policy_uses_known_structured_output_and_keeps_unknowns_optimistic() {
    assert!(proxy_route_enforces_response_schema("deepseek-v4-pro"));
    assert!(!proxy_route_enforces_response_schema("mistral-7b"));
    assert!(proxy_route_enforces_response_schema(
        "unknown/provider-model"
    ));
}

#[test]
fn every_capability_entry_is_reachable() {
    for entry in REFERENCE_MODELS {
        if let Some(expected) = entry.capabilities {
            let actual = get_reference_capabilities(entry.pattern)
                .unwrap_or_else(|| panic!("missing capabilities for {}", entry.pattern));
            assert_same_capabilities(actual, expected);

            let props = get_reference_model_properties(entry.pattern)
                .unwrap_or_else(|| panic!("missing properties for {}", entry.pattern));
            assert_eq!(props.capability_pattern, Some(entry.pattern));
            assert_same_capabilities(props.capabilities.unwrap(), expected);
        }
    }
}

#[test]
fn every_pricing_entry_is_reachable() {
    for entry in REFERENCE_MODELS {
        if let Some(expected) = entry.pricing {
            let actual = get_reference_pricing(entry.pattern)
                .unwrap_or_else(|| panic!("missing pricing for {}", entry.pattern));
            assert_same_pricing(actual, expected);

            let props = get_reference_model_properties(entry.pattern)
                .unwrap_or_else(|| panic!("missing properties for {}", entry.pattern));
            assert_eq!(props.pricing_pattern, Some(entry.pattern));
            assert_same_pricing(props.pricing.unwrap(), expected);
        }
    }
}
#[test]
fn august_2026_additions_resolve() {
    // Seed 2.1 family (ByteDance, Aug 2026)
    let p = get_reference_pricing("seed-2-1-turbo").unwrap();
    assert_eq!(p.input_price_per_1m, 0.50);
    assert_eq!(p.output_price_per_1m, 2.50);
    let caps = get_reference_capabilities("seed-2-1-turbo").unwrap();
    assert!(caps.vision);
    assert_eq!(caps.max_input_tokens, 262_144);

    let p = get_reference_pricing("seed-2-1-pro").unwrap();
    assert_eq!(p.input_price_per_1m, 0.85);
    assert_eq!(p.cache_read_price_per_1m, 0.17);

    // Qwen3.8-Flash production API (Aug 2026)
    let p = get_reference_pricing("qwen3.8-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.113);
    assert_eq!(p.output_price_per_1m, 0.382);
    assert_eq!(p.cache_read_price_per_1m, 0.0226);
    let caps = get_reference_capabilities("qwen3.8-flash").unwrap();
    assert!(caps.vision);
    assert!(caps.structured_output);
    assert_eq!(caps.max_input_tokens, 1_000_000);

    // Qwen3.8-27B open weights (Aug 2026)
    let p = get_reference_pricing("Qwen/Qwen3.8-27B").unwrap();
    assert_eq!(p.input_price_per_1m, 0.35);
    assert_eq!(p.output_price_per_1m, 2.75);
    let caps = get_reference_capabilities("qwen/qwen3.8-27b").unwrap();
    assert!(caps.vision);
    assert!(caps.video);
    assert_eq!(caps.max_input_tokens, 262_144);

    // Meta Muse family (Aug 2026)
    let p = get_reference_pricing("meta/muse-spark-1.2").unwrap();
    assert_eq!(p.input_price_per_1m, 1.25);
    assert_eq!(p.cache_read_price_per_1m, 0.15);
    let p = get_reference_pricing("meta-models/Muse-Glimmer-30B").unwrap();
    assert_eq!(p.input_price_per_1m, 0.30);
    assert_eq!(p.output_price_per_1m, 1.10);
    assert!(!proxy_route_enforces_response_schema("meta/muse-spark-1.2"));
}

/// Bedrock's Nova family resolves pricing and capabilities through the
/// reference table. Before these entries the whole family fell to the
/// 32_768 context default and unpriced usage, and the provider-level
/// `contains("nova")` vision shortcut claimed vision for text-only Micro.
#[test]
fn nova_family_resolves_pricing_and_capabilities() {
    // (Bedrock model ID, input, output, cache_read per 1M — us-east-1 rates)
    for (model, input, output, cache_read) in [
        ("amazon.nova-2-lite-v1:0", 0.30, 2.50, 0.075),
        ("global.amazon.nova-2-lite-v1:0", 0.30, 2.50, 0.075),
        // US geo cross-region routing bills 10% above the global tier.
        ("us.amazon.nova-2-lite-v1:0", 0.33, 2.75, 0.0825),
        ("amazon.nova-premier-v1:0", 2.50, 12.50, 0.625),
        ("amazon.nova-pro-v1:0", 0.80, 3.20, 0.20),
        ("amazon.nova-lite-v1:0", 0.06, 0.24, 0.015),
        ("amazon.nova-micro-v1:0", 0.035, 0.14, 0.00875),
    ] {
        let pricing = get_reference_pricing(model)
            .unwrap_or_else(|| panic!("{model} must resolve to reference pricing"));
        assert_eq!(pricing.input_price_per_1m, input);
        assert_eq!(pricing.output_price_per_1m, output);
        assert_eq!(pricing.cache_read_price_per_1m, cache_read);
        // AWS charges nothing for Nova cache writes.
        assert_eq!(pricing.cache_write_price_per_1m, 0.0);
    }

    let micro = get_reference_capabilities("amazon.nova-micro-v1:0")
        .expect("nova-micro must resolve to reference capabilities");
    assert!(!micro.vision);
    assert!(!micro.structured_output);
    assert_eq!(micro.max_input_tokens, 128_000);

    let two_lite = get_reference_capabilities("amazon.nova-2-lite-v1:0")
        .expect("nova-2-lite must resolve to reference capabilities");
    assert!(two_lite.vision);
    assert!(!two_lite.structured_output);
    assert_eq!(two_lite.max_input_tokens, 1_000_000);
}
