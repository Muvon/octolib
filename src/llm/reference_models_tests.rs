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

/// Ollama names V4.1 Flash by its model card, not by DeepSeek's `deepseek-flash`
/// route, and that spelling also contains the generic `deepseek-v4` pattern — a
/// miss here bills V4 Flash's rate and reports the model as blind.
#[test]
fn deepseek_v4_1_flash_resolves_by_its_model_card_name() {
    for model in ["deepseek-v4.1-flash:cloud", "deepseek-v4.1-flash"] {
        let caps = get_reference_capabilities(model)
            .expect("deepseek-v4.1-flash must resolve to a reference entry");
        assert!(caps.vision);
        assert_eq!(caps.max_input_tokens, 1_000_000);
        let pricing = get_reference_pricing(model)
            .expect("deepseek-v4.1-flash must resolve to reference pricing");
        assert_eq!(pricing.input_price_per_1m, 0.3);
        assert_eq!(pricing.output_price_per_1m, 1.2);
        assert_eq!(pricing.cache_read_price_per_1m, 0.006);
    }
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
        "gemini-3.1-flash-lite",
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

/// Models added to live provider catalogs in Aug-Sep 2026. Before these rows
/// the aggregator routes served them with no pricing and the 32K context
/// default, and `muse-spark-1.2-contributor` inherited the 12x-higher base rate.
#[test]
fn september_2026_additions_resolve() {
    // Contributor tiers must win over their base version.
    let p = get_reference_pricing("meta/muse-spark-1.2-contributor").unwrap();
    assert_eq!(p.input_price_per_1m, 0.10);
    let p = get_reference_pricing("meta/muse-spark-1.3").unwrap();
    assert_eq!(p.input_price_per_1m, 1.25);
    assert_eq!(p.cache_read_price_per_1m, 0.15);

    // Short aggregator ID resolves without stealing the NVIDIA-hosted route.
    let p = get_reference_pricing("nvidia/nemotron-3.5-lightning").unwrap();
    assert_eq!(p.input_price_per_1m, 0.08);
    let p = get_reference_pricing("nvidia/nemotron-3.5-lightning-30b-a3b").unwrap();
    assert_eq!(p.input_price_per_1m, 0.05);

    // Qwen 3.6 35B A3B: free on Hetzner, priced on OpenRouter.
    let caps = get_reference_capabilities("Qwen/Qwen3.6-35B-A3B-FP8").unwrap();
    assert!(caps.vision);
    assert!(caps.video);
    assert_eq!(caps.max_input_tokens, 262_144);
    let p = get_reference_pricing("qwen/qwen3.6-35b-a3b").unwrap();
    assert_eq!(p.output_price_per_1m, 0.90);

    // New families that previously fell through to the unpriced default.
    for (model, input, output) in [
        ("inception/mercury-2.5", 0.04, 0.15),
        ("inception/mercury-2", 0.25, 0.75),
        ("tencent/hy4-preview", 0.834, 2.501),
        ("ibm-granite/granite-4.2-8b", 0.06, 0.25),
        ("inclusionai/ling-3.0-flash", 0.021, 0.063),
        ("inclusionai/ling-3.0-flash-fin", 0.06, 0.18),
        ("sakana/sakana-namazu", 0.95, 4.00),
        ("upstage/solar-pro4", 0.03, 0.12),
        ("poolside/laguna-s-2.1", 0.09, 0.18),
        ("poolside/laguna-xs-2.1", 0.06, 0.12),
        ("meituan/longcat-2.0", 0.30, 1.20),
        ("kwaipilot/kat-coder-pro-v2.5", 0.74, 2.96),
        ("xiaomi/mimo-v2.5", 0.14, 0.28),
        ("xiaomi/mimo-v2.5-pro", 0.435, 0.87),
        ("qwen/qwen3.8-2.4t-a95b", 2.00, 6.00),
        ("nvidia/nemotron-3-super-120b-a12b", 0.085, 0.40),
    ] {
        let pricing = get_reference_pricing(model)
            .unwrap_or_else(|| panic!("{model} must resolve to reference pricing"));
        assert_eq!(pricing.input_price_per_1m, input);
        assert_eq!(pricing.output_price_per_1m, output);
    }
}

/// Dotted and dashed version spellings sanitize identically, so both must land
/// on the versioned row rather than the family fallback.
#[test]
fn dotted_versions_resolve_to_the_dashed_entry() {
    for (model, pattern) in [
        ("anthropic/claude-opus-4.8", "claude-opus-4-8"),
        ("anthropic/claude-haiku-4.5", "claude-haiku-4-5"),
        ("bytedance-seed/seed-2.0-code", "seed-2-0-code"),
        ("o3", "o3"),
        ("openai/o3-2025-04-16", "o3"),
    ] {
        let props = get_reference_model_properties(model)
            .unwrap_or_else(|| panic!("{model} must resolve to a reference entry"));
        assert_eq!(props.pricing_pattern, Some(pattern), "{model}");
        assert_eq!(props.capability_pattern, Some(pattern), "{model}");
    }
}

/// Bare patterns such as `o3` and `o1` only match on separator boundaries,
/// so they no longer fire inside `nano-3-30b` or `unslopnemo-12b`.
#[test]
fn bare_patterns_do_not_match_inside_unrelated_names() {
    for (model, bare) in [
        ("nvidia/nemotron-3-nano-30b-a3b", "o3"),
        ("upstage/solar-pro-3", "o3"),
        ("thedrummer/unslopnemo-12b", "o1"),
        ("sao10k/l3.3-euryale-70b", "o1"),
    ] {
        let props = get_reference_model_properties(model);
        assert_ne!(props.and_then(|p| p.pricing_pattern), Some(bare), "{model}");
        assert_ne!(
            props.and_then(|p| p.capability_pattern),
            Some(bare),
            "{model}"
        );
    }
}

/// Families confirmed on the live OpenRouter, Mistral and Bedrock catalogs in
/// late Sep 2026. Bedrock spells several IDs differently from the aggregator
/// routes (`nemotron-super-3-120b`, `ministral-3-14b-instruct`), so those
/// resolve through their own rows.
#[test]
fn late_september_2026_additions_resolve() {
    for (model, input, output) in [
        ("amazon.nova-2-pro-v1:0", 1.25, 10.00),
        ("amazon.nova-2-omni-v1:0", 0.30, 2.50),
        ("amazon.nova-2-sonic-v1:0", 0.33, 2.75),
        ("qwen.qwen3-next-80b-a3b", 0.15, 1.20),
        ("qwen.qwen3-vl-235b-a22b-instruct", 0.53, 2.66),
        ("qwen.qwen3-coder-30b-a3b-v1:0", 0.15, 0.60),
        ("mistral.devstral-2-123b", 0.40, 2.00),
        ("mistral.magistral-small-2509", 0.50, 1.50),
        ("mistral.ministral-3-14b-instruct", 0.20, 0.20),
        ("mistral.ministral-3-8b-instruct", 0.15, 0.15),
        ("mistral.ministral-3-3b-instruct", 0.10, 0.10),
        ("mistralai/ministral-14b-2512", 0.20, 0.20),
        ("ministral-8b-2512", 0.15, 0.15),
        ("ministral-3b-2512", 0.10, 0.10),
        ("mistral-small-2603", 0.15, 0.60),
        ("mistral-small-4", 0.15, 0.60),
        ("nvidia.nemotron-super-3-120b", 0.15, 0.65),
        ("openai.gpt-oss-safeguard-120b", 0.15, 0.60),
        ("openai.gpt-oss-safeguard-20b", 0.07, 0.20),
        ("writer.palmyra-vision-7b", 0.15, 0.60),
        ("ai21.jamba-1-5-large-v1:0", 2.00, 8.00),
        ("nvidia/nemotron-3.5-content-safety", 0.20, 0.20),
        ("xiaomi/mimo-v2.6-pro-ultraspeed", 4.35, 8.70),
        ("xiaomi/mimo-v2.6-pro", 0.435, 0.87),
        ("xiaomi/mimo-v2.6-flash", 0.14, 0.28),
        ("sakana/fugu-max", 2.00, 6.00),
        ("sakana/fugu-ultra", 5.00, 30.00),
        ("sakana/fugu-ultra-v2", 5.00, 30.00),
        ("tencent/hy3", 0.132, 0.528),
        ("tencent/hy3-preview", 0.18, 0.60),
        ("tencent/hy-mt2-1.8b", 0.044, 0.177),
        ("tencent/hy-mt2-7b", 0.074, 0.295),
        ("tencent/hy-mt2-30b-a3b", 0.074, 0.295),
        ("nex-agi/nex-n2.5-pro", 0.075, 0.25),
        ("nex-agi/nex-n2.5-mini", 0.025, 0.10),
        ("stepfun/step-3.7-flash", 0.20, 1.15),
        ("aion-labs/aion-3.0", 3.00, 6.00),
        ("aion-labs/aion-3.0-mini", 0.70, 1.40),
        ("unbiased/pareto", 2.50, 7.50),
        ("prism-ml/ternary-bonsai-2-27b", 0.075, 0.50),
        ("inference-net/schematron-v2-small", 0.05, 0.23),
        ("inference-net/schematron-v2-turbo", 0.03, 0.15),
        ("perceptron/perceptron-mk1", 0.15, 1.50),
        ("nvidia/nemotron-3-nano-30b-a3b", 0.05, 0.20),
        ("stepfun/step-3.5-flash", 0.10, 0.30),
        ("aion-labs/aion-2.0", 0.80, 1.60),
        ("mistralai/devstral-2512", 0.40, 2.00),
    ] {
        let pricing = get_reference_pricing(model)
            .unwrap_or_else(|| panic!("{model} must resolve to reference pricing"));
        assert_eq!(pricing.input_price_per_1m, input, "{model}");
        assert_eq!(pricing.output_price_per_1m, output, "{model}");
    }

    // Preview Nova 2 models publish a rate but no context window.
    assert!(get_reference_capabilities("amazon.nova-2-pro-v1:0").is_none());
    assert!(get_reference_capabilities("amazon.nova-2-omni-v1:0").is_none());
    assert_eq!(
        get_reference_capabilities("amazon.nova-2-sonic-v1:0")
            .unwrap()
            .max_input_tokens,
        1_000_000
    );

    // Nemotron 3 Nano Omni only has a free route, so it carries capabilities
    // but no rate.
    let omni = get_reference_model_properties("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
        .unwrap();
    assert_eq!(
        omni.capability_pattern,
        Some("nemotron-3-nano-omni-30b-a3b")
    );
    assert!(omni.capabilities.unwrap().video);
    assert!(omni.pricing.is_none());

    // Bedrock serves Ministral 3 at 128K while the Mistral API serves 256K.
    assert_eq!(
        get_reference_capabilities("mistral.ministral-3-14b-instruct")
            .unwrap()
            .max_input_tokens,
        131_072
    );
    assert_eq!(
        get_reference_capabilities("ministral-14b-2512")
            .unwrap()
            .max_input_tokens,
        262_144
    );
    assert_eq!(
        get_reference_pricing("ministral-14b-2512")
            .unwrap()
            .cache_read_price_per_1m,
        0.02
    );

    let mimo = get_reference_capabilities("xiaomi/mimo-v2.6-pro").unwrap();
    assert!(mimo.vision);
    assert!(mimo.video);
    assert!(proxy_route_enforces_response_schema("tencent/hy3"));
    assert!(!proxy_route_enforces_response_schema("tencent/hy3-preview"));
    let palmyra = get_reference_capabilities("writer.palmyra-vision-7b").unwrap();
    assert!(palmyra.vision);
    assert_eq!(palmyra.max_input_tokens, 4_096);

    // Existing rows keep their own routes.
    let p = get_reference_pricing("nvidia/nemotron-3-super-120b-a12b").unwrap();
    assert_eq!(p.input_price_per_1m, 0.085);
    let p = get_reference_pricing("mistralai/mistral-small-3.2-24b-instruct").unwrap();
    assert_eq!(p.input_price_per_1m, 0.10);
}
