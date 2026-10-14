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

fn catalog_fixture() -> Vec<CatalogModel> {
    serde_json::from_value::<CatalogResponse>(serde_json::json!({
        "data": [
            {
                "id": "google/gemini-3.5-flash-lite",
                "reasoning": {
                    "mandatory": true,
                    "default_enabled": true,
                    "supported_efforts": ["high", "medium", "low", "minimal"],
                    "default_effort": "minimal"
                }
            },
            {
                "id": "google/gemini-3.1-flash-lite",
                "reasoning": {
                    "mandatory": false,
                    "default_enabled": true,
                    "supported_efforts": ["high", "medium", "low", "minimal"],
                    "default_effort": "minimal"
                }
            },
            {
                "id": "deepseek/deepseek-v4.1-flash",
                "reasoning": {
                    "mandatory": false,
                    "default_enabled": true,
                    "supported_efforts": ["max", "high", "low"],
                    "default_effort": "high"
                }
            },
            {
                "id": "example/no-reasoning",
                "reasoning": null
            },
            {
                "id": "liquid/lfm-2.5-24b-a2b",
                "reasoning": {
                    "mandatory": true
                }
            },
            {
                "id": "example/optional-reasoning-without-levels",
                "reasoning": {
                    "mandatory": false
                }
            },
            {
                "id": "example/malformed-reasoning",
                "reasoning": {
                    "mandatory": "unknown",
                    "supported_efforts": "high"
                }
            }
        ]
    }))
    .unwrap()
    .data
}

#[test]
fn catalog_effort_none_uses_mandatory_minimum_or_disables_reasoning() {
    let catalog = catalog_fixture();
    let mandatory = catalog_model(&catalog, "google/gemini-3.5-flash-lite").unwrap();
    assert_eq!(
        select_effort(mandatory, ReasoningEffort::None),
        Some(ReasoningSelection::Effort("minimal"))
    );

    let optional = catalog_model(&catalog, "google/gemini-3.1-flash-lite").unwrap();
    assert_eq!(
        select_effort(optional, ReasoningEffort::None),
        Some(ReasoningSelection::Disabled)
    );
}

#[test]
fn catalog_without_effort_levels_omits_or_disables_reasoning() {
    let catalog = catalog_fixture();
    assert_eq!(catalog.len(), 7);
    assert!(catalog_model(&catalog, "example/malformed-reasoning")
        .unwrap()
        .reasoning
        .is_none());

    let mut mandatory = serde_json::json!({});
    apply_reasoning_effort(
        &mut mandatory,
        Some(&catalog),
        "liquid/lfm-2.5-24b-a2b",
        ReasoningEffort::None,
    );
    assert_eq!(mandatory, serde_json::json!({}));

    let mut optional = serde_json::json!({});
    apply_reasoning_effort(
        &mut optional,
        Some(&catalog),
        "example/optional-reasoning-without-levels",
        ReasoningEffort::None,
    );
    assert_eq!(
        optional,
        serde_json::json!({"reasoning": {"enabled": false}})
    );

    let mut requested = serde_json::json!({});
    apply_reasoning_effort(
        &mut requested,
        Some(&catalog),
        "liquid/lfm-2.5-24b-a2b",
        ReasoningEffort::XHigh,
    );
    assert_eq!(requested, serde_json::json!({"reasoning_effort": "high"}));
}

#[test]
fn catalog_effort_maps_to_the_nearest_supported_level_ties_up() {
    let catalog = catalog_fixture();
    // low/high/max: the middle request lands on the middle level, not the lowest.
    let deepseek = catalog_model(&catalog, "deepseek/deepseek-v4.1-flash").unwrap();
    assert_eq!(
        select_effort(deepseek, ReasoningEffort::Medium),
        Some(ReasoningSelection::Effort("high"))
    );
    assert_eq!(
        select_effort(deepseek, ReasoningEffort::XHigh),
        Some(ReasoningSelection::Effort("max"))
    );
    // minimal..high: nothing above "high", so the top requests stay at it.
    let gemini = catalog_model(&catalog, "google/gemini-3.5-flash-lite").unwrap();
    assert_eq!(
        select_effort(gemini, ReasoningEffort::Max),
        Some(ReasoningSelection::Effort("high"))
    );
}

#[test]
fn catalog_lookup_strips_openrouter_routing_variant() {
    let catalog = catalog_fixture();
    assert_eq!(
        catalog_reasoning_selection(
            &catalog,
            "google/gemini-3.5-flash-lite:nitro",
            ReasoningEffort::None
        ),
        Some(ReasoningSelection::Effort("minimal"))
    );
}

#[test]
fn absent_catalog_model_keeps_legacy_reasoning_mapping() {
    let catalog = catalog_fixture();
    let mut disabled = serde_json::json!({});
    apply_reasoning_effort(
        &mut disabled,
        Some(&catalog),
        "unknown/model",
        ReasoningEffort::None,
    );
    assert_eq!(
        disabled,
        serde_json::json!({"reasoning": {"enabled": false}})
    );

    let mut capped = serde_json::json!({});
    apply_reasoning_effort(
        &mut capped,
        Some(&catalog),
        "unknown/model",
        ReasoningEffort::Max,
    );
    assert_eq!(capped, serde_json::json!({"reasoning_effort": "high"}));
}

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

#[test]
fn tool_images_follow_tool_results_without_losing_call_association() {
    let messages = [shared::tests::tool_image_message("")];
    let value = serde_json::to_value(convert_messages(&messages).unwrap()).unwrap();
    assert_eq!(value[0]["role"], "tool");
    assert_eq!(value[0]["tool_call_id"], "call_image");
    assert_eq!(value[0]["content"], "");
    assert_eq!(value[1]["role"], "user");
    assert_eq!(
        value[1]["content"][1]["image_url"]["url"],
        "data:image/png;base64,aW1hZ2U="
    );
}
