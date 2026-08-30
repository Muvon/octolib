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
    let zen = OpenCodeZenProvider::new();
    assert!(zen.supports_model("claude-opus-5"));
    assert!(zen.supports_model("gpt-5.5"));
    assert!(zen.supports_model("deepseek-v4-flash-free"));
    assert!(!zen.supports_model(""));

    let go = OpenCodeGoProvider::new();
    assert!(go.supports_model("kimi-k2.7-code"));
    assert!(go.supports_model("glm-5.2"));
    assert!(!go.supports_model(""));
}

#[test]
fn test_sampling_support_mirrors_upstream_restrictions() {
    // Kimi K2.7/K3 pin temperature=1 and top_p=0.95 upstream (verified live on Go)
    assert!(!sampling_support("kimi-k2.7-code").temperature);
    assert!(!sampling_support("kimi-k2.7-code").top_p);
    assert!(!sampling_support("kimi-k3").temperature);
    assert!(!sampling_support("Kimi-K3").temperature);

    // GPT-5 family reasoning models reject non-default temperature/top_p
    assert!(!sampling_support("gpt-5.6-luna").temperature);
    assert!(!sampling_support("gpt-5.5").top_p);

    // Claude Fable/Opus 5 reject all sampling; Sonnet 4.5 rejects top_p only
    assert_eq!(sampling_support("claude-fable-5"), SamplingSupport::NONE);
    assert!(!sampling_support("claude-opus-5").temperature);
    assert!(sampling_support("claude-sonnet-4-5").temperature);
    assert!(!sampling_support("claude-sonnet-4-5").top_p);

    // Other families pass through untouched
    assert_eq!(sampling_support("glm-5.2"), SamplingSupport::ALL);
    assert_eq!(sampling_support("minimax-m3"), SamplingSupport::ALL);
    assert_eq!(sampling_support("deepseek-v4-pro"), SamplingSupport::ALL);
}

#[test]
fn test_adjust_reasoning_effort_kimi_rules() {
    // K3: five levels floored onto Moonshot's low/high/max tiers
    assert_eq!(
        adjust_reasoning_effort("kimi-k3", Some(ReasoningEffort::Low)),
        Some(ReasoningEffort::Low)
    );
    assert_eq!(
        adjust_reasoning_effort("kimi-k3", Some(ReasoningEffort::Medium)),
        Some(ReasoningEffort::Low)
    );
    assert_eq!(
        adjust_reasoning_effort("kimi-k3", Some(ReasoningEffort::High)),
        Some(ReasoningEffort::High)
    );
    assert_eq!(
        adjust_reasoning_effort("kimi-k3", Some(ReasoningEffort::XHigh)),
        Some(ReasoningEffort::High)
    );
    assert_eq!(
        adjust_reasoning_effort("kimi-k3", Some(ReasoningEffort::Max)),
        Some(ReasoningEffort::Max)
    );
    // Unset stays unset — K3 applies its own default ("max")
    assert_eq!(adjust_reasoning_effort("kimi-k3", None), None);

    // Other Kimi models don't support the field at all
    assert_eq!(
        adjust_reasoning_effort("kimi-k2.7-code", Some(ReasoningEffort::High)),
        None
    );
    assert_eq!(
        adjust_reasoning_effort("kimi-k2.6", Some(ReasoningEffort::Max)),
        None
    );
    assert_eq!(
        adjust_reasoning_effort("Kimi-K2.5", Some(ReasoningEffort::Low)),
        None
    );

    // Non-Kimi families pass through untouched
    assert_eq!(
        adjust_reasoning_effort("gpt-5.5", Some(ReasoningEffort::Medium)),
        Some(ReasoningEffort::Medium)
    );
    assert_eq!(
        adjust_reasoning_effort("glm-5.2", Some(ReasoningEffort::Max)),
        Some(ReasoningEffort::Max)
    );
    assert_eq!(adjust_reasoning_effort("claude-opus-5", None), None);
}

#[test]
fn test_go_zero_response_cost_falls_back_to_reference_pricing() {
    let usage = TokenUsage {
        input_tokens: 1_000,
        cache_read_tokens: 500,
        cache_write_tokens: 0,
        output_tokens: 100,
        reasoning_tokens: 50,
        total_tokens: 1_650,
        cost: Some(0.0),
        request_time_ms: None,
    };

    let cost = resolve_opencode_cost("opencode-go", "kimi-k3", None, &usage).unwrap();
    let expected =
        1_000.0 / 1_000_000.0 * 3.00 + 500.0 / 1_000_000.0 * 0.30 + 150.0 / 1_000_000.0 * 15.00;
    assert!((cost - expected).abs() < 1e-12);
    assert!(cost > 0.0);
}

#[test]
fn test_opencode_reported_cost_precedence() {
    let usage = TokenUsage {
        input_tokens: 1_000,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens: 100,
        reasoning_tokens: 0,
        total_tokens: 1_100,
        cost: Some(0.25),
        request_time_ms: None,
    };

    assert_eq!(
        resolve_opencode_cost("opencode-go", "kimi-k3", None, &usage),
        Some(0.25)
    );
    assert_eq!(
        resolve_opencode_cost("opencode-zen", "kimi-k3", Some(0.125), &usage),
        Some(0.125)
    );
}

#[test]
fn test_empty_kimi_assistant_placeholders_are_removed() {
    let mut messages = vec![
        Message::tool("Memory stored", "memorize_9", "memorize"),
        Message::assistant(""),
        Message::user("proceed"),
    ];

    remove_empty_kimi_assistant_messages("kimi-k3", &mut messages);

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "tool");
    assert_eq!(messages[1].role, "user");
}

#[test]
fn test_kimi_assistant_state_is_never_removed() {
    let mut tool_call = Message::assistant("");
    tool_call.tool_calls = Some(serde_json::json!([{
        "id": "memorize_9",
        "name": "memorize",
        "arguments": {}
    }]));
    let mut thinking = Message::assistant("");
    thinking.thinking = Some(crate::llm::types::ThinkingBlock {
        content: "reasoning".to_string(),
        tokens: 1,
    });
    let mut messages = vec![Message::assistant("answer"), tool_call, thinking];

    remove_empty_kimi_assistant_messages("kimi-k3", &mut messages);

    assert_eq!(messages.len(), 3);
}

#[test]
fn test_empty_non_kimi_assistant_is_untouched() {
    let mut messages = vec![Message::assistant("")];

    remove_empty_kimi_assistant_messages("gpt-5.5", &mut messages);

    assert_eq!(messages.len(), 1);
}

#[test]
fn test_default_capabilities() {
    let zen = OpenCodeZenProvider::new();
    assert_eq!(zen.name(), "opencode-zen");
    assert!(!zen.supports_caching("any-model"));

    let go = OpenCodeGoProvider::new();
    assert_eq!(go.name(), "opencode-go");
    assert!(go.supports_caching("any-model"));
    assert!(go.supports_structured_output("any-model"));
}
