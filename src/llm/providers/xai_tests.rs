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
use crate::llm::tool_calls::GenericToolCall;
use crate::llm::types::{ImageAttachment, ImageData, SourceType};
use std::path::PathBuf;

#[test]
fn supports_current_models_aliases_and_redirects() {
    let provider = XaiProvider::new();
    for model in [
        "grok-4.5",
        "grok-4.5-latest",
        "grok-build-latest",
        "grok-build-0.1",
        "grok-code-fast-1",
        "grok-4.3",
        "grok-latest",
        "grok-4.20-0309-reasoning",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-multi-agent-0309",
        "grok-4.20-reasoning-latest",
        "grok-4.20",
        "grok-4.20-reasoning",
        "grok-4.20-0309",
        "grok-4.20-beta-0309-reasoning",
        "grok-4.20-beta",
        "grok-4.20-beta-0309",
        "grok-4.20-beta-latest",
        "grok-4.20-beta-latest-reasoning",
        "grok-4.20-beta-reasoning",
        "grok-4.20-experimental-beta-0304-reasoning",
        "grok-4.20-experimental-beta-0304",
        "grok-4.20-experimental-beta-reasoning-latest",
        "grok-4.20-experimental-beta-latest",
        "grok-4.20-reasoning-gv2",
        "grok-4.20-non-reasoning",
        "grok-4.20-non-reasoning-latest",
        "grok-4.20-beta-non-reasoning",
        "grok-4.20-beta-latest-non-reasoning",
        "grok-4.20-experimental-beta-0304-non-reasoning",
        "grok-4.20-experimental-beta-non-reasoning-latest",
        "grok-4.20-beta-0309-non-reasoning",
        "grok-4.20-non-reasoning-gv2",
        "grok-4.20-multi-agent",
        "grok-4.20-multi-agent-latest",
        "grok-4.20-multi-agent-beta-latest",
        "grok-4.20-multi-agent-experimental-beta-0304",
        "grok-4.20-multi-agent-experimental-beta-latest",
        "grok-4.20-multi-agent-beta-0309",
        "grok-3",
    ] {
        assert!(provider.supports_model(model), "missing {model}");
    }
    assert!(!provider.supports_model("grok-imagine-image"));
    assert!(!provider.supports_model("grok-4.20-typo"));
    assert!(!provider.supports_model("grok-4"));
    assert!(!provider.supports_model("grok-unknown"));
}

#[test]
fn capabilities_and_pricing_follow_model_family() {
    let provider = XaiProvider::new();
    assert_eq!(provider.get_max_input_tokens("grok-4.5"), 500_000);
    assert_eq!(provider.get_max_input_tokens("grok-build-0.1"), 256_000);
    assert_eq!(
        provider.get_max_input_tokens("grok-4.20-0309-reasoning"),
        1_000_000
    );
    assert!(provider.supports_vision("grok-4.3"));
    assert!(provider.enforces_response_schema("grok-4.20-multi-agent-0309"));
    let pricing = provider.get_model_pricing("grok-4.5-latest").unwrap();
    assert_eq!(pricing.input_price_per_1m, 2.0);
    assert_eq!(pricing.cache_read_price_per_1m, 0.3);
    assert_eq!(pricing.output_price_per_1m, 6.0);
}

#[test]
fn long_context_pricing_applies_at_threshold() {
    assert_eq!(
        usage_pricing("grok-4.3", 199_999)
            .unwrap()
            .input_price_per_1m,
        1.25
    );
    let pricing = usage_pricing("grok-4.3", 200_000).unwrap();
    assert_eq!(pricing.input_price_per_1m, 2.5);
    assert_eq!(pricing.cache_read_price_per_1m, 0.4);
    assert_eq!(pricing.output_price_per_1m, 5.0);
    assert!((ticks_to_usd(37_756_000) - 0.0037756).abs() < f64::EPSILON);
}

#[test]
fn usage_splits_reasoning_out_of_output_without_double_counting() {
    let usage: XaiUsage = serde_json::from_value(json!({
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
        "input_tokens_details": {"cached_tokens": 20},
        "output_tokens_details": {"reasoning_tokens": 12},
        "cost_in_usd_ticks": 37_756_000
    }))
    .unwrap();
    let usage = normalize_usage("grok-4.5", usage, 9);
    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.cache_read_tokens, 20);
    // xAI bills reasoning inside output_tokens; reported apart from it.
    assert_eq!(usage.output_tokens, 18);
    assert_eq!(usage.reasoning_tokens, 12);
    assert_eq!(usage.total_tokens, 150);
    assert_eq!(
        usage.input_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
            + usage.output_tokens
            + usage.reasoning_tokens,
        usage.total_tokens
    );
    // Cost is billed on the raw output counter, reasoning included.
    assert_eq!(usage.cost, Some(0.0037756));
    assert_eq!(usage.request_time_ms, Some(9));
}

#[test]
fn reasoning_effort_is_only_sent_where_documented() {
    assert_eq!(
        reasoning_effort("grok-4.5", Some(ReasoningEffort::Max)),
        Some("high")
    );
    assert_eq!(
        reasoning_effort("grok-4.3", Some(ReasoningEffort::Low)),
        Some("low")
    );
    assert_eq!(
        reasoning_effort("grok-4.20-multi-agent-0309", Some(ReasoningEffort::Max)),
        Some("xhigh")
    );
    assert_eq!(
        reasoning_effort("grok-4.20-0309-reasoning", Some(ReasoningEffort::High)),
        None
    );
    assert_eq!(
        reasoning_effort("grok-build-0.1", Some(ReasoningEffort::High)),
        None
    );
}

#[test]
fn parses_reasoning_and_preserves_raw_items_on_tool_calls() {
    let output = vec![
        json!({"type":"reasoning","summary":[{"type":"summary_text","text":"Checked the inputs"}],"encrypted_content":"opaque"}),
        json!({"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"rust\"}"}),
    ];
    assert_eq!(
        reasoning_from_output(&output).as_deref(),
        Some("Checked the inputs")
    );
    let calls = tool_calls_from_output(&output);
    assert_eq!(calls[0].arguments, json!({"q":"rust"}));
    let meta = reasoning_meta(&output).unwrap();
    assert_eq!(meta[REASONING_META_KEY][0]["encrypted_content"], "opaque");
}

#[test]
fn rebased_history_replays_encrypted_reasoning_before_function_calls() {
    let generic = GenericToolCall {
        id: "call_1".to_string(),
        name: "lookup".to_string(),
        arguments: json!({"q":"rust"}),
        meta: Some(Map::from_iter([(
            REASONING_META_KEY.to_string(),
            json!([{"type":"reasoning","encrypted_content":"opaque","summary":[]}]),
        )])),
    };
    let mut assistant = Message::assistant("");
    assistant.tool_calls = Some(serde_json::to_value([generic]).unwrap());
    let input = messages_to_input(&[Message::user("find it"), assistant], None);
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[2]["type"], "function_call");
}

#[test]
fn xai_ids_gate_automatic_stateful_continuation() {
    let mut assistant = Message::assistant("");
    assistant.id = Some("xai_response:abc-123".to_string());
    let messages = vec![assistant, Message::tool("done", "call_1", "lookup")];
    let previous = resolve_previous_response_id(&messages, None);
    assert_eq!(previous.as_deref(), Some("abc-123"));
    let input = messages_to_input(&messages, previous.as_deref());
    assert_eq!(
        input,
        vec![json!({"type":"function_call_output","call_id":"call_1","output":"done"})]
    );

    let mut foreign = Message::assistant("");
    foreign.id = Some("resp_openai".to_string());
    assert_eq!(resolve_previous_response_id(&[foreign], None), None);
}

#[test]
fn request_always_asks_for_encrypted_reasoning() {
    let params =
        ChatCompletionParams::new(&[Message::user("hello")], "grok-4.5", 0.2, 0.9, 40, 100)
            .with_reasoning_effort(ReasoningEffort::High);
    let request = build_request(&params, None);
    assert_eq!(request["include"][0], "reasoning.encrypted_content");
    assert_eq!(request["reasoning"]["effort"], "high");
}

#[test]
fn vision_input_uses_responses_api_parts() {
    let message = Message::user("inspect").with_images(vec![ImageAttachment {
        data: ImageData::Base64("aGVsbG8=".to_string()),
        media_type: "image/png".to_string(),
        source_type: SourceType::File(PathBuf::from("test.png")),
        dimensions: None,
        size_bytes: None,
    }]);
    let input = messages_to_input(&[message], None);
    assert_eq!(input[0]["content"][0]["type"], "input_text");
    assert_eq!(input[0]["content"][1]["type"], "input_image");
    assert_eq!(
        input[0]["content"][1]["image_url"],
        "data:image/png;base64,aGVsbG8="
    );
}
