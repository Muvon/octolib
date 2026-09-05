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
fn test_tool_call_and_result_round_trip() {
    use crate::llm::tool_calls::GenericToolCall;
    use crate::llm::types::Message;

    let tool_calls = serde_json::to_value(vec![GenericToolCall {
        id: "call_weather".to_string(),
        name: "get_weather".to_string(),
        arguments: serde_json::json!({"city": "Bangkok"}),
        meta: None,
    }])
    .unwrap();

    let assistant = Message {
        tool_calls: Some(tool_calls),
        ..Message::assistant("")
    };
    let tool_result = Message::tool(
        r#"{"temperature_c":31,"condition":"sunny"}"#,
        "call_weather",
        "get_weather",
    );

    let converted = convert_messages(&[assistant, tool_result]);
    let serialized = serde_json::to_value(converted).unwrap();

    assert_eq!(serialized[0]["tool_calls"][0]["id"], "call_weather");
    assert_eq!(
        serialized[0]["tool_calls"][0]["function"]["arguments"],
        r#"{"city":"Bangkok"}"#
    );
    assert_eq!(serialized[1]["role"], "tool");
    assert_eq!(serialized[1]["tool_call_id"], "call_weather");
    assert_eq!(
        serialized[1]["content"],
        r#"{"temperature_c":31,"condition":"sunny"}"#
    );
    assert!(serialized[1].get("tool_calls_id").is_none());
}

#[test]
fn test_model_support() {
    let provider = ZaiProvider::new();
    assert!(provider.supports_model("glm-5.1"));
    assert!(provider.supports_model("glm-5.3"));
    assert!(provider.supports_model("glm-5.3-flash"));
    assert!(provider.supports_model("glm-5.1-turbo"));
    assert!(provider.supports_model("glm-5"));
    assert!(provider.supports_model("glm-5-turbo"));
    assert!(provider.supports_model("glm-5v-turbo"));
    assert!(provider.supports_model("glm-4.7"));
    assert!(provider.supports_model("glm-4.7-flash"));
    assert!(provider.supports_model("glm-4.6"));
    assert!(provider.supports_model("glm-4.5"));
    // Near-miss rejections: invalid models that contain no pricing entry as a substring
    assert!(!provider.supports_model("glm5.3-flash"));
    assert!(!provider.supports_model("glmm-5.3-flash"));
    // Substring convention: a variant of a known family matches its base entry
    // (same mechanism that bills glm-4.7-flashx via its own specific-first entry),
    // so a hypothetical glm-5.3-flashx is accepted at flash pricing until z.ai
    // ships it with distinct prices and it gets its own entry.
    assert!(provider.supports_model("glm-5.3-flashx"));
    // Deprecated models
    assert!(!provider.supports_model("glm-4"));
    assert!(!provider.supports_model("glm-4-flash"));
    assert!(!provider.supports_model("gpt-4"));
}

#[test]
fn test_model_support_case_insensitive() {
    let provider = ZaiProvider::new();
    // Test uppercase
    assert!(provider.supports_model("GLM-5-Turbo"));
    assert!(provider.supports_model("GLM-5.1"));
    // Test mixed case
    assert!(provider.supports_model("Glm-4.7"));
    assert!(provider.supports_model("GLM-4.6"));
}

#[test]
fn test_cost_calculation() {
    // Test GLM-5-Turbo: $1.20 input, $4.00 output
    let cost = calculate_cost("glm-5-turbo", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 5.20).abs() < 0.01); // 1.20 + 4.00

    // Test GLM-5.1: $1.40 input, $4.40 output
    let cost = calculate_cost("glm-5.1", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 5.80).abs() < 0.01); // 1.40 + 4.40

    // Test GLM-5.3: mirrors GLM-5.2 pricing ($1.40 input, $4.40 output)
    let cost = calculate_cost("glm-5.3", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 5.80).abs() < 0.01); // 1.40 + 4.40

    // Test GLM-4.5: $0.60 input, $2.20 output
    let cost = calculate_cost("glm-4.5", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 2.80).abs() < 0.01); // 0.60 + 2.20

    // Test GLM-4.7: $0.60 input, $2.20 output
    let cost = calculate_cost("glm-4.7", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 2.80).abs() < 0.01); // 0.60 + 2.20

    // Test GLM-4.7-flash: free model
    let cost = calculate_cost("glm-4.7-flash", 1_000_000, 0, 1_000_000);
    assert_eq!(cost.unwrap(), 0.0);
}

#[test]
fn glm_5_3_flash_promo_switches_to_list_price_at_documented_cutoff() {
    use std::time::{Duration, SystemTime};

    let before =
        SystemTime::UNIX_EPOCH + Duration::from_secs(GLM_5_3_FLASH_PROMO_END_UNIX_SECS - 1);
    let promo = calculate_cost_at("glm-5.3-flash", 1_000_000, 0, 1_000_000, before).unwrap();
    assert!((promo - 0.325).abs() < 0.0001);

    let cutoff = SystemTime::UNIX_EPOCH + Duration::from_secs(GLM_5_3_FLASH_PROMO_END_UNIX_SECS);
    let list = calculate_cost_at("glm-5.3-flash", 1_000_000, 0, 1_000_000, cutoff).unwrap();
    assert!((list - 0.65).abs() < 0.0001);
}

#[test]
fn test_extract_thinking_from_reasoning_content() {
    let content = "Final answer";
    let (thinking, clean) = extract_thinking(content, Some("step by step".to_string()));

    assert_eq!(clean, "Final answer");
    assert!(thinking.is_some());
    assert_eq!(thinking.as_ref().unwrap().content, "step by step");
}

#[test]
fn test_extract_thinking_from_think_tags() {
    let content = "before <think>internal reasoning</think> after";
    let (thinking, clean) = extract_thinking(content, None);

    assert_eq!(clean, "beforeafter");
    assert!(thinking.is_some());
    assert_eq!(thinking.as_ref().unwrap().content, "internal reasoning");
}

#[test]
fn test_cost_calculation_case_insensitive() {
    let cost = calculate_cost("GLM-5-TURBO", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 5.20).abs() < 0.01);

    let cost = calculate_cost("gLm-4.7-FlAsH", 1_000_000, 0, 1_000_000);
    assert_eq!(cost.unwrap(), 0.0);

    let cost = calculate_cost("glm-4.5-AIR", 1_000_000, 0, 1_000_000);
    assert!((cost.unwrap() - 1.30).abs() < 0.01); // 0.20 + 1.10
}

#[test]
fn test_cost_with_partial_tokens() {
    let cost = calculate_cost("glm-4.5", 500_000, 0, 500_000);
    assert!((cost.unwrap() - 1.40).abs() < 0.01); // 0.60 * 0.5 + 2.20 * 0.5
}

#[test]
fn test_unknown_model() {
    let cost = calculate_cost("unknown-model", 1_000_000, 0, 1_000_000);
    assert_eq!(cost, None);

    // Deprecated models should return None
    let cost = calculate_cost("glm-4", 1_000_000, 0, 1_000_000);
    assert_eq!(cost, None);
}

#[test]
fn test_cost_with_cache_read_tokens() {
    // GLM-4.7 pricing: input 0.60, cache_read 0.11, output 2.20
    // regular_input=100K => 0.06
    // cache_read=200K => 0.022
    // output=100K => 0.22
    // total => 0.302
    let cost = calculate_cost("glm-4.7", 100_000, 200_000, 100_000).unwrap();
    assert!((cost - 0.302).abs() < 0.0001);
}

#[test]
fn historical_reasoning_is_preserved_for_zai() {
    use crate::llm::types::{Message, ThinkingBlock};
    let think = |c: &str| ThinkingBlock {
        content: c.to_string(),
        tokens: 0,
    };
    let mut older = Message::assistant("older");
    older.thinking = Some(think("old reasoning"));
    let mut newer = Message::assistant("newer");
    newer.thinking = Some(think("current reasoning"));

    let out = serde_json::to_value(convert_messages(&[
        Message::user("go"),
        older,
        Message::user("again"),
        newer,
    ]))
    .unwrap();

    // Z.ai requires the complete unmodified reasoning history back: dropping
    // it to save context breaks the chain and the model re-derives.
    assert_eq!(out[1]["reasoning_content"], "old reasoning");
    assert_eq!(out[3]["reasoning_content"], "current reasoning");
}

#[test]
fn test_zai_usage_deserialize_prompt_tokens_shape() {
    let parsed: ZaiUsage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 173,
        "completion_tokens": 104,
        "total_tokens": 277,
        "prompt_tokens_details": { "cached_tokens": 43 }
    }))
    .expect("prompt_tokens shape should deserialize");
    assert_eq!(parsed.prompt_tokens, 173);
    assert_eq!(parsed.prompt_tokens_details.cached_tokens, 43);
}

#[test]
fn test_provider_capabilities() {
    let provider = ZaiProvider::new();
    assert!(provider.supports_caching("glm-4.7"));
    assert!(provider.supports_structured_output("glm-4.7"));
    // json_object only — schema is NOT enforced (no json_schema response_format)
    assert!(!provider.enforces_response_schema("glm-4.7"));
    // Vision models
    assert!(provider.supports_vision("glm-5v-turbo"));
    assert!(provider.supports_vision("glm-4.6v"));
    assert!(provider.supports_vision("glm-4.6v-flash"));
    assert!(provider.supports_vision("glm-4.5v"));
    assert!(provider.supports_vision("glm-ocr"));
    // GLM-5.3-Flash: native image/video input; both 5.3 variants have 1M context.
    assert!(provider.supports_vision("glm-5.3-flash"));
    assert!(provider.supports_video("glm-5.3-flash"));
    assert_eq!(provider.get_max_input_tokens("glm-5.3-flash"), 1_000_000);
    assert_eq!(provider.get_max_input_tokens("glm-5.3"), 1_000_000);
    // Non-vision models
    assert!(!provider.supports_vision("glm-5.1"));
    assert!(!provider.supports_vision("glm-5-turbo"));
    assert!(!provider.supports_vision("glm-5"));
    assert!(!provider.supports_vision("glm-4.7"));
}

#[test]
fn image_attachments_become_multimodal_content() {
    use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};
    // The regression this pins: a vision turn used to serialize as a plain text
    // string, so GLM-5.3-Flash never received the image and answered as if blind.
    let msg = Message::user("what is on the image?").with_images(vec![ImageAttachment {
        data: ImageData::Base64("QUJD".to_string()),
        media_type: "image/jpeg".to_string(),
        source_type: SourceType::Clipboard,
        dimensions: None,
        size_bytes: None,
    }]);
    let converted = convert_messages(std::slice::from_ref(&msg));
    let content = converted[0].content.clone().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[0]["text"], "what is on the image?");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(
        content[1]["image_url"]["url"],
        "data:image/jpeg;base64,QUJD"
    );

    // A text-only turn keeps the plain string shape Z.ai expects.
    let plain = convert_messages(&[Message::user("hi")]);
    assert_eq!(plain[0].content, Some(serde_json::json!("hi")));
}
