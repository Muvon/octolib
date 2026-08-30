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

fn calculate_cost(
    pricing: &[PricingTuple],
    model: &str,
    input_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    calculate_cost_with_cache(pricing, model, input_tokens, 0, completion_tokens)
}

#[test]
fn test_supports_model() {
    let provider = DeepSeekProvider::new();
    assert!(provider.supports_model("deepseek-v4-flash"));
    assert!(provider.supports_model("deepseek-v4-pro"));
    // Legacy aliases removed by DeepSeek 2026-07-24
    assert!(!provider.supports_model("deepseek-chat"));
    assert!(!provider.supports_model("deepseek-reasoner"));
    assert!(!provider.supports_model("gpt-4"));
    assert!(!provider.supports_model("deepseek-coder")); // Not in current API
}

#[test]
fn test_vision_route() {
    use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};

    let provider = DeepSeekProvider::new();
    assert!(provider.supports_model("deepseek-v4-flash-vision-exp"));
    assert!(provider.supports_vision("deepseek-v4-flash-vision-exp"));
    assert!(!provider.supports_vision("deepseek-v4-flash"));
    assert!(!provider.supports_vision("deepseek-v4-pro"));

    // Vision route bills at v4-flash rates
    assert_eq!(
        calculate_cost(PRICING_PEAK, "deepseek-v4-flash-vision-exp", 1_000_000, 0),
        calculate_cost(PRICING_PEAK, "deepseek-v4-flash", 1_000_000, 0)
    );

    let msg = Message::user("what is this?").with_images(vec![ImageAttachment {
        data: ImageData::Base64("QUJD".to_string()),
        media_type: "image/png".to_string(),
        source_type: SourceType::Clipboard,
        dimensions: None,
        size_bytes: None,
    }]);
    let content = convert_messages(std::slice::from_ref(&msg), false)[0]
        .content
        .clone()
        .unwrap();
    assert_eq!(content[0]["text"], "what is this?");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,QUJD");

    // Text-only turns keep the plain string shape
    let plain = convert_messages(&[Message::user("hi")], false)[0]
        .content
        .clone();
    assert_eq!(plain, Some(serde_json::json!("hi")));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = DeepSeekProvider::new();
    assert!(provider.supports_model("DEEPSEEK-V4-FLASH"));
    assert!(provider.supports_model("DEEPSEEK-V4-PRO"));
    assert!(provider.supports_model("DeepSeek-V4-Flash"));
}

#[test]
fn test_max_input_tokens() {
    let provider = DeepSeekProvider::new();
    assert_eq!(
        provider.get_max_input_tokens("deepseek-v4-flash"),
        1_000_000
    );
    assert_eq!(provider.get_max_input_tokens("deepseek-v4-pro"), 1_000_000);
}

#[test]
fn test_map_reasoning_effort() {
    use crate::llm::types::ReasoningEffort;
    assert_eq!(
        map_reasoning_effort(Some(ReasoningEffort::Low)),
        Some("low")
    );
    assert_eq!(
        map_reasoning_effort(Some(ReasoningEffort::Medium)),
        Some("low")
    );
    assert_eq!(
        map_reasoning_effort(Some(ReasoningEffort::High)),
        Some("high")
    );
    assert_eq!(
        map_reasoning_effort(Some(ReasoningEffort::XHigh)),
        Some("high")
    );
    assert_eq!(
        map_reasoning_effort(Some(ReasoningEffort::Max)),
        Some("max")
    );
    // None = provider default (thinking on, effort "high"); field omitted.
    assert_eq!(map_reasoning_effort(None), None);
}

#[test]
fn test_build_request_uses_native_thinking_tool_contract() {
    use crate::llm::types::{FunctionDefinition, Message, ReasoningEffort, ThinkingBlock};

    let messages = [
        Message::assistant("previous answer").with_thinking(ThinkingBlock {
            content: "previous reasoning".to_string(),
            tokens: 3,
        }),
    ];
    let mut params = ChatCompletionParams::new(&messages, "deepseek-v4-flash", 0.7, 0.9, 40, 1024)
        .with_reasoning_effort(ReasoningEffort::Medium);
    params.tools = Some(vec![FunctionDefinition {
        name: "inspect".to_string(),
        description: "Inspect state".to_string(),
        parameters: serde_json::json!({"type": "object"}),
        cache_control: None,
    }]);

    let request = serde_json::to_value(build_request(&params)).unwrap();
    assert_eq!(request["thinking"]["type"], "enabled");
    assert_eq!(request["reasoning_effort"], "low");
    assert_eq!(
        request["messages"][0]["reasoning_content"],
        "previous reasoning"
    );
    assert_eq!(request["messages"][0]["content"], "previous answer");
    assert_eq!(request["tools"][0]["function"]["name"], "inspect");
    assert!(request.get("tool_choice").is_none());
    assert!(request.get("temperature").is_none());
    assert!(request.get("top_p").is_none());
}

#[test]
fn test_thinking_models_do_not_advertise_sampling_controls() {
    let provider = DeepSeekProvider::new();
    assert_eq!(
        provider.supported_sampling_params("deepseek-v4-flash"),
        SamplingSupport::NONE
    );
}

#[test]
fn test_tiered_pricing_peak_and_off_peak() {
    // Peak: flash $0.44 in / $1.32 out per 1M
    let peak = calculate_cost(PRICING_PEAK, "deepseek-v4-flash", 1_000_000, 500_000).unwrap();
    assert!((peak - (0.44 + 0.5 * 1.32)).abs() < 0.01);

    // Off-peak is exactly half of peak
    let off_peak =
        calculate_cost(PRICING_OFF_PEAK, "deepseek-v4-flash", 1_000_000, 500_000).unwrap();
    assert!((off_peak - peak / 2.0).abs() < 0.01);

    // Peak: pro $1.32 in / $3.96 out per 1M
    let pro = calculate_cost(PRICING_PEAK, "deepseek-v4-pro", 1_000_000, 500_000).unwrap();
    assert!((pro - (1.32 + 0.5 * 3.96)).abs() < 0.01);

    // Peak cache-hit rate: flash $0.014/1M
    let cached =
        calculate_cost_with_cache(PRICING_PEAK, "deepseek-v4-flash", 0, 1_000_000, 0).unwrap();
    assert!((cached - 0.014).abs() < 0.0001);
}

#[test]
fn test_pricing_table_at_selects_tier() {
    use std::time::{Duration, SystemTime};

    let at = |secs: u64| SystemTime::UNIX_EPOCH + Duration::from_secs(secs);

    // Monday 2026-08-17 00:00 UTC — walk a full weekday hour by hour.
    let monday_midnight = 1_786_924_800_u64;
    for hour in 0..24u64 {
        let table = pricing_table_at(at(monday_midnight + hour * 3_600));
        let expected = if is_peak_window(monday_midnight / 86_400, hour) {
            PRICING_PEAK
        } else {
            PRICING_OFF_PEAK
        };
        assert_eq!(table, expected, "hour {} misclassified", hour);
    }

    // Saturday 2026-08-22 is off-peak for the entire day, including hours
    // that would be peak on weekdays.
    let saturday_midnight = monday_midnight + 5 * 86_400;
    for hour in 0..24u64 {
        assert_eq!(
            pricing_table_at(at(saturday_midnight + hour * 3_600)),
            PRICING_OFF_PEAK,
            "Saturday hour {} must be off-peak",
            hour
        );
    }
}

/// Deserializes real payload shapes on purpose: the bug this guards lived in
/// the `#[serde(default)]` fields, so constructing the struct by hand would
/// step right over it.
#[test]
fn test_split_prompt_tokens_handles_both_usage_shapes() {
    let parse = |v: serde_json::Value| -> DeepSeekUsage { serde_json::from_value(v).unwrap() };

    // Native shape — hit and miss both reported.
    let native = parse(serde_json::json!({
        "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
        "prompt_cache_hit_tokens": 400, "prompt_cache_miss_tokens": 600
    }));
    assert_eq!(split_prompt_tokens(&native), (600, 400));

    // OpenAI-compatible shape — cached_tokens only, NO miss field anywhere.
    // Reading the miss field directly yielded 0 here, so the whole uncached
    // prompt was billed free and reported as 0 input tokens.
    let compat = parse(serde_json::json!({
        "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010,
        "prompt_tokens_details": {"cached_tokens": 400}
    }));
    assert_eq!(
        split_prompt_tokens(&compat),
        (600, 400),
        "uncached prompt tokens must not bill as free"
    );

    // No cache information at all — every prompt token is a miss.
    let plain = parse(serde_json::json!({
        "prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010
    }));
    assert_eq!(split_prompt_tokens(&plain), (1000, 0));

    // Inconsistent provider data must saturate, never underflow into a
    // near-u64::MAX token count and a catastrophic charge.
    let bogus = parse(serde_json::json!({
        "prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101,
        "prompt_tokens_details": {"cached_tokens": 500}
    }));
    assert_eq!(split_prompt_tokens(&bogus), (0, 500));
}

#[test]
fn test_thinking_block_extraction() {
    // Test with reasoning_content present
    let message_with_thinking = DeepSeekMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("The answer is 9.11")),
        reasoning_content: Some("Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    // Verify reasoning_content is properly stored
    assert!(message_with_thinking.reasoning_content.is_some());
    let reasoning = message_with_thinking.reasoning_content.as_ref().unwrap();
    assert_eq!(reasoning, "Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.");

    // Test token estimation (length / 4)
    let estimated_tokens = (reasoning.len() / 4) as u64;
    assert!(estimated_tokens > 0);

    // Test without reasoning_content
    let message_without_thinking = DeepSeekMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("Hello")),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    assert!(message_without_thinking.reasoning_content.is_none());

    // Test with empty reasoning_content
    let message_empty_thinking = DeepSeekMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("Hello")),
        reasoning_content: Some("".to_string()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    };

    assert!(message_empty_thinking.reasoning_content.is_some());
    assert!(message_empty_thinking
        .reasoning_content
        .as_ref()
        .unwrap()
        .is_empty());

    // Response deserialization accepts null content from the provider.
    let message_tool_call = DeepSeekMessage {
        role: "assistant".to_string(),
        content: None,
        reasoning_content: None,
        tool_calls: Some(vec![DeepSeekToolCall {
            id: "call_123".to_string(),
            tool_type: "function".to_string(),
            function: DeepSeekFunction {
                name: "get_weather".to_string(),
                arguments: "{}".to_string(),
            },
        }]),
        tool_call_id: None,
        name: None,
    };

    assert!(message_tool_call.content.is_none());
    assert!(message_tool_call.tool_calls.is_some());
}

#[test]
fn test_convert_messages_reasoning_content_replay() {
    use crate::llm::tool_calls::GenericToolCall;
    use crate::llm::types::{Message, ThinkingBlock};

    let tool_calls_json = serde_json::to_value(vec![GenericToolCall {
        id: "call_123".to_string(),
        name: "list_files".to_string(),
        arguments: serde_json::json!({"path": "."}),
        meta: None,
    }])
    .unwrap();

    // Assistant turn with tool_calls + thinking → reasoning_content must be replayed.
    let assistant_with_tools = Message {
        role: "assistant".to_string(),
        content: String::new(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(tool_calls_json.clone()),
        images: None,
        videos: None,
        thinking: Some(ThinkingBlock {
            content: "I should list the files first.".to_string(),
            tokens: 8,
        }),
        id: None,
    };
    let converted = convert_messages(std::slice::from_ref(&assistant_with_tools), true);
    assert_eq!(converted.len(), 1);
    assert_eq!(
        converted[0].reasoning_content.as_deref(),
        Some("I should list the files first.")
    );
    assert!(converted[0].tool_calls.is_some());
    assert_eq!(converted[0].content, Some(serde_json::json!("")));

    // Assistant turn with tool_calls but no stored thinking → field omitted entirely (None).
    // DeepSeek does not require reasoning_content when there was no thinking; unlike
    // Moonshot it does NOT require an empty string sentinel.
    let assistant_tools_no_thinking = Message {
        thinking: None,
        ..assistant_with_tools.clone()
    };
    let converted = convert_messages(std::slice::from_ref(&assistant_tools_no_thinking), true);
    assert!(converted[0].reasoning_content.is_none());

    // With tools on the current request, assistant reasoning is replayed even
    // when that historical assistant turn did not itself call a tool.
    let assistant_plain = Message::assistant("Hello").with_thinking(ThinkingBlock {
        content: "trivial".to_string(),
        tokens: 1,
    });
    let converted = convert_messages(std::slice::from_ref(&assistant_plain), true);
    assert_eq!(converted[0].reasoning_content.as_deref(), Some("trivial"));

    // Without tools, DeepSeek ignores historical reasoning, so omit it.
    let converted = convert_messages(std::slice::from_ref(&assistant_plain), false);
    assert!(converted[0].reasoning_content.is_none());

    // User / tool / system messages → never carry reasoning_content.
    let user_msg = Message::user("hi");
    let tool_msg = Message::tool("ok", "call_123", "list_files");
    let system_msg = Message::system("be helpful");
    for msg in [user_msg, tool_msg, system_msg] {
        let converted = convert_messages(std::slice::from_ref(&msg), true);
        assert!(converted[0].reasoning_content.is_none());
    }

    // Verify JSON serialization: None is omitted, Some("") is preserved.
    let json = serde_json::to_value(
        &convert_messages(std::slice::from_ref(&assistant_with_tools), true)[0],
    )
    .unwrap();
    assert_eq!(
        json.get("reasoning_content").and_then(|v| v.as_str()),
        Some("I should list the files first.")
    );

    let json_plain =
        serde_json::to_value(&convert_messages(std::slice::from_ref(&assistant_plain), true)[0])
            .unwrap();
    assert_eq!(
        json_plain
            .get("reasoning_content")
            .and_then(serde_json::Value::as_str),
        Some("trivial")
    );
}
