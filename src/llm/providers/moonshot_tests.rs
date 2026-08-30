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
    let provider = MoonshotProvider::new();
    // Kimi K3
    assert!(provider.supports_model("kimi-k3"));
    // Kimi K2 series
    assert!(provider.supports_model("kimi-k2"));
    assert!(provider.supports_model("kimi-k2-thinking"));
    assert!(provider.supports_model("kimi-k2-thinking-turbo"));
    assert!(provider.supports_model("kimi-k2-turbo-preview"));
    assert!(provider.supports_model("kimi-k2.5"));
    assert!(provider.supports_model("kimi-k2.6"));
    assert!(provider.supports_model("kimi-k2.7-code"));
    assert!(provider.supports_model("kimi-k2.7-code-highspeed"));
    assert!(provider.supports_model("kimi-k2-0711-preview"));
    assert!(provider.supports_model("kimi-k2-0905-preview"));
    assert!(provider.supports_model("KIMI-K2"));
    // Moonshot V1 series
    assert!(provider.supports_model("moonshot-v1-8k"));
    assert!(provider.supports_model("moonshot-v1-32k"));
    assert!(provider.supports_model("moonshot-v1-128k"));
    assert!(provider.supports_model("moonshot-v1-8k-vision-preview"));
    // Not supported
    assert!(!provider.supports_model("gpt-4"));
}

#[test]
fn test_supports_vision() {
    let provider = MoonshotProvider::new();
    assert!(provider.supports_vision("kimi-k3"));
    assert!(provider.supports_vision("kimi-k2.5"));
    assert!(provider.supports_vision("kimi-k2.6"));
    assert!(provider.supports_vision("kimi-k2.7-code"));
    assert!(!provider.supports_vision("kimi-k2"));
}

#[test]
fn test_supports_caching() {
    let provider = MoonshotProvider::new();
    // K2/K3 families support automatic context caching
    assert!(provider.supports_caching("kimi-k3"));
    assert!(provider.supports_caching("kimi-k2"));
    assert!(provider.supports_caching("kimi-k2.5"));
    assert!(provider.supports_caching("kimi-k2.6"));
    assert!(provider.supports_caching("kimi-k2.7-code"));
    assert!(provider.supports_caching("kimi-k2-thinking"));
    // V1 legacy models do NOT support caching
    assert!(!provider.supports_caching("moonshot-v1-8k"));
    assert!(!provider.supports_caching("moonshot-v1-32k"));
    assert!(!provider.supports_caching("moonshot-v1-128k"));
}

#[test]
fn test_calculate_cost() {
    // kimi-k2: Input: $0.60/1M, Output: $2.50/1M
    let cost = calculate_cost("kimi-k2", 1_000_000, 500_000);
    assert!(cost.is_some());
    let expected = 0.60 + (0.5 * 2.50);
    assert!((cost.unwrap() - expected).abs() < 0.01);
}

#[test]
fn test_calculate_cost_with_cache() {
    // kimi-k2: Cache hit: $0.15/1M, Cache miss: $0.60/1M, Output: $2.50/1M
    let cost = calculate_cost_with_cache("kimi-k2", 500_000, 500_000, 250_000);
    assert!(cost.is_some());
    let expected = (0.5 * 0.60) + (0.5 * 0.15) + (0.25 * 2.50);
    assert!((cost.unwrap() - expected).abs() < 0.01);

    // Cost with cache should be less
    let cost_no_cache = calculate_cost("kimi-k2", 1_000_000, 250_000);
    assert!(cost.unwrap() < cost_no_cache.unwrap());

    // kimi-k2.5: Cache hit: $0.10/1M, Cache miss: $0.60/1M, Output: $3.00/1M
    let cost_k25 = calculate_cost_with_cache("kimi-k2.5", 400_000, 600_000, 500_000);
    assert!(cost_k25.is_some());
    let expected_k25 = (0.4 * 0.60) + (0.6 * 0.10) + (0.5 * 3.00);
    assert!((cost_k25.unwrap() - expected_k25).abs() < 0.01);

    // kimi-k2-thinking-turbo: Cache hit: $0.15/1M, Cache miss: $1.15/1M, Output: $8.00/1M
    let cost_turbo = calculate_cost_with_cache("kimi-k2-thinking-turbo", 500_000, 500_000, 250_000);
    assert!(cost_turbo.is_some());
    let expected_turbo = (0.5 * 1.15) + (0.5 * 0.15) + (0.25 * 8.00);
    assert!((cost_turbo.unwrap() - expected_turbo).abs() < 0.01);
}

#[test]
fn test_get_max_input_tokens() {
    let provider = MoonshotProvider::new();
    // Kimi K3 — 1M = 1_048_576
    assert_eq!(provider.get_max_input_tokens("kimi-k3"), 1_048_576);
    // Kimi K2 family — 256K = 262_144 (kimi-k2-0711 is 128K = 131_072)
    assert_eq!(provider.get_max_input_tokens("kimi-k2"), 262_144);
    assert_eq!(provider.get_max_input_tokens("kimi-k2.5"), 262_144);
    assert_eq!(provider.get_max_input_tokens("kimi-k2.6"), 262_144);
    assert_eq!(provider.get_max_input_tokens("kimi-k2.7-code"), 262_144);
    assert_eq!(
        provider.get_max_input_tokens("kimi-k2-0905-preview"),
        262_144
    );
    assert_eq!(provider.get_max_input_tokens("kimi-k2-thinking"), 262_144);
    assert_eq!(
        provider.get_max_input_tokens("kimi-k2-thinking-turbo"),
        262_144
    );
    assert_eq!(
        provider.get_max_input_tokens("kimi-k2-0711-preview"),
        131_072
    );
    // Moonshot V1 series
    assert_eq!(provider.get_max_input_tokens("moonshot-v1-128k"), 131_072);
    assert_eq!(provider.get_max_input_tokens("moonshot-v1-32k"), 32_768);
    assert_eq!(provider.get_max_input_tokens("moonshot-v1-8k"), 8_192);
    // Unknown
    assert_eq!(provider.get_max_input_tokens("unknown"), 128_000);
}

#[test]
fn test_calculate_cost_k26() {
    // kimi-k2.6: Input (cache miss): $0.95/1M, Output: $4.00/1M, Cache hit: $0.16/1M
    let cost = calculate_cost("kimi-k2.6", 1_000_000, 500_000);
    assert!(cost.is_some());
    let expected = 0.95 + (0.5 * 4.00);
    assert!((cost.unwrap() - expected).abs() < 0.01);

    // kimi-k2.6 with cache (50/50 split, 250k output)
    let cost_cached = calculate_cost_with_cache("kimi-k2.6", 500_000, 500_000, 250_000);
    assert!(cost_cached.is_some());
    let expected_cached = (0.5 * 0.95) + (0.5 * 0.16) + (0.25 * 4.00);
    assert!((cost_cached.unwrap() - expected_cached).abs() < 0.01);

    // Cached version should be cheaper than full cache-miss for same total prompt
    let cost_full_miss = calculate_cost("kimi-k2.6", 1_000_000, 250_000);
    assert!(cost_cached.unwrap() < cost_full_miss.unwrap());
}

#[test]
fn test_calculate_cost_k27() {
    // kimi-k2.7-code: Input (cache miss): $0.95/1M, Output: $4.00/1M, Cache hit: $0.19/1M
    let cost = calculate_cost("kimi-k2.7-code", 1_000_000, 500_000);
    assert!(cost.is_some());
    let expected = 0.95 + (0.5 * 4.00);
    assert!((cost.unwrap() - expected).abs() < 0.01);

    // kimi-k2.7-code with cache (50/50 split, 250k output)
    let cost_cached = calculate_cost_with_cache("kimi-k2.7-code", 500_000, 500_000, 250_000);
    assert!(cost_cached.is_some());
    let expected_cached = (0.5 * 0.95) + (0.5 * 0.19) + (0.25 * 4.00);
    assert!((cost_cached.unwrap() - expected_cached).abs() < 0.01);

    // highspeed must resolve to its own pricing ($1.90/$8.00), NOT the base
    // kimi-k2.7 entry (substring-ordering regression guard)
    let cost_hs = calculate_cost("kimi-k2.7-code-highspeed", 1_000_000, 500_000);
    assert!(cost_hs.is_some());
    let expected_hs = 1.90 + (0.5 * 8.00);
    assert!((cost_hs.unwrap() - expected_hs).abs() < 0.01);
}
#[test]
fn test_calculate_cost_k3() {
    // kimi-k3: Input (cache miss): $3.00/1M, Output: $15.00/1M, Cache hit: $0.30/1M
    let cost = calculate_cost("kimi-k3", 1_000_000, 500_000);
    assert!(cost.is_some());
    let expected = 3.00 + (0.5 * 15.00);
    assert!((cost.unwrap() - expected).abs() < 0.01);

    // kimi-k3 with cache (50/50 split, 250k output)
    let cost_cached = calculate_cost_with_cache("kimi-k3", 500_000, 500_000, 250_000);
    assert!(cost_cached.is_some());
    let expected_cached = (0.5 * 3.00) + (0.5 * 0.30) + (0.25 * 15.00);
    assert!((cost_cached.unwrap() - expected_cached).abs() < 0.01);
}

#[test]
fn test_moonshot_v1_pricing() {
    // moonshot-v1-8k: Input: $0.20/1M, Output: $2.00/1M (no cache)
    let cost = calculate_cost("moonshot-v1-8k", 1_000_000, 500_000);
    assert!(cost.is_some());
    let expected = 0.20 + (0.5 * 2.00);
    assert!((cost.unwrap() - expected).abs() < 0.01);

    // moonshot-v1-32k: Input: $1.00/1M, Output: $3.00/1M
    let cost_32k = calculate_cost("moonshot-v1-32k", 1_000_000, 500_000);
    assert!(cost_32k.is_some());
    let expected_32k = 1.00 + (0.5 * 3.00);
    assert!((cost_32k.unwrap() - expected_32k).abs() < 0.01);

    // moonshot-v1-128k: Input: $2.00/1M, Output: $5.00/1M
    let cost_128k = calculate_cost("moonshot-v1-128k", 1_000_000, 500_000);
    assert!(cost_128k.is_some());
    let expected_128k = 2.00 + (0.5 * 5.00);
    assert!((cost_128k.unwrap() - expected_128k).abs() < 0.01);
}

#[test]
fn test_k3_reasoning_effort() {
    // Only kimi-k3 gets the reasoning_effort field
    assert_eq!(
        k3_reasoning_effort("kimi-k3", Some(ReasoningEffort::Low)),
        Some("low")
    );
    assert_eq!(
        k3_reasoning_effort("kimi-k3", Some(ReasoningEffort::Medium)),
        Some("low")
    );
    assert_eq!(
        k3_reasoning_effort("kimi-k3", Some(ReasoningEffort::High)),
        Some("high")
    );
    assert_eq!(
        k3_reasoning_effort("kimi-k3", Some(ReasoningEffort::XHigh)),
        Some("high")
    );
    assert_eq!(
        k3_reasoning_effort("kimi-k3", Some(ReasoningEffort::Max)),
        Some("max")
    );
    // Unset effort → field omitted → K3 applies its own default ("max")
    assert_eq!(k3_reasoning_effort("kimi-k3", None), None);
    // Non-K3 models never get the field
    assert_eq!(
        k3_reasoning_effort("kimi-k2.6", Some(ReasoningEffort::Max)),
        None
    );
}

#[test]
fn test_usage_parses_reasoning_and_cached_tokens() {
    // Real K3 response shape: top-level cached_tokens, plus
    // prompt/completion token details (see platform.kimi.ai/docs/api/chat)
    let usage: MoonshotUsage = serde_json::from_value(serde_json::json!({
        "prompt_tokens": 208,
        "completion_tokens": 98,
        "total_tokens": 306,
        "cached_tokens": 208,
        "prompt_tokens_details": { "cached_tokens": 208 },
        "completion_tokens_details": { "reasoning_tokens": 35 }
    }))
    .unwrap();
    assert_eq!(usage.cached_tokens, 208);
    assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, 208);
    assert_eq!(
        usage.completion_tokens_details.unwrap().reasoning_tokens,
        35
    );
}

#[test]
fn test_reasoning_content_serialization() {
    use crate::llm::types::ThinkingBlock;

    // Test 1: Thinking model - Assistant message with tool calls and thinking
    let msg_with_thinking = crate::llm::types::Message {
        role: "assistant".to_string(),
        content: "I'll help you with that.".to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(serde_json::json!([{
            "id": "call_123",
            "name": "get_weather",
            "arguments": {"city": "Beijing"}
        }])),
        images: None,
        videos: None,
        thinking: Some(ThinkingBlock {
            content: "Let me check the weather".to_string(),
            tokens: 0,
        }),
        id: None,
    };

    // For thinking models, reasoning_content should be present
    let converted = convert_messages(std::slice::from_ref(&msg_with_thinking), "kimi-k2-thinking");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_some());
    assert_eq!(
        converted[0].reasoning_content.as_ref().unwrap(),
        "Let me check the weather"
    );

    // For Kimi K2 models, reasoning_content should be present
    let converted = convert_messages(std::slice::from_ref(&msg_with_thinking), "kimi-k2");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_some());

    // Test 2: Thinking model - Assistant message with tool calls but no thinking (empty reasoning_content)
    let msg_no_thinking = crate::llm::types::Message {
        role: "assistant".to_string(),
        content: "I'll help you with that.".to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: Some(serde_json::json!([{
            "id": "call_456",
            "name": "search",
            "arguments": {"query": "test"}
        }])),
        images: None,
        videos: None,
        thinking: None,
        id: None,
    };

    // For thinking models, should have Some("") for tool calls even without thinking
    let converted = convert_messages(std::slice::from_ref(&msg_no_thinking), "kimi-k2.5");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_some());
    assert_eq!(converted[0].reasoning_content.as_ref().unwrap(), "");

    // For Kimi K2 models, should be Some("") even without thinking
    let converted = convert_messages(std::slice::from_ref(&msg_no_thinking), "kimi-k2");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_some());
    assert_eq!(converted[0].reasoning_content.as_ref().unwrap(), "");

    // Test 3: Regular assistant message without tool calls or stored thinking
    let regular_msg = crate::llm::types::Message {
        role: "assistant".to_string(),
        content: "Hello, how can I help?".to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: None,
        name: None,
        tool_calls: None,
        images: None,
        videos: None,
        thinking: None,
        id: None,
    };

    // Regular assistant messages without tool calls: no reasoning_content
    let converted = convert_messages(std::slice::from_ref(&regular_msg), "kimi-k2-thinking");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_none());

    let converted = convert_messages(std::slice::from_ref(&regular_msg), "kimi-k2");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_none());

    // Test 4: K2.7 preserves thinking even on assistant turns without tools.
    let regular_msg_with_thinking = crate::llm::types::Message {
        thinking: Some(ThinkingBlock {
            content: "I chose the first three numbers and retained two.".to_string(),
            tokens: 12,
        }),
        ..regular_msg
    };
    let converted = convert_messages(&[regular_msg_with_thinking], "kimi-k2.7-code");
    assert_eq!(
        converted[0].reasoning_content.as_deref(),
        Some("I chose the first three numbers and retained two.")
    );

    let k25_plain_with_thinking = crate::llm::types::Message {
        thinking: Some(ThinkingBlock {
            content: "K2.5 trace".to_string(),
            tokens: 3,
        }),
        ..crate::llm::types::Message::assistant("answer")
    };
    let converted = convert_messages(&[k25_plain_with_thinking], "kimi-k2.5");
    assert!(converted[0].reasoning_content.is_none());

    // Test 5: Tool response message (no reasoning_content)
    let tool_msg = crate::llm::types::Message {
        role: "tool".to_string(),
        content: "Weather is sunny".to_string(),
        timestamp: 0,
        cached: false,
        cache_ttl: None,
        tool_call_id: Some("call_123".to_string()),
        name: Some("get_weather".to_string()),
        tool_calls: None,
        images: None,
        videos: None,
        thinking: None,
        id: None,
    };

    let converted = convert_messages(&[tool_msg], "kimi-k2-thinking");
    assert_eq!(converted.len(), 1);
    assert!(converted[0].reasoning_content.is_none());

    // Test 6: Verify JSON serialization behavior
    let msg_with_reasoning = MoonshotMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("test")),
        tool_calls: Some(vec![]),
        tool_call_id: None,
        name: None,
        reasoning_content: Some("thinking".to_string()),
    };

    let json = serde_json::to_value(&msg_with_reasoning).unwrap();
    assert!(json.get("reasoning_content").is_some());
    assert_eq!(
        json.get("reasoning_content").unwrap().as_str().unwrap(),
        "thinking"
    );

    // Test 7: None reasoning_content should be omitted
    let msg_without_reasoning = MoonshotMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("test")),
        tool_calls: None,
        tool_call_id: None,
        name: None,
        reasoning_content: None,
    };

    let json = serde_json::to_value(&msg_without_reasoning).unwrap();
    assert!(json.get("reasoning_content").is_none());

    // Test 8: Empty reasoning_content (Some("")) should be serialized
    let msg_with_empty_reasoning = MoonshotMessage {
        role: "assistant".to_string(),
        content: Some(serde_json::json!("test")),
        tool_calls: Some(vec![]),
        tool_call_id: None,
        name: None,
        reasoning_content: Some("".to_string()),
    };

    let json = serde_json::to_value(&msg_with_empty_reasoning).unwrap();
    assert!(json.get("reasoning_content").is_some());
    assert_eq!(json.get("reasoning_content").unwrap().as_str().unwrap(), "");

    // Test 9: Some Kimi-compatible servers return `reasoning`; store it
    // through the same internal field while continuing to emit Moonshot's
    // documented `reasoning_content` request field.
    let aliased: MoonshotMessage = serde_json::from_value(serde_json::json!({
        "role": "assistant",
        "content": "answer",
        "reasoning": "retained trace"
    }))
    .unwrap();
    assert_eq!(aliased.reasoning_content.as_deref(), Some("retained trace"));
    let json = serde_json::to_value(&aliased).unwrap();
    assert_eq!(
        json.get("reasoning_content")
            .and_then(serde_json::Value::as_str),
        Some("retained trace")
    );
    assert!(json.get("reasoning").is_none());
}

#[test]
fn test_preserved_thinking_model_contract() {
    assert!(preserves_historical_thinking("kimi-k2.6"));
    assert!(preserves_historical_thinking("kimi-k2.7-code"));
    assert!(preserves_historical_thinking("kimi-k3"));
    assert!(!preserves_historical_thinking("kimi-k2.5"));

    assert_eq!(
        thinking_config("kimi-k2.6"),
        Some(serde_json::json!({"type": "enabled", "keep": "all"}))
    );
    assert!(thinking_config("kimi-k2.7-code").is_none());
    assert!(thinking_config("kimi-k2.5").is_none());
    assert!(thinking_config("kimi-k3").is_none());
}

#[test]
fn test_sanitize_schema_inlines_ref_with_sibling_description() {
    // Repro: schemars 1.x emits {"$ref": "#/$defs/Foo", "description": "..."}
    // which Moonshot rejects as "conflicting keywords found after $ref expansion: description"
    let schema = serde_json::json!({
        "type": "object",
        "$defs": {
            "RelationshipKind": {
                "type": "string",
                "enum": ["related_to", "depends_on"]
            }
        },
        "properties": {
            "relationship_type": {
                "$ref": "#/$defs/RelationshipKind",
                "description": "Relationship type"
            }
        },
        "required": ["relationship_type"]
    });

    let sanitized = sanitize_schema_for_moonshot(&schema);

    // $defs should be gone
    assert!(sanitized.get("$defs").is_none(), "$defs should be removed");

    let prop = &sanitized["properties"]["relationship_type"];
    // No $ref left
    assert!(prop.get("$ref").is_none(), "$ref should be inlined");
    // Inlined enum from def
    assert_eq!(prop["type"], "string");
    assert_eq!(prop["enum"][0], "related_to");
    // Sibling description preserved
    assert_eq!(prop["description"], "Relationship type");
}

#[test]
fn test_sanitize_schema_handles_nested_refs_in_anyof() {
    // schemars also produces $refs inside anyOf — must be recursively inlined
    let schema = serde_json::json!({
        "type": "object",
        "$defs": {
            "MemoryType": {
                "type": "string",
                "enum": ["code", "architecture"]
            }
        },
        "properties": {
            "memory_type": {
                "anyOf": [
                    {"$ref": "#/$defs/MemoryType"},
                    {"type": "null"}
                ],
                "description": "Memory category"
            }
        }
    });

    let sanitized = sanitize_schema_for_moonshot(&schema);

    let any_of = &sanitized["properties"]["memory_type"]["anyOf"];
    assert!(
        any_of[0].get("$ref").is_none(),
        "nested $ref should be inlined"
    );
    assert_eq!(any_of[0]["type"], "string");
    assert_eq!(any_of[0]["enum"][0], "code");
    assert_eq!(any_of[1]["type"], "null");
}

#[test]
fn test_sanitize_schema_no_defs_passthrough() {
    // Schemas without $defs should pass through unchanged
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A name"}
        },
        "required": ["name"]
    });
    let sanitized = sanitize_schema_for_moonshot(&schema);
    assert_eq!(sanitized, schema);
}
