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
fn test_model_support() {
    let provider = MinimaxProvider::new();
    assert!(provider.supports_model("MiniMax-M3"));
    assert!(provider.supports_model("MiniMax-M2.7"));
    assert!(provider.supports_model("MiniMax-M2.7-highspeed"));
    assert!(provider.supports_model("MiniMax-M2.5"));
    assert!(provider.supports_model("MiniMax-M2.5-highspeed"));
    assert!(provider.supports_model("MiniMax-M2.1"));
    assert!(provider.supports_model("MiniMax-M2.1-highspeed"));
    assert!(provider.supports_model("MiniMax-M2"));
    // Not on the official pricing page
    assert!(!provider.supports_model("M2-her"));
    assert!(!provider.supports_model("gpt-4"));
    assert!(!provider.supports_model("claude-3"));
}

#[test]
fn test_model_support_case_insensitive() {
    let provider = MinimaxProvider::new();
    // Test lowercase
    assert!(provider.supports_model("minimax-m2.5"));
    assert!(provider.supports_model("minimax-m2.5-highspeed"));
    assert!(provider.supports_model("minimax-m2.1"));
    assert!(provider.supports_model("minimax-m2"));
    // Test uppercase
    assert!(provider.supports_model("MINIMAX-M2.5"));
    assert!(provider.supports_model("MINIMAX-M2.1"));
    assert!(provider.supports_model("MINIMAX-M2"));
    // Test mixed case
    assert!(provider.supports_model("Minimax-M2.5"));
    assert!(provider.supports_model("MINIMAX-m2.1"));
}

#[test]
fn test_cost_calculation() {
    // Test MiniMax-M3: $0.30 input, $1.20 output (≤512k input tokens)
    let cost = calculate_minimax_cost("MiniMax-M3", 500_000, 1_000_000, 0, 0);
    assert!((cost.unwrap() - 1.35).abs() < 1e-9); // 0.15 (0.30 × 0.5M) + 1.20

    // Test MiniMax-M3 above 512k: $0.60 input, $2.40 output
    let cost = calculate_minimax_cost("MiniMax-M3", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(3.00)); // 0.60 + 2.40

    // Test MiniMax-M2.7: $0.30 input, $1.20 output
    let cost = calculate_minimax_cost("MiniMax-M2.7", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(1.50)); // 0.30 + 1.20

    // Test MiniMax-M2.7-highspeed: $0.60 input, $2.40 output
    let cost = calculate_minimax_cost("MiniMax-M2.7-highspeed", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(3.00)); // 0.60 + 2.40

    // Test MiniMax-M2.5: $0.30 input, $1.20 output (Feb 2026)
    let cost = calculate_minimax_cost("MiniMax-M2.5", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(1.50)); // 0.30 + 1.20

    // Test MiniMax-M2.5-highspeed: $0.60 input, $2.40 output (Feb 2026)
    let cost = calculate_minimax_cost("MiniMax-M2.5-highspeed", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(3.00)); // 0.60 + 2.40

    // Test MiniMax-M2.1: $0.30 input, $1.20 output
    let cost = calculate_minimax_cost("MiniMax-M2.1", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(1.50)); // 0.30 + 1.20

    // Test MiniMax-M2.1-highspeed: $0.60 input, $2.40 output — must resolve to
    // its own row, not the base M2.1 substring match; cache read/write shared
    let cost = calculate_minimax_cost("MiniMax-M2.1-highspeed", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(3.00)); // 0.60 + 2.40
    let cost = calculate_minimax_cost("MiniMax-M2.1-highspeed", 0, 0, 1_000_000, 1_000_000);
    assert!((cost.unwrap() - 0.405).abs() < 1e-9); // 0.375 write + 0.03 read

    // Test MiniMax-M2: $0.30 input, $1.20 output
    let cost = calculate_minimax_cost("MiniMax-M2", 1_000_000, 1_000_000, 0, 0);
    assert_eq!(cost, Some(1.50)); // 0.30 + 1.20
}

#[test]
fn test_m3_long_context_tier_boundary() {
    // Exactly 512k input tokens stays on the ≤512k rate
    let at_limit = calculate_minimax_cost("MiniMax-M3", 512_000, 0, 0, 0).unwrap();
    assert!((at_limit - 0.512 * 0.30).abs() < 1e-9);

    // One token over switches the whole request to the >512k rate
    let over = calculate_minimax_cost("MiniMax-M3", 512_001, 0, 0, 0).unwrap();
    assert!((over - 0.512_001 * 0.60).abs() < 1e-9);

    // Cache reads count toward the threshold and are billed at the tier's
    // cache-read rate: $0.06 ≤512k, $0.12 >512k
    let cached_short = calculate_minimax_cost("MiniMax-M3", 0, 0, 0, 500_000).unwrap();
    assert!((cached_short - 0.5 * 0.06).abs() < 1e-9);
    let cached_long = calculate_minimax_cost("MiniMax-M3", 0, 0, 0, 600_000).unwrap();
    assert!((cached_long - 0.6 * 0.12).abs() < 1e-9);

    // Only M3 is tiered: M2.7 keeps its flat rate above 512k
    let m27 = calculate_minimax_cost("MiniMax-M2.7", 1_000_000, 0, 0, 0).unwrap();
    assert!((m27 - 0.30).abs() < 1e-9);
}

#[test]
fn test_provider_capabilities() {
    let provider = MinimaxProvider::new();
    assert!(provider.supports_caching("MiniMax-M2.1"));
    assert!(!provider.supports_vision("MiniMax-M2.1"));
    assert!(!provider.supports_video("MiniMax-M2.1"));
    assert!(!provider.supports_structured_output("MiniMax-M2.1"));

    // MiniMax-M3 is natively multimodal (image + video)
    assert!(provider.supports_vision("MiniMax-M3"));
    assert!(provider.supports_video("MiniMax-M3"));
    assert!(provider.supports_caching("MiniMax-M3"));
}

#[test]
fn parallel_tool_results_share_one_user_message() {
    let mut assistant = Message::assistant("");
    assistant.tool_calls = Some(serde_json::json!([
        {"id": "toolu_a", "name": "first", "arguments": {}, "meta": null},
        {"id": "toolu_b", "name": "second", "arguments": {}, "meta": null}
    ]));

    let messages = vec![
        assistant,
        Message::tool("result A", "toolu_a", "first"),
        Message::tool("result B", "toolu_b", "second"),
    ];

    let converted = convert_messages(&messages);
    assert_eq!(converted.len(), 2);
    assert_eq!(converted[0].role, "assistant");
    assert_eq!(converted[1].role, "user");

    let user_content = serde_json::to_value(&converted[1].content).unwrap();
    let blocks = user_content.as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
    assert_eq!(blocks[1]["type"], "tool_result");
    assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
}

#[test]
fn tool_results_merge_following_user_hint() {
    let messages = vec![
        Message::tool("result A", "toolu_a", "first"),
        Message::tool("result B", "toolu_b", "second"),
        Message::user("Please use those results."),
    ];

    let converted = convert_messages(&messages);
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].role, "user");

    let content = serde_json::to_value(&converted[0].content).unwrap();
    let blocks = content.as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
    assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
    assert_eq!(blocks[2]["type"], "text");
    assert_eq!(blocks[2]["text"], "Please use those results.");
}
