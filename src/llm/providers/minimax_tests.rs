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
use crate::test_support::env::EnvGuard;

#[test]
fn test_model_support() {
	let provider = MinimaxProvider::new();
	assert!(provider.supports_model("MiniMax-M3.1-Flash"));
	assert!(provider.supports_model("MiniMax-M3.1-Flash-Preview"));
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
fn test_context_windows_follow_model_table() {
	let provider = MinimaxProvider::new();
	assert_eq!(provider.get_max_input_tokens("MiniMax-M3"), 1_000_000);
	assert_eq!(provider.get_max_input_tokens("MiniMax-M3.1-Flash-Preview"), 1_000_000);
	assert_eq!(provider.get_max_input_tokens("MiniMax-M3.1-Flash"), 1_000_000);
	for model in [
		"MiniMax-M2.7",
		"MiniMax-M2.7-highspeed",
		"MiniMax-M2.5",
		"MiniMax-M2.1",
		"MiniMax-M2",
	] {
		assert_eq!(provider.get_max_input_tokens(model), 204_800, "{model}");
	}
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
fn test_m3_1_flash_preview_has_no_per_token_cost() {
	// Token Plan only: must not fall through to the MiniMax-M3 substring row
	let cost = calculate_minimax_cost("MiniMax-M3.1-Flash-Preview", 600_000, 1_000_000, 1_000, 1_000);
	assert_eq!(cost, Some(0.0));
}

#[test]
fn test_m3_1_flash_single_tier_pricing() {
	// Must not fall through to the MiniMax-M3 row, and has no >512k tier
	let cost = calculate_minimax_cost("MiniMax-M3.1-Flash", 600_000, 1_000_000, 0, 1_000_000).unwrap();
	assert!((cost - (0.09 + 0.60 + 0.015)).abs() < 1e-9);
}

#[test]
fn test_effort_for_m3_1_flash_ga() {
	assert_eq!(effort_value("MiniMax-M3.1-Flash", ReasoningEffort::None), Some("low"));
	assert_eq!(effort_value("MiniMax-M3.1-Flash", ReasoningEffort::Max), Some("max"));
	assert_eq!(thinking_value("MiniMax-M3.1-Flash", ReasoningEffort::High), None);
}

#[test]
fn test_effort_only_for_m3_1_flash_preview() {
	let model = "MiniMax-M3.1-Flash-Preview";
	assert_eq!(effort_value(model, ReasoningEffort::None), Some("low"));
	assert_eq!(effort_value(model, ReasoningEffort::Low), Some("low"));
	assert_eq!(effort_value(model, ReasoningEffort::Medium), Some("medium"));
	assert_eq!(effort_value(model, ReasoningEffort::High), Some("high"));
	assert_eq!(effort_value(model, ReasoningEffort::XHigh), Some("xhigh"));
	assert_eq!(effort_value(model, ReasoningEffort::Max), Some("max"));

	assert_eq!(effort_value("MiniMax-M3", ReasoningEffort::High), None);
	assert_eq!(effort_value("MiniMax-M2.7", ReasoningEffort::High), None);
}

#[test]
fn test_m3_thinking_enabled_by_any_effort_except_none() {
	let adaptive = Some(serde_json::json!({ "type": "adaptive" }));
	for effort in [
		ReasoningEffort::Low,
		ReasoningEffort::Medium,
		ReasoningEffort::High,
		ReasoningEffort::XHigh,
		ReasoningEffort::Max,
	] {
		assert_eq!(thinking_value("MiniMax-M3", effort), adaptive);
	}
	// Thinking is off by default on M3, so None sends nothing
	assert_eq!(thinking_value("MiniMax-M3", ReasoningEffort::None), None);

	// Flash-Preview always thinks; M2.x cannot toggle thinking
	assert_eq!(
		thinking_value("MiniMax-M3.1-Flash-Preview", ReasoningEffort::High),
		None
	);
	assert_eq!(thinking_value("MiniMax-M2.7", ReasoningEffort::High), None);
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
	assert!(provider.supports_vision("MiniMax-M3.1-Flash-Preview"));
	assert!(provider.supports_video("MiniMax-M3.1-Flash-Preview"));
	assert!(provider.supports_vision("MiniMax-M3.1-Flash"));
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

#[test]
fn tool_images_stay_inside_the_matching_tool_result() {
	for text in ["caption", ""] {
		let converted = convert_messages(&[shared::tests::tool_image_message(text)]);
		let value = serde_json::to_value(&converted).unwrap();
		let result = &value[0]["content"][0];
		assert_eq!(result["type"], "tool_result");
		assert_eq!(result["tool_use_id"], "call_image");
		let offset = usize::from(!text.is_empty());
		assert_eq!(result["content"][offset]["source"]["data"], "aW1hZ2U=");
		assert_eq!(
			result["content"][offset + 1]["source"]["url"],
			"https://example.com/screenshot.jpg"
		);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_splits_text_thinking_tools_and_cache() {
	let body = r#"{"id":"mm-1","content":[{"type":"text","text":"final answer"},{"type":"thinking","thinking":"pondering deeply"},{"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Oslo"}}],"usage":{"input_tokens":10,"output_tokens":30,"cache_read_input_tokens":4,"cache_creation_input_tokens":2},"stop_reason":"tool_use"}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("MINIMAX_API_URL", mock.url.clone());

	let provider = MinimaxProvider::new();
	let params = ChatCompletionParams::new(&[], "MiniMax-M2.7", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	// Only text parts become content; thinking rides its own block.
	assert_eq!(response.content, "final answer");
	assert_eq!(
		response.thinking.as_ref().expect("thinking present").content,
		"pondering deeply"
	);

	let calls = response.tool_calls.expect("tool calls present");
	assert_eq!(calls.len(), 1);
	assert_eq!(calls[0].name, "get_weather");
	assert_eq!(calls[0].arguments["city"], "Oslo");

	// input stays clean; cache read/write are separate; thinking is carved out of output.
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 10);
	assert_eq!(usage.cache_read_tokens, 4);
	assert_eq!(usage.cache_write_tokens, 2);
	// "pondering deeply" is 16 chars, estimated at 4 tokens and carved out of the 30.
	assert_eq!(usage.output_tokens, 26);
	assert_eq!(usage.reasoning_tokens, 4);
	// MiniMax-M2.7 per 1M: 0.30 input, 0.375 cache write, 0.06 cache read, 1.20
	// output. Thinking bills as output, so all 30 completion tokens are priced.
	let expected = (10.0 * 0.30 + 2.0 * 0.375 + 4.0 * 0.06 + 30.0 * 1.20) / 1_000_000.0;
	assert!(
		(usage.cost.expect("cost computed from the pricing table") - expected).abs() < 1e-12,
		"unexpected cost: {:?}",
		usage.cost
	);
	assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));

	// Request targeted the mock with the priced model.
	let requests = mock.requests();
	let req_body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(req_body["model"], "MiniMax-M2.7");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	let _env = EnvGuard::new().set("MINIMAX_API_URL", mock.url.clone());

	let provider = MinimaxProvider::new();
	let params = ChatCompletionParams::new(&[], "MiniMax-M2.7", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("MiniMax API error"), "unexpected error: {err}");
}
