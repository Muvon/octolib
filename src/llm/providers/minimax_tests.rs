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
#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_splits_text_thinking_tools_and_cache() {
	let body = r#"{"id":"mm-1","content":[{"type":"text","text":"final answer"},{"type":"thinking","thinking":"pondering deeply"},{"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Oslo"}}],"usage":{"input_tokens":10,"output_tokens":30,"cache_read_input_tokens":4,"cache_creation_input_tokens":2},"stop_reason":"tool_use"}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("MINIMAX_API_URL", format!("{}/v1/messages", mock.base_url()));

	let provider = MinimaxProvider::new();
	let params = ChatCompletionParams::new(&[], "MiniMax-M2.7", 10).with_api_key("mock-key".to_owned());
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
	for field in ["temperature", "top_p", "top_k"] {
		assert!(req_body.get(field).is_none(), "{field} must not be sent");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_surfaces_http_error() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		400,
		r#"{"error":{"message":"bad request"}}"#.to_string(),
	)]);
	let _env = EnvGuard::new().set("MINIMAX_API_URL", format!("{}/v1/messages", mock.base_url()));

	let provider = MinimaxProvider::new();
	let params = ChatCompletionParams::new(&[], "MiniMax-M2.7", 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("minimax API error"), "unexpected error: {err}");
}

#[test]
fn thinking_knobs_follow_the_model() {
	let mut request = serde_json::json!({});
	configure_thinking("MiniMax-M3", None, &mut request);
	configure_thinking("MiniMax-M2.7", Some(ReasoningEffort::High), &mut request);
	assert_eq!(
		request,
		serde_json::json!({}),
		"no effort, or a model without knobs, sends nothing"
	);

	let mut request = serde_json::json!({});
	configure_thinking("MiniMax-M3", Some(ReasoningEffort::Low), &mut request);
	assert_eq!(request, serde_json::json!({ "thinking": { "type": "adaptive" } }));

	let mut request = serde_json::json!({});
	configure_thinking("MiniMax-M3.1-Flash", Some(ReasoningEffort::Max), &mut request);
	assert_eq!(request, serde_json::json!({ "output_config": { "effort": "max" } }));
}

#[test]
fn openai_thinking_follows_the_documented_switches() {
	// With no effort requested nothing is sent: the endpoint thinks by default.
	for model in ["MiniMax-M3", "MiniMax-M3.1-Flash", "MiniMax-M2.7"] {
		let mut request = serde_json::json!({});
		openai_thinking(model, None, &mut request);
		assert_eq!(request, serde_json::json!({}), "{model}");
	}

	// M3.1-Flash tunes depth through reasoning_effort; `none` is a 400 there.
	for (effort, expected) in [
		(ReasoningEffort::None, "low"),
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "xhigh"),
		(ReasoningEffort::Max, "max"),
	] {
		let mut request = serde_json::json!({});
		openai_thinking("MiniMax-M3.1-Flash", Some(effort), &mut request);
		assert_eq!(request, serde_json::json!({ "reasoning_effort": expected }));
	}

	// M3 thinks adaptively unless disabled, and ignores reasoning_effort.
	let mut request = serde_json::json!({});
	openai_thinking("MiniMax-M3", Some(ReasoningEffort::High), &mut request);
	assert_eq!(request, serde_json::json!({}));
	let mut request = serde_json::json!({});
	openai_thinking("MiniMax-M3", Some(ReasoningEffort::None), &mut request);
	assert_eq!(request, serde_json::json!({ "thinking": { "type": "disabled" } }));

	// M2.x documents neither knob.
	let mut request = serde_json::json!({});
	openai_thinking("MiniMax-M2.7", Some(ReasoningEffort::Max), &mut request);
	assert_eq!(request, serde_json::json!({}));
}

#[test]
fn the_default_endpoint_follows_the_mode() {
	// (MINIMAX_API_MODE, MINIMAX_API_URL, the protocol and URL that follow)
	for (mode, api_url, want_mode, want_url) in [
		(None, None, ApiMode::Anthropic, MINIMAX_API_URL),
		(Some(""), None, ApiMode::Anthropic, MINIMAX_API_URL),
		(Some("anthropic"), None, ApiMode::Anthropic, MINIMAX_API_URL),
		(Some(" OpenAI "), None, ApiMode::OpenAi, MINIMAX_CHAT_API_URL),
		// A URL naming no endpoint was always served by Messages, and goes
		// out exactly as configured.
		(
			None,
			Some("https://proxy.example/v4"),
			ApiMode::Anthropic,
			"https://proxy.example/v4",
		),
	] {
		assert_eq!(
			select_endpoint(mode, api_url).expect("selected"),
			(want_mode, want_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
	// An empty URL is no URL.
	assert_eq!(
		select_endpoint(Some("openai"), Some("")).expect("selected"),
		(ApiMode::OpenAi, MINIMAX_CHAT_API_URL.to_owned())
	);
	assert_eq!(
		select_endpoint(None, Some("https://proxy.example/v4")).expect("selected"),
		(ApiMode::Anthropic, "https://proxy.example/v4".to_owned())
	);
}

#[tokio::test]
#[serial_test::serial]
async fn openai_mode_speaks_chat_completions_on_the_configured_url() {
	use crate::llm::types::Message;
	let body = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"MiniMax-M3","choices":[{"index":0,"message":{"role":"assistant","content":"chat answer","reasoning_content":"pondering"},"finish_reason":"stop"}],"usage":{"total_tokens":19,"prompt_tokens":12,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":4}}}"#;
	let messages_body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"messages answer"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":7}}"#;
	// (MINIMAX_API_MODE, path under the mock's origin, answer, whether Messages is spoken)
	let cases = [
		("", "/api/prefix/v1/messages", "messages answer", true),
		("", "/api/prefix/chat/completions", "chat answer", false),
		// A URL naming no endpoint was always served by Messages.
		("", "/api/prefix/v4", "messages answer", true),
		("openai", "/api/prefix/v1/messages", "chat answer", false),
		("anthropic", "/api/prefix/chat/completions", "messages answer", true),
	];
	for (mode, path, answer, is_messages) in cases {
		let mock = crate::test_support::mock_http::MockHttp::start(vec![(
			200,
			if is_messages { messages_body } else { body }.to_string(),
		)]);
		let _env = EnvGuard::new()
			.set("MINIMAX_API_URL", format!("{}{path}", mock.base_url()))
			.set("MINIMAX_API_MODE", mode);

		let provider = MinimaxProvider::new();
		let params =
			ChatCompletionParams::new(&[Message::user("hi")], "MiniMax-M3", 100).with_api_key("mock-key".to_owned());
		let response = provider.chat_completion(params).await.expect("completion");
		assert_eq!(response.content, answer);
		let request = &mock.requests()[0];
		assert_eq!(request.method, "POST");
		assert_eq!(request.path, path, "the URL is not rewritten");
		assert_eq!(request.header("authorization"), Some("Bearer mock-key"));
		assert_eq!(
			request.header("anthropic-version").is_some(),
			is_messages,
			"path {path}"
		);
		let sent: serde_json::Value = serde_json::from_str(&request.body).expect("request json");
		assert_eq!(sent["model"], "MiniMax-M3");
		if !is_messages {
			// Usage: 12 prompt of which 4 cached, 7 completion, some of it the
			// returned reasoning; MiniMax-M3 per 1M: 0.30 input, 0.06 cache read,
			// 1.20 output on every completion token, billed from PRICING.
			assert_eq!(
				response.thinking.as_ref().map(|thinking| thinking.content.as_str()),
				Some("pondering")
			);
			let usage = response.exchange.usage.expect("usage");
			assert_eq!(
				(
					usage.input_tokens,
					usage.cache_read_tokens,
					usage.output_tokens + usage.reasoning_tokens
				),
				(8, 4, 7)
			);
			let expected = (8.0 * 0.30 + 4.0 * 0.06 + 7.0 * 1.20) / 1_000_000.0;
			assert!((usage.cost.expect("priced") - expected).abs() < 1e-12, "path {path}");
			assert_eq!(sent["messages"][0]["role"], "user");
			assert_eq!(sent["messages"][0]["content"], "hi");
			assert_eq!(sent["stream"], true, "the endpoint documents stream_options");
			assert_eq!(sent["stream_options"]["include_usage"], true);
			for field in [
				"thinking",
				"reasoning_effort",
				"response_format",
				"parallel_tool_calls",
				"system",
			] {
				assert!(sent.get(field).is_none(), "{field} must not be sent");
			}
		}
	}
}

#[tokio::test]
#[serial_test::serial]
async fn messages_keep_the_fallback_prompt_of_a_conversation_without_a_system_turn() {
	use crate::llm::types::Message;
	let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string()), (200, body.to_string())]);
	let _env = EnvGuard::new()
		.set("MINIMAX_API_URL", format!("{}/v1/messages", mock.base_url()))
		.set("MINIMAX_API_MODE", "");

	let provider = MinimaxProvider::new();
	provider
		.chat_completion(
			ChatCompletionParams::new(&[Message::user("hi")], "MiniMax-M3", 10).with_api_key("k".to_owned()),
		)
		.await
		.expect("completion");
	provider
		.chat_completion(
			ChatCompletionParams::new(&[Message::system("Be terse."), Message::user("hi")], "MiniMax-M3", 10)
				.with_api_key("k".to_owned()),
		)
		.await
		.expect("completion");
	let first: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	let second: serde_json::Value = serde_json::from_str(&mock.requests()[1].body).expect("request json");
	assert_eq!(first["system"], "You are a helpful assistant.");
	assert_eq!(second["system"], "Be terse.");
}

#[tokio::test]
#[serial_test::serial]
async fn an_unknown_mode_is_refused_before_any_request() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, "{}".to_string())]);
	let _env = EnvGuard::new()
		.set("MINIMAX_API_URL", mock.url.clone())
		.set("MINIMAX_API_MODE", "claude");

	let provider = MinimaxProvider::new();
	let params = ChatCompletionParams::new(&[], "MiniMax-M3", 10).with_api_key("mock-key".to_owned());
	let error = provider.chat_completion(params).await.expect_err("refused");
	assert!(
		error.to_string().contains("MINIMAX_API_MODE must be"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "nothing is sent");
}

#[tokio::test]
#[serial_test::serial]
async fn m3_reads_inline_images_and_m2_refuses_them() {
	use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};
	let image = Message::user("what is this?").with_images(vec![ImageAttachment {
		data: ImageData::Base64("aW1hZ2U=".into()),
		media_type: "image/png".into(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}]);
	let body = r#"{"id":"mm-1","content":[{"type":"text","text":"stripes"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":2}}"#;
	let provider = MinimaxProvider::new();

	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("MINIMAX_API_URL", format!("{}/v1/messages", mock.base_url()));
	let params =
		ChatCompletionParams::new(std::slice::from_ref(&image), "MiniMax-M3", 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("M3 takes images");
	assert_eq!(response.content, "stripes");
	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["messages"][0]["content"][0]["type"], "text");
	assert_eq!(request["messages"][0]["content"][1]["type"], "image");
	assert_eq!(request["messages"][0]["content"][1]["source"]["data"], "aW1hZ2U=");

	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let params = ChatCompletionParams::new(&[image], "MiniMax-M2.7", 10).with_api_key("mock-key".to_owned());
	let error = provider
		.chat_completion(params)
		.await
		.expect_err("M2.x has no image input");
	assert!(
		error
			.to_string()
			.contains("minimax model MiniMax-M2.7 does not accept image input"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "a refused request is not sent");
}
