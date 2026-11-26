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
fn test_glm_5_3_flashx_legal_pricing() {
	let provider = ZaiProvider::new();
	assert!(provider.supports_model("glm-5.3-flashx-legal"));

	// GLM-5.3-FlashX-Legal: $0.37 input, $1.25 output (same list rates as
	// glm-5.3-flashx, so this pins the price, not the row order)
	let cost = calculate_cost("glm-5.3-flashx-legal", 1_000_000, 0, 1_000_000);
	assert!((cost.unwrap() - 1.62).abs() < 0.01); // 0.37 + 1.25

	// Cached input: $0.075
	let cost = calculate_cost("glm-5.3-flashx-legal", 0, 1_000_000, 0);
	assert!((cost.unwrap() - 0.075).abs() < 0.0001);
}

#[test]
fn test_glm_5_4_placeholder_pricing_and_capabilities() {
	let provider = ZaiProvider::new();
	assert!(provider.supports_model("glm-5.4"));

	// Placeholder at GLM-5.3 rates until Z.ai publishes a per-token price
	let cost = calculate_cost("glm-5.4", 1_000_000, 0, 1_000_000);
	assert!((cost.unwrap() - 5.80).abs() < 0.01); // 1.40 + 4.40
	let cost = calculate_cost("glm-5.4", 0, 1_000_000, 0);
	assert!((cost.unwrap() - 0.26).abs() < 0.0001);

	assert!(provider.supports_vision("glm-5.4"));
	// The adapter delivers no video on either route, whatever the model lists.
	assert!(!provider.supports_video("glm-5.4"));
	assert_eq!(provider.get_max_input_tokens("glm-5.4"), 1_000_000);
}

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
	assert_eq!(serialized[1]["content"], r#"{"temperature_c":31,"condition":"sunny"}"#);
	assert!(serialized[1].get("tool_calls_id").is_none());
}

#[test]
fn test_model_support() {
	let provider = ZaiProvider::new();
	assert!(provider.supports_model("glm-5.1"));
	assert!(provider.supports_model("glm-5.3"));
	assert!(provider.supports_model("glm-5.3-flash"));
	assert!(provider.supports_model("glm-5.3-flashx"));
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

	// Test GLM-5.3-FlashX: $0.37 input, $1.25 output — must resolve to its own
	// entry, not the glm-5.3-flash substring match
	let cost = calculate_cost("glm-5.3-flashx", 1_000_000, 0, 1_000_000);
	assert!((cost.unwrap() - 1.62).abs() < 0.01); // 0.37 + 1.25
											   // GLM-5.3-FlashX cached input: $0.075
	let cost = calculate_cost("glm-5.3-flashx", 0, 1_000_000, 0);
	assert!((cost.unwrap() - 0.075).abs() < 0.0001);

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

	let before = SystemTime::UNIX_EPOCH + Duration::from_secs(GLM_5_3_FLASH_PROMO_END_UNIX_SECS - 1);
	let promo = calculate_cost_at("glm-5.3-flash", 1_000_000, 0, 1_000_000, before).unwrap();
	assert!((promo - 0.325).abs() < 0.0001);

	let cutoff = SystemTime::UNIX_EPOCH + Duration::from_secs(GLM_5_3_FLASH_PROMO_END_UNIX_SECS);
	let list = calculate_cost_at("glm-5.3-flash", 1_000_000, 0, 1_000_000, cutoff).unwrap();
	assert!((list - 0.65).abs() < 0.0001);

	// FlashX was never part of the Flash promo: list price even before the cutoff
	let flashx = calculate_cost_at("glm-5.3-flashx", 1_000_000, 0, 1_000_000, before).unwrap();
	assert!((flashx - 1.62).abs() < 0.0001);
}

#[test]
fn test_reasoning_effort_floors_to_glm_5_3_supported_levels() {
	// GLM-5.3 / GLM-5.3-Flash reject anything but low/high/max with a 400.
	assert_eq!(reasoning_effort_value("glm-5.3", ReasoningEffort::Medium), "low");
	assert_eq!(reasoning_effort_value("glm-5.3", ReasoningEffort::XHigh), "high");
	assert_eq!(reasoning_effort_value("glm-5.3-flash", ReasoningEffort::Medium), "low");
	assert_eq!(reasoning_effort_value("GLM-5.3", ReasoningEffort::Max), "max");

	// GLM-5.2 maps intermediate levels itself, so the caller's level is kept.
	assert_eq!(reasoning_effort_value("glm-5.2", ReasoningEffort::Medium), "medium");
	assert_eq!(reasoning_effort_value("glm-5.2", ReasoningEffort::XHigh), "xhigh");
	assert_eq!(reasoning_effort_value("glm-5.2", ReasoningEffort::Max), "max");

	// GLM-5.4 accepted every level live, so the caller's level is kept.
	assert_eq!(reasoning_effort_value("glm-5.4", ReasoningEffort::Medium), "medium");
	assert_eq!(reasoning_effort_value("glm-5.4", ReasoningEffort::XHigh), "xhigh");
}

/// GLM-5.3 cannot stop thinking, so no reasoning runs it at its lowest level;
/// other models get `thinking: disabled`.
#[test]
fn test_no_reasoning_floors_glm_5_3_and_disables_other_models() {
	let enabled = || Some(serde_json::json!({ "type": "enabled" }));
	assert_eq!(
		thinking_fields("glm-5.3", Some(ReasoningEffort::None)),
		(enabled(), Some("low"))
	);
	assert_eq!(
		thinking_fields("glm-5.3-flash", Some(ReasoningEffort::None)),
		(enabled(), Some("low"))
	);
	assert_eq!(
		thinking_fields("glm-5.2", Some(ReasoningEffort::None)),
		(Some(serde_json::json!({ "type": "disabled" })), None)
	);
	assert_eq!(
		thinking_fields("glm-5.3", Some(ReasoningEffort::High)),
		(enabled(), Some("high"))
	);
	assert_eq!(thinking_fields("glm-5.3", None), (None, None));
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
	// GLM-5.3-Flash: native image input; both 5.3 variants have 1M context.
	assert!(provider.supports_vision("glm-5.3-flash"));
	assert!(!provider.supports_video("glm-5.3-flash"));
	assert_eq!(provider.get_max_input_tokens("glm-5.3-flash"), 1_000_000);
	assert_eq!(provider.get_max_input_tokens("glm-5.3"), 1_000_000);
	// GLM-5.3-FlashX shares Flash's modalities and 1M context
	assert!(provider.supports_vision("glm-5.3-flashx"));
	assert!(!provider.supports_video("glm-5.3-flashx"));
	assert_eq!(provider.get_max_input_tokens("glm-5.3-flashx"), 1_000_000);
	// GLM-5 / GLM-5-Turbo: 200K per the model pages
	assert_eq!(provider.get_max_input_tokens("glm-5"), 200_000);
	assert_eq!(provider.get_max_input_tokens("glm-5-turbo"), 200_000);
	// Context windows from the Z.ai models overview
	assert_eq!(provider.get_max_input_tokens("glm-5.2"), 1_000_000);
	assert_eq!(provider.get_max_input_tokens("glm-5.1"), 200_000);
	assert_eq!(provider.get_max_input_tokens("glm-4.6"), 200_000);
	assert_eq!(provider.get_max_input_tokens("glm-4.6v"), 128_000);
	assert_eq!(provider.get_max_input_tokens("glm-4.5-flash"), 200_000);
	assert_eq!(provider.get_max_input_tokens("glm-4.5v"), 64_000);
	assert_eq!(provider.get_max_input_tokens("glm-4.5-air"), 131_072);
	// Non-vision models
	assert!(!provider.supports_vision("glm-5.1"));
	assert!(!provider.supports_vision("glm-5-turbo"));
	assert!(!provider.supports_vision("glm-5"));
	assert!(!provider.supports_vision("glm-4.7"));
	// Video is refused on both routes, so the answer does not depend on the
	// model id at all: empty, unknown, vision and non-vision ids all say no.
	for model in ["", "glm-9-unknown", "glm-5.4", "glm-5.3-flash", "glm-5.1"] {
		assert!(!provider.supports_video(model), "{model} must report no video");
	}
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
	assert_eq!(content[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD");

	// A text-only turn keeps the plain string shape Z.ai expects.
	let plain = convert_messages(&[Message::user("hi")]);
	assert_eq!(plain[0].content, Some(serde_json::json!("hi")));
}

#[test]
fn test_zai_replays_thinking_for_every_model() {
	let provider = ZaiProvider::new();
	assert!(provider.replays_thinking("glm-5.3"));
	assert!(provider.replays_thinking("glm-4.7"));
}

#[test]
fn tool_images_follow_tool_results_without_losing_call_association() {
	let messages = [shared::tests::tool_image_message("")];
	let value = serde_json::to_value(convert_messages(&messages)).unwrap();
	assert_eq!(value[0]["role"], "tool");
	assert_eq!(value[0]["tool_call_id"], "call_image");
	assert_eq!(value[0]["content"], "");
	assert_eq!(value[1]["role"], "user");
	assert_eq!(
		value[1]["content"][1]["image_url"]["url"],
		"data:image/png;base64,aW1hZ2U="
	);
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_separates_cached_input_and_prices_from_table() {
	let body = r#"{"id":"z-1","object":"chat.completion","created":1,"model":"glm-5.3-flash","choices":[{"index":0,"message":{"role":"assistant","content":"zai answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":50,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":20}}}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("ZAI_API_URL", mock.url.clone());

	let provider = ZaiProvider::new();
	let params = ChatCompletionParams::new(&[], "glm-5.3-flash", 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "zai answer");
	let usage = response.exchange.usage.expect("usage present");
	// prompt_tokens is raw: cached reads are split out of clean input.
	assert_eq!(usage.input_tokens, 80);
	assert_eq!(usage.cache_read_tokens, 20);
	assert_eq!(usage.output_tokens, 50);
	assert_eq!(usage.total_tokens, 150);
	assert!(usage.cost.unwrap_or(0.0) > 0.0, "cost computed from the pricing table");

	let requests = mock.requests();
	let req_body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(req_body["model"], "glm-5.3-flash");
	for field in ["do_sample", "temperature", "top_p", "top_k"] {
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
	let _env = EnvGuard::new().set("ZAI_API_URL", mock.url.clone());

	let provider = ZaiProvider::new();
	let params = ChatCompletionParams::new(&[], "glm-5.3-flash", 10).with_api_key("mock-key".to_owned());
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("Z.ai API error"), "unexpected error: {err}");
}

#[test]
fn cache_lifetime_is_the_measured_hour() {
	let provider = ZaiProvider::new();
	assert_eq!(
		provider.cache_lifetime("glm-5.3", false),
		Some(std::time::Duration::from_secs(3600))
	);
}

#[test]
fn the_mode_and_the_url_path_decide_the_protocol() {
	let provider = ZaiProvider::new();
	let selected = |mode: &str, api_url: &str, model: &str| {
		provider
			.select_endpoint(
				(!mode.is_empty()).then_some(mode),
				(!api_url.is_empty()).then_some(api_url),
				model,
			)
			.expect("selected")
	};

	const MESSAGES: &str = "https://api.z.ai/api/anthropic/v1/messages/";
	const CHAT: &str = "https://api.z.ai/api/paas/v4/chat/completions";
	const OPAQUE: &str = "https://plan.example/api/v4";

	// A text-only model: no configuration goes to Messages; the URL's own path
	// decides for a configured one, and an explicit mode decides over both.
	let cases = [
		("", "", ApiMode::Anthropic, ZAI_MESSAGES_API_URL),
		("", MESSAGES, ApiMode::Anthropic, MESSAGES),
		("", CHAT, ApiMode::OpenAi, CHAT),
		// A URL naming no endpoint was always served by Chat Completions.
		("", OPAQUE, ApiMode::OpenAi, OPAQUE),
		("anthropic", "", ApiMode::Anthropic, ZAI_MESSAGES_API_URL),
		(" OpenAI ", "", ApiMode::OpenAi, ZAI_API_URL),
		("anthropic", CHAT, ApiMode::Anthropic, CHAT),
		("openai", MESSAGES, ApiMode::OpenAi, MESSAGES),
		("openai", OPAQUE, ApiMode::OpenAi, OPAQUE),
	];
	for (mode, api_url, want_mode, want_url) in cases {
		assert_eq!(
			selected(mode, api_url, "glm-4.6"),
			(want_mode, want_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}

	for raw in ["claude", "openai2", " "] {
		let error = provider
			.select_endpoint(Some(raw), Some(CHAT), "glm-4.6")
			.expect_err("refused");
		assert!(
			error.to_string().contains("ZAI_API_MODE must be"),
			"unexpected error: {error}"
		);
	}
}

/// The transport is a fact about the model, not about one request: a
/// conversation keeps one protocol from its first turn to its last, and only an
/// explicit `ZAI_API_MODE` or `ZAI_API_URL` moves it.
#[test]
fn a_vision_model_keeps_chat_for_its_whole_conversation() {
	let provider = ZaiProvider::new();
	let default_for = |model: &str| provider.select_endpoint(None, None, model).expect("selected");
	let chat = (ApiMode::OpenAi, ZAI_API_URL.to_owned());
	let messages = (ApiMode::Anthropic, ZAI_MESSAGES_API_URL.to_owned());

	// One conversation, three turns: the text-only turn before an image, the
	// turn that carries it, and the text-only history a compaction leaves
	// behind. All three are the same model, so all three take the same
	// protocol — reading the media out of the turn sent the first and the third
	// to Messages and only the second to Chat.
	for vision_model in ["glm-4.6v", "glm-5v-turbo", "glm-5.3-flash"] {
		assert_eq!(default_for(vision_model), chat, "{vision_model} keeps Chat");
	}

	// A model that reads no media has no Chat route: Messages is its only
	// endpoint, and it refuses an image when the request is built.
	for text_model in ["glm-4.6", "glm-5.3"] {
		assert_eq!(default_for(text_model), messages, "{text_model} stays on Messages");
	}
}

/// An explicit mode or URL is the caller's decision: it moves a model off the
/// protocol it would otherwise keep, in either direction.
#[test]
fn an_explicit_choice_moves_a_vision_model_and_a_text_model() {
	let provider = ZaiProvider::new();
	let selected = |mode: &str, api_url: &str, model: &str| {
		provider
			.select_endpoint(
				(!mode.is_empty()).then_some(mode),
				(!api_url.is_empty()).then_some(api_url),
				model,
			)
			.expect("selected")
	};
	let chat = (ApiMode::OpenAi, ZAI_API_URL.to_owned());
	let messages = (ApiMode::Anthropic, ZAI_MESSAGES_API_URL.to_owned());

	for vision_model in ["glm-4.6v", "glm-5v-turbo"] {
		assert_eq!(
			selected("anthropic", "", vision_model),
			messages,
			"{vision_model} off Chat by mode"
		);
		assert_eq!(
			selected("", "https://plan.example/v1/messages", vision_model),
			(ApiMode::Anthropic, "https://plan.example/v1/messages".to_owned()),
			"{vision_model} off Chat by URL, sent as configured"
		);
		assert_eq!(
			selected("openai", "", vision_model),
			chat,
			"a mode that agrees with the model changes nothing"
		);
	}

	for text_model in ["glm-4.6", "glm-5.3"] {
		assert_eq!(selected("", "", text_model), messages, "{text_model} keeps Messages");
		assert_eq!(selected("openai", "", text_model), chat);
	}
}

#[serial_test::serial]
#[tokio::test]
async fn each_protocol_is_sent_to_its_url_unchanged() {
	use crate::llm::types::Message;
	// (ZAI_API_MODE, path under the mock's origin, answer, whether Messages is spoken)
	let messages_body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"glm-5.3","content":[{"type":"text","text":"messages answer"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}"#;
	let chat_body = r#"{"id":"z-1","object":"chat.completion","created":1,"model":"glm-5.3","choices":[{"index":0,"message":{"role":"assistant","content":"chat answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}"#;
	let cases = [
		("", "/api/prefix/v1/messages", "messages answer", true),
		("", "/api/prefix/chat/completions?x=1", "chat answer", false),
		// A URL naming no endpoint is served as before, not switched to Messages.
		("", "/api/prefix/v4", "chat answer", false),
		// A mode states the protocol of the URL; the URL itself is not touched.
		("anthropic", "/api/prefix/chat/completions", "messages answer", true),
		("openai", "/api/prefix/v1/messages", "chat answer", false),
	];
	for (mode, path, answer, is_messages) in cases {
		let mock = crate::test_support::mock_http::MockHttp::start(vec![(
			200,
			if is_messages { messages_body } else { chat_body }.to_string(),
		)]);
		let _env = EnvGuard::new()
			.set("ZAI_API_URL", format!("{}{path}", mock.base_url()))
			.set("ZAI_API_MODE", mode);

		let provider = ZaiProvider::new();
		let params = ChatCompletionParams::new(&[Message::user("Weather in Paris?")], "glm-5.3", 100)
			.with_api_key("mock-key".to_owned());
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
		let body: serde_json::Value = serde_json::from_str(&request.body).expect("request json");
		if is_messages {
			assert_eq!(body["max_tokens"], 100);
			assert_eq!(body["messages"][0]["content"][0]["text"], "Weather in Paris?");
		} else {
			assert_eq!(body["messages"][0]["content"], "Weather in Paris?");
		}
	}
}

#[tokio::test]
#[serial_test::serial]
async fn a_video_is_refused_on_both_endpoints_instead_of_dropped() {
	use crate::llm::types::{Message, VideoAttachment, VideoData};
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, "{}".to_string())]);
	let _env = EnvGuard::new()
		.set("ZAI_API_URL", mock.url.clone())
		.set("ZAI_API_MODE", "openai");
	let video = Message::user("what happens?").with_videos(vec![VideoAttachment {
		data: VideoData::Url("https://example.com/clip.mp4".to_owned()),
		media_type: "video/mp4".to_owned(),
		source_type: crate::llm::types::SourceType::Url,
		dimensions: None,
		size_bytes: None,
		duration_secs: None,
	}]);
	let provider = ZaiProvider::new();

	let params = ChatCompletionParams::new(std::slice::from_ref(&video), "glm-5.3-flash", 10)
		.with_api_key("mock-key".to_owned());
	let error = provider
		.chat_completion(params)
		.await
		.expect_err("no silent video loss");
	assert!(
		error.to_string().contains("does not accept video input"),
		"unexpected error: {error}"
	);

	// Messages refuses it too, whatever the mode says.
	let _env = EnvGuard::new()
		.set("ZAI_API_URL", format!("{}/v1/messages", mock.base_url()))
		.set("ZAI_API_MODE", "anthropic");
	let params = ChatCompletionParams::new(&[video], "glm-5.3-flash", 10).with_api_key("mock-key".to_owned());
	let error = provider
		.chat_completion(params)
		.await
		.expect_err("no silent video loss");
	assert!(error.to_string().contains("video input"), "unexpected error: {error}");
	assert!(mock.requests().is_empty(), "nothing is sent");
}

#[tokio::test]
#[serial_test::serial]
async fn an_unknown_mode_is_refused_before_any_request() {
	use crate::llm::types::Message;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, "{}".to_string())]);
	let _env = EnvGuard::new()
		.set("ZAI_API_URL", mock.url.clone())
		.set("ZAI_API_MODE", "claude");

	let provider = ZaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("Weather in Paris?")], "glm-5.3", 100)
		.with_api_key("mock-key".to_owned());
	let error = provider.chat_completion(params).await.expect_err("refused");
	assert!(
		error.to_string().contains("ZAI_API_MODE must be"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "nothing is sent");
}
#[tokio::test]
#[serial_test::serial]
async fn messages_endpoint_gets_the_thinking_switch_and_bills_from_the_pricing_table() {
	use crate::llm::types::Message;
	let body = r#"{"id":"msg_1","type":"message","role":"assistant","model":"glm-4.6","content":[{"type":"thinking","thinking":"pondering","signature":"sig"},{"type":"text","text":"Paris: 18C"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"cache_read_input_tokens":4,"output_tokens":20}}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("ZAI_API_URL", format!("{}/v1/messages", mock.base_url()));

	let provider = ZaiProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("Weather in Paris?")], "glm-4.6", 100)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::High);
	let response = provider.chat_completion(params).await.expect("mock completion");
	assert_eq!(response.content, "Paris: 18C");
	assert_eq!(
		response.thinking.as_ref().map(|thinking| thinking.content.as_str()),
		Some("pondering")
	);

	let requests = mock.requests();
	assert_eq!(requests[0].path, "/v1/messages");
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["thinking"], serde_json::json!({ "type": "enabled" }));
	assert_eq!(request["max_tokens"], 100);
	for field in ["reasoning_effort", "stream_options", "response_format"] {
		assert!(request.get(field).is_none(), "{field} is a Chat API field");
	}
	// glm-4.6 per 1M: 0.60 input, 0.11 cache read, 2.20 output; thinking bills as output.
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(
			usage.input_tokens,
			usage.cache_read_tokens,
			usage.billable_output_tokens()
		),
		(10, 4, 20)
	);
	let expected = (10.0 * 0.60 + 4.0 * 0.11 + 20.0 * 2.20) / 1_000_000.0;
	assert!((usage.cost.expect("cost") - expected).abs() < 1e-12);
	// The next case binds its own mock: an unset URL would reach api.z.ai.
	let body = r#"{"id":"msg_2","type":"message","role":"assistant","content":[{"type":"text","text":"Paris: 18C"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}"#;
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("ZAI_API_URL", format!("{}/v1/messages", mock.base_url()));
	let params = ChatCompletionParams::new(&[Message::user("Weather in Paris?")], "glm-4.6", 100)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::None);
	provider.chat_completion(params).await.expect("mock completion");
	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["thinking"], serde_json::json!({ "type": "disabled" }));
}
