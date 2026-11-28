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
use crate::llm::types::Message;
use crate::test_support::env::EnvGuard;

#[test]
fn test_short_names_resolve_consistently() {
	let provider = TinkerProvider::new();
	// Absolute rates, not "same as whatever the alias resolves to" — comparing
	// the alias against its own resolution is true for any mapping, including
	// the wrong one. These are the Serverless Inference rows of the models page.
	for (alias, canonical, input, output, cached) in [
		(
			"inkling",
			"thinkingmachines/Inkling:peft:262144:sampling-nvfp4",
			1.00,
			4.05,
			0.17,
		),
		(
			"INKLING",
			"thinkingmachines/Inkling:peft:262144:sampling-nvfp4",
			1.00,
			4.05,
			0.17,
		),
		(
			"inkling-small",
			"thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4",
			0.30,
			1.20,
			0.06,
		),
		(
			"INKLING-SMALL",
			"thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4",
			0.30,
			1.20,
			0.06,
		),
	] {
		assert_eq!(resolve_model(alias), canonical);
		assert!(provider.supports_model(alias));
		assert_eq!(provider.get_max_input_tokens(alias), 262_144);
		let pricing = provider.get_model_pricing(alias).unwrap();
		assert_eq!(pricing.input_price_per_1m, input);
		assert_eq!(pricing.output_price_per_1m, output);
		assert_eq!(pricing.cache_read_price_per_1m, cached);
	}
}

/// The bare ids are the TRAINING table's spelling of the same two models and
/// keep that table's rate and window. Naming one explicitly still means the
/// training tier — the alias fix must not collapse the two.
#[test]
fn test_bare_ids_keep_the_training_tier() {
	let provider = TinkerProvider::new();
	for (model, input, output, context) in [
		("thinkingmachines/Inkling", 1.87, 4.68, 65_536),
		("thinkingmachines/Inkling-Small", 0.58, 1.44, 65_536),
	] {
		assert_eq!(resolve_model(model), model);
		let pricing = provider.get_model_pricing(model).unwrap();
		assert_eq!(pricing.input_price_per_1m, input);
		assert_eq!(pricing.output_price_per_1m, output);
		assert_eq!(provider.get_max_input_tokens(model), context);
	}
}

#[test]
fn test_qualified_model_ids_are_preserved() {
	for model in [
		"thinkingmachines/Inkling:peft:262144:sampling-nvfp4",
		"thinkingmachines/Inkling-Small",
		"moonshotai/Kimi-K2.6",
		"tinker://CaseSensitive-ID:train:0/sampler_weights/000080",
		"unknown-model",
	] {
		assert_eq!(resolve_model(model), model);
	}
}

#[test]
fn test_supports_model() {
	let provider = TinkerProvider::new();
	// Known Tinker IDs (colons intact — the factory splits only the first)
	assert!(provider.supports_model("thinkingmachines/Inkling"));
	assert!(provider.supports_model("thinkingmachines/Inkling-Small:peft:262144"));
	assert!(provider.supports_model("thinkingmachines/Inkling:peft:262144:sampling-nvfp4"));
	assert!(provider.supports_model("moonshotai/Kimi-K2.6"));
	// Arbitrary sampler checkpoint paths
	assert!(provider.supports_model("tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_pricing_ordering() {
	let provider = TinkerProvider::new();

	// Serverless variant wins over the `:peft:` training ID
	let serverless = provider
		.get_model_pricing("thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4")
		.unwrap();
	assert_eq!(serverless.input_price_per_1m, 0.30);
	assert_eq!(serverless.output_price_per_1m, 1.20);
	assert_eq!(serverless.cache_read_price_per_1m, 0.06);

	// 256K `:peft:` variant wins over the base ID
	let peft = provider
		.get_model_pricing("thinkingmachines/Inkling:peft:262144")
		.unwrap();
	assert_eq!(peft.input_price_per_1m, 3.74);
	assert_eq!(peft.output_price_per_1m, 9.36);

	// Base IDs
	let base = provider.get_model_pricing("thinkingmachines/Inkling").unwrap();
	assert_eq!(base.input_price_per_1m, 1.87);
	assert_eq!(base.output_price_per_1m, 4.68);

	// Inkling-Small must not be captured by the Inkling pattern
	let small = provider.get_model_pricing("thinkingmachines/Inkling-Small").unwrap();
	assert_eq!(small.input_price_per_1m, 0.58);
	assert_eq!(small.output_price_per_1m, 1.44);

	// Inkling-2-Small must not be captured by the Inkling pattern on any of its ids
	for (model, context) in [
		("thinkingmachines/Inkling-2-Small:peft:262144:sampling-nvfp4", 262_144),
		("thinkingmachines/Inkling-2-Small:peft:262144", 262_144),
		("thinkingmachines/Inkling-2-Small", 65_536),
	] {
		let pricing = provider.get_model_pricing(model).unwrap();
		assert_eq!(pricing.input_price_per_1m, 0.30, "{model}");
		assert_eq!(pricing.output_price_per_1m, 1.00, "{model}");
		assert_eq!(pricing.cache_read_price_per_1m, 0.03, "{model}");
		assert_eq!(provider.get_max_input_tokens(model), context, "{model}");
	}

	// Inkling-2-Mini must not be captured by the Inkling pattern either
	for (model, context) in [
		("thinkingmachines/Inkling-2-Mini:peft:262144", 262_144),
		("thinkingmachines/Inkling-2-Mini", 65_536),
	] {
		let pricing = provider.get_model_pricing(model).unwrap();
		assert_eq!(pricing.input_price_per_1m, 0.15, "{model}");
		assert_eq!(pricing.output_price_per_1m, 0.50, "{model}");
		assert_eq!(pricing.cache_read_price_per_1m, 0.015, "{model}");
		assert_eq!(provider.get_max_input_tokens(model), context, "{model}");
	}

	// `-base` suffix must not be captured by the instruct pattern
	let base_suffix = provider.get_model_pricing("Qwen/Qwen3.5-9B-Base").unwrap();
	assert_eq!(base_suffix.input_price_per_1m, 0.66);

	// Unknown checkpoint paths carry no pricing
	assert!(provider
		.get_model_pricing("tinker://0034d8c9-0a88-52a9-b2b7-bce7cb1e6fef:train:0/sampler_weights/000080")
		.is_none());
}

#[test]
fn test_context_windows() {
	let provider = TinkerProvider::new();
	assert_eq!(provider.get_max_input_tokens("thinkingmachines/Inkling"), 65_536);
	assert_eq!(
		provider.get_max_input_tokens("thinkingmachines/Inkling:peft:262144:sampling-nvfp4"),
		262_144
	);
	assert_eq!(provider.get_max_input_tokens("moonshotai/Kimi-K2.6"), 32_768);
	assert_eq!(
		provider.get_max_input_tokens("moonshotai/Kimi-K2.6:peft:131072"),
		131_072
	);
	// Unknown checkpoints fall back to the library default
	assert_eq!(provider.get_max_input_tokens("tinker://some/checkpoint"), 262_144);
}

/// The checkpoints added in the October 2026 model watch, with the rates and
/// windows published on https://tinker-docs.thinkingmachines.ai/tinker/models/
/// The official mixed-case ids resolve to the lowercase table rows.
#[test]
fn test_october_2026_checkpoints_use_the_published_rates() {
	let provider = TinkerProvider::new();
	for (model, input, output, cache_read, context) in [
		(
			"nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
			0.57,
			1.44,
			0.114,
			65_536,
		),
		(
			"nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:peft:262144",
			0.57,
			1.44,
			0.114,
			262_144,
		),
		("zai-org/GLM-5.3-Flash:peft:262144", 0.84, 2.11, 0.168, 262_144),
		("deepseek-ai/DeepSeek-V4.1-Flash:peft:262144", 0.60, 3.60, 0.12, 262_144),
		(
			"deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144",
			0.60,
			3.60,
			0.12,
			262_144,
		),
	] {
		let pricing = provider
			.get_model_pricing(model)
			.unwrap_or_else(|| panic!("{model} must have a pricing entry"));
		assert_eq!(pricing.input_price_per_1m, input, "{model}");
		assert_eq!(pricing.output_price_per_1m, output, "{model}");
		assert_eq!(pricing.cache_write_price_per_1m, input, "{model}");
		assert_eq!(pricing.cache_read_price_per_1m, cache_read, "{model}");
		assert_eq!(provider.get_max_input_tokens(model), context, "{model}");
	}
}

#[test]
#[serial_test::serial]
fn test_conservative_capabilities() {
	let provider = TinkerProvider::new();
	let _env = EnvGuard::new().remove("TINKER_API_MODE").remove("TINKER_API_URL");
	assert_eq!(provider.name(), "tinker");
	assert!(provider.supports_caching("thinkingmachines/Inkling"));
	assert!(!provider.supports_structured_output("thinkingmachines/Inkling"));
	assert!(!provider.enforces_response_schema("thinkingmachines/Inkling"));
}

/// Images belong to Messages only; the cache lifetime was measured on both
/// routes and is the same.
#[test]
#[serial_test::serial]
fn vision_follows_the_route_and_cache_lifetime_does_not() {
	let provider = TinkerProvider::new();
	let hour = Some(std::time::Duration::from_secs(3600));
	let _env = EnvGuard::new().remove("TINKER_API_MODE").remove("TINKER_API_URL");
	// Messages is the default.
	assert!(provider.supports_vision("inkling"));
	assert!(provider.supports_vision("thinkingmachines/Inkling-Small:peft:262144:sampling-nvfp4"));
	assert!(!provider.supports_vision("openai/gpt-oss-120b"));
	assert_eq!(provider.cache_lifetime("openai/gpt-oss-120b", false), hour);

	let _env = _env.set("TINKER_API_MODE", "openai");
	assert!(!provider.supports_vision("inkling"));
	assert_eq!(provider.cache_lifetime("openai/gpt-oss-120b", false), hour);
}

#[test]
fn endpoint_selection_follows_mode_then_url_path() {
	assert_eq!(
		select_endpoint(None, None).expect("selected"),
		(ApiMode::Anthropic, TINKER_MESSAGES_API_URL.to_owned())
	);
	assert_eq!(
		select_endpoint(Some("openai"), None).expect("selected"),
		(ApiMode::OpenAi, TINKER_CHAT_API_URL.to_owned())
	);
	let chat = "https://proxy.example/v1/chat/completions";
	assert_eq!(
		select_endpoint(None, Some(chat)).expect("selected"),
		(ApiMode::OpenAi, chat.to_owned())
	);
	let messages = "https://proxy.example/v1/messages";
	assert_eq!(
		select_endpoint(None, Some(messages)).expect("selected"),
		(ApiMode::Anthropic, messages.to_owned())
	);
	// A URL naming neither endpoint keeps Chat Completions, as before the option.
	let custom = "https://proxy.example/inference";
	assert_eq!(
		select_endpoint(None, Some(custom)).expect("selected"),
		(ApiMode::OpenAi, custom.to_owned())
	);
	// An explicit mode wins over the path, and the URL is not rewritten.
	assert_eq!(
		select_endpoint(Some("anthropic"), Some(chat)).expect("selected"),
		(ApiMode::Anthropic, chat.to_owned())
	);
	// Empty variables are unset ones.
	assert_eq!(
		select_endpoint(Some(""), Some("")).expect("selected"),
		(ApiMode::Anthropic, TINKER_MESSAGES_API_URL.to_owned())
	);
	let error = select_endpoint(Some("claude"), None).expect_err("refused");
	assert!(
		error.to_string().contains("TINKER_API_MODE must be"),
		"unexpected error: {error}"
	);
}

#[test]
fn the_messages_thinking_hook_follows_the_probed_ladders() {
	let effort = |model: &str, effort| {
		let mut body = serde_json::json!({});
		messages_thinking(model, Some(effort), &mut body);
		body
	};
	let inkling = "thinkingmachines/Inkling:peft:262144:sampling-nvfp4";
	for (level, want) in [
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "xhigh"),
		(ReasoningEffort::Max, "xhigh"),
	] {
		assert_eq!(effort(inkling, level)["output_config"]["effort"], want, "{level:?}");
	}
	assert_eq!(
		effort(inkling, ReasoningEffort::None),
		serde_json::json!({ "thinking": { "type": "disabled" } })
	);

	// gpt-oss answers 400 to xhigh, max and disabled thinking.
	for (level, want) in [
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "high"),
		(ReasoningEffort::Max, "high"),
	] {
		assert_eq!(
			effort("openai/gpt-oss-120b", level)["output_config"]["effort"],
			want,
			"{level:?}"
		);
	}
	assert_eq!(
		effort("openai/gpt-oss-120b", ReasoningEffort::None),
		serde_json::json!({})
	);

	// Unprobed families send nothing: Nemotron takes a boolean on Chat Completions.
	for level in [ReasoningEffort::None, ReasoningEffort::High] {
		assert_eq!(
			effort("nvidia/nvidia-nemotron-3-super-120b-a12b-bf16", level),
			serde_json::json!({}),
			"{level:?}"
		);
	}

	let mut unset = serde_json::json!({});
	messages_thinking(inkling, None, &mut unset);
	assert_eq!(unset, serde_json::json!({}), "an unset effort sends nothing");
}

/// Usage as the live endpoint reported it on 2026-11-28: the uncached prompt in
/// `cache_creation_input_tokens`, `input_tokens` 0.
const MESSAGES_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"bands","signature":""},{"type":"text","text":"red green blue"}],"stop_reason":"end_turn","usage":{"input_tokens":0,"cache_creation_input_tokens":1000000,"cache_read_input_tokens":1000000,"output_tokens":1000000}}"#;
/// Inkling serverless at 1M uncached input, 1M cached input and 1M output:
/// $1.00 + $0.17 + $4.05.
const INKLING_MILLION_COST: f64 = 5.22;

fn png_message() -> Message {
	use crate::llm::types::{ImageAttachment, ImageData, SourceType};
	Message::user("what colours?").with_images(vec![ImageAttachment {
		data: ImageData::Base64("aW1hZ2U=".into()),
		media_type: "image/png".into(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}])
}

#[tokio::test]
#[serial_test::serial]
async fn messages_is_the_default_protocol_with_x_api_key() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_BODY.to_string())]);
	let path = "/services/tinker-prod/anthropic/api/v1/messages";
	let _env = EnvGuard::new()
		.set("TINKER_API_URL", format!("{}{path}", mock.base_url()))
		.set("TINKER_API_MODE", "");

	let params = ChatCompletionParams::new(&[png_message()], "inkling", 100)
		.with_api_key("mock-key".to_owned())
		.with_reasoning_effort(ReasoningEffort::None);
	let response = TinkerProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	assert_eq!(response.content, "red green blue");
	let request = &mock.requests()[0];
	assert_eq!(request.path, path, "the URL is not rewritten");
	assert_eq!(request.header("x-api-key"), Some("mock-key"));
	assert_eq!(request.header("authorization"), None);

	let sent: serde_json::Value = serde_json::from_str(&request.body).expect("request json");
	assert_eq!(sent["model"], "thinkingmachines/Inkling:peft:262144:sampling-nvfp4");
	assert_eq!(sent["stream"], true);
	assert_eq!(sent["messages"][0]["content"][1]["type"], "image");
	assert_eq!(sent["messages"][0]["content"][1]["source"]["type"], "base64");
	assert_eq!(sent["thinking"]["type"], "disabled");

	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(usage.input_tokens, usage.cache_write_tokens, usage.cache_read_tokens),
		(0, 1_000_000, 1_000_000)
	);
	let cost = usage.cost.expect("cost");
	assert!((cost - INKLING_MILLION_COST).abs() < 1e-9, "cost {cost}");
}

#[tokio::test]
#[serial_test::serial]
async fn messages_refuse_images_for_models_outside_inkling_before_any_request() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set("TINKER_API_URL", format!("{}/v1/messages", mock.base_url()))
		.remove("TINKER_API_MODE");

	let error = TinkerProvider::new()
		.chat_completion(
			ChatCompletionParams::new(&[png_message()], "openai/gpt-oss-120b", 10).with_api_key("k".to_owned()),
		)
		.await
		.expect_err("gpt-oss takes no images on Messages");
	assert!(
		error.to_string().contains("does not accept image input"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "a refused request is not sent");
}

#[tokio::test]
#[serial_test::serial]
async fn request_omits_sampling_fields() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("ok"),
	)]);
	let _env = EnvGuard::new()
		.set("TINKER_API_URL", mock.url.clone())
		.remove("TINKER_API_MODE");
	let params = ChatCompletionParams::new(&[], "inkling", 10).with_api_key("mock-key".to_owned());
	TinkerProvider::new()
		.chat_completion(params)
		.await
		.expect("mock completion");
	let body: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	for field in ["temperature", "top_p", "top_k"] {
		assert!(body.get(field).is_none(), "{field} must not be sent");
	}
}
