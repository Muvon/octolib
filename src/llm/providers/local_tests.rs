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

const CHAT: &str = "http://localhost:1234/api/prefix/chat/completions?x=1";
const MESSAGES: &str = "http://localhost:1234/api/prefix/v1/messages";
const OPAQUE: &str = "http://localhost:8080/v1";

/// The selection with `LOCAL_ANTHROPIC_MODELS` empty: no model is opted in.
fn unlisted(mode: Option<&str>, api_url: Option<&str>) -> Result<(ApiMode, String)> {
	let _env = EnvGuard::new().set("LOCAL_ANTHROPIC_MODELS", "");
	select_local_endpoint(mode, api_url, "llama-3.1-8b")
}

#[test]
#[serial_test::serial]
fn endpoint_follows_the_mode_then_the_url_path_then_the_ollama_default() {
	let cases = [
		(None, None, ApiMode::OpenAi, LOCAL_API_URL),
		(None, Some(""), ApiMode::OpenAi, LOCAL_API_URL),
		(None, Some(CHAT), ApiMode::OpenAi, CHAT),
		(None, Some(MESSAGES), ApiMode::Anthropic, MESSAGES),
		// A URL that names no endpoint was always served as Chat Completions.
		(None, Some(OPAQUE), ApiMode::OpenAi, OPAQUE),
		(Some("anthropic"), None, ApiMode::Anthropic, LOCAL_MESSAGES_URL),
		(Some(" OpenAI "), None, ApiMode::OpenAi, LOCAL_API_URL),
		// A mode states the protocol of the URL; the URL itself is not touched.
		(Some("anthropic"), Some(CHAT), ApiMode::Anthropic, CHAT),
		(Some("openai"), Some(MESSAGES), ApiMode::OpenAi, MESSAGES),
		(Some("openai"), Some(OPAQUE), ApiMode::OpenAi, OPAQUE),
	];
	for (mode, api_url, want_mode, want_url) in cases {
		assert_eq!(
			unlisted(mode, api_url).expect("selected"),
			(want_mode, want_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
	for raw in ["claude", "openai2", " "] {
		let error = unlisted(Some(raw), Some(CHAT)).expect_err("refused");
		assert!(
			error.to_string().contains("LOCAL_API_MODE must be"),
			"unexpected error: {error}"
		);
	}
}

#[test]
#[serial_test::serial]
fn an_allowlisted_model_speaks_messages_over_the_mode_and_the_url() {
	let _env = EnvGuard::new().set("LOCAL_ANTHROPIC_MODELS", "other-model, llama-3.1-8b");
	let selected = |mode: Option<&str>, api_url: Option<&str>| select_local_endpoint(mode, api_url, "llama-3.1-8b");
	assert_eq!(
		selected(None, None).expect("selected"),
		(ApiMode::Anthropic, LOCAL_MESSAGES_URL.to_owned())
	);
	assert_eq!(
		selected(None, Some(MESSAGES)).expect("selected"),
		(ApiMode::Anthropic, MESSAGES.to_owned())
	);
	// The list is the strongest statement: whatever the URL's path or the
	// mode says, the model speaks Messages and the URL is not rewritten.
	for (mode, api_url) in [
		(None, None),
		(None, Some(OPAQUE)),
		(None, Some(CHAT)),
		(Some("openai"), None),
		(Some("openai"), Some(CHAT)),
		(Some("anthropic"), Some(CHAT)),
	] {
		let want_url = api_url.unwrap_or(LOCAL_MESSAGES_URL);
		assert_eq!(
			selected(mode, api_url).expect("selected"),
			(ApiMode::Anthropic, want_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
	// The mode is still read: an unknown one is refused before any request.
	let error = selected(Some("claude"), Some(CHAT)).expect_err("refused");
	assert!(
		error.to_string().contains("LOCAL_API_MODE must be"),
		"unexpected error: {error}"
	);
	// Other models are not affected by the list.
	assert_eq!(
		select_local_endpoint(None, None, "some-other-model").expect("selected"),
		(ApiMode::OpenAi, LOCAL_API_URL.to_owned())
	);
}

#[test]
fn test_supports_model() {
	let provider = LocalProvider::new();

	assert!(provider.supports_model("llama3.2"));
	assert!(provider.supports_model("mistral-7b"));
	assert!(provider.supports_model("gpt4all-j"));
	assert!(provider.supports_model("any-model-name"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_get_api_key_optional() {
	let provider = LocalProvider::new();
	let result = provider.get_api_key();
	assert!(result.is_ok());
}

#[test]
fn test_default_capabilities() {
	let provider = LocalProvider::new();
	assert_eq!(provider.name(), "local");
	assert!(!provider.supports_caching("any-model"));
}

#[test]
fn test_capabilities_model_specific() {
	let provider = LocalProvider::new();
	// Vision models
	assert!(provider.supports_vision("llava:latest"));
	assert!(provider.supports_vision("gemma-3-27b"));
	// Text-only models
	assert!(!provider.supports_vision("llama-3.1-8b"));
	// Structured output
	assert!(provider.supports_structured_output("llama-3.1-8b"));
	assert!(!provider.supports_structured_output("mistral-7b"));
	// Context windows
	assert_eq!(provider.get_max_input_tokens("llama-3.1-8b"), 131_072);
	assert_eq!(provider.get_max_input_tokens("mistral-7b"), 32_768);
}

#[test]
fn anthropic_models_are_matched_whole() {
	let allowlist = "qwen3, glm-4.6 ,";
	assert!(is_anthropic_model(allowlist, "qwen3"));
	assert!(is_anthropic_model(allowlist, "glm-4.6"));
	assert!(
		!is_anthropic_model(allowlist, "qwen3:8b"),
		"a shorter name does not opt a tag in"
	);
	assert!(!is_anthropic_model(allowlist, ""));
	assert!(!is_anthropic_model("", "qwen3"), "an unset flag opts nobody in");
}

#[test]
fn messages_thinking_maps_every_effort() {
	let sent = |effort: Option<ReasoningEffort>| {
		let mut request = json!({ "model": "m" });
		(ANTHROPIC_COMPAT.thinking)("m", effort, &mut request);
		request
	};
	assert_eq!(sent(None), json!({ "model": "m" }), "an unset effort sends nothing");
	assert_eq!(
		sent(Some(ReasoningEffort::None)),
		json!({ "model": "m", "thinking": { "type": "disabled" } })
	);
	for (effort, want) in [
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "high"),
		(ReasoningEffort::Max, "high"),
	] {
		assert_eq!(
			sent(Some(effort)),
			json!({ "model": "m", "output_config": { "effort": want } }),
			"{effort:?}"
		);
	}
}

#[test]
fn messages_thinking_keeps_an_existing_output_format() {
	let mut request = json!({ "output_config": { "format": { "type": "json_schema" } } });
	(ANTHROPIC_COMPAT.thinking)("m", Some(ReasoningEffort::Medium), &mut request);
	assert_eq!(
		request["output_config"],
		json!({ "format": { "type": "json_schema" }, "effort": "medium" })
	);
}

#[test]
fn messages_take_base64_images_only() {
	assert_eq!((ANTHROPIC_COMPAT.images)("gemma-3-27b"), Images::Base64);
	assert_eq!((ANTHROPIC_COMPAT.images)("llama-3.1-8b"), Images::None);
}
