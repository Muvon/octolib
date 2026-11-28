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

//! HTTP-path coverage for the local provider: request shaping through the
//! MockHttp loopback seam, reference-cost fill for cloud-equivalent models,
//! and error surfacing.

use super::*;
use crate::llm::types::Message;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_fills_reference_cost_for_cloud_equivalent_model() {
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("local answer"),
	)]);
	let _env = EnvGuard::new().set("LOCAL_API_URL", mock.url.clone());

	let provider = LocalProvider::new();
	let params =
		ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	assert_eq!(response.content, "local answer");
	// The mock reports 12 prompt + 7 completion tokens; llama-3.1-8b reference
	// pricing is a flat $0.10/M for input and output → 19 tokens ≈ $0.0000019.
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);
	let expected = (12.0 / 1_000_000.0) * 0.10 + (7.0 / 1_000_000.0) * 0.10;
	assert!(
		(usage.cost.expect("reference cost filled") - expected).abs() < 1e-12,
		"unexpected cost: {:?}",
		usage.cost
	);

	let requests = mock.requests();
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["model"], "llama-3.1-8b");
	assert_eq!(request["messages"][0]["role"], "user");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_leaves_unpriced_local_models_costless() {
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("free"),
	)]);
	let _env = EnvGuard::new().set("LOCAL_API_URL", mock.url.clone());

	let provider = LocalProvider::new();
	let params = ChatCompletionParams::new(&[Message::user("hello")], "some-unknown-local-model", 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.cost, None, "unpriced model must stay costless");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_retries_server_errors_then_surfaces_them() {
	let mock = MockHttp::start(vec![(500, r#"{"error":{"message":"boom"}}"#.to_string())]);
	let _env = EnvGuard::new().set("LOCAL_API_URL", mock.url.clone());

	let provider = LocalProvider::new();
	// 5xx is retryable; a 1ms backoff keeps the retry from sleeping for seconds.
	let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100)
		.with_api_key("mock-key".to_owned())
		.with_max_retries(1)
		.with_retry_timeout(std::time::Duration::from_millis(1));
	let err = provider.chat_completion(params).await.expect_err("http error surfaces");
	assert!(err.to_string().contains("local API error"), "unexpected error: {err}");
	assert_eq!(mock.requests().len(), 2, "one attempt plus one retry");
}

#[tokio::test]
#[serial_test::serial]
async fn allowlisted_model_speaks_messages_on_the_same_server() {
	let body = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"local answer"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":7}}"#;
	let mock = MockHttp::start(vec![(200, body.to_string())]);
	// A Messages endpoint at a custom path: sent to as it is.
	let messages_url = format!("{}/api/prefix/v1/messages", mock.base_url());
	let _env = EnvGuard::new()
		.set("LOCAL_API_URL", messages_url.clone())
		.set("LOCAL_ANTHROPIC_MODELS", "other-model, llama-3.1-8b")
		.set("LOCAL_API_MODE", "");

	let provider = LocalProvider::new();
	let params =
		ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");
	assert_eq!(response.content, "local answer");

	let requests = mock.requests();
	assert!(
		requests[0].header("anthropic-version").is_some(),
		"the Messages protocol"
	);
	assert_eq!(requests[0].path, "/api/prefix/v1/messages", "the URL is not rewritten");
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let request: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(request["model"], "llama-3.1-8b");
	assert_eq!(
		request["messages"][0],
		serde_json::json!({ "role": "user", "content": [{ "type": "text", "text": "hello" }] })
	);
	// Reference cost is filled the same way as on the Chat route.
	let usage = response.exchange.usage.expect("usage present");
	let expected = (12.0 / 1_000_000.0) * 0.10 + (7.0 / 1_000_000.0) * 0.10;
	assert!((usage.cost.expect("reference cost filled") - expected).abs() < 1e-12);

	// Models outside the list keep Chat Completions: the URL's path decides.
	let mock = MockHttp::start(vec![(
		200,
		crate::test_support::mock_http::chat_completion_body("chat answer"),
	)]);
	let _env = EnvGuard::new()
		.set("LOCAL_API_URL", mock.url.clone())
		.set("LOCAL_ANTHROPIC_MODELS", "other-model, llama-3.1-8b")
		.set("LOCAL_API_MODE", "");
	let params = ChatCompletionParams::new(&[Message::user("hello")], "some-other-model", 100)
		.with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");
	assert_eq!(response.content, "chat answer");
	assert_eq!(mock.requests()[0].path, "/v1/chat/completions");
	assert!(mock.requests()[0].header("anthropic-version").is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn an_allowlisted_model_speaks_messages_to_the_url_as_configured() {
	let messages = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"messages answer"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":7}}"#;
	// The list over a Chat Completions URL and over an opposite mode: the
	// protocol is Messages, the URL goes out as it is.
	for mode in ["", "openai"] {
		let mock = MockHttp::start(vec![(200, messages.to_string())]);
		let _env = EnvGuard::new()
			.set("LOCAL_API_URL", format!("{}/v1/chat/completions", mock.base_url()))
			.set("LOCAL_ANTHROPIC_MODELS", "llama-3.1-8b")
			.set("LOCAL_API_MODE", mode);

		let provider = LocalProvider::new();
		let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100)
			.with_api_key("mock-key".to_owned());
		let response = provider.chat_completion(params).await.expect("mock completion");
		assert_eq!(response.content, "messages answer");
		assert!(
			mock.requests()[0].header("anthropic-version").is_some(),
			"mode {mode:?}"
		);
		assert_eq!(
			mock.requests()[0].path,
			"/v1/chat/completions",
			"the URL is not rewritten"
		);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn the_mode_states_the_protocol_of_the_url() {
	let messages = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"messages answer"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":7}}"#;
	// Messages on the mock's canned Chat Completions path, and Chat Completions
	// on a path named messages: both sent as configured.
	let chat = crate::test_support::mock_http::chat_completion_body("chat answer");
	for (mode, path, reply, answer, is_messages) in [
		("anthropic", "/v1/chat/completions", messages, "messages answer", true),
		("openai", "/v1/messages", chat.as_str(), "chat answer", false),
	] {
		let mock = MockHttp::start(vec![(200, reply.to_string())]);
		let _env = EnvGuard::new()
			.set("LOCAL_API_URL", format!("{}{path}", mock.base_url()))
			.set("LOCAL_ANTHROPIC_MODELS", "")
			.set("LOCAL_API_MODE", mode);

		let provider = LocalProvider::new();
		let params = ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100)
			.with_api_key("mock-key".to_owned());
		let response = provider.chat_completion(params).await.expect("completion");
		assert_eq!(response.content, answer);
		assert_eq!(
			mock.requests()[0].header("anthropic-version").is_some(),
			is_messages,
			"mode {mode}"
		);
		assert_eq!(mock.requests()[0].path, path, "the URL is not rewritten");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn an_unknown_mode_is_refused_before_any_request() {
	let mock = MockHttp::start(vec![(200, "{}".to_string())]);
	let _env = EnvGuard::new()
		.set("LOCAL_API_URL", mock.url.clone())
		.set("LOCAL_ANTHROPIC_MODELS", "")
		.set("LOCAL_API_MODE", "claude");

	let provider = LocalProvider::new();
	let params =
		ChatCompletionParams::new(&[Message::user("hello")], "llama-3.1-8b", 100).with_api_key("mock-key".to_owned());
	let error = provider.chat_completion(params).await.expect_err("refused");
	assert!(
		error.to_string().contains("LOCAL_API_MODE must be"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "nothing is sent");
}

#[tokio::test]
#[serial_test::serial]
async fn messages_refuse_url_images_before_any_request() {
	use crate::llm::types::{ImageAttachment, ImageData, SourceType};

	let mock = MockHttp::start(vec![(200, "{}".to_string())]);
	let _env = EnvGuard::new()
		.set("LOCAL_API_URL", format!("{}/v1/messages", mock.base_url()))
		.set("LOCAL_ANTHROPIC_MODELS", "")
		.set("LOCAL_API_MODE", "");

	// Ollama answered `400 invalid image source type: url` on 2026-11-28, so a
	// URL image is refused locally rather than sent to fail.
	let url_image = Message::user("look").with_images(vec![ImageAttachment {
		data: ImageData::Url("https://example.com/a.png".into()),
		media_type: "image/png".to_owned(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}]);
	let error = LocalProvider::new()
		.chat_completion(ChatCompletionParams::new(&[url_image], "gemma-3-27b", 10).with_api_key("k".to_owned()))
		.await
		.expect_err("a URL image is refused");
	assert!(
		error.to_string().contains("does not accept URL images"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "a refused request is not sent");
}
