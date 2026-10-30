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
use crate::test_support::mock_http::MockHttp;

fn accumulate(format: Format, events: &[Value], done: bool) -> Result<Value> {
	let mut accumulator = Accumulator::new(format);
	for event in events {
		accumulator.push(&event.to_string())?;
	}
	if done {
		accumulator.push("[DONE]")?;
	}
	accumulator.finish()
}

fn chat_events() -> Vec<Value> {
	vec![
		json!({"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello ","reasoning":"think "}}]}),
		json!({"choices":[{"index":0,"delta":{"content":"世界","reasoning":"more"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5}}),
		json!({"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":9,"total_tokens":21,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":3},"cost":0.12}}),
	]
}

fn wire(events: &[Value], done: bool) -> String {
	let mut body = events
		.iter()
		.map(|event| format!("data: {event}\n\n"))
		.collect::<String>();
	if done {
		body.push_str("data: [DONE]\n\n");
	}
	body
}

#[test]
fn chat_preserves_content_reasoning_and_final_usage_snapshot() {
	let response = accumulate(Format::Chat, &chat_events(), true).unwrap();
	assert_eq!(response["id"], "c1");
	assert_eq!(response["choices"][0]["message"]["content"], "Hello 世界");
	assert_eq!(response["choices"][0]["message"]["reasoning"], "think more");
	assert_eq!(response["usage"]["completion_tokens"], 9);
	assert_eq!(response["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
	assert_eq!(response["usage"]["completion_tokens_details"]["reasoning_tokens"], 3);
	assert_eq!(response["usage"]["cost"], 0.12);
}

#[test]
fn fragmented_parallel_tools_and_reasoning_signatures_round_trip() {
	let response = accumulate(Format::Chat, &[
		json!({"choices":[{"index":0,"delta":{"tool_calls":[
			{"index":1,"id":"b","type":"function","function":{"name":"other","arguments":"{}"}},
			{"index":0,"id":"a","type":"function","function":{"name":"get_","arguments":"{\"city\":"},"extra_content":{"google":{"thought_signature":"signed"}}}],
			"reasoning_details":[{"index":0,"type":"reasoning.encrypted","data":"part1","id":"r1","format":"anthropic-claude-v1"}]}}]}),
		json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"weather","arguments":"\"NYC\"}"}}],
			"reasoning_details":[{"index":0,"data":"part2"}]},"finish_reason":"tool_calls"}]}),
	], true).unwrap();
	let message = &response["choices"][0]["message"];
	assert_eq!(message["tool_calls"][0]["id"], "a");
	assert_eq!(message["tool_calls"][1]["id"], "b");
	assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
	assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{\"city\":\"NYC\"}");
	assert_eq!(
		message["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
		"signed"
	);
	assert_eq!(message["reasoning_details"][0]["data"], "part1part2");
	assert_eq!(message["reasoning_details"][0]["id"], "r1");
}

#[test]
fn decoder_handles_every_byte_boundary_multiline_bom_and_line_endings() {
	for ending in ["\n", "\r\n", "\r"] {
		let body = format!("\u{feff}: heartbeat{ending}event: chunk{ending}data: {{\"choices\": [{{{ending}data: \"index\":0,\"delta\":{{\"content\":\"世界\"}},\"finish_reason\":\"stop\"}}]}}{ending}{ending}data: [DONE]{ending}{ending}");
		for split in 0..=body.len() {
			let mut decoder = Decoder::default();
			let mut accumulator = Accumulator::new(Format::Chat);
			decoder
				.push(&body.as_bytes()[..split], |data| accumulator.push(data))
				.unwrap();
			decoder
				.push(&body.as_bytes()[split..], |data| accumulator.push(data))
				.unwrap();
			decoder.finish().unwrap();
			assert_eq!(
				accumulator.finish().unwrap()["choices"][0]["message"]["content"],
				"世界"
			);
		}
	}
}

#[test]
fn malformed_truncated_empty_and_provider_error_streams_fail() {
	assert!(accumulate(Format::Chat, &chat_events(), false)
		.unwrap_err()
		.is::<IncompleteStream>());
	assert!(accumulate(Format::Chat, &[], true).is_err());
	assert!(accumulate(
		Format::Chat,
		&[json!({"choices":[{"index":0,"delta":{"content":"partial"}}]})],
		true
	)
	.is_err());
	for data in [
		"not-json",
		"[]",
		r#"{"error":{"message":"overloaded"}}"#,
		r#"{"choices":[{"index":-1,"delta":{}}]}"#,
	] {
		assert!(Accumulator::new(Format::Chat).push(data).is_err(), "{data}");
	}
	let mut decoder = Decoder::default();
	decoder.push(b"data: {", |_| Ok(())).unwrap();
	assert!(decoder.finish().is_err());
	assert!(Decoder::default().push(b"data: \xff\n\n", |_| Ok(())).is_err());
}

#[test]
fn missing_tool_fields_and_sparse_indices_fail_without_allocating_by_index() {
	for call in [
		json!({"index":u64::MAX,"id":"x","function":{"name":"f","arguments":"{}"}}),
		json!({"index":0,"function":{"name":"f","arguments":"{}"}}),
		json!({"index":0,"id":"x","function":{"arguments":"{}"}}),
		json!({"index":0,"id":"x","function":{"name":"f","arguments":"{"}}),
	] {
		assert!(accumulate(
			Format::Chat,
			&[json!({"choices":[{"index":0,"delta":{"tool_calls":[call]},"finish_reason":"tool_calls"}]})],
			true
		)
		.is_err());
	}
}

#[test]
fn messages_preserve_thinking_signature_tools_and_cache_usage() {
	let events = vec![
		json!({"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":20,"cache_creation_input_tokens":30}}}),
		json!({"type":"ping"}),
		json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"think"}}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}}),
		json!({"type":"content_block_stop","index":0}),
		json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t1","name":"run","input":{}}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}),
		json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
		json!({"type":"content_block_stop","index":1}),
		json!({"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":40}}),
		json!({"type":"message_stop"}),
	];
	let response = accumulate(Format::Messages, &events, false).unwrap();
	assert_eq!(response["content"][0]["thinking"], "think");
	assert_eq!(response["content"][0]["signature"], "signed");
	assert_eq!(response["content"][1]["input"], json!({"x":1}));
	assert_eq!(response["usage"]["output_tokens"], 40);
	assert_eq!(response["usage"]["cache_read_input_tokens"], 20);
	assert_eq!(response["usage"]["cache_creation_input_tokens"], 30);
	assert_eq!(response["stop_reason"], "tool_use");
	assert!(accumulate(Format::Messages, &events[..events.len() - 1], false).is_err());
}

#[test]
fn responses_keep_terminal_output_usage_and_encrypted_reasoning() {
	let response = json!({"id":"r1","status":"completed","output":[{"type":"reasoning","encrypted_content":"opaque"},{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"}],"usage":{"input_tokens":10,"output_tokens":5}});
	for event_type in ["response.completed", "response.incomplete"] {
		assert_eq!(
			accumulate(
				Format::Responses,
				&[json!({"type":event_type,"response":response})],
				false
			)
			.unwrap(),
			response
		);
	}
	assert!(accumulate(
		Format::Responses,
		&[json!({"type":"response.output_text.delta","delta":"partial"})],
		false
	)
	.is_err());
	assert!(accumulate(
		Format::Responses,
		&[json!({"type":"response.failed","response":{"error":{"code":"server_error"}}})],
		false
	)
	.is_err());
}

#[test]
fn groq_usage_envelope_is_preserved() {
	let response = accumulate(Format::Chat, &[json!({"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"x_groq":{"usage":{"prompt_tokens":2,"completion_tokens":3}}})], true).unwrap();
	assert_eq!(response["usage"]["completion_tokens"], 3);
}

#[tokio::test]
async fn http_accumulates_sse_and_preserves_status_and_json_error_bodies() {
	let mock = MockHttp::start(vec![(200, wire(&chat_events(), true))]);
	let response = send_and_read(shared::http_client().post(&mock.url), None, None, Format::Chat)
		.await
		.unwrap();
	assert_eq!(response.status, 200);
	let body: Value = serde_json::from_str(&response.body).unwrap();
	assert_eq!(body["choices"][0]["message"]["content"], "Hello 世界");
	let mock = MockHttp::start(vec![(429, r#"{"error":{"message":"slow down"}}"#.to_owned())]);
	let response = send_and_read(shared::http_client().post(&mock.url), None, None, Format::Chat)
		.await
		.unwrap();
	assert_eq!(response.status, 429);
	assert!(response.body.contains("slow down"));
}

#[tokio::test]
#[serial_test::serial]
async fn native_chat_providers_accumulate_and_keep_usage() {
	use crate::llm::providers::{
		DeepSeekProvider, MoonshotProvider, OpenRouterProvider, TogetherProvider, ZaiProvider,
	};
	use crate::llm::traits::AiProvider;
	use crate::llm::types::ChatCompletionParams;
	use crate::test_support::env::EnvGuard;
	let cases: Vec<(Box<dyn AiProvider>, &str, &str)> = vec![
		(Box::new(DeepSeekProvider::new()), "DEEPSEEK_API_URL", "deepseek-v4-pro"),
		(Box::new(MoonshotProvider::new()), "MOONSHOT_API_URL", "kimi-k3"),
		(
			Box::new(OpenRouterProvider::new()),
			"OPENROUTER_API_URL",
			"openai/gpt-5",
		),
		(
			Box::new(TogetherProvider::new()),
			"TOGETHER_API_URL",
			"Qwen/Qwen3.8-Flash",
		),
		(Box::new(ZaiProvider::new()), "ZAI_API_URL", "glm-5.3"),
	];
	for (provider, env, model) in cases {
		let mut events = chat_events();
		events[0]["created"] = json!(1);
		events[0]["model"] = json!(model);
		let mock = MockHttp::start(vec![(200, wire(&events, true))]);
		let _env = EnvGuard::new().set(env, &mock.url);
		let mut params = ChatCompletionParams::new(&[], model, 1.0, 1.0, 0, 100).with_api_key("mock-key".to_owned());
		params.max_retries = 0;
		let response = provider
			.chat_completion(params)
			.await
			.unwrap_or_else(|error| panic!("{}: {error:#}", provider.name()));
		assert_eq!(response.content, "Hello 世界", "{}", provider.name());
		let usage = response.exchange.usage.unwrap();
		assert_eq!(usage.billable_output_tokens(), 9, "{}", provider.name());
		assert!(usage.cost.is_some(), "{}", provider.name());
		let request: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
		assert_eq!(request["stream"], true, "{}", provider.name());
	}
}

#[tokio::test]
#[serial_test::serial]
async fn messages_providers_accumulate_text_and_bill_final_usage() {
	use crate::llm::providers::{AnthropicProvider, MinimaxProvider};
	use crate::llm::traits::AiProvider;
	use crate::llm::types::ChatCompletionParams;
	use crate::test_support::env::EnvGuard;
	let cases: Vec<(Box<dyn AiProvider>, &str, &str)> = vec![
		(
			Box::new(AnthropicProvider::new()),
			"ANTHROPIC_API_URL",
			"claude-opus-4-7",
		),
		(Box::new(MinimaxProvider::new()), "MINIMAX_API_URL", "MiniMax-M3"),
	];
	for (provider, env, model) in cases {
		let events = vec![
			json!({"type":"message_start","message":{"id":"msg1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":20}}}),
			json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
			json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}),
			json!({"type":"content_block_stop","index":0}),
			json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}),
			json!({"type":"message_stop"}),
		];
		let mock = MockHttp::start(vec![(200, wire(&events, false))]);
		let _env = EnvGuard::new().set(env, &mock.url);
		let mut params = ChatCompletionParams::new(&[], model, 1.0, 1.0, 0, 100).with_api_key("mock-key".to_owned());
		params.max_retries = 0;
		let response = provider
			.chat_completion(params)
			.await
			.unwrap_or_else(|error| panic!("{}: {error:#}", provider.name()));
		assert_eq!(response.content, "hello");
		let usage = response.exchange.usage.unwrap();
		assert_eq!(usage.input_tokens, 10);
		assert_eq!(usage.cache_read_tokens, 20);
		assert_eq!(usage.billable_output_tokens(), 5);
		let request: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
		assert_eq!(request["stream"], true);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn responses_providers_keep_the_complete_terminal_response() {
	use crate::llm::providers::{OpenAiProvider, XaiProvider};
	use crate::llm::traits::AiProvider;
	use crate::llm::types::ChatCompletionParams;
	use crate::test_support::env::EnvGuard;
	let cases: Vec<(Box<dyn AiProvider>, &str, &str)> = vec![
		(Box::new(OpenAiProvider::new()), "OPENAI_API_URL", "gpt-5"),
		(Box::new(XaiProvider::new()), "XAI_API_URL", "grok-4.5"),
	];
	for (provider, env, model) in cases {
		let body = json!({"id":"resp1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":10,"output_tokens":9,"total_tokens":19,"output_tokens_details":{"reasoning_tokens":3}}});
		let mock = MockHttp::start(vec![(
			200,
			wire(&[json!({"type":"response.completed","response":body})], false),
		)]);
		let _env = EnvGuard::new().set(env, &mock.url);
		let mut params = ChatCompletionParams::new(&[], model, 1.0, 1.0, 0, 100).with_api_key("mock-key".to_owned());
		params.max_retries = 0;
		let response = provider
			.chat_completion(params)
			.await
			.unwrap_or_else(|error| panic!("{}: {error:#}", provider.name()));
		assert_eq!(response.content, "hello");
		let usage = response.exchange.usage.unwrap();
		assert_eq!(usage.output_tokens, 6);
		assert_eq!(usage.reasoning_tokens, 3);
		let request: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
		assert_eq!(request["stream"], true);
	}
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn truncated_stream_retries_but_malformed_and_permanent_errors_do_not() {
	use crate::llm::retry;
	use crate::llm::types::ChatCompletionParams;
	for (first_body, retry_expected) in [
		(wire(&chat_events(), false), true),
		("data: {".to_owned(), true),
		("data: {\n\n".to_owned(), false),
		(
			wire(&[json!({"error":{"code":401,"message":"bad private-key"}})], false),
			false,
		),
		(
			wire(&[json!({"error":{"type":"overloaded_error","message":"busy"}})], false),
			true,
		),
	] {
		let mock = MockHttp::start(vec![(200, first_body), (200, wire(&chat_events(), true))]);
		let params = ChatCompletionParams::new(&[], "test", 1.0, 1.0, 0, 100).with_api_key("private-key".to_owned());
		let result = retry::retry_provider_request(
			|| {
				Box::pin(send_and_read(
					shared::http_client().post(&mock.url),
					None,
					None,
					Format::Chat,
				))
			},
			1,
			Duration::ZERO,
			None,
			params.api_key.as_ref(),
			None,
		)
		.await;
		assert_eq!(result.is_ok(), retry_expected);
		assert_eq!(mock.requests().len(), if retry_expected { 2 } else { 1 });
		if let Err(error) = result {
			assert!(!format!("{error:#}").contains("private-key"));
		}
	}
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn timeout_and_cancellation_cover_the_stream_body() {
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	use tokio::net::TcpListener;
	use tokio::sync::{oneshot, watch};
	for cancel in [false, true] {
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		let (started, ready) = oneshot::channel();
		let server = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = Vec::new();
			while !request.ends_with(b"\r\n\r\n") {
				request.push(socket.read_u8().await.unwrap());
			}
			socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\nD\r\n: heartbeat\n\n\r\n").await.unwrap();
			started.send(()).unwrap();
			let mut byte = [0];
			assert_eq!(
				socket.read(&mut byte).await.unwrap(),
				0,
				"aborted request must close the body"
			);
		});
		let (sender, token) = watch::channel(false);
		let client = tokio::spawn(async move {
			crate::llm::retry::retry_provider_request(
				|| {
					Box::pin(send_and_read(
						shared::http_client().get(&url),
						Some(Duration::from_millis(250)),
						None,
						Format::Chat,
					))
				},
				0,
				Duration::ZERO,
				Some(&token),
				None,
				None,
			)
			.await
		});
		ready.await.unwrap();
		if cancel {
			sender.send(true).unwrap();
		}
		let error = client.await.unwrap().err().expect("must abort");
		if cancel {
			assert!(matches!(
				error.downcast_ref::<crate::errors::ProviderError>(),
				Some(crate::errors::ProviderError::Cancelled)
			));
		} else {
			assert_eq!(
				shared::transport_error_kind(&error),
				Some(shared::TransportErrorKind::Timeout)
			);
		}
		tokio::time::timeout(Duration::from_secs(2), server)
			.await
			.unwrap()
			.unwrap();
	}
}

#[tokio::test]
async fn http_chunk_boundaries_do_not_define_event_boundaries() {
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}", listener.local_addr().unwrap());
	let body = wire(&chat_events(), true);
	let server = tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.unwrap();
		let mut request = Vec::new();
		while !request.ends_with(b"\r\n\r\n") {
			request.push(socket.read_u8().await.unwrap());
		}
		socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
		for byte in body.bytes() {
			socket
				.write_all(&[b'1', b'\r', b'\n', byte, b'\r', b'\n'])
				.await
				.unwrap();
		}
		socket.write_all(b"0\r\n\r\n").await.unwrap();
	});
	let response = send_and_read(
		shared::http_client().get(&url),
		Some(Duration::from_secs(2)),
		None,
		Format::Chat,
	)
	.await
	.unwrap();
	let body: Value = serde_json::from_str(&response.body).unwrap();
	assert_eq!(body["choices"][0]["message"]["content"], "Hello 世界");
	server.await.unwrap();
}

#[tokio::test]
async fn compatible_adapter_validates_assembled_json_and_preserves_billing() {
	use crate::llm::providers::openai_compat::{chat_completion, OpenAiCompatConfig};
	use crate::llm::types::{ChatCompletionParams, StructuredOutputRequest};
	let schema =
		json!({"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false});
	let events = vec![
		json!({"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"{\"ok\":"}}]}),
		json!({"choices":[{"index":0,"delta":{"content":"true}"},"finish_reason":"stop"}]}),
		json!({"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":5},"completion_tokens_details":{"reasoning_tokens":3},"cost":0.25}}),
	];
	let mock = MockHttp::start(vec![(200, wire(&events, true))]);
	let mut params = ChatCompletionParams::new(&[], "test", 1.0, 1.0, 0, 100);
	params.response_format = Some(StructuredOutputRequest::json_schema(schema));
	params.max_retries = 0;
	let response = chat_completion(
		OpenAiCompatConfig {
			provider_name: "nvidia",
			usage_fallback_cost: None,
			use_response_cost: true,
			enforces_response_schema: true,
			supports_required_tool_choice: true,
			supports_parallel_tool_calls: true,
		},
		"mock-key".to_owned(),
		mock.url.clone(),
		params,
	)
	.await
	.unwrap();
	assert_eq!(response.structured_output, Some(json!({"ok":true})));
	let usage = response.exchange.usage.unwrap();
	assert_eq!(usage.input_tokens, 15);
	assert_eq!(usage.cache_read_tokens, 5);
	assert_eq!(usage.output_tokens, 7);
	assert_eq!(usage.reasoning_tokens, 3);
	assert_eq!(usage.cost, Some(0.25));
	let request: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
	assert_eq!(request["stream"], true);
	assert_eq!(request["stream_options"]["include_usage"], true);
	assert_eq!(request["response_format"]["type"], "json_schema");
}

#[test]
fn repeated_finish_reason_on_openrouter_usage_chunk_is_not_a_second_completion() {
	let mut events = chat_events();
	events[2]["choices"] = json!([{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop","native_finish_reason":"stop"}]);
	let response = accumulate(Format::Chat, &events, true).unwrap();
	assert_eq!(response["choices"][0]["message"]["content"], "Hello 世界");
	assert_eq!(response["usage"]["cost"], 0.12);
	assert_eq!(response["choices"][0]["native_finish_reason"], "stop");
}

#[test]
fn messages_reject_unclosed_blocks_and_unknown_content_deltas() {
	let start = json!({"type":"message_start","message":{"id":"m1","type":"message","content":[]}});
	let block = json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}});
	for last in [
		json!({"type":"message_stop"}),
		json!({"type":"content_block_delta","index":0,"delta":{"type":"unknown_delta"}}),
	] {
		assert!(accumulate(Format::Messages, &[start.clone(), block.clone(), last], false).is_err());
	}
}

#[test]
fn stream_failure_finish_reason_and_invalid_accumulated_shapes_are_errors() {
	assert!(accumulate(
		Format::Chat,
		&[json!({"choices":[{"index":0,"delta":{},"finish_reason":"error"}]})],
		true
	)
	.is_err());
	assert!(accumulate(
		Format::Chat,
		&[
			json!({"choices":[{"index":0,"delta":{},"message":"invalid"}]}),
			json!({"choices":[{"index":0,"delta":{"content":"text"},"finish_reason":"stop"}]}),
		],
		true
	)
	.is_err());
}
