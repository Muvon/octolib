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

//! Provider-level loopback coverage of shared conversion and selectable transports.

use super::*;
use crate::llm::types::{
	FunctionDefinition, ImageAttachment, ImageData, ReasoningEffort, SourceType, StructuredOutputRequest,
	VideoAttachment, VideoData,
};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;
use serde_json::{json, Value};
use std::time::Duration;

fn completion_body() -> Value {
	json!({
		"id": "completion_transport",
		"output": [
			{"type": "reasoning", "content": [{"type": "output_text", "text": "consider 日本語 🦀"}]},
			{"type": "message", "content": [{"type": "output_text", "text": "{\"answer\":1}"}]},
			{"type": "function_call", "call_id": "call_string", "name": "lookup",
			 "arguments": "{\"q\":\"日本語 🦀\"}"},
			{"type": "function_call", "call_id": "call_object", "name": "save",
			 "arguments": {"answer": 42}}
		],
		"usage": {
			"input_tokens": 100, "cache_read_tokens": 300, "cache_write_tokens": 20,
			"output_tokens": 40, "reasoning_tokens": 10, "total_tokens": 470,
			"cost": 0.0125, "request_time_ms": 321
		},
		"finish_reason": "tool_calls",
		"structured_output": {"answer": 42}
	})
}

fn params() -> ChatCompletionParams {
	ChatCompletionParams::new(&[Message::user("hello 日本語 🦀")], "test-model", 0.5, 0.75, 50, 128)
		.with_api_key("transport-test-key".to_owned())
		.with_max_retries(0)
		.with_request_timeout(Some(Duration::from_secs(5)))
}

fn rich_params() -> ChatCompletionParams {
	let mut previous = Message::assistant("already stored");
	previous.id = Some("completion_previous".to_owned());
	let media = Message::user("compare 日本語 🦀")
		.with_images(vec![
			ImageAttachment {
				data: ImageData::Url("https://media.invalid/image.png".to_owned()),
				media_type: "image/png".to_owned(),
				source_type: SourceType::Url,
				dimensions: None,
				size_bytes: None,
			},
			ImageAttachment {
				data: ImageData::Base64("aW1hZ2U=".to_owned()),
				media_type: "image/png".to_owned(),
				source_type: SourceType::Clipboard,
				dimensions: None,
				size_bytes: None,
			},
		])
		.with_videos(vec![
			VideoAttachment {
				data: VideoData::Url("https://media.invalid/clip.mp4".to_owned()),
				media_type: "video/mp4".to_owned(),
				source_type: SourceType::Url,
				dimensions: None,
				size_bytes: None,
				duration_secs: Some(1.0),
			},
			VideoAttachment {
				data: VideoData::Base64("dmlkZW8=".to_owned()),
				media_type: "video/mp4".to_owned(),
				source_type: SourceType::Clipboard,
				dimensions: None,
				size_bytes: None,
				duration_secs: None,
			},
		]);
	let mut request = params()
		.with_previous_id("completion_previous")
		.with_reasoning_effort(ReasoningEffort::Max)
		.with_tools(vec![
			FunctionDefinition {
				name: "save".to_owned(),
				description: "Save result".to_owned(),
				parameters: json!({"type":"object", "properties":{"answer":{"type":"integer"}}}),
				cache_control: None,
			},
			FunctionDefinition {
				name: "lookup".to_owned(),
				description: "Look up result".to_owned(),
				parameters: json!({"type":"object", "properties":{"q":{"type":"string"}}}),
				cache_control: Some(json!({"type":"ephemeral"})),
			},
		])
		.with_structured_output(
			StructuredOutputRequest::json_schema(json!({
				"type":"object", "properties":{"answer":{"type":"integer"}}, "required":["answer"]
			}))
			.with_strict_mode(),
		);
	request.messages = vec![
		Message::system("System instructions"),
		Message::user("old question"),
		previous,
		Message::tool("tool result", "call_previous", "lookup"),
		media,
	];
	request
}

fn assert_rich_request(request: &Value) {
	assert_eq!(request["model"], "test-model");
	assert_eq!(request["instructions"], "System instructions");
	assert_eq!(request["previous_completion_id"], "completion_previous");
	assert!(request.get("previous_response_id").is_none());
	assert_eq!(
		request["input"],
		json!([
			{"type":"function_call_output", "call_id":"call_previous", "output":"tool result"},
			{"type":"message", "role":"user", "content":[
				{"type":"input_text", "text":"compare 日本語 🦀"},
				{"type":"input_image", "image_url":"https://media.invalid/image.png"},
				{"type":"input_image", "image_url":"data:image/png;base64,aW1hZ2U="},
				{"type":"input_video", "video_url":"https://media.invalid/clip.mp4"},
				{"type":"input_video", "video_url":"data:video/mp4;base64,dmlkZW8="}
			]}
		])
	);
	assert_eq!(request["temperature"], 0.5);
	assert_eq!(request["top_p"], 0.75);
	assert!(request.get("top_k").is_none());
	assert_eq!(request["max_output_tokens"], 128);
	assert_eq!(request["reasoning_effort"], "max");
	assert_eq!(
		request["tools"],
		json!([
			{"type":"function", "name":"lookup", "description":"Look up result",
			 "parameters":{"type":"object", "properties":{"q":{"type":"string"}}},
			 "cache_control":{"type":"ephemeral"}},
			{"type":"function", "name":"save", "description":"Save result",
			 "parameters":{"type":"object", "properties":{"answer":{"type":"integer"}}}}
		])
	);
	assert_eq!(
		request["text"]["format"],
		json!({
			"type":"json_schema", "name":"response_schema", "strict":true,
			"schema":{"type":"object", "properties":{"answer":{"type":"integer"}},
					  "required":["answer"], "additionalProperties":false}
		})
	);
	assert!(request.get("octohub_transport").is_none());
	assert!(!request.to_string().contains("transport-test-key"));
}

fn assert_rich_response(response: &ProviderResponse) {
	assert_eq!(response.id.as_deref(), Some("completion_transport"));
	assert_eq!(response.content, "{\"answer\":1}");
	let thinking = response.thinking.as_ref().expect("thinking preserved");
	assert_eq!(thinking.content, "consider 日本語 🦀");
	assert_eq!(thinking.tokens, 10);
	let calls = response.tool_calls.as_ref().expect("tool calls preserved");
	assert_eq!(calls.len(), 2);
	assert_eq!(calls[0].id, "call_string");
	assert_eq!(calls[0].name, "lookup");
	assert_eq!(calls[0].arguments, json!({"q":"日本語 🦀"}));
	assert_eq!(calls[1].id, "call_object");
	assert_eq!(calls[1].name, "save");
	assert_eq!(calls[1].arguments, json!({"answer":42}));
	assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
	// The canonical server value wins over the different JSON in the text.
	assert_eq!(response.structured_output, Some(json!({"answer":42})));
	let usage = response.exchange.usage.as_ref().expect("usage preserved");
	assert_eq!(usage.input_tokens, 100, "already clean, do not subtract cache tokens");
	assert_eq!(usage.cache_read_tokens, 300);
	assert_eq!(usage.cache_write_tokens, 20);
	assert_eq!(
		usage.output_tokens, 40,
		"already split, do not subtract reasoning twice"
	);
	assert_eq!(usage.reasoning_tokens, 10);
	assert_eq!(usage.billable_output_tokens(), 50);
	assert_eq!(usage.total_tokens, 470);
	assert_eq!(usage.cost, Some(0.0125));
	assert_eq!(usage.request_time_ms, Some(321));
}

#[tokio::test]
#[serial_test::serial]
async fn octohub_default_and_explicit_http_post_the_same_request_and_parse_the_same_response() {
	let mock = MockHttp::start(vec![(200, completion_body().to_string())]);
	let _env = EnvGuard::new()
		.set("OCTOHUB_API_URL", mock.base_url())
		.remove("OCTOHUB_API_KEY");
	let provider = OctoHubProvider::new();
	let request = rich_params();
	assert_eq!(request.octohub_transport, OctoHubTransport::Http);
	for request in [request.clone(), request.with_octohub_transport(OctoHubTransport::Http)] {
		let response = provider.chat_completion(request).await.expect("HTTP completion");
		assert_rich_response(&response);
		assert_rich_request(&response.exchange.request);
	}
	let requests = mock.requests();
	assert_eq!(requests.len(), 2);
	for request in &requests {
		assert_eq!(request.method, "POST");
		assert_eq!(request.path, "/v1/completions");
		assert_rich_request(&serde_json::from_str::<Value>(&request.body).expect("HTTP JSON"));
	}
	assert_eq!(requests[0].body, requests[1].body);
}

#[cfg(feature = "websockets")]
mod websocket_integration {
	use super::*;
	use crate::llm::retry::{RetryEvent, RetryObserver, RetryableStatusError};
	use futures_util::{SinkExt, StreamExt};
	use std::collections::HashMap;
	use std::future::Future;
	use std::sync::{Arc, Mutex};
	use std::time::Instant;
	use tokio::net::{TcpListener, TcpStream};
	use tokio::sync::oneshot;
	use tokio_tungstenite::tungstenite::{
		handshake::server::{Callback, ErrorResponse, Request, Response},
		http::HeaderMap,
		Message as Frame,
	};
	use tokio_tungstenite::{accept_hdr_async, WebSocketStream};

	type Socket = WebSocketStream<TcpStream>;
	const TEST_DEADLINE: Duration = Duration::from_secs(10);

	async fn bounded<T>(future: impl Future<Output = T>) -> T {
		tokio::time::timeout(TEST_DEADLINE, future)
			.await
			.expect("loopback operation timed out")
	}

	// A failed assertion must not leave a server task alive or lose its panic.
	struct Peer(Option<tokio::task::JoinHandle<()>>);

	impl Peer {
		fn spawn(future: impl Future<Output = ()> + Send + 'static) -> Self {
			Self(Some(tokio::spawn(future)))
		}

		async fn finish(mut self) {
			bounded(self.0.take().expect("server task"))
				.await
				.expect("server assertion");
		}
	}

	impl Drop for Peer {
		fn drop(&mut self) {
			if let Some(task) = self.0.take() {
				task.abort();
			}
		}
	}

	async fn server() -> (TcpListener, String) {
		let listener = TcpListener::bind("127.0.0.1:0").await.expect("loopback bind");
		let url = format!("http://{}", listener.local_addr().expect("loopback address"));
		(listener, url)
	}

	async fn accept(listener: &TcpListener) -> (Socket, HeaderMap) {
		let (stream, _) = bounded(listener.accept()).await.expect("loopback accept");
		upgrade(stream).await
	}

	struct CaptureHandshake(Arc<Mutex<Option<HeaderMap>>>);

	impl Callback for CaptureHandshake {
		fn on_request(self, request: &Request, response: Response) -> std::result::Result<Response, ErrorResponse> {
			assert_eq!(request.method().as_str(), "GET");
			assert_eq!(request.uri().path(), "/v1/completions");
			*self.0.lock().expect("handshake lock") = Some(request.headers().clone());
			Ok(response)
		}
	}

	async fn upgrade(stream: TcpStream) -> (Socket, HeaderMap) {
		let captured = Arc::new(Mutex::new(None));
		let socket = bounded(accept_hdr_async(stream, CaptureHandshake(captured.clone())))
			.await
			.expect("WebSocket handshake");
		let headers = captured
			.lock()
			.expect("handshake lock")
			.take()
			.expect("captured headers");
		(socket, headers)
	}

	async fn receive(socket: &mut Socket) -> Value {
		loop {
			let frame = bounded(socket.next())
				.await
				.expect("request frame")
				.expect("read frame");
			if matches!(frame, Frame::Ping(_)) {
				socket.flush().await.expect("flush pong");
				continue;
			}
			assert!(frame.is_text(), "completion request must be one JSON text frame");
			return serde_json::from_str(frame.to_text().expect("text frame")).expect("request JSON");
		}
	}

	async fn send(socket: &mut Socket, status: u16, headers: Value, body: Value) {
		let envelope = json!({"status":status, "headers":headers, "body":body});
		bounded(socket.send(Frame::Text(envelope.to_string().into())))
			.await
			.expect("send envelope");
	}

	async fn success(socket: &mut Socket) {
		send(
			socket,
			200,
			json!({"content-type":"application/json"}),
			completion_body(),
		)
		.await;
	}

	fn observe(request: &mut ChatCompletionParams) -> Arc<Mutex<Vec<RetryEvent>>> {
		let events = Arc::new(Mutex::new(Vec::new()));
		let copy = events.clone();
		request.retry_observer = Some(RetryObserver::new(move |event| {
			copy.lock().expect("retry lock").push(event.clone());
		}));
		events
	}

	fn ws_params() -> ChatCompletionParams {
		params().with_octohub_transport(OctoHubTransport::WebSocket)
	}

	// Keep the endpoint AND old socket observable until the provider returns.
	// A resend on the socket, a fresh upgrade, or an HTTP fallback all fail.
	async fn no_replay(listener: &TcpListener, socket: &mut Socket, mut done: oneshot::Receiver<()>) {
		loop {
			tokio::select! {
				result = &mut done => { result.expect("caller finished"); return; }
				connection = listener.accept() => {
					connection.expect("unexpected connection");
					panic!("provider reconnected or fell back to HTTP");
				}
				message = socket.next() => match message {
					Some(Ok(Frame::Text(_) | Frame::Binary(_))) => panic!("provider resent a terminal request"),
					Some(Ok(Frame::Ping(_))) => { socket.flush().await.expect("flush pong"); }
					Some(Ok(_)) => {}
					Some(Err(_)) | None => {
						tokio::select! {
							result = &mut done => { result.expect("caller finished"); return; }
							connection = listener.accept() => {
								connection.expect("unexpected connection");
								panic!("provider reconnected or fell back to HTTP");
							}
						}
					}
				}
			}
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_http_then_websocket_then_explicit_http_preserves_wire_and_parse_contracts() {
		let http = MockHttp::start(vec![(200, completion_body().to_string())]);
		let (listener, ws_url) = server().await;
		let (captured, received) = oneshot::channel();
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			captured.send(receive(&mut socket).await).expect("capture request");
			success(&mut socket).await;
			no_replay(&listener, &mut socket, finished).await;
		});
		let _env = EnvGuard::new()
			.set("OCTOHUB_API_URL", http.base_url())
			.remove("OCTOHUB_API_KEY");
		let provider = OctoHubProvider::new();
		let base = rich_params().with_prompt_cache_key("provider-switch");
		assert_eq!(base.octohub_transport, OctoHubTransport::Http);
		let default = bounded(provider.chat_completion(base.clone()))
			.await
			.expect("default HTTP");
		let ws = base.with_octohub_transport(OctoHubTransport::WebSocket);
		let switched = EnvGuard::new().set("OCTOHUB_API_URL", &ws_url);
		let websocket = bounded(provider.chat_completion(ws.clone()))
			.await
			.expect("WebSocket completion");
		drop(switched);
		let explicit = bounded(provider.chat_completion(ws.with_octohub_transport(OctoHubTransport::Http)))
			.await
			.expect("explicit HTTP after WebSocket");
		let wire = bounded(received).await.expect("captured WebSocket request");
		assert_rich_request(&wire);
		for response in [&default, &websocket, &explicit] {
			assert_rich_response(response);
			assert_eq!(response.exchange.request, wire);
			assert_eq!(response.exchange.response, default.exchange.response);
		}
		let requests = http.requests();
		assert_eq!(requests.len(), 2, "only the default and explicit HTTP calls POST");
		for request in requests {
			assert_eq!(request.method, "POST");
			assert_eq!(request.path, "/v1/completions");
			assert_eq!(serde_json::from_str::<Value>(&request.body).expect("HTTP JSON"), wire);
		}
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_old_server_recovers_json_text_and_absent_optional_fields() {
		let (listener, url) = server().await;
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			receive(&mut socket).await;
			send(
				&mut socket,
				200,
				json!({}),
				json!({
					"output":[{"type":"message", "content":[{"type":"output_text", "text":"{\"answer\":0}"}]}],
					"usage":{"input_tokens":0, "output_tokens":0, "total_tokens":0}
				}),
			)
			.await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let response = bounded(OctoHubProvider::new().chat_completion(ws_params()))
			.await
			.expect("old server completion");
		assert_eq!(response.content, "{\"answer\":0}");
		assert_eq!(response.structured_output, Some(json!({"answer":0})));
		assert!(response.id.is_none());
		assert!(response.finish_reason.is_none());
		assert!(response.thinking.is_none());
		assert!(response.tool_calls.is_none());
		let usage = response.exchange.usage.expect("usage");
		assert_eq!((usage.input_tokens, usage.output_tokens, usage.total_tokens), (0, 0, 0));
		assert_eq!(
			(
				usage.cache_read_tokens,
				usage.cache_write_tokens,
				usage.reasoning_tokens
			),
			(0, 0, 0)
		);
		assert_eq!(usage.cost, None);
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_handshake_uses_effective_auth_attribution_purpose_and_last_upsert() {
		let (listener, url) = server().await;
		let peer = Peer::spawn(async move {
			for overridden in [false, true] {
				let (mut socket, headers) = accept(&listener).await;
				let expected = if overridden {
					"Bearer extra-key"
				} else {
					"Bearer transport-test-key"
				};
				assert_eq!(headers["authorization"], expected);
				assert_eq!(headers.get_all("authorization").iter().count(), 1);
				assert_eq!(headers["x-title"], if overridden { "Extra title" } else { "Env title" });
				assert_eq!(headers.get_all("x-title").iter().count(), 1);
				assert_eq!(headers["http-referer"], "https://app.invalid");
				assert_eq!(headers["x-openrouter-categories"], "coding,tests");
				assert_eq!(headers["x-model-purpose"], "transport-contract");
				assert_eq!(headers["x-route"], "loopback");
				assert_eq!(
					headers["upgrade"].to_str().expect("upgrade").to_ascii_lowercase(),
					"websocket"
				);
				assert!(headers["connection"]
					.to_str()
					.expect("connection")
					.split(',')
					.any(|value| value.trim().eq_ignore_ascii_case("upgrade")));
				assert_eq!(headers["sec-websocket-version"], "13");
				assert!(!headers["sec-websocket-key"].is_empty());
				let body = receive(&mut socket).await;
				assert_eq!(body["input"][0]["content"], "hello 日本語 🦀");
				assert!(!body.to_string().contains("extra-key"));
				assert!(!body.to_string().contains("transport-test-key"));
				success(&mut socket).await;
			}
		});
		let _env = EnvGuard::new()
			.set("OCTOHUB_API_URL", url)
			.set("OCTOHUB_API_KEY", "env-key-not-used")
			.set("OPENROUTER_APP_TITLE", "Env title")
			.set("OPENROUTER_HTTP_REFERER", "https://app.invalid")
			.set("OPENROUTER_APP_CATEGORIES", "coding,tests");
		for overridden in [false, true] {
			let mut extra = HashMap::from([
				("X-Model-Purpose".to_owned(), "transport-contract".to_owned()),
				("X-Route".to_owned(), "loopback".to_owned()),
				("invalid header name".to_owned(), "ignored".to_owned()),
			]);
			if overridden {
				extra.insert("Authorization".to_owned(), "Bearer extra-key".to_owned());
				extra.insert("X-Title".to_owned(), "Extra title".to_owned());
			}
			assert_rich_response(
				&bounded(OctoHubProvider::new().chat_completion(ws_params().with_extra_headers(extra)))
					.await
					.expect("header completion"),
			);
		}
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_401_maps_missing_stored_and_per_call_auth_without_retry() {
		for mode in ["missing", "stored", "per-call"] {
			let (listener, url) = server().await;
			let (done, finished) = oneshot::channel();
			let peer = Peer::spawn(async move {
				let (mut socket, headers) = accept(&listener).await;
				match mode {
					"missing" => assert!(headers.get("authorization").is_none()),
					"stored" => assert_eq!(headers["authorization"], "Bearer stored-test-key"),
					_ => assert_eq!(headers["authorization"], "Bearer transport-test-key"),
				}
				receive(&mut socket).await;
				send(&mut socket, 401, json!({}), json!({"error":{"message":"invalid key"}})).await;
				no_replay(&listener, &mut socket, finished).await;
			});
			let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
			let _stored = (mode != "missing").then(|| EnvGuard::new().set("OCTOHUB_API_KEY", "stored-test-key"));
			let mut request = ChatCompletionParams::new(&[Message::user("auth")], "test-model", 0.5, 0.75, 0, 128)
				.with_octohub_transport(OctoHubTransport::WebSocket)
				.with_max_retries(3)
				.with_retry_timeout(Duration::ZERO)
				.with_request_timeout(Some(Duration::from_secs(5)));
			if mode == "per-call" {
				request = request.with_api_key("transport-test-key".to_owned());
			}
			let events = observe(&mut request);
			let error = bounded(OctoHubProvider::new().chat_completion(request))
				.await
				.expect_err("401 rejected");
			let message = error.to_string();
			assert!(message.contains("OctoHub API error 401"), "{error}");
			assert!(message.contains("invalid key"), "{error}");
			match mode {
				"missing" => assert!(message.contains("no OctoHub key is set") && message.contains("octomind login")),
				"stored" => {
					assert!(message.contains("stored OctoHub key was rejected") && message.contains("octomind login"))
				}
				_ => assert!(message.contains("passed for this request") && !message.contains("octomind login")),
			}
			assert!(events.lock().expect("events").is_empty());
			done.send(()).expect("finish server");
			peer.finish().await;
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_upgrade_401_403_and_400_map_auth_and_redact_without_post_fallback() {
		for (status, message) in [
			(401, "invalid key"),
			(403, "model is not permitted"),
			(400, "unsupported video modality"),
		] {
			let mock = MockHttp::start(vec![(
				status,
				json!({"error":{"message":format!("{message}: transport-test-key")}}).to_string(),
			)]);
			let _env = EnvGuard::new()
				.set("OCTOHUB_API_URL", mock.base_url())
				.remove("OCTOHUB_API_KEY");
			let mut request = ws_params().with_max_retries(3).with_retry_timeout(Duration::ZERO);
			let events = observe(&mut request);
			let error = bounded(OctoHubProvider::new().chat_completion(request))
				.await
				.expect_err("upgrade rejected");
			assert!(
				error.to_string().contains(&format!("OctoHub API error {status}")),
				"{error:#}"
			);
			assert!(
				error.to_string().contains(message),
				"upstream error context was lost: {error:#}"
			);
			if status == 401 {
				assert!(error.to_string().contains("passed for this request"));
			}
			assert!(!error.to_string().contains("octomind login"));
			for rendered in [error.to_string(), format!("{error:#}"), format!("{error:?}")] {
				assert!(
					!rendered.contains("transport-test-key"),
					"upgrade error leaked a credential"
				);
			}
			assert!(events.lock().expect("events").is_empty());
			let requests = mock.requests();
			assert_eq!(requests.len(), 1, "nonretryable upgrade failure is terminal");
			assert_eq!(requests[0].method, "GET", "must not fall back to HTTP POST");
			assert_eq!(requests[0].path, "/v1/completions");
			assert!(
				requests[0].body.is_empty(),
				"completion must not be sent before upgrade"
			);
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_upgrade_503_retries_only_handshake_and_preserves_typed_status() {
		let mock = MockHttp::start(vec![(
			503,
			json!({"error":{"message":"private upstream detail transport-test-key"}}).to_string(),
		)]);
		let _env = EnvGuard::new()
			.set("OCTOHUB_API_URL", mock.base_url())
			.remove("OCTOHUB_API_KEY");
		let mut request = ws_params()
			.with_max_retries(1)
			.with_retry_timeout(Duration::from_millis(1));
		let events = observe(&mut request);
		let error = bounded(OctoHubProvider::new().chat_completion(request))
			.await
			.expect_err("503 upgrade exhausted");
		let status = error
			.downcast_ref::<RetryableStatusError>()
			.expect("typed upgrade status");
		assert_eq!(status.status, 503);
		assert_eq!(status.retry_after, None);
		assert_eq!(
			*events.lock().expect("events"),
			vec![RetryEvent {
				attempt: 1,
				max_retries: 1,
				delay: Duration::from_millis(1),
				reason: "HTTP 503".to_owned(),
			}]
		);
		assert!(!format!("{error:?}").contains("transport-test-key"));
		assert!(!format!("{:?}", events.lock().expect("events")).contains("private upstream detail"));
		let requests = mock.requests();
		assert_eq!(requests.len(), 2, "one handshake plus one explicit-status retry");
		for request in requests {
			assert_eq!(request.method, "GET", "no HTTP POST fallback");
			assert_eq!(request.path, "/v1/completions");
			assert!(request.body.is_empty());
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_403_and_unsupported_image_video_400_surface_once_without_login_hint() {
		for (status, message) in [
			(403, "model is not permitted for this API key"),
			(400, "unsupported image modality"),
			(400, "unsupported video modality"),
		] {
			let (listener, url) = server().await;
			let (done, finished) = oneshot::channel();
			let peer = Peer::spawn(async move {
				let (mut socket, _) = accept(&listener).await;
				assert_rich_request(&receive(&mut socket).await);
				send(
					&mut socket,
					status,
					json!({"retry-after":"1"}),
					json!({"error":{"message":message}}),
				)
				.await;
				no_replay(&listener, &mut socket, finished).await;
			});
			let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
			let mut request = rich_params()
				.with_octohub_transport(OctoHubTransport::WebSocket)
				.with_max_retries(3)
				.with_retry_timeout(Duration::ZERO);
			let events = observe(&mut request);
			let error = bounded(OctoHubProvider::new().chat_completion(request.clone()))
				.await
				.expect_err("client error");
			assert_eq!(error.to_string(), format!("OctoHub API error {status}: {message}"));
			assert!(!error.to_string().contains("octomind login"));
			let http = MockHttp::start(vec![(status, json!({"error":{"message":message}}).to_string())]);
			let _http_env = EnvGuard::new().set("OCTOHUB_API_URL", http.base_url());
			let http_error =
				bounded(OctoHubProvider::new().chat_completion(request.with_octohub_transport(OctoHubTransport::Http)))
					.await
					.expect_err("same HTTP client error");
			assert_eq!(
				http_error.to_string(),
				error.to_string(),
				"transport-independent error mapping"
			);
			assert_eq!(http.requests().len(), 1, "HTTP client error must not retry either");
			assert!(events.lock().expect("events").is_empty());
			done.send(()).expect("finish server");
			peer.finish().await;
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_503_retries_identical_request_after_retry_after_with_secret_free_observer() {
		let (listener, url) = server().await;
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			let original = receive(&mut socket).await;
			send(
				&mut socket,
				503,
				json!({"retry-after":"1"}),
				json!({"error":{"message":"private upstream detail transport-test-key"}}),
			)
			.await;
			let started = Instant::now();
			let retry = tokio::select! {
				frame = socket.next() => match frame {
					Some(Ok(Frame::Text(text))) => serde_json::from_str::<Value>(&text).expect("retry JSON"),
					_ => { socket = accept(&listener).await.0; receive(&mut socket).await }
				},
				fresh = listener.accept() => {
					let (stream, _) = fresh.expect("retry connection");
					socket = upgrade(stream).await.0;
					receive(&mut socket).await
				}
			};
			assert_eq!(retry, original, "retry the full request, never a partial continuation");
			assert!(
				started.elapsed() >= Duration::from_millis(900),
				"Retry-After was skipped"
			);
			success(&mut socket).await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let mut request = ws_params()
			.with_prompt_cache_key("retry-503")
			.with_max_retries(1)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut request);
		assert_rich_response(
			&bounded(OctoHubProvider::new().chat_completion(request))
				.await
				.expect("retry succeeded"),
		);
		assert_eq!(
			*events.lock().expect("events"),
			vec![RetryEvent {
				attempt: 1,
				max_retries: 1,
				delay: Duration::from_secs(1),
				reason: "HTTP 503".to_owned(),
			}]
		);
		let debug = format!("{:?}", events.lock().expect("events"));
		assert!(!debug.contains("private upstream detail"));
		assert!(!debug.contains("transport-test-key"));
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_exhausted_503_preserves_retry_metadata_and_redacts_error_chain() {
		let (listener, url) = server().await;
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			receive(&mut socket).await;
			send(
				&mut socket,
				503,
				json!({"retry-after":"7"}),
				json!({"error":{"message":"busy transport-test-key"}}),
			)
			.await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let error = bounded(OctoHubProvider::new().chat_completion(ws_params()))
			.await
			.expect_err("503 exhausted");
		let status = error
			.downcast_ref::<RetryableStatusError>()
			.expect("typed retryable error");
		assert_eq!(status.status, 503);
		assert_eq!(status.retry_after, Some(Duration::from_secs(7)));
		for rendered in [error.to_string(), format!("{error:#}"), format!("{error:?}")] {
			assert!(rendered.contains("503"));
			assert!(
				!rendered.contains("transport-test-key"),
				"credential leaked in error rendering"
			);
		}
		for cause in error.chain() {
			assert!(!cause.to_string().contains("transport-test-key"));
		}
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_nonretryable_errors_redact_echoed_request_key() {
		for status in [400, 401, 403] {
			let (listener, url) = server().await;
			let peer = Peer::spawn(async move {
				let (mut socket, _) = accept(&listener).await;
				receive(&mut socket).await;
				send(
					&mut socket,
					status,
					json!({}),
					json!({"error":{"message":"rejected transport-test-key"}}),
				)
				.await;
			});
			let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
			let error = bounded(OctoHubProvider::new().chat_completion(ws_params()))
				.await
				.expect_err("rejected key");
			for rendered in [error.to_string(), format!("{error:#}"), format!("{error:?}")] {
				assert!(
					!rendered.contains("transport-test-key"),
					"credential leaked for status {status}"
				);
				assert!(rendered.contains("rejected"), "nonsecret error context lost");
			}
			peer.finish().await;
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_ambiguous_close_never_resends_or_falls_back_despite_retry_budget() {
		let (listener, url) = server().await;
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			assert_eq!(receive(&mut socket).await["input"][0]["content"], "hello 日本語 🦀");
			drop(socket); // Execution/billing may have happened; no completed envelope arrived.
			tokio::select! {
				result = finished => result.expect("caller finished"),
				connection = listener.accept() => {
					connection.expect("unexpected connection");
					panic!("ambiguous completion was replayed or fell back to HTTP");
				}
			}
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let mut request = ws_params().with_max_retries(3).with_retry_timeout(Duration::ZERO);
		let events = observe(&mut request);
		bounded(OctoHubProvider::new().chat_completion(request))
			.await
			.expect_err("ambiguous close");
		assert!(events.lock().expect("events").is_empty());
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	fn assert_timeout(error: &anyhow::Error) {
		let timeout = error
			.chain()
			.find_map(|cause| cause.downcast_ref::<crate::errors::ProviderError>());
		assert!(
			matches!(timeout, Some(crate::errors::ProviderError::TimeoutError { provider }) if provider == "octohub"),
			"expected OctoHub timeout, got {error:#}"
		);
	}

	async fn no_reconnect(listener: &TcpListener, done: oneshot::Receiver<()>) {
		tokio::select! {
			result = done => result.expect("caller finished"),
			connection = listener.accept() => {
				connection.expect("unexpected connection");
				panic!("timed-out request reconnected or fell back to HTTP");
			}
		}
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_some_timeout_bounds_stalled_setup_without_retry_or_http_fallback() {
		let (listener, url) = server().await;
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (stream, _) = bounded(listener.accept()).await.expect("accept stalled handshake");
			// Keep TCP open but never reply to the upgrade request.
			no_reconnect(&listener, finished).await;
			drop(stream);
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let mut request = ws_params()
			.with_request_timeout(Some(Duration::from_millis(200)))
			.with_max_retries(3)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut request);
		let started = Instant::now();
		let error = bounded(OctoHubProvider::new().chat_completion(request))
			.await
			.expect_err("setup deadline");
		assert_timeout(&error);
		assert!(
			started.elapsed() < Duration::from_secs(2),
			"setup ignored total request timeout"
		);
		assert!(events.lock().expect("events").is_empty());
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_some_timeout_is_one_budget_across_setup_and_response_read() {
		let (listener, url) = server().await;
		let (done, mut finished) = oneshot::channel();
		let (read_request, received) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (stream, _) = bounded(listener.accept()).await.expect("accept delayed handshake");
			tokio::time::sleep(Duration::from_millis(350)).await;
			let (mut socket, _) = upgrade(stream).await;
			receive(&mut socket).await;
			read_request.send(()).expect("request reached read phase");
			// Each phase alone fits 600 ms; their sum does not. Resetting the
			// timeout after setup incorrectly turns this into a successful call.
			tokio::select! {
				result = &mut finished => { result.expect("caller finished"); return; }
				_ = tokio::time::sleep(Duration::from_millis(400)) => success(&mut socket).await,
			}
			no_replay(&listener, &mut socket, finished).await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let error = bounded(
			OctoHubProvider::new().chat_completion(ws_params().with_request_timeout(Some(Duration::from_millis(600)))),
		)
		.await
		.expect_err("one total deadline, not a fresh read timeout");
		assert_timeout(&error);
		bounded(received)
			.await
			.expect("setup completed and request was sent before timeout");
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_some_timeout_includes_session_lease_wait_without_sending_queued_request() {
		let (listener, url) = server().await;
		let (ready, received) = oneshot::channel();
		let (release, released) = oneshot::channel();
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			assert_eq!(receive(&mut socket).await["input"][0]["content"], "first holds lease");
			ready.send(()).expect("first request in flight");
			tokio::select! {
				result = released => result.expect("release first request"),
				frame = socket.next() => panic!("queued request escaped the session lease: {frame:?}"),
				connection = listener.accept() => {
					connection.expect("unexpected connection");
					panic!("queued request opened a second connection");
				}
			}
			success(&mut socket).await;
			no_replay(&listener, &mut socket, finished).await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let provider = OctoHubProvider::new();
		let mut first_params = ws_params().with_prompt_cache_key("lease-deadline");
		first_params.messages = vec![Message::user("first holds lease")];
		let mut first = Box::pin(provider.chat_completion(first_params));
		tokio::select! {
			result = &mut first => panic!("first completed before server release: {result:?}"),
			result = received => result.expect("first request arrived"),
		}
		let mut second = ws_params()
			.with_prompt_cache_key("lease-deadline")
			.with_request_timeout(Some(Duration::from_millis(200)))
			.with_max_retries(3)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut second);
		let started = Instant::now();
		let error = bounded(provider.chat_completion(second))
			.await
			.expect_err("lease wait deadline");
		assert_timeout(&error);
		assert!(
			started.elapsed() < Duration::from_secs(2),
			"lease wait escaped total deadline"
		);
		assert!(events.lock().expect("events").is_empty());
		release.send(()).expect("release first request");
		assert_rich_response(&bounded(first).await.expect("first request still succeeds"));
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_some_timeout_bounds_backpressured_send_without_resend() {
		// A small receive window plus an unread 15 MiB message keeps the send
		// pending, rather than accidentally testing only a missing reply.
		let socket = tokio::net::TcpSocket::new_v4().expect("IPv4 socket");
		socket.set_recv_buffer_size(1024).expect("small receive buffer");
		socket
			.bind("127.0.0.1:0".parse().expect("loopback address"))
			.expect("bind");
		let listener = socket.listen(8).expect("listen");
		let url = format!("http://{}", listener.local_addr().expect("loopback address"));
		let (ready, received) = oneshot::channel();
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (socket, _) = accept(&listener).await;
			ready.send(()).expect("handshake finished");
			// Do not poll this WebSocket, so its receive window fills.
			no_reconnect(&listener, finished).await;
			drop(socket);
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let mut request = ws_params()
			.with_request_timeout(Some(Duration::from_millis(500)))
			.with_max_retries(3)
			.with_retry_timeout(Duration::ZERO);
		request.messages = vec![Message::user(&"x".repeat(15 * 1024 * 1024))];
		let events = observe(&mut request);
		let error = bounded(OctoHubProvider::new().chat_completion(request))
			.await
			.expect_err("send deadline");
		bounded(received).await.expect("send reached an upgraded socket");
		assert_timeout(&error);
		assert!(events.lock().expect("events").is_empty());
		done.send(()).expect("finish server");
		peer.finish().await;
	}

	#[tokio::test]
	#[serial_test::serial]
	async fn octohub_websocket_some_timeout_bounds_retry_after_in_the_same_total_budget() {
		let (listener, url) = server().await;
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let (mut socket, _) = accept(&listener).await;
			receive(&mut socket).await;
			send(
				&mut socket,
				503,
				json!({"retry-after":"30"}),
				json!({"error":{"message":"busy"}}),
			)
			.await;
			no_replay(&listener, &mut socket, finished).await;
		});
		let _env = EnvGuard::new().set("OCTOHUB_API_URL", url).remove("OCTOHUB_API_KEY");
		let mut request = ws_params()
			.with_request_timeout(Some(Duration::from_millis(200)))
			.with_max_retries(3)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut request);
		let started = Instant::now();
		let error = bounded(OctoHubProvider::new().chat_completion(request))
			.await
			.expect_err("retry deadline");
		assert_timeout(&error);
		assert!(started.elapsed() < Duration::from_secs(2));
		assert_eq!(
			*events.lock().expect("events"),
			vec![RetryEvent {
				attempt: 1,
				max_retries: 3,
				delay: Duration::from_secs(30),
				reason: "HTTP 503".to_owned(),
			}]
		);
		done.send(()).expect("finish server");
		peer.finish().await;
	}
}
