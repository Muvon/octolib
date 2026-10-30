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
use crate::llm::retry::{RetryEvent, RetryObserver};
use crate::llm::types::ChatCompletionParams;
use futures_util::{SinkExt, StreamExt};
use reqwest::header::HeaderMap;
use serde_json::{json, Value};
use std::future::Future;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::{
	handshake::server::{Callback, ErrorResponse, Request, Response},
	protocol::{
		frame::{
			coding::{Data, OpCode},
			Frame,
		},
		WebSocketConfig,
	},
	Message as ServerMessage,
};
use tokio_tungstenite::{accept_hdr_async_with_config, WebSocketStream};

type ServerSocket = WebSocketStream<TcpStream>;
const TEST_DEADLINE: Duration = Duration::from_secs(10);
const WIRE_REQUEST_LIMIT: usize = 16 * 1024 * 1024;
const WIRE_REPLY_LIMIT: usize = 64 * 1024 * 1024;

// Every wait is bounded, and dropping a failed test aborts its mock server.
// Each test gets its own ephemeral endpoint, including its global-pool namespace.
async fn bounded<T>(future: impl Future<Output = T>) -> T {
	tokio::time::timeout(TEST_DEADLINE, future)
		.await
		.expect("mock WebSocket operation did not finish")
}

struct Peer(Option<tokio::task::JoinHandle<()>>);

impl Peer {
	fn spawn(future: impl Future<Output = ()> + Send + 'static) -> Self {
		Self(Some(tokio::spawn(future)))
	}

	async fn finish(mut self) {
		bounded(self.0.take().unwrap()).await.unwrap();
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
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
	(listener, url)
}

fn headers(token: &str, route: &str) -> HeaderMap {
	let mut headers = HeaderMap::new();
	headers.insert("authorization", format!("Bearer {token}").parse().unwrap());
	headers.insert("x-test-route", route.parse().unwrap());
	headers
}

fn params(key: Option<&str>) -> ChatCompletionParams {
	let mut params = ChatCompletionParams::new(&[], "test-model", 0.0, 1.0, 0, 32)
		.with_max_retries(0)
		.with_request_timeout(Some(Duration::from_secs(5)));
	params.prompt_cache_key = key.map(str::to_owned);
	params
}

fn body(id: &str) -> Value {
	json!({"model":"test-model", "messages":[{"role":"user", "content":id}], "stream":false})
}

fn response_body(id: &str) -> Value {
	json!({"id":id, "choices":[{"message":{"role":"assistant", "content":id}}]})
}

fn envelope(status: u16, body: Value) -> Value {
	json!({"status":status, "headers":{"content-type":"application/json"}, "body":body})
}

struct CheckHandshake<'a>(&'a HeaderMap);

impl Callback for CheckHandshake<'_> {
	fn on_request(self, request: &Request, response: Response) -> std::result::Result<Response, ErrorResponse> {
		assert_eq!(request.method().as_str(), "GET");
		assert_eq!(
			request.version(),
			tokio_tungstenite::tungstenite::http::Version::HTTP_11
		);
		assert_eq!(request.uri().path(), "/v1/completions");
		assert_eq!(
			request.headers()["upgrade"].to_str().unwrap().to_ascii_lowercase(),
			"websocket"
		);
		assert!(request.headers()["connection"]
			.to_str()
			.unwrap()
			.split(',')
			.any(|part| part.trim().eq_ignore_ascii_case("upgrade")));
		for (name, value) in self.0 {
			assert_eq!(request.headers()[name.as_str()].as_bytes(), value.as_bytes());
		}
		Ok(response)
	}
}

async fn accept(listener: &TcpListener, expected_headers: &HeaderMap) -> ServerSocket {
	let (stream, _) = bounded(listener.accept()).await.unwrap();
	let callback = CheckHandshake(expected_headers);
	// Let the application, not the mock's frame-size default, enforce 16 MiB.
	let mut config = WebSocketConfig::default();
	config.max_frame_size = Some(WIRE_REQUEST_LIMIT * 2);
	config.max_message_size = Some(WIRE_REQUEST_LIMIT * 2);
	bounded(accept_hdr_async_with_config(stream, callback, Some(config)))
		.await
		.unwrap()
}

async fn frame(socket: &mut ServerSocket) -> ServerMessage {
	bounded(socket.next()).await.unwrap().unwrap()
}

async fn receive(socket: &mut ServerSocket) -> Value {
	loop {
		let message = frame(socket).await;
		if matches!(message, ServerMessage::Ping(_)) {
			socket.flush().await.unwrap();
			continue;
		}
		assert!(message.is_text(), "request must be one JSON text message");
		return serde_json::from_str(message.to_text().unwrap()).unwrap();
	}
}

async fn send(socket: &mut ServerSocket, value: Value) {
	bounded(socket.send(ServerMessage::Text(value.to_string().into())))
		.await
		.unwrap();
}

async fn succeed(socket: &mut ServerSocket, id: &str) {
	send(socket, envelope(200, response_body(id))).await;
}

async fn call(
	params: &ChatCompletionParams,
	url: &str,
	headers: &HeaderMap,
	body: &Value,
) -> anyhow::Result<shared::CapturedResponse> {
	bounded(complete(params, url, "resolved-key", headers, body)).await
}

fn assert_success(response: shared::CapturedResponse, id: &str) {
	assert_eq!(response.status.as_u16(), 200);
	assert_eq!(response.headers["content-type"], "application/json");
	assert_eq!(
		serde_json::from_str::<Value>(&response.body).unwrap(),
		response_body(id)
	);
}

fn observe(params: &mut ChatCompletionParams) -> Arc<StdMutex<Vec<RetryEvent>>> {
	let events = Arc::new(StdMutex::new(Vec::new()));
	let captured = events.clone();
	params.retry_observer = Some(RetryObserver::new(move |event| {
		captured.lock().unwrap().push(event.clone());
	}));
	events
}

// Keep both the listener and the old socket live until complete() returns.
// A replay must not be hidden by dropping the listener after the first reply.
async fn no_replay(listener: &TcpListener, socket: &mut ServerSocket, mut done: oneshot::Receiver<()>) {
	loop {
		tokio::select! {
			result = &mut done => { result.unwrap(); return; }
			result = listener.accept() => {
				result.unwrap();
				panic!("terminal outcome opened another socket");
			}
			message = socket.next() => match message {
				Some(Ok(ServerMessage::Text(_) | ServerMessage::Binary(_))) => panic!("terminal outcome resent the request"),
				Some(Ok(ServerMessage::Ping(_))) => { socket.flush().await.unwrap(); }
				Some(Ok(_)) => {}
				Some(Err(_)) | None => {
					tokio::select! {
						result = &mut done => { result.unwrap(); return; }
						result = listener.accept() => { result.unwrap(); panic!("terminal outcome reconnected"); }
					}
				}
			}
		}
	}
}

#[tokio::test]
async fn complete_reuses_socket_and_transmits_full_json_without_a_protocol_wrapper() {
	let (listener, url) = server().await;
	let auth = headers("reuse", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		for id in ["first", "next: 日本語 🦀"] {
			assert_eq!(receive(&mut socket).await, body(id));
			succeed(&mut socket, id).await;
		}
	});
	let params = params(Some("reuse"));
	for id in ["first", "next: 日本語 🦀"] {
		assert_success(call(&params, &url, &auth, &body(id)).await.unwrap(), id);
	}
	peer.finish().await;
}

#[tokio::test]
async fn complete_isolates_auth_headers_and_session_keys_on_live_sockets() {
	let (listener, url) = server().await;
	let cases = [
		("token-a", "route-a", "session-a"),
		("token-b", "route-a", "session-a"),
		("token-a", "route-b", "session-a"),
		("token-a", "route-a", "session-b"),
	];
	let peer = Peer::spawn(async move {
		let mut sockets = Vec::new();
		for (index, (token, route, _)) in cases.into_iter().enumerate() {
			let mut socket = accept(&listener, &headers(token, route)).await;
			let id = index.to_string();
			assert_eq!(receive(&mut socket).await, body(&id));
			succeed(&mut socket, &id).await;
			sockets.push(socket);
		}
		// Returning to the original identity must find its original socket.
		assert_eq!(receive(&mut sockets[0]).await, body("original"));
		succeed(&mut sockets[0], "original").await;
	});
	for (index, (token, route, key)) in cases.into_iter().enumerate() {
		let id = index.to_string();
		assert_success(
			call(&params(Some(key)), &url, &headers(token, route), &body(&id))
				.await
				.unwrap(),
			&id,
		);
	}
	assert_success(
		call(
			&params(Some("session-a")),
			&url,
			&headers("token-a", "route-a"),
			&body("original"),
		)
		.await
		.unwrap(),
		"original",
	);
	peer.finish().await;
}

#[tokio::test]
async fn complete_isolates_endpoints_even_with_identical_auth_and_session_keys() {
	let (first_listener, first_url) = server().await;
	let (second_listener, second_url) = server().await;
	let auth = headers("endpoints", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut first = accept(&first_listener, &expected).await;
		assert_eq!(receive(&mut first).await, body("a1"));
		succeed(&mut first, "a1").await;
		let mut second = accept(&second_listener, &expected).await;
		assert_eq!(receive(&mut second).await, body("b1"));
		succeed(&mut second, "b1").await;
		assert_eq!(receive(&mut first).await, body("a2"));
		succeed(&mut first, "a2").await;
		assert_eq!(receive(&mut second).await, body("b2"));
		succeed(&mut second, "b2").await;
	});
	let params = params(Some("same-session"));
	for (url, id) in [
		(&first_url, "a1"),
		(&second_url, "b1"),
		(&first_url, "a2"),
		(&second_url, "b2"),
	] {
		assert_success(call(&params, url, &auth, &body(id)).await.unwrap(), id);
	}
	peer.finish().await;
}

#[tokio::test]
async fn complete_without_a_cache_key_opens_a_fresh_socket_for_every_request() {
	let (listener, url) = server().await;
	let auth = headers("no-key", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut sockets = Vec::new();
		for id in ["first", "second"] {
			let mut socket = accept(&listener, &expected).await;
			assert_eq!(receive(&mut socket).await, body(id));
			succeed(&mut socket, id).await;
			sockets.push(socket);
		}
	});
	let params = params(None);
	for id in ["first", "second"] {
		assert_success(call(&params, &url, &auth, &body(id)).await.unwrap(), id);
	}
	peer.finish().await;
}

#[tokio::test]
async fn complete_serializes_concurrent_requests_and_keeps_responses_with_their_callers() {
	let (listener, url) = server().await;
	let auth = headers("concurrent", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		let first = receive(&mut socket).await;
		let first_id = first["messages"][0]["content"].as_str().unwrap();
		assert!(matches!(first_id, "first" | "second"));
		// The second future is polled by join!, but must not put a frame on
		// this socket or create a new socket while the first reply is held.
		tokio::select! {
			_ = tokio::time::sleep(Duration::from_millis(75)) => {}
			_ = socket.next() => panic!("second request was pipelined before the first response"),
			_ = listener.accept() => panic!("concurrent calls opened separate sockets"),
		}
		succeed(&mut socket, first_id).await;
		let second = receive(&mut socket).await;
		let second_id = second["messages"][0]["content"].as_str().unwrap();
		assert_ne!(first_id, second_id);
		assert!(matches!(second_id, "first" | "second"));
		succeed(&mut socket, second_id).await;
	});
	let params = params(Some("serial"));
	let first = body("first");
	let second = body("second");
	let (first_result, second_result) =
		tokio::join!(call(&params, &url, &auth, &first), call(&params, &url, &auth, &second),);
	assert_success(first_result.unwrap(), "first");
	assert_success(second_result.unwrap(), "second");
	peer.finish().await;
}

#[derive(Clone, Copy, Debug)]
enum Interruption {
	Cancel,
	DropFuture,
	AbortTask,
	Timeout,
}

async fn interruption_evicts_socket(mode: Interruption) {
	let (listener, url) = server().await;
	let auth = headers("interrupt", "one");
	let expected = auth.clone();
	let (started, received) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut abandoned = accept(&listener, &expected).await;
		assert_eq!(receive(&mut abandoned).await, body("prime"));
		succeed(&mut abandoned, "prime").await;
		assert_eq!(receive(&mut abandoned).await, body("abandon"));
		started.send(()).unwrap();
		// Keep the abandoned socket open: reconnecting must be the client's
		// decision, not the consequence of the mock closing it first.
		let mut fresh = accept(&listener, &expected).await;
		assert_eq!(receive(&mut fresh).await, body("fresh"));
		succeed(&mut fresh, "fresh").await;
		drop(abandoned);
	});
	let clean = params(Some("interrupted"));
	assert_success(call(&clean, &url, &auth, &body("prime")).await.unwrap(), "prime");
	let mut interrupted = clean.clone().with_max_retries(3).with_retry_timeout(Duration::ZERO);
	let events = observe(&mut interrupted);
	let (cancel, cancellation) = tokio::sync::watch::channel(false);
	if matches!(mode, Interruption::Cancel) {
		interrupted.cancellation_token = Some(cancellation);
	}
	if matches!(mode, Interruption::Timeout) {
		interrupted.request_timeout = Some(Duration::from_millis(150));
	}
	let request = body("abandon");
	if matches!(mode, Interruption::AbortTask) {
		let task_url = url.clone();
		let task_auth = auth.clone();
		let task =
			tokio::spawn(async move { complete(&interrupted, &task_url, "resolved-key", &task_auth, &request).await });
		bounded(received).await.unwrap();
		task.abort();
		assert!(bounded(task).await.err().unwrap().is_cancelled());
	} else {
		let mut pending = Box::pin(complete(&interrupted, &url, "resolved-key", &auth, &request));
		tokio::select! {
			_ = &mut pending => panic!("request finished before interruption: {mode:?}"),
			result = bounded(received) => result.unwrap(),
		}
		match mode {
			Interruption::Cancel => {
				cancel.send(true).unwrap();
				assert!(bounded(&mut pending).await.is_err());
			}
			Interruption::Timeout => assert!(bounded(&mut pending).await.is_err()),
			Interruption::DropFuture => {}
			Interruption::AbortTask => unreachable!(),
		}
		drop(pending);
	}
	assert!(events.lock().unwrap().is_empty(), "interruptions must not be retried");
	assert_success(call(&clean, &url, &auth, &body("fresh")).await.unwrap(), "fresh");
	peer.finish().await;
}

#[tokio::test]
async fn complete_cancelled_in_flight_evicts_the_leased_socket() {
	interruption_evicts_socket(Interruption::Cancel).await;
}

#[tokio::test]
async fn complete_dropped_future_evicts_the_leased_socket() {
	interruption_evicts_socket(Interruption::DropFuture).await;
}

#[tokio::test]
async fn complete_aborted_task_evicts_the_leased_socket() {
	interruption_evicts_socket(Interruption::AbortTask).await;
}

#[tokio::test]
async fn complete_timed_out_in_flight_evicts_the_leased_socket() {
	interruption_evicts_socket(Interruption::Timeout).await;
}

#[tokio::test]
async fn complete_flushes_matching_pong_before_waiting_for_the_envelope() {
	let (listener, url) = server().await;
	let auth = headers("ping", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("ping"));
		for payload in [vec![], vec![0, 1, 127, 255]] {
			socket.send(ServerMessage::Ping(payload.clone().into())).await.unwrap();
			match frame(&mut socket).await {
				ServerMessage::Pong(actual) => assert_eq!(actual.as_ref(), payload.as_slice()),
				message => panic!("expected matching pong, got {message:?}"),
			}
		}
		succeed(&mut socket, "ping").await;
	});
	assert_success(
		call(&params(Some("ping")), &url, &auth, &body("ping")).await.unwrap(),
		"ping",
	);
	peer.finish().await;
}

#[tokio::test]
async fn complete_rejects_malformed_envelopes_without_retry_and_reconnects() {
	let valid = envelope(200, json!({}));
	let mut invalid = vec![
		"not JSON".to_owned(),
		"{".to_owned(),
		"null".to_owned(),
		"[]".to_owned(),
		json!({"headers":{}, "body":{}}).to_string(),
		json!({"status":200, "body":{}}).to_string(),
		json!({"status":200, "headers":{}}).to_string(),
	];
	for status in [
		json!(99),
		json!(600),
		json!(-1),
		json!(200.5),
		json!("200"),
		Value::Null,
	] {
		let mut value = valid.clone();
		value["status"] = status;
		invalid.push(value.to_string());
	}
	for header in [
		json!([]),
		Value::Null,
		json!({"x-test":7}),
		json!({"x-test":null}),
		json!({"Content-Type":"application/json"}),
		json!({"bad name":"value"}),
		json!({"x-test":"bad\r\nvalue"}),
	] {
		let mut value = valid.clone();
		value["headers"] = header;
		invalid.push(value.to_string());
	}
	for non_object in [Value::Null, json!([]), json!("body"), json!(1), json!(true)] {
		let mut value = valid.clone();
		value["body"] = non_object;
		invalid.push(value.to_string());
	}
	for (index, wire) in invalid.into_iter().enumerate() {
		let (listener, url) = server().await;
		let auth = headers("malformed", "one");
		let expected = auth.clone();
		let peer = Peer::spawn(async move {
			let mut broken = accept(&listener, &expected).await;
			assert_eq!(receive(&mut broken).await, body("bad"));
			broken.send(ServerMessage::Text(wire.into())).await.unwrap();
			let mut fresh = accept(&listener, &expected).await;
			assert_eq!(receive(&mut fresh).await, body("fresh"));
			succeed(&mut fresh, "fresh").await;
			drop(broken);
		});
		let clean = params(Some("malformed"));
		let mut failing = clean.clone().with_max_retries(2).with_retry_timeout(Duration::ZERO);
		let events = observe(&mut failing);
		assert!(
			call(&failing, &url, &auth, &body("bad")).await.is_err(),
			"malformed case {index} was accepted"
		);
		assert!(events.lock().unwrap().is_empty(), "malformed case {index} retried");
		assert_success(call(&clean, &url, &auth, &body("fresh")).await.unwrap(), "fresh");
		peer.finish().await;
	}
}

#[tokio::test]
async fn complete_rejects_binary_even_when_its_bytes_are_a_valid_json_envelope() {
	let (listener, url) = server().await;
	let auth = headers("binary", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut broken = accept(&listener, &expected).await;
		assert_eq!(receive(&mut broken).await, body("binary"));
		broken
			.send(ServerMessage::Binary(
				envelope(200, response_body("binary")).to_string().into_bytes().into(),
			))
			.await
			.unwrap();
		let mut fresh = accept(&listener, &expected).await;
		assert_eq!(receive(&mut fresh).await, body("fresh"));
		succeed(&mut fresh, "fresh").await;
		drop(broken);
	});
	let mut params = params(Some("binary"))
		.with_max_retries(2)
		.with_retry_timeout(Duration::ZERO);
	let events = observe(&mut params);
	assert!(call(&params, &url, &auth, &body("binary")).await.is_err());
	assert!(events.lock().unwrap().is_empty());
	assert_success(call(&params, &url, &auth, &body("fresh")).await.unwrap(), "fresh");
	peer.finish().await;
}

#[tokio::test]
async fn complete_preserves_envelope_status_headers_and_object_body_at_status_boundaries() {
	// 599 is valid but retryable; zero retries still returns its typed error.
	for status in [100, 200, 299, 399, 499, 599] {
		let (listener, url) = server().await;
		let auth = headers("status", "one");
		let expected = auth.clone();
		let peer = Peer::spawn(async move {
			let mut socket = accept(&listener, &expected).await;
			assert_eq!(receive(&mut socket).await, body("status"));
			send(&mut socket, json!({"status":status, "headers":{"content-type":"application/json", "x-trace-id":"trace-123"}, "body":{"nested":[1,"日本語",null]}})).await;
		});
		let result = call(&params(None), &url, &auth, &body("status")).await;
		if status == 599 {
			let error = result.err().expect("599 must be a retryable status error");
			assert_eq!(
				error
					.downcast_ref::<crate::llm::retry::RetryableStatusError>()
					.unwrap()
					.status,
				599
			);
		} else {
			let response = result.unwrap();
			assert_eq!(response.status.as_u16(), status);
			assert_eq!(response.headers["x-trace-id"], "trace-123");
			assert_eq!(response.headers["content-type"], "application/json");
			assert_eq!(
				serde_json::from_str::<Value>(&response.body).unwrap(),
				json!({"nested":[1,"日本語",null]})
			);
		}
		peer.finish().await;
	}
}

fn sized_envelope(bytes: usize) -> String {
	let mut value = envelope(200, json!({"payload":""}));
	let overhead = value.to_string().len();
	value["body"]["payload"] = Value::String("x".repeat(bytes - overhead));
	let wire = value.to_string();
	assert_eq!(wire.len(), bytes);
	wire
}

#[tokio::test]
async fn complete_accepts_single_frame_envelopes_above_sixteen_mib_through_exactly_sixty_four_mib() {
	for bytes in [WIRE_REQUEST_LIMIT + 1, WIRE_REPLY_LIMIT] {
		let (listener, url) = server().await;
		let auth = headers("large-reply", "one");
		let expected = auth.clone();
		let wire = sized_envelope(bytes);
		let payload_bytes = bytes - envelope(200, json!({"payload":""})).to_string().len();
		let peer = Peer::spawn(async move {
			let mut socket = accept(&listener, &expected).await;
			assert_eq!(receive(&mut socket).await, body("large"));
			let reply = Frame::message(wire.into_bytes(), OpCode::Data(Data::Text), true);
			bounded(socket.send(ServerMessage::Frame(reply))).await.unwrap();
			assert_eq!(receive(&mut socket).await, body("reused"));
			succeed(&mut socket, "reused").await;
		});
		let params = params(Some("large-reply"));
		let response = call(&params, &url, &auth, &body("large")).await.unwrap();
		assert_eq!(response.status.as_u16(), 200);
		assert_eq!(response.headers["content-type"], "application/json");
		let body: Value = serde_json::from_str(&response.body).unwrap();
		let payload = body["payload"].as_str().unwrap();
		assert_eq!(payload.len(), payload_bytes);
		assert!(payload.bytes().all(|byte| byte == b'x'));
		assert_success(
			call(&params, &url, &auth, &self::body("reused")).await.unwrap(),
			"reused",
		);
		peer.finish().await;
	}
}

#[tokio::test]
async fn complete_rejects_reply_frames_one_byte_over_sixty_four_mib_without_replay() {
	let (listener, url) = server().await;
	let auth = headers("reply-frame-over", "one");
	let expected = auth.clone();
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("bill-once"));
		// The frame header alone must fail before buffering an oversized payload.
		let mut header = vec![0x81, 127];
		header.extend_from_slice(&((WIRE_REPLY_LIMIT + 1) as u64).to_be_bytes());
		bounded(socket.get_mut().write_all(&header)).await.unwrap();
		no_replay(&listener, &mut socket, finished).await;
	});
	let mut params = params(Some("reply-frame-over"))
		.with_max_retries(3)
		.with_retry_timeout(Duration::ZERO);
	let events = observe(&mut params);
	let error = call(&params, &url, &auth, &body("bill-once")).await.err().unwrap();
	assert!(
		matches!(
			error.downcast_ref::<reqwest_websocket::Error>(),
			Some(reqwest_websocket::Error::Tungstenite(tungstenite::Error::Capacity(
				tungstenite::error::CapacityError::MessageTooLong { size, max_size }
			))) if *size == WIRE_REPLY_LIMIT + 1 && *max_size == WIRE_REPLY_LIMIT
		),
		"{error:#}"
	);
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;
}

#[tokio::test]
async fn complete_rejects_aggregate_replies_one_byte_over_sixty_four_mib_without_replay() {
	let (listener, url) = server().await;
	let auth = headers("reply-message-over", "one");
	let expected = auth.clone();
	let wire = sized_envelope(WIRE_REPLY_LIMIT + 1).into_bytes();
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("bill-once"));
		for (index, chunk) in wire.chunks(WIRE_REQUEST_LIMIT).enumerate() {
			let opcode = if index == 0 { Data::Text } else { Data::Continue };
			let reply = Frame::message(
				chunk.to_vec(),
				OpCode::Data(opcode),
				(index + 1) * WIRE_REQUEST_LIMIT >= wire.len(),
			);
			bounded(socket.send(ServerMessage::Frame(reply))).await.unwrap();
		}
		no_replay(&listener, &mut socket, finished).await;
	});
	let mut params = params(Some("reply-message-over"))
		.with_max_retries(3)
		.with_retry_timeout(Duration::ZERO);
	let events = observe(&mut params);
	let error = call(&params, &url, &auth, &body("bill-once")).await.err().unwrap();
	assert!(
		matches!(
			error.downcast_ref::<reqwest_websocket::Error>(),
			Some(reqwest_websocket::Error::Tungstenite(tungstenite::Error::Capacity(
				tungstenite::error::CapacityError::MessageTooLong { size, max_size }
			))) if *size == WIRE_REPLY_LIMIT + 1 && *max_size == WIRE_REPLY_LIMIT
		),
		"{error:#}"
	);
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;
}

fn sized_request(bytes: usize) -> Value {
	let overhead = json!({"payload":""}).to_string().len();
	let payload_bytes = bytes - overhead;
	// Non-ASCII catches character-count caps accidentally used as byte caps.
	let mut payload = "🦀".repeat(payload_bytes / 4);
	payload.push_str(&"x".repeat(payload_bytes % 4));
	let request = json!({"payload":payload});
	assert_eq!(serde_json::to_vec(&request).unwrap().len(), bytes);
	request
}

#[tokio::test]
async fn complete_accepts_a_request_of_exactly_sixteen_mib_serialized_utf8() {
	let (listener, url) = server().await;
	let auth = headers("cap-exact", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		let message = frame(&mut socket).await;
		assert!(message.is_text());
		assert_eq!(message.to_text().unwrap().len(), WIRE_REQUEST_LIMIT);
		assert_eq!(
			serde_json::from_str::<Value>(message.to_text().unwrap()).unwrap(),
			sized_request(WIRE_REQUEST_LIMIT)
		);
		succeed(&mut socket, "at-limit").await;
	});
	assert_success(
		call(&params(None), &url, &auth, &sized_request(WIRE_REQUEST_LIMIT))
			.await
			.unwrap(),
		"at-limit",
	);
	peer.finish().await;
}

#[tokio::test]
async fn complete_rejects_a_request_one_byte_over_sixteen_mib_without_sending_it() {
	let (listener, url) = server().await;
	let auth = headers("cap-over", "one");
	let expected = auth.clone();
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		// A size check may run before or after connecting, but never after
		// transmitting the oversized request. Accept either valid ordering.
		let mut finished = finished;
		tokio::select! {
			result = &mut finished => result.unwrap(),
			mut socket = accept(&listener, &expected) => no_replay(&listener, &mut socket, finished).await,
		}
	});
	let mut params = params(Some("over-limit"))
		.with_max_retries(3)
		.with_retry_timeout(Duration::ZERO);
	let events = observe(&mut params);
	assert!(call(&params, &url, &auth, &sized_request(WIRE_REQUEST_LIMIT + 1))
		.await
		.is_err());
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;
}

#[tokio::test]
async fn complete_retries_explicit_429_and_503_and_observes_retry_after_without_leaking_body() {
	for status in [429, 503] {
		let (listener, url) = server().await;
		let auth = headers("private-test-token", "one");
		let expected = auth.clone();
		let peer = Peer::spawn(async move {
			let mut socket = accept(&listener, &expected).await;
			assert_eq!(receive(&mut socket).await, body("retry"));
			send(
				&mut socket,
				json!({"status":status, "headers":{"retry-after":"1"}, "body":{"error":"private upstream detail"}}),
			)
			.await;
			let sent_at = Instant::now();
			// A fully framed HTTP error can be followed on the same socket or
			// on a replacement; retry correctness does not require one policy.
			tokio::select! {
				request = socket.next() => {
					match request {
						Some(Ok(ServerMessage::Text(text))) => {
							assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), body("retry"));
						}
						_ => {
							socket = accept(&listener, &expected).await;
							assert_eq!(receive(&mut socket).await, body("retry"));
						}
					}
				}
				fresh = accept(&listener, &expected) => {
					socket = fresh;
					assert_eq!(receive(&mut socket).await, body("retry"));
				}
			}
			assert!(
				sent_at.elapsed() >= Duration::from_millis(900),
				"Retry-After sleep was skipped"
			);
			succeed(&mut socket, "retry").await;
		});
		let mut params = params(Some("retry"))
			.with_max_retries(1)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut params);
		assert_success(call(&params, &url, &auth, &body("retry")).await.unwrap(), "retry");
		assert_eq!(
			*events.lock().unwrap(),
			vec![RetryEvent {
				attempt: 1,
				max_retries: 1,
				delay: Duration::from_secs(1),
				reason: format!("HTTP {status}"),
			}]
		);
		peer.finish().await;
	}
}

#[tokio::test]
async fn complete_401_and_403_are_captured_once_without_retries() {
	for status in [401, 403] {
		let (listener, url) = server().await;
		let auth = headers("unauthorized-test-token", "one");
		let expected = auth.clone();
		let (done, finished) = oneshot::channel();
		let peer = Peer::spawn(async move {
			let mut socket = accept(&listener, &expected).await;
			assert_eq!(receive(&mut socket).await, body("denied"));
			send(
				&mut socket,
				json!({"status":status, "headers":{"retry-after":"1"}, "body":{"error":"denied"}}),
			)
			.await;
			no_replay(&listener, &mut socket, finished).await;
		});
		let mut params = params(Some("denied"))
			.with_max_retries(3)
			.with_retry_timeout(Duration::ZERO);
		let events = observe(&mut params);
		let response = call(&params, &url, &auth, &body("denied")).await.unwrap();
		assert_eq!(response.status.as_u16(), status);
		assert_eq!(response.headers["retry-after"], "1");
		assert_eq!(
			serde_json::from_str::<Value>(&response.body).unwrap(),
			json!({"error":"denied"})
		);
		assert!(events.lock().unwrap().is_empty());
		done.send(()).unwrap();
		peer.finish().await;
	}
}

#[tokio::test]
async fn complete_ambiguous_close_after_request_is_not_replayed_even_with_retries_enabled() {
	let (listener, url) = server().await;
	let auth = headers("ambiguous", "one");
	let expected = auth.clone();
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("bill-once"));
		// No envelope: the server may already have billed/executed this call.
		drop(socket);
		tokio::select! {
			result = finished => result.unwrap(),
			result = listener.accept() => { result.unwrap(); panic!("ambiguous request was replayed"); }
		}
	});
	let mut params = params(Some("ambiguous"))
		.with_max_retries(3)
		.with_retry_timeout(Duration::ZERO);
	let events = observe(&mut params);
	assert!(call(&params, &url, &auth, &body("bill-once")).await.is_err());
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;
}

#[tokio::test]
async fn complete_total_deadline_bounds_retry_after_backoff_and_next_call_can_reconnect() {
	let (listener, url) = server().await;
	let auth = headers("bounded-retry", "one");
	let expected = auth.clone();
	let peer = Peer::spawn(async move {
		let mut old = accept(&listener, &expected).await;
		assert_eq!(receive(&mut old).await, body("rate-limited"));
		send(
			&mut old,
			json!({"status":429, "headers":{"retry-after":"30"}, "body":{"error":"busy"}}),
		)
		.await;
		let mut fresh = accept(&listener, &expected).await;
		assert_eq!(receive(&mut fresh).await, body("fresh"));
		succeed(&mut fresh, "fresh").await;
		drop(old);
	});
	let clean = params(Some("bounded-retry"));
	let mut limited = clean
		.clone()
		.with_max_retries(3)
		.with_retry_timeout(Duration::ZERO)
		.with_request_timeout(Some(Duration::from_millis(150)));
	let events = observe(&mut limited);
	let started = Instant::now();
	assert!(call(&limited, &url, &auth, &body("rate-limited")).await.is_err());
	assert!(
		started.elapsed() < Duration::from_secs(2),
		"request_timeout did not bound retry backoff"
	);
	assert_eq!(
		*events.lock().unwrap(),
		vec![RetryEvent {
			attempt: 1,
			max_retries: 3,
			delay: Duration::from_secs(30),
			reason: "HTTP 429".to_owned(),
		}]
	);
	assert_success(call(&clean, &url, &auth, &body("fresh")).await.unwrap(), "fresh");
	peer.finish().await;
}

#[test]
fn session_key_is_deterministic_order_independent_and_isolates_all_routing_inputs() {
	let url = "http://127.0.0.1:12345/v1/completions";
	let original = headers("a", "route-a");
	let key = session_key(Some("session 日本語"), url, "resolved-key", &original).unwrap();
	assert_eq!(session_key(None, url, "resolved-key", &original), None);
	assert_eq!(
		session_key(Some("session 日本語"), url, "resolved-key", &original),
		Some(key)
	);
	for (session, endpoint, token, effective) in [
		("other", url, "resolved-key", original.clone()),
		(
			"session 日本語",
			"http://127.0.0.1:12346/v1/completions",
			"resolved-key",
			original.clone(),
		),
		(
			"session 日本語",
			"https://127.0.0.1:12345/v1/completions",
			"resolved-key",
			original.clone(),
		),
		("session 日本語", url, "other-resolved-key", original.clone()),
		("session 日本語", url, "resolved-key", headers("b", "route-a")),
		("session 日本語", url, "resolved-key", headers("a", "route-b")),
	] {
		assert_ne!(session_key(Some(session), endpoint, token, &effective), Some(key));
	}
	let mut reversed = HeaderMap::new();
	reversed.insert("x-test-route", "route-a".parse().unwrap());
	reversed.insert("authorization", "Bearer a".parse().unwrap());
	assert_eq!(
		session_key(Some("session 日本語"), url, "resolved-key", &reversed),
		Some(key)
	);
	let mut without_auth = original.clone();
	without_auth.remove("authorization");
	assert_ne!(
		session_key(Some("session 日本語"), url, "resolved-key", &without_auth),
		Some(key)
	);
}

#[test]
fn session_key_does_not_alias_concatenated_fields_or_ignore_repeated_header_values() {
	let url = "http://127.0.0.1:12345/v1/completions";
	let mut first = HeaderMap::new();
	first.insert("x-ab", "c".parse().unwrap());
	let mut second = HeaderMap::new();
	second.insert("x-a", "bc".parse().unwrap());
	assert_ne!(
		session_key(Some("key"), url, "resolved-key", &first),
		session_key(Some("key"), url, "resolved-key", &second)
	);
	let mut repeated = headers("a", "one");
	repeated.append("x-test-route", "two".parse().unwrap());
	let original = session_key(Some("key"), url, "resolved-key", &repeated);
	repeated.append("x-test-route", "three".parse().unwrap());
	assert_ne!(session_key(Some("key"), url, "resolved-key", &repeated), original);
}

#[tokio::test]
async fn setup_timeout_and_server_close_1009_are_terminal_without_replay() {
	let (listener, url) = server().await;
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let (_stream, _) = bounded(listener.accept()).await.unwrap();
		bounded(finished).await.unwrap();
	});
	let mut request = params(None)
		.with_request_timeout(Some(Duration::from_millis(100)))
		.with_max_retries(3);
	let events = observe(&mut request);
	let error = call(&request, &url, &headers("setup", "one"), &body("setup"))
		.await
		.err()
		.unwrap();
	assert!(matches!(
		error.downcast_ref::<ProviderError>(),
		Some(ProviderError::TimeoutError { .. })
	));
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;

	let (listener, url) = server().await;
	let auth = headers("close", "one");
	let expected = auth.clone();
	let (done, finished) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		receive(&mut socket).await;
		socket
			.send(ServerMessage::Close(Some(
				tokio_tungstenite::tungstenite::protocol::CloseFrame {
					code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::Size,
					reason: "request too large".into(),
				},
			)))
			.await
			.unwrap();
		no_replay(&listener, &mut socket, finished).await;
	});
	let mut request = params(None).with_max_retries(3);
	let events = observe(&mut request);
	assert!(call(&request, &url, &auth, &body("oversized")).await.is_err());
	assert!(events.lock().unwrap().is_empty());
	done.send(()).unwrap();
	peer.finish().await;
}

#[tokio::test]
async fn timed_out_queued_call_does_not_steal_or_evict_an_active_lease() {
	let (listener, url) = server().await;
	let auth = headers("queue", "one");
	let expected = auth.clone();
	let (started, received) = oneshot::channel();
	let (release, released) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("first"));
		started.send(()).unwrap();
		released.await.unwrap();
		succeed(&mut socket, "first").await;
		assert_eq!(receive(&mut socket).await, body("next"));
		succeed(&mut socket, "next").await;
	});
	let clean = params(Some("queue"));
	let first_body = body("first");
	let mut first = Box::pin(call(&clean, &url, &auth, &first_body));
	tokio::select! {
		_ = &mut first => panic!("reply held by server"),
		result = received => result.unwrap(),
	}
	let limited = clean.clone().with_request_timeout(Some(Duration::from_millis(50)));
	assert!(call(&limited, &url, &auth, &body("abandoned")).await.is_err());
	release.send(()).unwrap();
	assert_success(first.await.unwrap(), "first");
	assert_success(call(&clean, &url, &auth, &body("next")).await.unwrap(), "next");
	peer.finish().await;
}

#[tokio::test]
async fn idle_ping_and_close_reconnect_before_sending_new_work_with_no_retry_budget() {
	assert_eq!(DEFAULT_REQUEST_TIMEOUT, Duration::from_secs(600));
	let (listener, url) = server().await;
	let auth = headers("idle", "one");
	let expected = auth.clone();
	let (closed, observed) = oneshot::channel();
	let peer = Peer::spawn(async move {
		let mut socket = accept(&listener, &expected).await;
		assert_eq!(receive(&mut socket).await, body("first"));
		succeed(&mut socket, "first").await;
		socket.send(ServerMessage::Ping(vec![4, 5].into())).await.unwrap();
		assert_eq!(frame(&mut socket).await, ServerMessage::Pong(vec![4, 5].into()));
		socket.close(None).await.unwrap();
		assert!(matches!(frame(&mut socket).await, ServerMessage::Close(_)));
		closed.send(()).unwrap();
		let mut fresh = accept(&listener, &expected).await;
		assert_eq!(receive(&mut fresh).await, body("second"));
		succeed(&mut fresh, "second").await;
	});
	let mut request = params(Some("idle-close"));
	let events = observe(&mut request);
	assert_success(call(&request, &url, &auth, &body("first")).await.unwrap(), "first");
	bounded(observed).await.unwrap();
	assert_success(call(&request, &url, &auth, &body("second")).await.unwrap(), "second");
	assert!(events.lock().unwrap().is_empty());
	peer.finish().await;
}

#[tokio::test]
async fn rejected_upgrade_body_failure_keeps_operation_status_and_cause() {
	use tokio::io::AsyncReadExt;
	let (listener, url) = server().await;
	let peer = Peer::spawn(async move {
		let (mut stream, _) = listener.accept().await.unwrap();
		let mut request = Vec::new();
		while !request.ends_with(b"\r\n\r\n") {
			request.push(stream.read_u8().await.unwrap());
		}
		stream
			.write_all(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{")
			.await
			.unwrap();
	});
	let error = call(&params(None), &url, &headers("rejected", "one"), &body("first"))
		.await
		.err()
		.unwrap();
	assert!(error
		.to_string()
		.contains("OctoHub WebSocket upgrade rejected (HTTP 503"));
	assert!(error.is::<reqwest::Error>());
	peer.finish().await;
}
