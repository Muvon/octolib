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
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message as ServerMessage, WebSocketStream};

type ServerSocket = WebSocketStream<TcpStream>;

async fn server() -> (TcpListener, String) {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}/v1/responses", listener.local_addr().unwrap());
	(listener, url)
}

async fn accept(listener: &TcpListener) -> ServerSocket {
	let (stream, _) = listener.accept().await.unwrap();
	accept_async(stream).await.unwrap()
}

async fn receive(socket: &mut ServerSocket) -> Value {
	let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
		.await
		.unwrap()
		.unwrap()
		.unwrap();
	serde_json::from_str(message.to_text().unwrap()).unwrap()
}

async fn finish(socket: &mut ServerSocket, id: &str, output: Value) {
	socket
		.send(ServerMessage::Text(
			json!({"type":"response.completed", "response": {
				"id":id, "output":output, "usage":{"input_tokens":10,"output_tokens":2}
			}})
			.to_string()
			.into(),
		))
		.await
		.unwrap();
}

fn full_request(input: Value) -> Value {
	json!({"type":"response.create", "model":"gpt-6.1-sol", "store":false, "input":input})
}

fn output() -> Value {
	json!([{"type":"message", "role":"assistant", "id":"msg_1", "content":[{"type":"output_text", "text":"hello"}]}])
}

async fn request(session: &Session, full: &Value, url: &str) -> Result<(Value, Value)> {
	tokio::time::timeout(
		Duration::from_secs(5),
		exchange(
			session,
			full,
			&HeaderMap::new(),
			"test-token",
			url,
			Some(Duration::from_secs(5)),
			None,
		),
	)
	.await
	.unwrap()
}

#[tokio::test]
async fn connection_reuses_socket_and_sends_only_new_input() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		let first = receive(&mut socket).await;
		assert!(first.get("previous_response_id").is_none());
		assert!(first.get("stream").is_none());
		socket.send(ServerMessage::Ping(vec![1, 2, 3].into())).await.unwrap();
		finish(&mut socket, "resp_1", output()).await;
		let pong = socket.next().await.unwrap().unwrap();
		assert!(matches!(pong, ServerMessage::Pong(_)));
		let second = receive(&mut socket).await;
		assert_eq!(second["previous_response_id"], "resp_1");
		assert_eq!(second["input"], json!([{"role":"user", "content":"next"}]));
		finish(&mut socket, "resp_2", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let mut full = full_request(json!([{"role":"user", "content":"hi"}]));
	request(&session, &full, &url).await.unwrap();
	full["input"]
		.as_array_mut()
		.unwrap()
		.extend(output().as_array().unwrap().clone());
	full["input"]
		.as_array_mut()
		.unwrap()
		.push(json!({"role":"user", "content":"next"}));
	request(&session, &full, &url).await.unwrap();
	peer.await.unwrap();
}

#[test]
fn continuation_matches_encrypted_reasoning_and_namespaced_tool_calls() {
	let full = full_request(json!([{"role":"user", "content":"read"}]));
	let response = json!({"id":"resp_1", "output":[
		{"type":"reasoning", "encrypted_content":"enc"},
		{"type":"function_call", "call_id":"call_1", "name":"read", "arguments":"{}"}
	]});
	let previous = Continuation::from_response(&full, &response).unwrap();
	let mut next = full.clone();
	next["input"].as_array_mut().unwrap().extend(
		response["output"]
			.as_array()
			.unwrap()
			.iter()
			.cloned()
			.map(normalize_input_item),
	);
	let result = json!({"type":"function_call_output", "call_id":"call_1", "output":"file"});
	next["input"].as_array_mut().unwrap().push(result.clone());
	let delta = previous.request(&next).unwrap();
	assert_eq!(delta["previous_response_id"], "resp_1");
	assert_eq!(delta["input"], json!([result]));
	next["input"][1]["encrypted_content"] = json!("different");
	assert_eq!(previous.request(&next).unwrap(), next);
	assert_eq!(previous.request(&full).unwrap(), full);
}

#[tokio::test]
async fn changed_history_uses_full_input_on_the_same_socket() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		finish(&mut socket, "resp_1", output()).await;
		let second = receive(&mut socket).await;
		assert!(second.get("previous_response_id").is_none());
		assert_eq!(second["input"][0]["content"], "compacted");
		finish(&mut socket, "resp_2", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	request(&session, &full_request(json!([{"role":"user", "content":"hi"}])), &url)
		.await
		.unwrap();
	request(
		&session,
		&full_request(json!([{"role":"user", "content":"compacted"}])),
		&url,
	)
	.await
	.unwrap();
	peer.await.unwrap();
}

#[tokio::test]
async fn cancellation_discards_socket_and_next_request_replays_full_context() {
	let (listener, url) = server().await;
	let (started, received) = tokio::sync::oneshot::channel();
	let peer = tokio::spawn(async move {
		let mut abandoned = accept(&listener).await;
		receive(&mut abandoned).await;
		started.send(()).unwrap();
		let mut socket = accept(&listener).await;
		let next = receive(&mut socket).await;
		assert!(next.get("previous_response_id").is_none());
		assert_eq!(next["input"][0]["content"], "retry");
		finish(&mut socket, "resp_2", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let full = full_request(json!([{"role":"user", "content":"first"}]));
	let mut pending = Box::pin(request(&session, &full, &url));
	tokio::select! {
		result = &mut pending => panic!("request completed before cancellation: {result:?}"),
		result = received => result.unwrap(),
	}
	drop(pending);
	assert!(session.try_lock().unwrap().is_none());
	request(
		&session,
		&full_request(json!([{"role":"user", "content":"retry"}])),
		&url,
	)
	.await
	.unwrap();
	peer.await.unwrap();
}

#[tokio::test]
async fn missing_continuation_is_retryable_and_reconnects_without_previous_id() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		finish(&mut socket, "resp_1", output()).await;
		assert_eq!(receive(&mut socket).await["previous_response_id"], "resp_1");
		socket
			.send(ServerMessage::Text(
				json!({"type":"error", "error":{"code":"previous_response_not_found"}})
					.to_string()
					.into(),
			))
			.await
			.unwrap();
		let mut new_socket = accept(&listener).await;
		let full = receive(&mut new_socket).await;
		assert!(full.get("previous_response_id").is_none());
		assert_eq!(full["input"].as_array().unwrap().len(), 3);
		finish(&mut new_socket, "resp_2", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let mut full = full_request(json!([{"role":"user", "content":"hi"}]));
	request(&session, &full, &url).await.unwrap();
	full["input"]
		.as_array_mut()
		.unwrap()
		.extend(output().as_array().unwrap().clone());
	full["input"]
		.as_array_mut()
		.unwrap()
		.push(json!({"role":"user", "content":"next"}));
	assert!(is_retryable(&request(&session, &full, &url).await.unwrap_err()));
	assert!(session.try_lock().unwrap().is_none());
	request(&session, &full, &url).await.unwrap();
	peer.await.unwrap();
}

#[tokio::test]
async fn usage_limit_is_terminal_and_partial_output_is_not_a_success() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		socket
			.send(ServerMessage::Text(
				json!({"type":"response.output_item.done", "output_index":0, "item": output()[0]})
					.to_string()
					.into(),
			))
			.await
			.unwrap();
		socket.send(ServerMessage::Text(json!({"type":"response.failed", "response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}).to_string().into())).await.unwrap();
	});
	let session = Arc::new(AsyncMutex::new(None));
	let error = request(&session, &full_request(json!([])), &url).await.unwrap_err();
	assert!(!is_retryable(&error));
	assert!(error.to_string().contains("ChatGPT plan usage limit reached"));
	assert!(session.try_lock().unwrap().is_none());
	peer.await.unwrap();
}

#[test]
fn session_key_isolates_credentials_models_headers_and_sessions() {
	let params = ChatCompletionParams::new(&[], "gpt-6.1-sol", 0.0, 1.0, 0, 0).with_prompt_cache_key("session-a");
	let headers = HeaderMap::new();
	let key = session_key(params.prompt_cache_key.as_deref(), &params.model, "token-a", &headers).unwrap();
	assert_ne!(
		key,
		session_key(params.prompt_cache_key.as_deref(), &params.model, "token-b", &headers).unwrap()
	);
	let mut changed = params.clone();
	changed.model = "other-model".into();
	assert_ne!(
		key,
		session_key(changed.prompt_cache_key.as_deref(), &changed.model, "token-a", &headers).unwrap()
	);
	changed = params.clone().with_prompt_cache_key("session-b");
	assert_ne!(
		key,
		session_key(changed.prompt_cache_key.as_deref(), &changed.model, "token-a", &headers).unwrap()
	);
	let mut changed_headers = HeaderMap::new();
	changed_headers.insert("x-test", "changed".parse().unwrap());
	assert_ne!(
		key,
		session_key(
			params.prompt_cache_key.as_deref(),
			&params.model,
			"token-a",
			&changed_headers
		)
		.unwrap()
	);
	changed.prompt_cache_key = None;
	assert!(
		session_key(changed.prompt_cache_key.as_deref(), &changed.model, "token-a", &headers)
			.unwrap()
			.is_none()
	);
}

#[test]
fn pool_reuses_active_sessions_and_never_evicts_them() {
	let mut pool = SessionPool::default();
	let now = Instant::now();
	let sessions: Vec<_> = (0..MAX_SESSIONS)
		.map(|index| pool.get(Sha256::digest(index.to_be_bytes()).into(), now).unwrap())
		.collect();
	let key = Sha256::digest(0usize.to_be_bytes()).into();
	assert!(Arc::ptr_eq(&sessions[0], &pool.get(key, now + IDLE_TIMEOUT).unwrap()));
	assert!(pool.get([255; 32], now).is_err());
	drop(sessions);
	pool.get([255; 32], now + IDLE_TIMEOUT * 2).unwrap();
	assert_eq!(pool.entries.len(), 1);
}

#[test]
fn output_events_are_ordered_and_incomplete_responses_fail() {
	let mut events = ResponseEvents::default();
	events
		.push(json!({"type":"response.output_item.done", "output_index":1, "item":{"type":"message"}}))
		.unwrap();
	events
		.push(json!({"type":"response.output_item.done", "output_index":0, "item":{"type":"reasoning"}}))
		.unwrap();
	let response = events
		.push(json!({"type":"response.completed", "response":{"id":"resp_1", "output":[]}}))
		.unwrap()
		.unwrap();
	assert_eq!(response["output"][0]["type"], "reasoning");
	assert!(events
		.push(json!({"type":"response.incomplete", "response":{"incomplete_details":{"reason":"max_output_tokens"}}}))
		.is_err());
	assert!(events.push(json!({"type":"response.completed"})).is_err());
	assert!(events
		.push(json!({"type":"response.output_item.done", "item":{}}))
		.is_err());
}

#[tokio::test]
async fn concurrent_requests_share_one_socket_without_mixing_responses() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		for _ in 0..2 {
			let request = receive(&mut socket).await;
			let id = request["input"][0]["content"].as_str().unwrap();
			finish(&mut socket, id, json!([])).await;
		}
	});
	let session = Arc::new(AsyncMutex::new(None));
	let first = full_request(json!([{"role":"user", "content":"first"}]));
	let second = full_request(json!([{"role":"user", "content":"second"}]));
	let (first_result, second_result) = tokio::join!(request(&session, &first, &url), request(&session, &second, &url));
	assert_eq!(first_result.unwrap().1["id"], "first");
	assert_eq!(second_result.unwrap().1["id"], "second");
	peer.await.unwrap();
}

#[tokio::test]
async fn timeout_drops_an_in_flight_socket() {
	let (listener, url) = server().await;
	let (started, received) = tokio::sync::oneshot::channel();
	let (stop, stopped) = tokio::sync::oneshot::channel();
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		started.send(()).unwrap();
		stopped.await.unwrap();
	});
	let session = Arc::new(AsyncMutex::new(None));
	let full = full_request(json!([]));
	let mut pending = Box::pin(request(&session, &full, &url));
	tokio::select! {
		result = &mut pending => panic!("request completed before timeout: {result:?}"),
		result = received => result.unwrap(),
	}
	assert!(tokio::time::timeout(Duration::ZERO, pending).await.is_err());
	assert!(session.try_lock().unwrap().is_none());
	stop.send(()).unwrap();
	peer.await.unwrap();
}

#[tokio::test]
async fn expired_connections_reconnect_with_full_history() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		finish(&mut socket, "resp_1", output()).await;
		let mut next = accept(&listener).await;
		assert!(receive(&mut next).await.get("previous_response_id").is_none());
		finish(&mut next, "resp_2", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let full = full_request(json!([{"role":"user", "content":"hi"}]));
	request(&session, &full, &url).await.unwrap();
	session.lock().await.as_mut().unwrap().opened_at = Instant::now() - CONNECTION_LIFETIME;
	request(&session, &full, &url).await.unwrap();
	peer.await.unwrap();
}

#[tokio::test]
async fn closed_partial_response_is_retryable_and_never_cached() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		socket
			.send(ServerMessage::Text(
				json!({"type":"response.output_item.done", "output_index":0, "item":output()[0]})
					.to_string()
					.into(),
			))
			.await
			.unwrap();
		socket.close(None).await.unwrap();
	});
	let session = Arc::new(AsyncMutex::new(None));
	let error = request(&session, &full_request(json!([])), &url).await.unwrap_err();
	assert!(error.is::<RetryableStreamError>());
	assert!(session.try_lock().unwrap().is_none());
	peer.await.unwrap();
}

#[tokio::test]
async fn handshake_errors_keep_status_retry_after_and_usage_guidance() {
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	for (status, code) in [
		(401, "unauthorized"),
		(429, "rate_limit_exceeded"),
		(503, "subscription_sharing_usage_unavailable"),
		(403, "forbidden"),
	] {
		let (listener, url) = server().await;
		let peer = tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut headers = Vec::new();
			while !headers.ends_with(b"\r\n\r\n") {
				headers.push(socket.read_u8().await.unwrap());
			}
			let headers = String::from_utf8(headers).unwrap().to_lowercase();
			assert!(headers.contains("authorization: bearer test-token"));
			assert!(headers.contains("upgrade: websocket"));
			assert!(headers.contains("user-agent: "));
			let body = json!({"error":{"code":code}}).to_string();
			let response = format!(
				"HTTP/1.1 {status} Error\r\nContent-Length: {}\r\nRetry-After: 2\r\nConnection: close\r\n\r\n{body}",
				body.len()
			);
			socket.write_all(response.as_bytes()).await.unwrap();
		});
		let session = Arc::new(AsyncMutex::new(None));
		let error = request(&session, &full_request(json!([])), &url).await.unwrap_err();
		assert!(session.try_lock().unwrap().is_none());
		match status {
			401 => assert!(error.is::<Unauthorized>()),
			429 => {
				let status = error.downcast_ref::<retry::RetryableStatusError>().unwrap();
				assert_eq!(status.status, 429);
				assert_eq!(status.retry_after, Some(Duration::from_secs(2)));
			}
			503 => {
				assert!(!is_retryable(&error));
				assert!(error.to_string().starts_with("ChatGPT could not check"));
			}
			403 => assert!(!is_retryable(&error)),
			_ => unreachable!(),
		}
		peer.await.unwrap();
	}
}

#[tokio::test]
async fn response_events_extend_only_the_stream_deadline() {
	let frame = ServerMessage::Text(
		json!({"type":"response.output_text.delta", "delta":"working"})
			.to_string()
			.into(),
	);
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		for _ in 0..4 {
			tokio::time::sleep(Duration::from_millis(350)).await;
			socket.send(frame.clone()).await.unwrap();
		}
		finish(&mut socket, "long_response", output()).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let full = full_request(json!([]));
	let started = Instant::now();
	let response = tokio::time::timeout(
		Duration::from_secs(10),
		exchange(
			&session,
			&full,
			&HeaderMap::new(),
			"test-token",
			&url,
			Some(Duration::from_secs(10)),
			Some(Duration::from_secs(1)),
		),
	)
	.await
	.unwrap()
	.unwrap();
	assert!(started.elapsed() > Duration::from_secs(1));
	assert_eq!(response.1["id"], "long_response");
	peer.await.unwrap();
}

#[tokio::test]
async fn silent_response_times_out_and_reconnects_with_full_context() {
	for progress in [false, true] {
		let (listener, url) = server().await;
		let peer = tokio::spawn(async move {
			let mut silent = accept(&listener).await;
			receive(&mut silent).await;
			if progress {
				silent
					.send(ServerMessage::Text(
						json!({"type":"response.output_text.delta", "delta":"partial"})
							.to_string()
							.into(),
					))
					.await
					.unwrap();
				silent.send(ServerMessage::Ping(vec![1].into())).await.unwrap();
				assert!(matches!(silent.next().await.unwrap().unwrap(), ServerMessage::Pong(_)));
			}
			let mut next = accept(&listener).await;
			let request = receive(&mut next).await;
			assert!(request.get("previous_response_id").is_none());
			assert_eq!(request["input"][0]["content"], "retry");
			finish(&mut next, "recovered", json!([])).await;
		});
		let session = Arc::new(AsyncMutex::new(None));
		let error = exchange(
			&session,
			&full_request(json!([])),
			&HeaderMap::new(),
			"test-token",
			&url,
			None,
			Some(Duration::from_millis(100)),
		)
		.await
		.unwrap_err();
		assert!(error.is::<super::super::super::streaming::StreamTimeout>());
		assert_eq!(error.to_string(), "Stream timed out after 100ms without response data");
		assert_eq!(
			shared::transport_error_kind(&error),
			Some(shared::TransportErrorKind::Timeout)
		);
		assert!(is_retryable(&error));
		assert!(session.try_lock().unwrap().is_none());
		request(
			&session,
			&full_request(json!([{"role":"user", "content":"retry"}])),
			&url,
		)
		.await
		.unwrap();
		peer.await.unwrap();
	}
}

#[tokio::test]
async fn queue_wait_and_idle_connection_do_not_consume_response_timeout() {
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		for id in ["first", "second"] {
			receive(&mut socket).await;
			finish(&mut socket, id, json!([])).await;
		}
	});
	let session = Arc::new(AsyncMutex::new(None));
	let guard = session.lock().await;
	let full = full_request(json!([]));
	let headers = HeaderMap::new();
	let timeout = Duration::from_millis(200);
	let mut first = Box::pin(exchange(
		&session,
		&full,
		&headers,
		"test-token",
		&url,
		Some(timeout),
		None,
	));
	tokio::select! {
		result = &mut first => panic!("Request must wait for its session: {result:?}"),
		_ = tokio::time::sleep(timeout * 2) => {},
	}
	drop(guard);
	assert_eq!(first.await.unwrap().1["id"], "first");
	tokio::time::sleep(timeout * 2).await;
	assert_eq!(
		exchange(&session, &full, &headers, "test-token", &url, Some(timeout), None)
			.await
			.unwrap()
			.1["id"],
		"second"
	);
	peer.await.unwrap();
}

#[tokio::test]
async fn disabled_inactivity_timeout_waits_for_completion() {
	let (listener, url) = server().await;
	let (ready, received) = tokio::sync::oneshot::channel();
	let (release, wait) = tokio::sync::oneshot::channel();
	let peer = tokio::spawn(async move {
		let mut socket = accept(&listener).await;
		receive(&mut socket).await;
		ready.send(()).unwrap();
		wait.await.unwrap();
		finish(&mut socket, "completed", json!([])).await;
	});
	let session = Arc::new(AsyncMutex::new(None));
	let full = full_request(json!([]));
	let headers = HeaderMap::new();
	let mut pending = Box::pin(exchange(&session, &full, &headers, "test-token", &url, None, None));
	tokio::select! {
		result = &mut pending => panic!("Request completed before release: {result:?}"),
		result = received => result.unwrap(),
	}
	release.send(()).unwrap();
	assert_eq!(pending.await.unwrap().1["id"], "completed");
	peer.await.unwrap();
}

#[tokio::test]
async fn total_deadline_and_stream_inactivity_are_independent() {
	for (response_data, total, idle) in [
		(true, Duration::from_millis(400), Duration::from_secs(2)),
		(false, Duration::from_millis(400), Duration::from_secs(2)),
		(false, Duration::from_secs(2), Duration::from_millis(400)),
	] {
		let (listener, url) = server().await;
		let peer = tokio::spawn(async move {
			let mut socket = accept(&listener).await;
			receive(&mut socket).await;
			let mut tick = tokio::time::interval(Duration::from_millis(50));
			loop {
				tokio::select! {
					_ = tick.tick() => {
						let frame = if response_data {
							ServerMessage::Text(json!({"type":"response.output_text.delta","delta":"working"}).to_string().into())
						} else { ServerMessage::Ping(vec![1].into()) };
						if socket.send(frame).await.is_err() { break; }
					}
					message = socket.next() => match message {
						Some(Ok(ServerMessage::Pong(_))) => {},
						None | Some(Err(_)) | Some(Ok(ServerMessage::Close(_))) => break,
						other => panic!("unexpected client frame: {other:?}"),
					}
				}
			}
		});
		let session = Arc::new(AsyncMutex::new(None));
		let error = exchange(
			&session,
			&full_request(json!([])),
			&HeaderMap::new(),
			"test-token",
			&url,
			Some(total),
			Some(idle),
		)
		.await
		.unwrap_err();
		if total < idle {
			assert!(
				matches!(
					error.downcast_ref::<ProviderError>(),
					Some(ProviderError::TimeoutError { .. })
				),
				"{error:#}"
			);
		} else {
			assert!(error.is::<super::super::super::streaming::StreamTimeout>(), "{error:#}");
		}
		assert!(session.try_lock().unwrap().is_none());
		tokio::time::timeout(Duration::from_secs(3), peer)
			.await
			.unwrap()
			.unwrap();
	}
}

#[tokio::test]
async fn request_deadline_includes_websocket_handshake() {
	use tokio::io::AsyncReadExt;
	let (listener, url) = server().await;
	let peer = tokio::spawn(async move {
		let (mut socket, _) = listener.accept().await.unwrap();
		let mut bytes = Vec::new();
		socket.read_to_end(&mut bytes).await.unwrap();
		assert!(!bytes.is_empty());
	});
	let session = Arc::new(AsyncMutex::new(None));
	let error = exchange(
		&session,
		&full_request(json!([])),
		&HeaderMap::new(),
		"test-token",
		&url,
		Some(Duration::from_millis(100)),
		None,
	)
	.await
	.unwrap_err();
	assert!(matches!(
		error.downcast_ref::<ProviderError>(),
		Some(ProviderError::TimeoutError { .. })
	));
	assert!(session.try_lock().unwrap().is_none());
	peer.await.unwrap();
}
