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
use reqwest_websocket::RequestBuilderExt;
use std::future::Future;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::Message as Frame, WebSocketStream};

type Peer = WebSocketStream<TcpStream>;

async fn bounded<T>(future: impl Future<Output = T>) -> T {
	tokio::time::timeout(Duration::from_secs(5), future).await.unwrap()
}

async fn pair() -> (Connection<()>, Peer) {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}", listener.local_addr().unwrap());
	let client = async {
		let response = crate::http::websocket_client().get(url).upgrade().send().await.unwrap();
		Connection::new(response.into_websocket().await.unwrap(), "test", ())
	};
	let server = async {
		let (stream, _) = listener.accept().await.unwrap();
		accept_async(stream).await.unwrap()
	};
	bounded(async { tokio::join!(client, server) }).await
}

async fn frame(peer: &mut Peer) -> Frame {
	bounded(peer.next()).await.unwrap().unwrap()
}

async fn elapse(duration: Duration) {
	tokio::time::pause();
	tokio::time::advance(duration).await;
	tokio::time::resume();
	tokio::task::yield_now().await;
}

#[tokio::test]
async fn idle_connection_answers_server_pings_and_checks_liveness_before_reuse() {
	let (connection, mut peer) = pair().await;
	let idle = connection.park();
	peer.send(Frame::Ping(vec![7, 8, 9].into())).await.unwrap();
	assert_eq!(frame(&mut peer).await, Frame::Pong(vec![7, 8, 9].into()));
	let resumed = async { idle.resume().await.unwrap().unwrap() };
	let reply = async {
		assert!(matches!(frame(&mut peer).await, Frame::Ping(_)));
		peer.flush().await.unwrap();
	};
	let (mut connection, ()) = bounded(async { tokio::join!(resumed, reply) }).await;
	connection
		.socket
		.send(Message::Text("new request".into()))
		.await
		.unwrap();
	assert_eq!(frame(&mut peer).await.to_text().unwrap(), "new request");
}

#[tokio::test]
async fn idle_heartbeat_requires_the_matching_pong_and_releases_a_dead_socket() {
	let (connection, mut peer) = pair().await;
	let idle = connection.park();
	tokio::task::yield_now().await;
	elapse(PING_INTERVAL).await;
	let Frame::Ping(nonce) = frame(&mut peer).await else {
		panic!("expected heartbeat")
	};
	// A manual wrong pong replaces tungstenite's automatically queued pong.
	peer.send(Frame::Pong(vec![0; nonce.len()].into())).await.unwrap();
	elapse(PONG_TIMEOUT).await;
	assert!(bounded(idle.resume()).await.unwrap().is_none());
	assert!(matches!(bounded(peer.next()).await, None | Some(Err(_))));
}

#[tokio::test]
async fn active_heartbeat_keeps_silent_work_alive_and_does_not_extend_a_missing_pong_deadline() {
	let (mut connection, mut peer) = pair().await;
	connection.socket.next_ping = Instant::now();
	let read = tokio::spawn(async move {
		let result = connection.socket.next().await;
		(connection, result)
	});
	assert!(matches!(frame(&mut peer).await, Frame::Ping(_)));
	peer.flush().await.unwrap();
	peer.send(Frame::Text("progress".into())).await.unwrap();
	let (mut connection, result) = bounded(read).await.unwrap();
	assert!(matches!(result.unwrap(), Some(Event::Text(text)) if text == "progress"));
	assert!(connection.socket.pending_ping.is_none());

	connection.socket.next_ping = Instant::now();
	let read = tokio::spawn(async move {
		let result = connection.socket.next().await;
		(connection, result)
	});
	assert!(matches!(frame(&mut peer).await, Frame::Ping(_)));
	peer.send(Frame::Pong(vec![99].into())).await.unwrap();
	peer.send(Frame::Text("still progress".into())).await.unwrap();
	let (mut connection, result) = bounded(read).await.unwrap();
	assert!(result.unwrap().is_some());
	elapse(PONG_TIMEOUT).await;
	let error = connection.socket.next().await.unwrap_err();
	assert!(matches!(
		error.downcast_ref::<ConnectionError>(),
		Some(ConnectionError::HeartbeatTimeout)
	));
	assert_eq!(
		crate::llm::providers::shared::transport_error_kind(&error),
		Some(crate::llm::providers::shared::TransportErrorKind::Timeout)
	);
}

#[tokio::test]
async fn idle_close_is_acknowledged_and_cannot_be_reused() {
	let (connection, mut peer) = pair().await;
	let idle = connection.park();
	peer.close(None).await.unwrap();
	assert!(matches!(frame(&mut peer).await, Frame::Close(_)));
	assert!(idle.resume().await.unwrap().is_none());
}

#[tokio::test]
async fn unsolicited_idle_data_is_never_returned_as_the_next_response() {
	let (connection, mut peer) = pair().await;
	let idle = connection.park();
	peer.send(Frame::Text("late response".into())).await.unwrap();
	assert!(bounded(idle.resume()).await.unwrap().is_none());
}

#[tokio::test]
async fn idle_expiry_closes_without_another_pool_lookup_and_lifetime_still_rotates() {
	for lifetime in [false, true] {
		let (mut connection, mut peer) = pair().await;
		if lifetime {
			connection.opened_at -= CONNECTION_LIFETIME;
		}
		let idle = connection.park();
		tokio::task::yield_now().await;
		if !lifetime {
			elapse(IDLE_TIMEOUT).await;
		}
		loop {
			match frame(&mut peer).await {
				Frame::Ping(_) => peer.flush().await.unwrap(),
				Frame::Close(_) => break,
				other => panic!("expected heartbeat or expiry close, got {other:?}"),
			}
		}
		assert!(idle.resume().await.unwrap().is_none());
	}
}

#[tokio::test]
async fn dropping_idle_or_resuming_leases_terminates_the_reader_and_socket() {
	for during_probe in [false, true] {
		let (connection, mut peer) = pair().await;
		let idle = connection.park();
		let task = idle.task.abort_handle();
		if during_probe {
			let mut resume = Box::pin(idle.resume());
			assert!(futures_util::poll!(&mut resume).is_pending());
			assert!(matches!(frame(&mut peer).await, Frame::Ping(_)));
			drop(resume);
		} else {
			drop(idle);
		}
		assert!(matches!(bounded(peer.next()).await, None | Some(Err(_))));
		assert!(task.is_finished());
	}
}

#[tokio::test]
async fn idle_task_failure_is_reported_instead_of_reconnecting_silently() {
	let (connection, _peer) = pair().await;
	let idle = connection.park();
	idle.task.abort();
	let error = idle.resume().await.err().unwrap();
	assert!(error.to_string().contains("idle reader failed"));
}

#[test]
fn session_pool_reuses_existing_sessions_at_capacity_and_rejects_an_extra_active_session() {
	assert_eq!(MAX_SESSIONS, 64);
	assert_eq!(IDLE_TIMEOUT, Duration::from_secs(300));
	assert_eq!(crate::http::CONNECT_TIMEOUT, Duration::from_secs(20));
	let mut pool = SessionPool::<()>::default();
	let now = PoolInstant::now();
	let sessions: Vec<Session<()>> = (0..MAX_SESSIONS)
		.map(|index| {
			let mut key = [0; 32];
			key[..8].copy_from_slice(&(index as u64).to_be_bytes());
			pool.get(key, now).unwrap()
		})
		.collect();
	assert!(Arc::ptr_eq(&sessions[0], &pool.get([0; 32], now).unwrap()));
	assert!(pool.get([255; 32], now).is_err());
	// Even an old session cannot be evicted while a caller owns it.
	assert!(Arc::ptr_eq(
		&sessions[0],
		&pool.get([0; 32], now + IDLE_TIMEOUT * 2).unwrap()
	));
	assert!(pool.get([255; 32], now + IDLE_TIMEOUT * 2).is_err());
	drop(sessions);
	assert!(pool.get([255; 32], now + IDLE_TIMEOUT * 4).is_ok());
}

#[test]
fn session_pool_retains_an_unexpired_idle_session_and_reclaims_it_after_the_idle_timeout() {
	let mut pool = SessionPool::<()>::default();
	let now = PoolInstant::now();
	let session = pool.get([1; 32], now).unwrap();
	let old = Arc::downgrade(&session);
	drop(session);
	let before_expiry = now + IDLE_TIMEOUT - Duration::from_nanos(1);
	let same = pool.get([1; 32], before_expiry).unwrap();
	assert!(Arc::ptr_eq(&same, &old.upgrade().unwrap()));
	drop(same);
	// get() refreshes idle time; expire relative to that last access.
	let replacement = pool
		.get([1; 32], before_expiry + IDLE_TIMEOUT + Duration::from_nanos(1))
		.unwrap();
	assert!(old.upgrade().is_none());
	assert!(Arc::ptr_eq(
		&replacement,
		&pool
			.get([1; 32], before_expiry + IDLE_TIMEOUT + Duration::from_nanos(1))
			.unwrap()
	));
}

#[tokio::test]
async fn session_pool_measures_idle_time_from_release_not_from_a_long_lease() {
	let (connection, _peer) = pair().await;
	let mut pool = SessionPool::<()>::default();
	let released = PoolInstant::now();
	let checkout = released.checked_sub(IDLE_TIMEOUT * 2).unwrap();
	let session = pool.get([1; 32], checkout).unwrap();
	let old = Arc::downgrade(&session);
	*session.lock().await = Some(connection.park());
	drop(session);
	let same = pool.get([1; 32], released + Duration::from_secs(1)).unwrap();
	assert!(Arc::ptr_eq(&same, &old.upgrade().unwrap()));
}

#[tokio::test]
async fn checkout_detects_a_silent_dead_connection_before_sending_application_data() {
	let (connection, mut peer) = pair().await;
	let idle = connection.park();
	let mut resumed = Box::pin(idle.resume());
	assert!(futures_util::poll!(&mut resumed).is_pending());
	assert!(matches!(frame(&mut peer).await, Frame::Ping(_)));
	elapse(PONG_TIMEOUT).await;
	assert!(bounded(resumed).await.unwrap().is_none());
	assert!(matches!(bounded(peer.next()).await, None | Some(Err(_))));
}
