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

//! Shared socket health and idle ownership. Providers own framing and replay policy.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use reqwest_websocket::{Message, WebSocket};
use std::sync::Arc;
use std::time::{Duration, Instant as PoolInstant};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::Instant;

pub(super) const MAX_SESSIONS: usize = 64;
pub(super) const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECTION_LIFETIME: Duration = Duration::from_secs(55 * 60);
const PING_INTERVAL: Duration = Duration::from_secs(30);
const PONG_TIMEOUT: Duration = Duration::from_secs(10);
const WRITE_TIMEOUT: Duration = Duration::from_secs(20);

pub(super) type Session<T> = Arc<Mutex<Option<IdleConnection<T>>>>;

pub(super) struct SessionPool<T> {
	entries: Vec<PoolEntry<T>>,
}

impl<T> Default for SessionPool<T> {
	fn default() -> Self {
		Self { entries: Vec::new() }
	}
}

struct PoolEntry<T> {
	key: [u8; 32],
	session: Session<T>,
	used_at: PoolInstant,
}

impl<T> PoolEntry<T> {
	/// A lease can outlast `IDLE_TIMEOUT`, so idle time runs from the moment the
	/// socket was parked, never from checkout.
	fn is_expired(&self, now: PoolInstant) -> bool {
		if Arc::strong_count(&self.session) > 1 {
			return false;
		}
		let idle_since = self
			.session
			.try_lock()
			.ok()
			.and_then(|slot| slot.as_ref().map(|idle| idle.parked_at))
			.map_or(self.used_at, |parked_at| parked_at.max(self.used_at));
		now.duration_since(idle_since) >= IDLE_TIMEOUT
	}
}

impl<T> SessionPool<T> {
	pub(super) fn get(&mut self, key: [u8; 32], now: PoolInstant) -> Result<Session<T>> {
		// A waiting caller holds a lease too: evicting it would allow two sockets
		// to execute concurrently for the same provider session.
		self.entries.retain(|entry| !entry.is_expired(now));
		if let Some(index) = self.entries.iter().position(|entry| entry.key == key) {
			let mut entry = self.entries.remove(index);
			entry.used_at = now;
			let session = entry.session.clone();
			self.entries.push(entry);
			return Ok(session);
		}
		if self.entries.len() == MAX_SESSIONS {
			let index = self
				.entries
				.iter()
				.position(|entry| Arc::strong_count(&entry.session) == 1)
				.context("All provider WebSocket sessions are busy")?;
			self.entries.remove(index);
		}
		let session = Arc::new(Mutex::new(None));
		self.entries.push(PoolEntry {
			key,
			session: session.clone(),
			used_at: now,
		});
		Ok(session)
	}
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ConnectionError {
	#[error("WebSocket heartbeat timed out waiting for a matching pong")]
	HeartbeatTimeout,
	#[error("WebSocket write timed out")]
	WriteTimeout,
	#[error("WebSocket disconnected during its liveness check")]
	Disconnected,
	#[error("WebSocket received unexpected application data between requests")]
	UnexpectedData,
	#[error("WebSocket idle or connection lifetime expired")]
	Expired,
}

#[derive(Debug)]
pub(super) enum Event {
	Text(String),
	Binary,
	Close(reqwest_websocket::CloseCode),
}

pub(super) struct Socket {
	inner: WebSocket,
	next_ping: Instant,
	pending_ping: Option<([u8; 8], Instant)>,
	sequence: u64,
}

impl Socket {
	fn new(inner: WebSocket) -> Self {
		Self {
			inner,
			next_ping: Instant::now() + PING_INTERVAL,
			pending_ping: None,
			sequence: 0,
		}
	}

	pub(super) async fn send(&mut self, message: Message) -> Result<()> {
		tokio::time::timeout(WRITE_TIMEOUT, self.inner.send(message))
			.await
			.map_err(|_| ConnectionError::WriteTimeout)??;
		Ok(())
	}

	async fn flush(&mut self) -> Result<()> {
		tokio::time::timeout_at(
			self.pending_ping
				.map_or(Instant::now() + PONG_TIMEOUT, |(_, deadline)| deadline),
			self.inner.flush(),
		)
		.await
		.map_err(|_| ConnectionError::WriteTimeout)??;
		Ok(())
	}

	async fn ping(&mut self) -> Result<()> {
		self.sequence = self.sequence.wrapping_add(1);
		let payload = self.sequence.to_be_bytes();
		let deadline = Instant::now() + PONG_TIMEOUT;
		// Keep the nonce and deadline even if the idle reader is interrupted by
		// checkout during a partially flushed ping.
		self.pending_ping = Some((payload, deadline));
		tokio::time::timeout_at(deadline, self.inner.send(Message::Ping(payload.to_vec().into())))
			.await
			.map_err(|_| ConnectionError::HeartbeatTimeout)??;
		Ok(())
	}

	async fn read_frame(&mut self) -> Result<Option<Message>> {
		loop {
			let deadline = self.pending_ping.map_or(self.next_ping, |(_, deadline)| deadline);
			let message = tokio::select! {
				biased;
				_ = tokio::time::sleep_until(deadline) => {
					if self.pending_ping.is_some() {
						return Err(ConnectionError::HeartbeatTimeout.into());
					}
					self.ping().await?;
					continue;
				}
				message = self.inner.next() => message.transpose()?,
			};
			match &message {
				Some(Message::Ping(_)) => self.flush().await?,
				Some(Message::Pong(payload)) => {
					if self
						.pending_ping
						.is_some_and(|(expected, _)| payload.as_ref() == expected)
					{
						self.pending_ping = None;
						self.next_ping = Instant::now() + PING_INTERVAL;
					}
				}
				Some(Message::Close { .. }) => self.flush().await?,
				_ => {}
			}
			return Ok(message);
		}
	}

	pub(super) async fn next(&mut self) -> Result<Option<Event>> {
		loop {
			match self.read_frame().await? {
				Some(Message::Ping(_) | Message::Pong(_)) => {}
				Some(Message::Text(text)) => return Ok(Some(Event::Text(text))),
				Some(Message::Binary(_)) => return Ok(Some(Event::Binary)),
				Some(Message::Close { code, .. }) => return Ok(Some(Event::Close(code))),
				None => return Ok(None),
			}
		}
	}

	async fn probe(&mut self) -> Result<()> {
		if self.pending_ping.is_none() {
			self.ping().await?;
		}
		while self.pending_ping.is_some() {
			match self.read_frame().await? {
				Some(Message::Ping(_) | Message::Pong(_)) => {}
				Some(Message::Close { .. }) | None => return Err(ConnectionError::Disconnected.into()),
				Some(_) => return Err(ConnectionError::UnexpectedData.into()),
			}
		}
		Ok(())
	}
}

pub(super) struct Connection<T> {
	pub(super) socket: Socket,
	pub(super) state: T,
	provider: &'static str,
	opened_at: Instant,
}

impl<T: Send + 'static> Connection<T> {
	pub(super) fn new(socket: WebSocket, provider: &'static str, state: T) -> Self {
		Self {
			socket: Socket::new(socket),
			state,
			provider,
			opened_at: Instant::now(),
		}
	}

	pub(super) fn park(mut self) -> IdleConnection<T> {
		let resume = Arc::new(Notify::new());
		let resumed = resume.clone();
		let provider = self.provider;
		let deadline = (Instant::now() + IDLE_TIMEOUT).min(self.opened_at + CONNECTION_LIFETIME);
		let task = tokio::spawn(async move {
			tokio::select! {
				biased;
				_ = tokio::time::sleep_until(deadline) => {
					self.socket.send(Message::Close { code: reqwest_websocket::CloseCode::Normal, reason: String::new() }).await?;
					return Err(ConnectionError::Expired.into());
				}
				_ = resumed.notified() => {
					// Verify before any application message; replacing a dead idle
					// socket is safe even for providers that cannot replay requests.
					self.socket.probe().await?;
				}
				message = self.socket.next() => {
					return match message? {
						Some(Event::Close(_)) | None => Err(ConnectionError::Disconnected.into()),
						Some(_) => Err(ConnectionError::UnexpectedData.into()),
					};
				}
			}
			Ok(self)
		});
		IdleConnection {
			resume,
			task,
			provider,
			parked_at: PoolInstant::now(),
		}
	}
}

// No detached readers, queues, or retained provider state survive lease eviction
// or cancellation while checkout is waiting for its liveness probe.
pub(super) struct IdleConnection<T> {
	resume: Arc<Notify>,
	task: JoinHandle<Result<Connection<T>>>,
	provider: &'static str,
	parked_at: PoolInstant,
}

impl<T> IdleConnection<T> {
	pub(super) async fn resume(mut self) -> Result<Option<Connection<T>>> {
		self.resume.notify_one();
		match (&mut self.task).await.context("WebSocket idle reader failed")? {
			Ok(connection) => Ok(Some(connection)),
			Err(error) => {
				tracing::debug!(provider = self.provider, error = %error, "Discarding unavailable idle WebSocket");
				Ok(None)
			}
		}
	}
}

impl<T> Drop for IdleConnection<T> {
	fn drop(&mut self) {
		self.task.abort();
	}
}

#[cfg(test)]
#[path = "websocket_transport_tests.rs"]
mod tests;
