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

use super::{auth_aware_error, shared};
use crate::errors::ProviderError;
use crate::llm::{retry, ChatCompletionParams};
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest_websocket::{Message, RequestBuilderExt, WebSocket};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;

const MAX_SESSIONS: usize = 64;
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECTION_LIFETIME: Duration = Duration::from_secs(55 * 60);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

type Session = Arc<AsyncMutex<Option<Connection>>>;
static SESSIONS: LazyLock<Mutex<SessionPool>> = LazyLock::new(|| Mutex::new(SessionPool::default()));

#[derive(Default)]
struct SessionPool {
	entries: Vec<PoolEntry>,
}

struct PoolEntry {
	key: [u8; 32],
	session: Session,
	used_at: Instant,
}

impl SessionPool {
	fn get(&mut self, key: [u8; 32], now: Instant) -> Result<Session> {
		self.entries
			.retain(|entry| Arc::strong_count(&entry.session) > 1 || now.duration_since(entry.used_at) < IDLE_TIMEOUT);
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
				.context("All OctoHub WebSocket sessions are busy")?;
			self.entries.remove(index);
		}
		let session = Arc::new(AsyncMutex::new(None));
		self.entries.push(PoolEntry {
			key,
			session: session.clone(),
			used_at: now,
		});
		Ok(session)
	}
}

struct Connection {
	socket: WebSocket,
	opened_at: Instant,
}

fn session_key(cache_key: Option<&str>, url: &str, token: &str, headers: &HeaderMap) -> Option<[u8; 32]> {
	let session = cache_key?;
	let mut hasher = Sha256::new();
	let mut sorted: Vec<_> = headers
		.iter()
		.map(|(name, value)| (name.as_str(), value.as_bytes()))
		.collect();
	sorted.sort_by_key(|(name, _)| *name);
	for part in [session.as_bytes(), url.as_bytes(), token.as_bytes()]
		.into_iter()
		.chain(sorted.iter().flat_map(|(name, value)| [name.as_bytes(), *value]))
	{
		hasher.update((part.len() as u64).to_be_bytes());
		hasher.update(part);
	}
	Some(hasher.finalize().into())
}

pub(super) async fn complete(
	params: &ChatCompletionParams,
	url: &str,
	token: &str,
	headers: &HeaderMap,
	body: &Value,
) -> Result<shared::CapturedResponse> {
	let text = serde_json::to_string(body)?;
	if text.len() > MAX_REQUEST_BYTES {
		anyhow::bail!("OctoHub WebSocket request exceeds the 16 MiB limit");
	}
	let session = match session_key(params.prompt_cache_key.as_deref(), url, token, headers) {
		Some(key) => SESSIONS
			.lock()
			.map_err(|_| anyhow::anyhow!("OctoHub WebSocket pool lock poisoned"))?
			.get(key, Instant::now())?,
		None => Arc::new(AsyncMutex::new(None)),
	};
	let timeout = params.request_timeout.unwrap_or(DEFAULT_REQUEST_TIMEOUT);
	let operation = retry::retry_chat_operation(
		|| {
			let session = session.clone();
			let headers = headers.clone();
			let url = url.to_owned();
			let text = text.clone();
			let per_call_key = params.api_key.clone();
			Box::pin(async move {
				let result = exchange(&session, &url, &headers, text, per_call_key.is_some())
					.await
					.map(|mut response| {
						if !response.status.is_success() {
							response.body = shared::redact_text(&response.body, per_call_key.as_ref());
						}
						response
					})
					.map_err(|error| shared::redact_error(error, per_call_key.as_ref()));
				// Only an explicit server status is safe to retry. A lost reply may already be paid work.
				match result {
					Err(error) if error.is::<retry::RetryableStatusError>() => Err(error),
					result => Ok(result),
				}
			})
		},
		params,
	);
	tokio::time::timeout(timeout, operation)
		.await
		.map_err(|_| timeout_error(timeout))??
}

async fn exchange(
	session: &Session,
	url: &str,
	headers: &HeaderMap,
	text: String,
	per_call_key: bool,
) -> Result<shared::CapturedResponse> {
	let mut slot = session.lock().await;
	// The lease owns the socket until success. Cancellation, dropping the future,
	// or any protocol/transport error leaves the slot empty, never a late reply.
	let mut connection = match slot.take() {
		Some(connection) if connection.opened_at.elapsed() < CONNECTION_LIFETIME => connection,
		_ => connect(url, headers, per_call_key).await?,
	};
	connection.socket.send(Message::Text(text)).await?;
	let response = read_response(&mut connection.socket).await?;
	if retry::is_retryable_status(response.status.as_u16()) {
		return Err(retry::retryable_status_error(
			&response,
			format!("OctoHub API error {}: {}", response.status, response.body),
		));
	}
	if response.status.is_success() {
		*slot = Some(connection);
	}
	Ok(response)
}

async fn connect(url: &str, headers: &HeaderMap, per_call_key: bool) -> Result<Connection> {
	let setup = async {
		let response = crate::http::websocket_client()
			.get(url)
			.version(reqwest::Version::HTTP_11)
			.timeout(crate::http::CONNECT_TIMEOUT)
			.headers(headers.clone())
			.upgrade()
			.send()
			.await
			.context("Opening OctoHub WebSocket connection")?;
		if response.status() != reqwest::StatusCode::SWITCHING_PROTOCOLS {
			let status = response.status();
			let headers = response.headers().clone();
			let body = response.into_inner().text().await?;
			let captured = shared::CapturedResponse { status, headers, body };
			if retry::is_retryable_status(status.as_u16()) {
				return Err(retry::retryable_status_error(
					&captured,
					format!("OctoHub API error {status}"),
				));
			}
			return Err(auth_aware_error(status.as_u16(), &captured.body, per_call_key));
		}
		Ok(Connection {
			socket: response.into_websocket().await?,
			opened_at: Instant::now(),
		})
	};
	tokio::time::timeout(crate::http::CONNECT_TIMEOUT, setup)
		.await
		.map_err(|_| timeout_error(crate::http::CONNECT_TIMEOUT))?
}

fn timeout_error(timeout: Duration) -> anyhow::Error {
	anyhow::Error::new(ProviderError::TimeoutError {
		provider: "octohub".to_owned(),
	})
	.context(format!("OctoHub WebSocket timed out after {}s", timeout.as_secs_f64()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
	status: u16,
	headers: BTreeMap<String, String>,
	body: serde_json::Map<String, Value>,
}

fn parse_envelope(text: &str) -> Result<shared::CapturedResponse> {
	let envelope: Envelope = serde_json::from_str(text).context("Malformed OctoHub WebSocket envelope")?;
	if !(100..=599).contains(&envelope.status) {
		anyhow::bail!("Invalid OctoHub WebSocket envelope status");
	}
	let status = reqwest::StatusCode::from_u16(envelope.status)?;
	let mut headers = HeaderMap::new();
	for (name, value) in envelope.headers {
		if name != name.to_ascii_lowercase() {
			anyhow::bail!("OctoHub WebSocket envelope header names must be lowercase");
		}
		headers.insert(HeaderName::from_bytes(name.as_bytes())?, HeaderValue::from_str(&value)?);
	}
	Ok(shared::CapturedResponse {
		status,
		headers,
		body: serde_json::to_string(&envelope.body)?,
	})
}

async fn read_response(socket: &mut WebSocket) -> Result<shared::CapturedResponse> {
	while let Some(message) = socket.next().await {
		match message? {
			Message::Text(text) => return parse_envelope(&text),
			Message::Ping(_) => socket.flush().await?,
			Message::Pong(_) => {}
			Message::Close { code, .. } => anyhow::bail!(
				"OctoHub WebSocket closed before its reply (code {code:?}); request outcome is unknown, not retried"
			),
			Message::Binary(_) => {
				anyhow::bail!("OctoHub WebSocket returned binary data; expected a JSON text envelope")
			}
		}
	}
	anyhow::bail!("OctoHub WebSocket disconnected before its reply; request outcome is unknown, not retried")
}

#[cfg(test)]
#[path = "websocket_tests.rs"]
mod tests;
