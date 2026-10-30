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

use super::{auth, normalize_input_item, request_headers, with_usage_guidance, ResponseEvents, RetryableStreamError};
use crate::llm::{providers::shared, retry, types::ChatCompletionParams};
use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use reqwest::header::HeaderMap;
use reqwest_websocket::{Message, RequestBuilderExt, WebSocket};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";
const MAX_SESSIONS: usize = 64;
const IDLE_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECTION_LIFETIME: Duration = Duration::from_secs(55 * 60);

type Session = Arc<AsyncMutex<Option<Connection>>>;

// Callers recreate providers between turns. Only idle connections can be evicted;
// active leases keep the slot unique until the request releases it.
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
				.context("All ChatGPT WebSocket sessions are busy")?;
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
	continuation: Option<Continuation>,
}

struct Continuation {
	response_id: String,
	input_len: usize,
	prefix_hash: [u8; 32],
}

impl Continuation {
	fn from_response(request: &Value, response: &Value) -> Result<Self> {
		let input = request["input"]
			.as_array()
			.context("ChatGPT request is missing input")?;
		let output = response["output"]
			.as_array()
			.context("ChatGPT response is missing output")?;
		let response_id = response["id"]
			.as_str()
			.filter(|id| !id.is_empty())
			.context("ChatGPT response is missing id")?
			.to_owned();
		let mut hasher = Sha256::new();
		for item in input {
			hash_item(&mut hasher, item)?;
		}
		for item in output {
			hash_item(&mut hasher, &normalize_input_item(item.clone()))?;
		}
		Ok(Self {
			response_id,
			input_len: input.len() + output.len(),
			prefix_hash: hasher.finalize().into(),
		})
	}

	fn request(&self, full: &Value) -> Result<Value> {
		let input = full["input"].as_array().context("ChatGPT request is missing input")?;
		let mut request = full.clone();
		if input.len() >= self.input_len {
			let mut hasher = Sha256::new();
			for item in &input[..self.input_len] {
				hash_item(&mut hasher, item)?;
			}
			let hash: [u8; 32] = hasher.finalize().into();
			if hash == self.prefix_hash {
				request["previous_response_id"] = json!(self.response_id);
				request["input"] = json!(&input[self.input_len..]);
			}
		}
		Ok(request)
	}
}

fn hash_item(hasher: &mut Sha256, item: &Value) -> Result<()> {
	let bytes = serde_json::to_vec(item)?;
	hasher.update((bytes.len() as u64).to_be_bytes());
	hasher.update(bytes);
	Ok(())
}

fn session_key(cache_key: Option<&str>, model: &str, token: &str, headers: &HeaderMap) -> Result<Option<[u8; 32]>> {
	let Some(session) = cache_key else {
		return Ok(None);
	};
	let mut hasher = Sha256::new();
	hash_item(&mut hasher, &json!([session, model, token]))?;
	let sorted: BTreeMap<_, _> = headers
		.iter()
		.map(|(name, value)| (name.as_str(), value.as_bytes()))
		.collect();
	for (name, value) in sorted {
		hash_item(&mut hasher, &json!([name.as_bytes(), value]))?;
	}
	Ok(Some(hasher.finalize().into()))
}

pub(super) async fn complete(params: &ChatCompletionParams, full: Value) -> Result<(Value, Value)> {
	let headers = request_headers(params)?;
	let full = Arc::new(full);
	retry::retry_chat_operation(
		|| {
			let headers = headers.clone();
			let full = full.clone();
			let model = params.model.clone();
			let cache_key = params.prompt_cache_key.clone();
			let timeout = params.request_timeout;
			Box::pin(async move {
				let attempt = async {
					let token = auth::access_token(None).await?;
					match authenticated_request(cache_key.as_deref(), &model, &full, &headers, &token).await {
						Err(error) if error.is::<Unauthorized>() => {
							let refreshed = auth::access_token(Some(&token)).await?;
							authenticated_request(cache_key.as_deref(), &model, &full, &headers, &refreshed).await
						}
						result => result,
					}
				};
				let result = match timeout {
					Some(timeout) => match tokio::time::timeout(timeout, attempt).await {
						Ok(result) => result,
						Err(_) => Err(RetryableStreamError::Timeout.into()),
					},
					None => attempt.await,
				};
				match result {
					Err(error) if is_retryable(&error) => Err(error),
					result => Ok(result),
				}
			})
		},
		params,
	)
	.await?
}

fn is_retryable(error: &anyhow::Error) -> bool {
	error.is::<RetryableStreamError>()
		|| error.is::<retry::RetryableStatusError>()
		|| error.is::<reqwest::Error>()
		|| matches!(
			error.downcast_ref::<reqwest_websocket::Error>(),
			Some(reqwest_websocket::Error::Reqwest(_) | reqwest_websocket::Error::Tungstenite(_))
		)
}

async fn authenticated_request(
	cache_key: Option<&str>,
	model: &str,
	full: &Value,
	headers: &HeaderMap,
	token: &str,
) -> Result<(Value, Value)> {
	let session = match session_key(cache_key, model, token, headers)? {
		Some(key) => SESSIONS
			.lock()
			.map_err(|_| anyhow::anyhow!("ChatGPT WebSocket pool lock poisoned"))?
			.get(key, Instant::now())?,
		None => Arc::new(AsyncMutex::new(None)),
	};
	exchange(&session, full, headers, token, RESPONSES_URL).await
}

async fn exchange(
	session: &Session,
	full: &Value,
	headers: &HeaderMap,
	token: &str,
	url: &str,
) -> Result<(Value, Value)> {
	let mut slot = session.lock().await;
	// The future owns the socket until completion. Cancellation or any error
	// drops it, so a later request cannot consume a cancelled response's events.
	let mut connection = match slot.take() {
		Some(connection) if connection.opened_at.elapsed() < CONNECTION_LIFETIME => connection,
		_ => connect(url, token, headers).await?,
	};
	let request = match &connection.continuation {
		Some(previous) => previous.request(full)?,
		None => full.clone(),
	};
	connection
		.socket
		.send(Message::Text(serde_json::to_string(&request)?))
		.await?;
	let response = read_response(&mut connection.socket).await?;
	connection.continuation = Some(Continuation::from_response(full, &response)?);
	*slot = Some(connection);
	Ok((request, response))
}

#[derive(Debug, thiserror::Error)]
#[error("ChatGPT WebSocket authentication rejected; sign in again")]
struct Unauthorized;

async fn connect(url: &str, token: &str, headers: &HeaderMap) -> Result<Connection> {
	let response = crate::http::websocket_client()
		.get(url)
		.bearer_auth(token)
		.headers(headers.clone())
		.upgrade()
		.send()
		.await?;
	if response.status() == reqwest::StatusCode::UNAUTHORIZED {
		return Err(Unauthorized.into());
	}
	if response.status() != reqwest::StatusCode::SWITCHING_PROTOCOLS {
		let status = response.status();
		let headers = response.headers().clone();
		let body = response.into_inner().text().await?;
		let code = serde_json::from_str::<Value>(&body)
			.ok()
			.and_then(|body| body["error"]["code"].as_str().map(str::to_owned));
		let message = format!("ChatGPT WebSocket upgrade failed: {status}: {body}");
		if retry::is_retryable_status(status.as_u16())
			&& !code
				.as_deref()
				.is_some_and(|code| code.starts_with("subscription_sharing_usage"))
		{
			return Err(retry::retryable_status_error(
				&shared::CapturedResponse { status, headers, body },
				message,
			));
		}
		return Err(with_usage_guidance(code.as_deref(), message));
	}
	Ok(Connection {
		socket: response.into_websocket().await?,
		opened_at: Instant::now(),
		continuation: None,
	})
}

async fn read_response(socket: &mut WebSocket) -> Result<Value> {
	let mut events = ResponseEvents::default();
	while let Some(message) = socket.next().await {
		match message? {
			Message::Text(text) => {
				let event = serde_json::from_str(&text).context("Malformed ChatGPT WebSocket event")?;
				if let Some(response) = events.push(event)? {
					return Ok(response);
				}
			}
			Message::Ping(_) => socket.flush().await?,
			Message::Pong(_) => {}
			Message::Close { .. } => break,
			Message::Binary(_) => anyhow::bail!("ChatGPT WebSocket returned a binary event; expected JSON text"),
		}
	}
	Err(RetryableStreamError::CutOff.into())
}

#[cfg(test)]
#[path = "websocket_tests.rs"]
mod tests;
