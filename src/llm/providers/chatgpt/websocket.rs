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

use super::super::websocket_transport as transport;
use super::{auth, normalize_input_item, request_headers, with_usage_guidance, ResponseEvents, RetryableStreamError};
use crate::errors::ProviderError;
use crate::llm::{providers::shared, retry, types::ChatCompletionParams};
use anyhow::{Context, Result};
use reqwest::header::HeaderMap;
use reqwest_websocket::{Message, RequestBuilderExt};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as AsyncMutex;

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";

type Session = transport::Session<Option<Continuation>>;
type Connection = transport::Connection<Option<Continuation>>;
type SessionPool = transport::SessionPool<Option<Continuation>>;
static SESSIONS: LazyLock<Mutex<SessionPool>> = LazyLock::new(|| Mutex::new(SessionPool::default()));

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
			let stream_timeout = params.stream_timeout;
			Box::pin(async move {
				let result = async {
					let token = auth::access_token(None).await?;
					match authenticated_request(
						cache_key.as_deref(),
						&model,
						&full,
						&headers,
						&token,
						timeout,
						stream_timeout,
					)
					.await
					{
						Err(error) if error.is::<Unauthorized>() => {
							let refreshed = auth::access_token(Some(&token)).await?;
							authenticated_request(
								cache_key.as_deref(),
								&model,
								&full,
								&headers,
								&refreshed,
								timeout,
								stream_timeout,
							)
							.await
						}
						result => result,
					}
				}
				.await;
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
		|| matches!(
			error.downcast_ref::<transport::ConnectionError>(),
			Some(transport::ConnectionError::HeartbeatTimeout | transport::ConnectionError::WriteTimeout)
		) || error.is::<super::super::streaming::StreamTimeout>()
		|| matches!(
			error.downcast_ref::<ProviderError>(),
			Some(ProviderError::TimeoutError { .. })
		) || error.is::<retry::RetryableStatusError>()
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
	timeout: Option<Duration>,
	stream_timeout: Option<Duration>,
) -> Result<(Value, Value)> {
	let session = match session_key(cache_key, model, token, headers)? {
		Some(key) => SESSIONS
			.lock()
			.map_err(|_| anyhow::anyhow!("ChatGPT WebSocket pool lock poisoned"))?
			.get(key, Instant::now())?,
		None => Arc::new(AsyncMutex::new(None)),
	};
	exchange(&session, full, headers, token, RESPONSES_URL, timeout, stream_timeout).await
}

async fn exchange(
	session: &Session,
	full: &Value,
	headers: &HeaderMap,
	token: &str,
	url: &str,
	timeout: Option<Duration>,
	stream_timeout: Option<Duration>,
) -> Result<(Value, Value)> {
	let mut slot = session.lock().await;
	let operation = async {
		// Taking ownership before awaiting ensures cancellation or failure drops the
		// socket instead of exposing this response's remaining events to the next call.
		let connection = match slot.take() {
			Some(idle) => idle.resume().await?,
			None => None,
		};
		let mut connection = match connection {
			Some(connection) => connection,
			None => connect(url, token, headers).await?,
		};
		let request = match &connection.state {
			Some(previous) => previous.request(full)?,
			None => full.clone(),
		};
		let message = Message::Text(serde_json::to_string(&request)?);
		connection.socket.send(message).await?;
		let response = read_response(&mut connection.socket, stream_timeout).await?;
		connection.state = Some(Continuation::from_response(full, &response)?);
		*slot = Some(connection.park());
		Ok((request, response))
	};
	match timeout {
		Some(timeout) => tokio::time::timeout(timeout, operation)
			.await
			.map_err(|_| timeout_error(timeout, "before the complete response (request deadline)"))?,
		None => operation.await,
	}
}

#[derive(Debug, thiserror::Error)]
#[error("ChatGPT WebSocket authentication rejected; sign in again")]
struct Unauthorized;

async fn connect(url: &str, token: &str, headers: &HeaderMap) -> Result<Connection> {
	let setup = async {
		let response = crate::http::websocket_client()
			.get(url)
			.timeout(crate::http::CONNECT_TIMEOUT)
			.bearer_auth(token)
			.headers(headers.clone())
			.upgrade()
			.send()
			.await
			.context("Opening ChatGPT WebSocket connection")?;
		if response.status() == reqwest::StatusCode::UNAUTHORIZED {
			return Err(Unauthorized.into());
		}
		if response.status() != reqwest::StatusCode::SWITCHING_PROTOCOLS {
			let status = response.status();
			let headers = response.headers().clone();
			let body = response.into_inner().text().await.with_context(|| {
				format!("ChatGPT WebSocket upgrade rejected (HTTP {status}); failed to read error body")
			})?;
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
		Ok(Connection::new(
			response
				.into_websocket()
				.await
				.context("Completing ChatGPT WebSocket upgrade")?,
			"chatgpt",
			None,
		))
	};
	tokio::time::timeout(crate::http::CONNECT_TIMEOUT, setup)
		.await
		.map_err(|_| timeout_error(crate::http::CONNECT_TIMEOUT, "during connection setup"))?
}

fn timeout_error(timeout: Duration, phase: &str) -> anyhow::Error {
	anyhow::Error::new(ProviderError::TimeoutError {
		provider: "chatgpt".to_owned(),
	})
	.context(format!(
		"ChatGPT WebSocket timed out after {}s {phase}",
		timeout.as_secs_f64()
	))
}

async fn read_response(socket: &mut transport::Socket, timeout: Option<Duration>) -> Result<Value> {
	let mut events = ResponseEvents::default();
	let mut last_received = tokio::time::Instant::now();
	loop {
		let message = match timeout {
			Some(timeout) => tokio::time::timeout_at(last_received + timeout, socket.next())
				.await
				.map_err(|_| super::super::streaming::StreamTimeout(timeout))??,
			None => socket.next().await?,
		};
		let Some(message) = message else {
			break;
		};
		match message {
			transport::Event::Text(text) => {
				last_received = tokio::time::Instant::now();
				let event = serde_json::from_str(&text).context("Malformed ChatGPT WebSocket event")?;
				if let Some(response) = events.push(event)? {
					return Ok(response);
				}
			}
			transport::Event::Close(_) => break,
			transport::Event::Binary => anyhow::bail!("ChatGPT WebSocket returned a binary event; expected JSON text"),
		}
	}
	Err(RetryableStreamError::CutOff.into())
}

#[cfg(test)]
#[path = "websocket_tests.rs"]
mod tests;
