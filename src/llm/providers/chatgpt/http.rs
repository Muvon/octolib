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

use super::{auth, with_usage_guidance, ResponseEvents, RetryableStreamError, SESSION_HEADER};
use crate::llm::{providers::shared, retry, types::ChatCompletionParams};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

const RESPONSES_URL: &str = "https://api.openai.com/v1/responses";

/// Echoing the first server token throughout a turn eliminated all measured
/// cache breaks; later server values must not replace it.
const TURN_STATE_HEADER: &str = "x-codex-turn-state";
const TURN_STATE_STORE_CAPACITY: usize = 1_024;

#[derive(Debug)]
struct TurnStateEntry {
	cache_key: String,
	token: String,
}

#[derive(Debug, Default)]
struct TurnStateStore {
	entries: Vec<TurnStateEntry>,
}

impl TurnStateStore {
	fn clear(&mut self, cache_key: &str) {
		if let Some(index) = self.entries.iter().position(|entry| entry.cache_key == cache_key) {
			self.entries.remove(index);
		}
	}

	fn get(&mut self, cache_key: &str) -> Option<String> {
		let index = self.entries.iter().position(|entry| entry.cache_key == cache_key)?;
		let entry = self.entries.remove(index);
		let token = entry.token.clone();
		self.entries.push(entry);
		Some(token)
	}

	fn store_first(&mut self, cache_key: &str, token: &str) {
		if let Some(index) = self.entries.iter().position(|entry| entry.cache_key == cache_key) {
			let entry = self.entries.remove(index);
			self.entries.push(entry);
			return;
		}

		if self.entries.len() == TURN_STATE_STORE_CAPACITY {
			self.entries.remove(0);
		}
		self.entries.push(TurnStateEntry {
			cache_key: cache_key.to_string(),
			token: token.to_string(),
		});
	}
}

static TURN_STATES: LazyLock<Mutex<TurnStateStore>> = LazyLock::new(|| Mutex::new(TurnStateStore::default()));

pub(super) async fn complete(params: &ChatCompletionParams, request_body: &Value) -> Result<Value> {
	let extra_headers = request_headers(params)?;

	// Errors are retried; `Ok` holds the outcome of a delivered response,
	// which a retry would not change.
	let response_json = retry::retry_chat_operation(
		|| {
			let client = shared::http_client();
			let request_body = request_body.clone();
			let request_timeout = params.request_timeout;
			let extra_headers = extra_headers.clone();
			let capture_turn_state = params.prompt_cache_key.is_some();
			Box::pin(async move {
				let request = client.post(RESPONSES_URL).json(&request_body);
				let captured = match auth::send_authenticated(request, request_timeout, extra_headers.as_ref()).await {
					Ok(captured) => captured,
					Err(error) if error.is::<reqwest::Error>() => return Err(error),
					// Credential failures need sign-in, not another refresh attempt.
					Err(error) => return Ok(Err(error)),
				};

				if !captured.status.is_success() {
					let code = serde_json::from_str::<Value>(&captured.body)
						.ok()
						.and_then(|body| body["error"]["code"].as_str().map(str::to_owned));
					let message = format!("ChatGPT API error {}: {}", captured.status, captured.body);
					if retry::is_retryable_status(captured.status.as_u16())
						&& !code
							.as_deref()
							.is_some_and(|code| code.starts_with("subscription_sharing_usage"))
					{
						return Err(retry::retryable_status_error(&captured, message));
					}
					return Ok(Err(with_usage_guidance(code.as_deref(), message)));
				}

				let turn_state = if capture_turn_state {
					captured
						.headers
						.get(TURN_STATE_HEADER)
						.map(|value| value.to_str())
						.transpose()
						.context("ChatGPT response returned an invalid x-codex-turn-state header")?
						.map(str::to_owned)
				} else {
					None
				};

				match merge_stream(&captured.body) {
					Err(e) if e.is::<RetryableStreamError>() => Err(e),
					merged => Ok(merged.map(|body| (body, turn_state))),
				}
			})
		},
		params,
	)
	.await??;

	let (response_json, response_turn_state) = response_json;
	remember_turn_state(params.prompt_cache_key.as_deref(), response_turn_state.as_deref())?;
	Ok(response_json)
}

fn request_headers(params: &ChatCompletionParams) -> Result<Option<HashMap<String, String>>> {
	if params.prompt_cache_key.is_none() {
		return Ok(request_headers_without_turn_state(params));
	}
	let mut turn_states = TURN_STATES
		.lock()
		.map_err(|_| anyhow::anyhow!("ChatGPT turn-state store lock poisoned"))?;
	Ok(request_headers_with_store(params, &mut turn_states))
}

fn request_headers_with_store(
	params: &ChatCompletionParams,
	turn_states: &mut TurnStateStore,
) -> Option<HashMap<String, String>> {
	let mut headers = request_headers_without_turn_state(params);

	let Some(cache_key) = &params.prompt_cache_key else {
		return headers;
	};

	let request_headers = headers.get_or_insert_with(HashMap::new);
	request_headers
		.entry(SESSION_HEADER.to_string())
		.or_insert_with(|| cache_key.clone());

	if params.messages.last().is_some_and(|message| message.role == "user") {
		turn_states.clear(cache_key);
	} else if let Some(token) = turn_states.get(cache_key) {
		request_headers.insert(TURN_STATE_HEADER.to_string(), token);
	}
	headers
}

fn request_headers_without_turn_state(params: &ChatCompletionParams) -> Option<HashMap<String, String>> {
	let mut headers = params.extra_headers.clone();
	remove_header(&mut headers, TURN_STATE_HEADER);
	headers.filter(|headers| !headers.is_empty())
}

fn remove_header(headers: &mut Option<HashMap<String, String>>, name: &str) {
	if let Some(headers) = headers {
		headers.retain(|header, _| !header.eq_ignore_ascii_case(name));
	}
}

fn remember_turn_state(cache_key: Option<&str>, token: Option<&str>) -> Result<()> {
	let (Some(cache_key), Some(token)) = (cache_key, token) else {
		return Ok(());
	};
	let mut turn_states = TURN_STATES
		.lock()
		.map_err(|_| anyhow::anyhow!("ChatGPT turn-state store lock poisoned"))?;
	remember_turn_state_with_store(Some(cache_key), Some(token), &mut turn_states);
	Ok(())
}

fn remember_turn_state_with_store(cache_key: Option<&str>, token: Option<&str>, turn_states: &mut TurnStateStore) {
	if let (Some(cache_key), Some(token)) = (cache_key, token) {
		turn_states.store_first(cache_key, token);
	}
}

pub(super) fn merge_stream(body: &str) -> Result<Value> {
	let mut events = ResponseEvents::default();
	let mut data = String::new();
	for line in body.lines().chain(std::iter::once("")) {
		if line.is_empty() && !data.is_empty() {
			let event = serde_json::from_str(&data).context("Malformed ChatGPT SSE event")?;
			if let Some(response) = events.push(event)? {
				return Ok(response);
			}
			data.clear();
		} else if let Some(value) = line.strip_prefix("data:") {
			data.push_str(value);
			data.push('\n');
		}
	}
	Err(RetryableStreamError::CutOff.into())
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
