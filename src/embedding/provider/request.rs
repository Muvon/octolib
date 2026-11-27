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

//! Bounded retries for individual embedding HTTP requests.
use crate::errors::ProviderError;
use anyhow::{Context, Result};
use reqwest::{header::HeaderMap, RequestBuilder};
use serde_json::Value;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::{Duration, SystemTime};
use tokio::time::Instant;

const MAX_RETRIES: u32 = 4;
const MAX_ELAPSED: Duration = Duration::from_secs(300);

fn seconds(value: &str) -> Option<Duration> {
	let value: f64 = value.parse().ok()?;
	Duration::try_from_secs_f64(value).ok()
}

fn retry_after(headers: &HeaderMap, body: &str, now: SystemTime) -> Option<Duration> {
	if let Some(delay) = headers
		.get("retry-after-ms")
		.and_then(|v| v.to_str().ok())
		.and_then(seconds)
		.map(|d| d.div_f64(1000.0))
	{
		return Some(delay);
	}
	if let Some(value) = headers.get("retry-after").and_then(|v| v.to_str().ok()) {
		if let Some(delay) = seconds(value) {
			return Some(delay);
		}
		if let Ok(date) = httpdate::parse_http_date(value) {
			return Some(date.duration_since(now).unwrap_or_default());
		}
	}
	let json: Value = serde_json::from_str(body).ok()?;
	// Google exposes server-directed backoff as google.rpc.RetryInfo.
	if let Some(details) = json["error"]["details"].as_array() {
		for detail in details {
			if detail["@type"] == "type.googleapis.com/google.rpc.RetryInfo" {
				if let Some(delay) = detail["retryDelay"]
					.as_str()
					.and_then(|v| v.strip_suffix('s'))
					.and_then(seconds)
				{
					return Some(delay);
				}
			}
		}
	}
	// OpenAI sometimes supplies the wait only in its rate-limit message.
	let message = json["error"]["message"].as_str()?;
	let value = message
		.split_once("Please try again in ")?
		.1
		.split_whitespace()
		.next()?
		.trim_end_matches('.');
	if let Some(ms) = value.strip_suffix("ms") {
		return seconds(ms).map(|d| d.div_f64(1000.0));
	}
	value.strip_suffix('s').and_then(seconds)
}

fn retryable(status: u16, body: &str) -> bool {
	let json: Value = serde_json::from_str(body).unwrap_or(Value::Null);
	let permanent_quota = ["insufficient_quota", "billing_hard_limit_reached", "billing_not_active"]
		.iter()
		.any(|code| json["error"]["code"] == *code || json["error"]["type"] == *code);
	!permanent_quota && (matches!(status, 408 | 409 | 429) || (500..=599).contains(&status))
}

pub(super) async fn send_json(provider: &str, request: RequestBuilder) -> Result<Value> {
	let deadline = Instant::now() + MAX_ELAPSED;
	let mut attempt = 0;
	loop {
		let retry = request
			.try_clone()
			.context("embedding request body cannot be replayed")?;
		let operation = async {
			let response = retry.send().await?;
			let status = response.status();
			let headers = response.headers().clone();
			let body = response.bytes().await?;
			Ok::<_, reqwest::Error>((status, headers, body))
		};
		let remaining = deadline.saturating_duration_since(Instant::now());
		let result = tokio::time::timeout(remaining, operation)
			.await
			.map_err(|_| ProviderError::TimeoutError {
				provider: provider.to_owned(),
			})?;
		let (error, server_delay, reason) = match result {
			Ok((status, headers, body)) => {
				if status.is_success() {
					return serde_json::from_slice(&body).map_err(|e| ProviderError::ResponseParsingError(e).into());
				}
				let body = String::from_utf8_lossy(&body);
				let error = ProviderError::ApiError {
					provider: provider.to_owned(),
					status: status.as_u16(),
					message: body.to_string(),
				};
				if !retryable(status.as_u16(), &body) {
					return Err(error.into());
				}
				(
					error,
					retry_after(&headers, &body, SystemTime::now()),
					format!("HTTP {}", status.as_u16()),
				)
			}
			Err(error) => {
				// bytes() reports an interrupted body as Decode; JSON parsing occurs
				// separately above, so retrying this does not retry malformed JSON.
				let transient = error.is_timeout()
					|| error.is_connect()
					|| error.is_body()
					|| error.is_request()
					|| error.is_decode();
				let error = ProviderError::NetworkError(error.without_url());
				if !transient {
					return Err(error.into());
				}
				(error, None, "transport error".to_owned())
			}
		};
		if attempt == MAX_RETRIES {
			return Err(error.into());
		}
		// Independent jitter avoids synchronized retries across indexing processes.
		let jitter = Duration::from_millis(RandomState::new().build_hasher().finish() % 251);
		let delay = server_delay
			.unwrap_or_else(|| Duration::from_secs(1 << attempt))
			.saturating_add(jitter);
		if delay >= deadline.saturating_duration_since(Instant::now()) {
			return Err(error.into());
		}
		tracing::warn!(
			provider,
			attempt = attempt + 1,
			max_retries = MAX_RETRIES,
			delay_ms = delay.as_millis() as u64,
			reason,
			"retrying embedding request"
		);
		tokio::time::sleep(delay).await;
		attempt += 1;
	}
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
