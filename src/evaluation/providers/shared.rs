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

//! Transport and payload handling shared by the Jev hosts. The inner payload
//! (`{model, answers, usage}`) is identical on TypeSafe and Cloudflare; only
//! the envelope around it differs.

use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::types::{Answer, EvaluationPricing, EvaluationRequest, EvaluationResponse, EvaluationUsage};
use reqwest::header::HeaderMap;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

const RETRY_BACKOFF: Duration = Duration::from_millis(500);
const MAX_RETRY_AFTER: Duration = Duration::from_secs(30);

pub(crate) struct CapturedResponse {
	pub status: reqwest::StatusCode,
	pub headers: HeaderMap,
	pub body: Vec<u8>,
}

/// A blank value is no key: container env files pass unset variables through
/// as empty strings, and sending one only earns the provider's 403.
pub(crate) fn api_key(environment_variable: &str) -> EvaluationResult<String> {
	std::env::var(environment_variable)
		.ok()
		.filter(|key| !key.trim().is_empty())
		.ok_or_else(|| EvaluationError::MissingApiKey(environment_variable.to_string()))
}

/// POST with retries on 429, 529 and 5xx, honouring `Retry-After` and
/// `retry-after-ms`. Evaluations are side-effect free, so a replay after an
/// ambiguous failure cannot create duplicate work.
pub(crate) async fn send(
	provider: &str,
	request: &EvaluationRequest,
	build: impl Fn() -> reqwest::Result<reqwest::RequestBuilder>,
) -> EvaluationResult<CapturedResponse> {
	let mut attempt = 0_u32;
	loop {
		match build()?.timeout(request.timeout).send().await {
			Ok(response) => {
				let status = response.status();
				let headers = response.headers().clone();
				let body = response.bytes().await?.to_vec();
				if retryable(status) && attempt < request.max_retries {
					let delay = retry_after(&headers).unwrap_or(RETRY_BACKOFF * 2u32.pow(attempt));
					tracing::warn!(provider, attempt, %status, "retrying evaluation request");
					tokio::time::sleep(delay).await;
					attempt += 1;
					continue;
				}
				return Ok(CapturedResponse { status, headers, body });
			}
			Err(error) if attempt < request.max_retries && !error.is_timeout() => {
				crate::http::refresh_http_client();
				tracing::warn!(provider, attempt, error = %error, "retrying evaluation request");
				tokio::time::sleep(RETRY_BACKOFF * 2u32.pow(attempt)).await;
				attempt += 1;
			}
			Err(error) => return Err(EvaluationError::Transport(error)),
		}
	}
}

fn retryable(status: reqwest::StatusCode) -> bool {
	status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error()
}

/// `retry-after-ms` first (TypeSafe), then a delta-seconds `Retry-After`;
/// capped so a hostile header cannot park the caller.
pub(crate) fn retry_after(headers: &HeaderMap) -> Option<Duration> {
	let header = |name: &str| {
		headers
			.get(name)
			.and_then(|value| value.to_str().ok())
			.and_then(|value| value.trim().parse::<u64>().ok())
	};
	header("retry-after-ms")
		.map(Duration::from_millis)
		.or_else(|| header("retry-after").map(Duration::from_secs))
		.map(|delay| delay.min(MAX_RETRY_AFTER))
}

pub(crate) fn require_success(provider: &str, response: &CapturedResponse) -> EvaluationResult<()> {
	if response.status.is_success() {
		return Ok(());
	}
	let message = error_message(&response.body);
	Err(match response.status.as_u16() {
		401 => EvaluationError::Authentication {
			provider: provider.to_string(),
			message,
		},
		403 => EvaluationError::Permission {
			provider: provider.to_string(),
			message,
		},
		422 => EvaluationError::InvalidRequest(format!("{provider} rejected the request: {message}")),
		429 => EvaluationError::RateLimit {
			provider: provider.to_string(),
			message,
			retry_after_secs: retry_after(&response.headers).map(|delay| delay.as_secs()),
		},
		status => EvaluationError::Api {
			provider: provider.to_string(),
			status,
			message,
		},
	})
}

pub(crate) fn parse_json(provider: &str, response: &CapturedResponse) -> EvaluationResult<Value> {
	require_success(provider, response)?;
	serde_json::from_slice(&response.body).map_err(|error| EvaluationError::InvalidResponse {
		provider: provider.to_string(),
		message: format!("response was not valid JSON: {error}"),
	})
}

/// Error text from the common JSON error shapes, else the raw body.
pub(crate) fn error_message(body: &[u8]) -> String {
	let text = String::from_utf8_lossy(body);
	serde_json::from_str::<Value>(&text)
		.ok()
		.and_then(|value| {
			value
				.pointer("/errors/0/message")
				.or_else(|| value.pointer("/error/message"))
				.or_else(|| value.pointer("/detail/message"))
				.or_else(|| value.get("error"))
				.or_else(|| value.get("message"))
				.or_else(|| value.get("detail"))
				.map(|found| match found {
					Value::String(message) => message.clone(),
					other => other.to_string(),
				})
		})
		.unwrap_or_else(|| text.trim().chars().take(500).collect())
}

/// The Jev payload `{model, answers, usage}` into the typed response.
pub(crate) fn parse_payload(
	provider: &str,
	payload: &Value,
	pricing: Option<EvaluationPricing>,
) -> EvaluationResult<EvaluationResponse> {
	let invalid = |message: &str| EvaluationError::InvalidResponse {
		provider: provider.to_string(),
		message: message.to_string(),
	};
	let model = payload
		.get("model")
		.and_then(Value::as_str)
		.ok_or_else(|| invalid("payload is missing model"))?
		.to_string();
	let answers: BTreeMap<String, Answer> = serde_json::from_value(
		payload
			.get("answers")
			.cloned()
			.ok_or_else(|| invalid("payload is missing answers"))?,
	)
	.map_err(|error| invalid(&format!("answers did not match the Jev shape: {error}")))?;
	let tokens = |field: &str| {
		payload
			.pointer(&format!("/usage/{field}"))
			.and_then(Value::as_u64)
			.ok_or_else(|| invalid(&format!("usage is missing {field}")))
	};
	let input_tokens = tokens("input_tokens")?;
	let output_tokens = tokens("output_tokens")?;
	Ok(EvaluationResponse {
		model,
		answers,
		usage: EvaluationUsage {
			input_tokens,
			output_tokens,
			cost: pricing.map(|rate| rate.cost(input_tokens, output_tokens)),
		},
	})
}

#[cfg(test)]
#[path = "shared_tests.rs"]
mod tests;
