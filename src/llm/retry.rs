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

//! Retry logic with exponential backoff for AI provider requests.
//!
//! # Retry Behavior
//!
//! This module provides retry functionality for transient API failures:
//!
//! **Retryable HTTP Status Codes:**
//! - `429` - Rate limit exceeded (temporary, will resolve)
//! - `500-599` - Server errors (temporary infrastructure issues)
//!
//! **Non-Retryable Status Codes:**
//! - `400-499` (except 429) - Client errors (won't resolve with retry)
//! - `1xx`, `2xx`, `3xx` - Not errors
//!
//! # Backoff Strategy
//!
//! Exponential backoff with configurable base timeout:
//! - Delay grows as: `base_timeout * 2^attempt`
//! - A `Retry-After: <seconds>` header on a 429/503 response stretches the
//!   delay to at least that long (the HTTP-date form is ignored)
//! - Maximum delay capped at 5 minutes
//! - Example: base 1s → 1s, 2s, 4s, 8s, 16s...
//!
//! # Observing retries
//!
//! A [`RetryObserver`] on [`ChatCompletionParams`](crate::llm::types::ChatCompletionParams)
//! is told about every retry before its backoff sleep starts, so a caller can show
//! the wait instead of going silent for minutes. The event carries only a short
//! cause (`HTTP 503`, `connection error`) — never a response body or credential.
//!
//! # Cancellation
//!
//! All retry operations support optional cancellation tokens.
//! When the token is triggered, the operation immediately returns
//! a cancellation error without further retries.

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;

/// Longest single backoff sleep, whatever the exponent or `Retry-After` says.
const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

/// One retry about to happen, as told to a [`RetryObserver`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryEvent {
	/// The retry about to run, 1-based (`1..=max_retries`).
	pub attempt: u32,
	/// Retries allowed for this call.
	pub max_retries: u32,
	/// How long the call sleeps before that retry.
	pub delay: Duration,
	/// Short, secret-free cause of the failed attempt: `HTTP 503`,
	/// a transport failure category, or `request failed`.
	pub reason: String,
}

/// Callback told before every backoff sleep of a provider call.
///
/// It runs on the request's task, so it must return quickly — hand the event
/// to a channel or a print, never block on it.
#[derive(Clone)]
pub struct RetryObserver(Arc<dyn Fn(&RetryEvent) + Send + Sync>);

impl RetryObserver {
	pub fn new(observer: impl Fn(&RetryEvent) + Send + Sync + 'static) -> Self {
		Self(Arc::new(observer))
	}

	pub fn notify(&self, event: &RetryEvent) {
		(self.0)(event)
	}
}

impl std::fmt::Debug for RetryObserver {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.write_str("RetryObserver")
	}
}

/// A delivered HTTP response whose status is worth retrying (429 or 5xx).
///
/// Displays exactly as the provider's message, so callers see the same error
/// text as before; the typed fields feed the retry loop's [`RetryEvent`] and
/// its `Retry-After` handling.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct RetryableStatusError {
	pub status: u16,
	/// The response's `Retry-After`, in seconds form only, on 429/503.
	pub retry_after: Option<Duration>,
	message: String,
}

impl RetryableStatusError {
	pub(crate) fn with_message(&self, message: String) -> Self {
		Self {
			status: self.status,
			retry_after: self.retry_after,
			message,
		}
	}
}

/// Build the error a provider returns for a retryable status, keeping the
/// status and `Retry-After` the retry loop needs.
pub(crate) fn retryable_status_error(
	response: &crate::llm::providers::shared::CapturedResponse,
	message: String,
) -> anyhow::Error {
	let status = response.status.as_u16();
	RetryableStatusError {
		status,
		retry_after: parse_retry_after(status, &response.headers),
		message,
	}
	.into()
}

/// `Retry-After` in delay-seconds form, honored on 429 and 503 only. The
/// HTTP-date form is ignored: it needs a trusted clock and no provider we
/// call sends it.
fn parse_retry_after(status: u16, headers: &reqwest::header::HeaderMap) -> Option<Duration> {
	if status != 429 && status != 503 {
		return None;
	}
	headers
		.get(reqwest::header::RETRY_AFTER)?
		.to_str()
		.ok()?
		.trim()
		.parse::<u64>()
		.ok()
		.map(Duration::from_secs)
}

/// What the retry loop knows about a failed attempt.
struct RetryCause {
	reason: String,
	retry_after: Option<Duration>,
}

fn describe_error(error: &anyhow::Error) -> RetryCause {
	if let Some(status) = error.downcast_ref::<RetryableStatusError>() {
		return RetryCause {
			reason: format!("HTTP {}", status.status),
			retry_after: status.retry_after,
		};
	}
	let reason =
		crate::llm::providers::shared::transport_error_kind(error).map_or("request failed", |kind| kind.reason());
	RetryCause {
		reason: reason.to_owned(),
		retry_after: None,
	}
}

/// Backoff before the retry that follows failed attempt `attempt` (0-based):
/// `base_timeout * 2^attempt`, stretched to `retry_after` when the server
/// asked for longer, capped at [`MAX_RETRY_DELAY`].
fn backoff_delay(base_timeout: Duration, attempt: u32, retry_after: Option<Duration>) -> Duration {
	let delay = base_timeout.saturating_mul(2_u32.saturating_pow(attempt));
	let delay = retry_after.map_or(delay, |server| delay.max(server));
	delay.min(MAX_RETRY_DELAY)
}

/// Await a future while honoring an optional cancellation token.
pub async fn cancellable<F, T, E, C>(
	future: F,
	cancellation_token: Option<&watch::Receiver<bool>>,
	on_cancel: C,
) -> Result<T, E>
where
	F: Future<Output = Result<T, E>> + Send,
	C: Fn() -> E + Copy,
{
	if let Some(token) = cancellation_token {
		if *token.borrow() {
			return Err(on_cancel());
		}

		let mut token = token.clone();
		let mut future = Box::pin(future);

		loop {
			tokio::select! {
				result = &mut future => return result,
				changed = token.changed() => {
					if changed.is_err() || *token.borrow() {
						return Err(on_cancel());
					}
				}
			}
		}
	} else {
		future.await
	}
}

async fn sleep_cancellable<E, C>(
	duration: Duration,
	cancellation_token: Option<&watch::Receiver<bool>>,
	on_cancel: C,
) -> Result<(), E>
where
	C: Fn() -> E + Copy,
{
	cancellable(
		async {
			sleep(duration).await;
			Ok(())
		},
		cancellation_token,
		on_cancel,
	)
	.await
}

pub(crate) async fn retry_provider_request<F>(
	mut operation: F,
	max_retries: u32,
	base_timeout: Duration,
	cancellation_token: Option<&watch::Receiver<bool>>,
	api_key: Option<&crate::llm::types::PerCallApiKey>,
	retry_observer: Option<&RetryObserver>,
) -> Result<crate::llm::providers::shared::CapturedResponse>
where
	F: FnMut() -> Pin<Box<dyn Future<Output = Result<crate::llm::providers::shared::CapturedResponse>> + Send>>,
{
	use crate::errors::ProviderError;
	use crate::llm::providers::shared;
	let api_key = api_key.cloned();
	retry_with_backoff(
		|| {
			let response = operation();
			let api_key = api_key.clone();
			Box::pin(async move {
				let mut response = response
					.await
					.map_err(|error| shared::redact_error(error, api_key.as_ref()))?;
				if !response.status.is_success() {
					response.body = shared::redact_text(&response.body, api_key.as_ref());
				}
				Ok(response)
			})
		},
		max_retries,
		base_timeout,
		cancellation_token,
		|| ProviderError::Cancelled.into(),
		|error| {
			matches!(error.downcast_ref::<ProviderError>(), Some(ProviderError::Cancelled))
				|| shared::is_terminal_stream_error(error)
		},
		describe_error,
		retry_observer,
	)
	.await
}

/// Retry a provider call that is not a single HTTP request (a streamed
/// response, a CLI subprocess) with the retry settings, cancellation token
/// and observer carried on `params`.
pub(crate) async fn retry_chat_operation<F, T>(
	operation: F,
	params: &crate::llm::types::ChatCompletionParams,
) -> Result<T>
where
	F: FnMut() -> Pin<Box<dyn Future<Output = Result<T>> + Send>>,
{
	use crate::errors::ProviderError;
	retry_with_backoff(
		operation,
		params.max_retries,
		params.retry_timeout,
		params.cancellation_token.as_ref(),
		|| ProviderError::Cancelled.into(),
		|error| matches!(error.downcast_ref::<ProviderError>(), Some(ProviderError::Cancelled)),
		describe_error,
		params.retry_observer.as_ref(),
	)
	.await
}

/// Generic retry logic with exponential backoff for providers that don't have smart retry
///
/// This function implements exponential backoff with a configurable base timeout.
/// The delay grows as: base_timeout * 2^attempt, capped at 5 minutes.
///
/// Connection health and eviction remain owned by the HTTP transport. The
/// `on_connection_error` classifier supplies diagnostic context only.
///
/// # Arguments
/// * `operation` - The async operation to retry (must return Result<T, E>)
/// * `max_retries` - Maximum number of retry attempts (0 = no retries, just one attempt)
/// * `base_timeout` - Base delay for exponential backoff
/// * `cancellation_token` - Optional token to check for cancellation
///
/// # Returns
/// * `Ok(T)` - Success result from the operation
/// * `Err(E)` - The last error encountered after all retries exhausted
pub async fn retry_with_exponential_backoff<F, T, E>(
	operation: F,
	max_retries: u32,
	base_timeout: Duration,
	cancellation_token: Option<&watch::Receiver<bool>>,
	on_cancel: impl Fn() -> E + Copy,
	is_cancelled: impl Fn(&E) -> bool + Copy,
	on_connection_error: impl Fn(&E) -> bool + Copy,
) -> Result<T, E>
where
	F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
	E: std::fmt::Display,
{
	retry_with_backoff(
		operation,
		max_retries,
		base_timeout,
		cancellation_token,
		on_cancel,
		is_cancelled,
		|error| RetryCause {
			reason: if on_connection_error(error) {
				"connection error"
			} else {
				"request failed"
			}
			.to_owned(),
			retry_after: None,
		},
		None,
	)
	.await
}

#[allow(clippy::too_many_arguments)]
async fn retry_with_backoff<F, T, E>(
	mut operation: F,
	max_retries: u32,
	base_timeout: Duration,
	cancellation_token: Option<&watch::Receiver<bool>>,
	on_cancel: impl Fn() -> E + Copy,
	stop_retrying: impl Fn(&E) -> bool + Copy,
	describe: impl Fn(&E) -> RetryCause,
	retry_observer: Option<&RetryObserver>,
) -> Result<T, E>
where
	F: FnMut() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
	E: std::fmt::Display,
{
	let mut last_error = None;

	for attempt in 0..=max_retries {
		// Check for cancellation before each attempt
		if let Some(token) = cancellation_token {
			if *token.borrow() {
				return Err(on_cancel());
			}
		}

		match cancellable(operation(), cancellation_token, on_cancel).await {
			Ok(result) => return Ok(result),
			Err(e) => {
				if stop_retrying(&e) {
					return Err(e);
				}

				let cause = describe(&e);
				tracing::debug!(reason = %cause.reason, "API request attempt {} failed: {:#}", attempt + 1, e);

				// Don't sleep after the last attempt
				if attempt < max_retries {
					let delay = backoff_delay(base_timeout, attempt, cause.retry_after);

					tracing::debug!("Waiting {:?} before retry attempt {}", delay, attempt + 2);

					if let Some(observer) = retry_observer {
						observer.notify(&RetryEvent {
							attempt: attempt + 1,
							max_retries,
							delay,
							reason: cause.reason,
						});
					}

					last_error = Some(e);
					sleep_cancellable(delay, cancellation_token, on_cancel).await?;
				} else {
					last_error = Some(e);
				}
			}
		}
	}

	Err(last_error.unwrap())
}

/// Returns true for HTTP status codes that should trigger a retry.
/// Retries on 429 (rate limit) and all 5xx server errors.
/// Does NOT retry on 4xx client errors (400, 401, 403, 404, etc.) since those won't resolve.
pub fn is_retryable_status(status: u16) -> bool {
	status == 429 || status >= 500
}

/// Helper to wrap HTTP requests in retry logic
///
/// This is a convenience wrapper that creates the appropriate closure for HTTP requests.
/// It handles cloning of necessary data for each retry attempt.
pub async fn retry_http_request<T, E>(
	max_retries: u32,
	base_timeout: Duration,
	cancellation_token: Option<&watch::Receiver<bool>>,
	on_cancel: impl Fn() -> E + Copy,
	is_cancelled: impl Fn(&E) -> bool + Copy,
	on_connection_error: impl Fn(&E) -> bool + Copy,
	request_builder: impl Fn() -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>,
) -> Result<T, E>
where
	E: std::fmt::Display,
{
	retry_with_exponential_backoff(
		|| request_builder(),
		max_retries,
		base_timeout,
		cancellation_token,
		on_cancel,
		is_cancelled,
		on_connection_error,
	)
	.await
}
#[cfg(test)]
#[path = "retry_tests.rs"]
mod tests;
