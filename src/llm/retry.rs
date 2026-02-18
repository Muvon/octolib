// Copyright 2025 Muvon Un Limited
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
//! - Maximum delay capped at 5 minutes
//! - Example: base 1s → 1s, 2s, 4s, 8s, 16s...
//!
//! # Cancellation
//!
//! All retry operations support optional cancellation tokens.
//! When the token is triggered, the operation immediately returns
//! a cancellation error without further retries.

use anyhow::Result;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;

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

/// Generic retry logic with exponential backoff for providers that don't have smart retry
///
/// This function implements exponential backoff with a configurable base timeout.
/// The delay grows as: base_timeout * 2^attempt, capped at 5 minutes.
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
    mut operation: F,
    max_retries: u32,
    base_timeout: Duration,
    cancellation_token: Option<&watch::Receiver<bool>>,
    on_cancel: impl Fn() -> E + Copy,
    is_cancelled: impl Fn(&E) -> bool + Copy,
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
                if is_cancelled(&e) {
                    return Err(e);
                }

                // Simple debug logging without external dependencies
                tracing::debug!("API request attempt {} failed: {}", attempt + 1, e);

                last_error = Some(e);

                // Don't sleep after the last attempt
                if attempt < max_retries {
                    // Exponential backoff: base_timeout * 2^attempt
                    let delay = base_timeout * 2_u32.pow(attempt);
                    // Cap at 5 minutes for safety
                    let delay = std::cmp::min(delay, Duration::from_secs(300));

                    tracing::debug!("Waiting {:?} before retry attempt {}", delay, attempt + 2);

                    sleep_cancellable(delay, cancellation_token, on_cancel).await?;
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
    )
    .await
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_retryable_status_429_rate_limit() {
        // 429 (rate limit) should be retryable
        assert!(is_retryable_status(429), "429 should be retryable");
    }

    #[test]
    fn test_is_retryable_status_5xx_server_errors() {
        // All 5xx server errors should be retryable
        assert!(is_retryable_status(500), "500 should be retryable");
        assert!(is_retryable_status(501), "501 should be retryable");
        assert!(is_retryable_status(502), "502 should be retryable");
        assert!(is_retryable_status(503), "503 should be retryable");
        assert!(is_retryable_status(504), "504 should be retryable");
        assert!(is_retryable_status(599), "599 should be retryable");
    }

    #[test]
    fn test_is_retryable_status_4xx_client_errors_not_retryable() {
        // 4xx client errors should NOT be retryable (except 429)
        assert!(!is_retryable_status(400), "400 should not be retryable");
        assert!(!is_retryable_status(401), "401 should not be retryable");
        assert!(!is_retryable_status(403), "403 should not be retryable");
        assert!(!is_retryable_status(404), "404 should not be retryable");
        assert!(!is_retryable_status(405), "405 should not be retryable");
        assert!(!is_retryable_status(408), "408 should not be retryable");
        assert!(!is_retryable_status(418), "418 should not be retryable");
    }

    #[test]
    fn test_is_retryable_status_2xx_success_not_retryable() {
        // 2xx success codes should NOT be retryable (they're already successful)
        assert!(!is_retryable_status(200), "200 should not be retryable");
        assert!(!is_retryable_status(201), "201 should not be retryable");
        assert!(!is_retryable_status(204), "204 should not be retryable");
    }

    #[test]
    fn test_is_retryable_status_3xx_redirect_not_retryable() {
        // 3xx redirect codes should NOT be retryable
        assert!(!is_retryable_status(301), "301 should not be retryable");
        assert!(!is_retryable_status(302), "302 should not be retryable");
        assert!(!is_retryable_status(304), "304 should not be retryable");
    }

    #[test]
    fn test_is_retryable_status_1xx_informational_not_retryable() {
        // 1xx informational codes should NOT be retryable
        assert!(!is_retryable_status(100), "100 should not be retryable");
        assert!(!is_retryable_status(101), "101 should not be retryable");
    }
}
