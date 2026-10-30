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

#[derive(Clone)]
struct LogBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl std::io::Write for LogBuffer {
	fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
		self.0.lock().unwrap().extend_from_slice(bytes);
		Ok(bytes.len())
	}

	fn flush(&mut self) -> std::io::Result<()> {
		Ok(())
	}
}

// Every test that drives a failing attempt through the retry loop shares this
// key: tracing caches a callsite's interest process-wide, so one of them running
// beside this test without a subscriber can blank the debug lines it asserts on.
#[tokio::test]
#[serial_test::serial(retry_log)]
async fn test_upstream_key_is_redacted_in_returned_error_and_retry_logs() {
	use crate::llm::providers::shared;
	use crate::llm::types::ChatCompletionParams;
	use tracing::instrument::WithSubscriber;

	let params =
		ChatCompletionParams::new(&[], "model", 1.0, 1.0, 0, 10).with_api_key("private-request-key".to_owned());
	let buffer = LogBuffer(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
	let writer = buffer.clone();
	let subscriber = tracing_subscriber::fmt()
		.with_max_level(tracing::Level::DEBUG)
		.with_ansi(false)
		.without_time()
		.with_writer(move || writer.clone())
		.finish();
	let body = r#"{"error":"rejected private-request-key; private-request-key is invalid"}"#;
	let result = retry_provider_request(
		|| {
			Box::pin(async move {
				Err(anyhow::anyhow!("upstream API error 503: {body}").context("upstream request failed"))
			})
		},
		1,
		Duration::ZERO,
		None,
		params.api_key.as_ref(),
		None,
	)
	.with_subscriber(subscriber)
	.await;
	let error = match result {
		Ok(_) => panic!("upstream failure must be returned"),
		Err(error) => error,
	};
	for text in [error.to_string(), format!("{error:#}"), format!("{error:#?}")] {
		assert_eq!(text.matches("[redacted]").count(), 2);
		assert!(!text.contains("private-request-key"));
	}
	let log = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
	assert!(log.contains("API request attempt 1 failed"));
	assert!(log.contains("API request attempt 2 failed"));
	assert!(log.contains("upstream request failed: upstream API error 503:"));
	assert_eq!(log.matches("[redacted]").count(), 4);
	assert!(!log.contains("private-request-key"));
	assert_eq!(shared::transport_error_kind(&error), None);
}

#[tokio::test]
async fn test_non_retryable_error_body_is_redacted_and_success_body_is_preserved() {
	use crate::llm::providers::shared::CapturedResponse;
	use crate::llm::types::ChatCompletionParams;
	let params =
		ChatCompletionParams::new(&[], "model", 1.0, 1.0, 0, 10).with_api_key("private-request-key".to_owned());
	for status in [reqwest::StatusCode::UNAUTHORIZED, reqwest::StatusCode::OK] {
		let result = retry_provider_request(
			|| {
				Box::pin(async move {
					Ok(CapturedResponse {
						status,
						headers: reqwest::header::HeaderMap::new(),
						body: "private-request-key / private-request-key".to_owned(),
					})
				})
			},
			0,
			Duration::ZERO,
			None,
			params.api_key.as_ref(),
			None,
		)
		.await
		.unwrap();
		if status.is_success() {
			assert_eq!(result.body, "private-request-key / private-request-key");
		} else {
			let error = anyhow::anyhow!("upstream API error {}: {}", result.status, result.body);
			assert_eq!(
				error.to_string(),
				"upstream API error 401 Unauthorized: [redacted] / [redacted]"
			);
		}
	}
}

fn status_response(status: u16, retry_after: Option<&str>) -> crate::llm::providers::shared::CapturedResponse {
	let mut headers = reqwest::header::HeaderMap::new();
	if let Some(value) = retry_after {
		headers.insert(reqwest::header::RETRY_AFTER, value.parse().unwrap());
	}
	crate::llm::providers::shared::CapturedResponse {
		status: reqwest::StatusCode::from_u16(status).unwrap(),
		headers,
		body: "upstream down".to_owned(),
	}
}

fn recording_observer() -> (RetryObserver, std::sync::Arc<std::sync::Mutex<Vec<RetryEvent>>>) {
	let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
	let sink = events.clone();
	(
		RetryObserver::new(move |event| sink.lock().unwrap().push(event.clone())),
		events,
	)
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn test_observer_is_told_before_every_retry_but_not_after_the_last_attempt() {
	let (observer, events) = recording_observer();
	let result = retry_provider_request(
		|| {
			Box::pin(async {
				let response = status_response(503, None);
				Err(retryable_status_error(
					&response,
					format!("OctoHub API error {}: {}", response.status, response.body),
				))
			})
		},
		2,
		Duration::ZERO,
		None,
		None,
		Some(&observer),
	)
	.await;

	let error = match result {
		Ok(_) => panic!("a provider that keeps failing must return its error"),
		Err(error) => error,
	};
	// The typed error displays exactly as the provider's message always did.
	assert_eq!(
		error.to_string(),
		"OctoHub API error 503 Service Unavailable: upstream down"
	);
	assert_eq!(
		*events.lock().unwrap(),
		vec![
			RetryEvent {
				attempt: 1,
				max_retries: 2,
				delay: Duration::ZERO,
				reason: "HTTP 503".to_owned(),
			},
			RetryEvent {
				attempt: 2,
				max_retries: 2,
				delay: Duration::ZERO,
				reason: "HTTP 503".to_owned(),
			},
		]
	);
}

#[tokio::test]
async fn test_observer_is_silent_when_the_first_attempt_succeeds() {
	let (observer, events) = recording_observer();
	let result = retry_provider_request(
		|| Box::pin(async { Ok(status_response(200, None)) }),
		3,
		Duration::ZERO,
		None,
		None,
		Some(&observer),
	)
	.await;

	assert!(result.is_ok());
	assert!(events.lock().unwrap().is_empty());
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn test_retry_event_reason_never_carries_the_body_or_the_key() {
	let params = crate::llm::types::ChatCompletionParams::new(&[], "model", 1.0, 1.0, 0, 10)
		.with_api_key("private-request-key".to_owned());
	let (observer, events) = recording_observer();
	let result = retry_provider_request(
		|| {
			Box::pin(async {
				let mut response = status_response(429, Some("0"));
				response.body = "key private-request-key is rate limited".to_owned();
				Err(retryable_status_error(
					&response,
					format!("upstream API error {}: {}", response.status, response.body),
				))
			})
		},
		1,
		Duration::ZERO,
		None,
		params.api_key.as_ref(),
		Some(&observer),
	)
	.await;

	let error = match result {
		Ok(_) => panic!("upstream failure must be returned"),
		Err(error) => error,
	};
	assert!(!error.to_string().contains("private-request-key"));
	// Redaction keeps the typed status, so the loop can still name it.
	assert_eq!(
		error.downcast_ref::<RetryableStatusError>().map(|e| e.status),
		Some(429)
	);
	let events = events.lock().unwrap();
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].reason, "HTTP 429");
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn test_retry_chat_operation_uses_the_observer_on_params() {
	let (observer, events) = recording_observer();
	let params = crate::llm::types::ChatCompletionParams::new(&[], "model", 1.0, 1.0, 0, 10)
		.with_max_retries(1)
		.with_retry_timeout(Duration::ZERO)
		.with_retry_observer(observer);
	let result: Result<()> =
		retry_chat_operation(|| Box::pin(async { Err(anyhow::anyhow!("stream broke")) }), &params).await;

	assert!(result.is_err());
	assert_eq!(
		*events.lock().unwrap(),
		vec![RetryEvent {
			attempt: 1,
			max_retries: 1,
			delay: Duration::ZERO,
			reason: "request failed".to_owned(),
		}]
	);
}

#[test]
fn test_retry_after_seconds_are_read_on_429_and_503_only() {
	let with = |status: u16, value: &str| {
		let response = status_response(status, Some(value));
		parse_retry_after(status, &response.headers)
	};
	assert_eq!(with(429, "12"), Some(Duration::from_secs(12)));
	assert_eq!(with(503, " 3 "), Some(Duration::from_secs(3)));
	assert_eq!(with(500, "12"), None, "only 429/503 carry a meaningful Retry-After");
	assert_eq!(
		with(503, "Wed, 21 Oct 2026 07:28:00 GMT"),
		None,
		"HTTP-date form is ignored"
	);
	assert_eq!(with(429, "-1"), None);
	assert_eq!(parse_retry_after(503, &reqwest::header::HeaderMap::new()), None);
}

#[test]
fn test_backoff_delay_doubles_honors_longer_retry_after_and_caps() {
	let base = Duration::from_secs(30);
	assert_eq!(backoff_delay(base, 0, None), Duration::from_secs(30));
	assert_eq!(backoff_delay(base, 1, None), Duration::from_secs(60));
	assert_eq!(backoff_delay(base, 2, None), Duration::from_secs(120));
	// A server asking for longer than our backoff is waited for...
	assert_eq!(
		backoff_delay(base, 0, Some(Duration::from_secs(45))),
		Duration::from_secs(45)
	);
	// ...a shorter ask never makes us retry sooner than the configured backoff...
	assert_eq!(
		backoff_delay(base, 1, Some(Duration::from_secs(5))),
		Duration::from_secs(60)
	);
	// ...and neither can push a single sleep past the 5-minute cap.
	assert_eq!(backoff_delay(base, 0, Some(Duration::from_secs(3600))), MAX_RETRY_DELAY);
	assert_eq!(backoff_delay(base, 40, None), MAX_RETRY_DELAY);
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn test_retry_after_from_a_503_stretches_the_observed_delay() {
	let (observer, events) = recording_observer();
	let result = retry_provider_request(
		|| {
			Box::pin(async {
				let response = status_response(503, Some("1"));
				Err(retryable_status_error(&response, "down".to_owned()))
			})
		},
		1,
		Duration::ZERO,
		None,
		None,
		Some(&observer),
	)
	.await;

	assert!(result.is_err());
	assert_eq!(events.lock().unwrap()[0].delay, Duration::from_secs(1));
}

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

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn retry_logs_include_the_underlying_error_without_flattening_the_returned_error() {
	use tracing::instrument::WithSubscriber;
	let buffer = LogBuffer(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
	let writer = buffer.clone();
	let subscriber = tracing_subscriber::fmt()
		.with_max_level(tracing::Level::DEBUG)
		.with_ansi(false)
		.without_time()
		.with_writer(move || writer.clone())
		.finish();
	let result = retry_provider_request(
		|| {
			Box::pin(async {
				Err(
					anyhow::Error::new(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "peer closed"))
						.context("sending request"),
				)
			})
		},
		0,
		Duration::ZERO,
		None,
		None,
		None,
	)
	.with_subscriber(subscriber)
	.await;
	let error = match result {
		Ok(_) => panic!("the operation always fails"),
		Err(error) => error,
	};
	assert_eq!(
		error.downcast_ref::<std::io::Error>().unwrap().kind(),
		std::io::ErrorKind::ConnectionReset
	);
	let log = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
	assert!(log.contains("sending request: peer closed"));
}

#[tokio::test]
async fn typed_provider_timeout_is_reported_as_a_timeout_to_the_observer() {
	let (observer, events) = recording_observer();
	let params = crate::llm::types::ChatCompletionParams::new(&[], "model", 0.0, 1.0, 0, 0)
		.with_max_retries(1)
		.with_retry_timeout(Duration::ZERO)
		.with_retry_observer(observer);
	let result: anyhow::Result<()> = retry_chat_operation(
		|| {
			Box::pin(async {
				Err(anyhow::Error::new(crate::errors::ProviderError::TimeoutError {
					provider: "chatgpt".into(),
				})
				.context("ChatGPT WebSocket timed out after 100s without receiving a frame"))
			})
		},
		&params,
	)
	.await;
	assert!(result.is_err());
	let events = events.lock().unwrap();
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].reason, "request timed out");
}
