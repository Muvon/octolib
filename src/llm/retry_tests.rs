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

#[tokio::test]
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
		|| Box::pin(async move { Err(anyhow::anyhow!("upstream API error 503: {body}")) }),
		1,
		Duration::ZERO,
		None,
		params.api_key.as_ref(),
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
	assert_eq!(log.matches("[redacted]").count(), 4);
	assert!(!log.contains("private-request-key"));
	assert!(!shared::is_connection_error(&error));
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
