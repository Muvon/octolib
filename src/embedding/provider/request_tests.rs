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

use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[test]
fn reads_server_delays_and_rejects_invalid_values() {
	let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
	let mut headers = HeaderMap::new();
	headers.insert("retry-after", "2.5".parse().unwrap());
	assert_eq!(retry_after(&headers, "", now), Some(Duration::from_millis(2500)));
	headers.insert("retry-after-ms", "722".parse().unwrap());
	assert_eq!(retry_after(&headers, "", now), Some(Duration::from_millis(722)));
	headers.remove("retry-after-ms");
	headers.insert(
		"retry-after",
		httpdate::fmt_http_date(now + Duration::from_secs(7)).parse().unwrap(),
	);
	assert_eq!(retry_after(&headers, "", now), Some(Duration::from_secs(7)));
	headers.insert(
		"retry-after",
		httpdate::fmt_http_date(now - Duration::from_secs(7)).parse().unwrap(),
	);
	assert_eq!(retry_after(&headers, "", now), Some(Duration::ZERO));
	for invalid in ["-1", "NaN", "inf", "1e100", "invalid"] {
		headers.insert("retry-after", invalid.parse().unwrap());
		headers.insert("retry-after-ms", invalid.parse().unwrap());
		assert_eq!(retry_after(&headers, "{}", now), None, "{invalid}");
	}
}

#[test]
fn reads_openai_and_google_delay_bodies() {
	let headers = HeaderMap::new();
	let now = SystemTime::now();
	let body = json!({"error":{"code":"rate_limit_exceeded","message":
		"Rate limit reached on tokens per min (TPM). Please try again in 722ms. Visit https://platform.openai.com/account/rate-limits"}}).to_string();
	assert_eq!(retry_after(&headers, &body, now), Some(Duration::from_millis(722)));
	let body = json!({"error":{"details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"1.25s"}]}})
		.to_string();
	assert_eq!(retry_after(&headers, &body, now), Some(Duration::from_millis(1250)));
}

#[test]
fn distinguishes_temporary_errors_from_invalid_requests_and_exhausted_quota() {
	for status in [408, 409, 429, 500, 502, 503, 504] {
		assert!(retryable(status, "upstream unavailable"), "{status}");
	}
	for status in [400, 401, 403, 404, 422] {
		assert!(!retryable(status, "{}"), "{status}");
	}
	for field in ["code", "type"] {
		for code in ["insufficient_quota", "billing_hard_limit_reached", "billing_not_active"] {
			let mut error = json!({"error":{}});
			error["error"][field] = json!(code);
			assert!(!retryable(429, &error.to_string()));
		}
	}
}

// A real HTTP fixture exercises request cloning, response bodies and status propagation.
async fn fixture(responses: Vec<(u16, String, String)>) -> (String, tokio::task::JoinHandle<Vec<(String, Instant)>>) {
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}/embeddings", listener.local_addr().unwrap());
	let server = tokio::spawn(async move {
		let mut requests = Vec::new();
		for (status, headers, body) in responses {
			let (mut stream, _) = tokio::time::timeout(Duration::from_secs(10), listener.accept())
				.await
				.unwrap()
				.unwrap();
			let mut request = Vec::new();
			let mut buf = [0; 4096];
			loop {
				let n = stream.read(&mut buf).await.unwrap();
				assert!(n > 0);
				request.extend_from_slice(&buf[..n]);
				if let Some(offset) = request.windows(4).position(|s| s == b"\r\n\r\n") {
					let header = String::from_utf8_lossy(&request[..offset]).to_lowercase();
					let len: usize = header
						.lines()
						.find_map(|l| l.strip_prefix("content-length: "))
						.unwrap()
						.parse()
						.unwrap();
					if request.len() >= offset + 4 + len {
						break;
					}
				}
			}
			requests.push((String::from_utf8(request).unwrap(), Instant::now()));
			if status == 0 {
				continue;
			} // Drop the connection before any response.
			if status == 1 {
				stream
					.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{")
					.await
					.unwrap();
				continue; // A truncated response body, not invalid JSON from a complete response.
			}
			let response = format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nConnection: close\r\n{headers}Content-Length: {}\r\n\r\n{body}", body.len());
			stream.write_all(response.as_bytes()).await.unwrap();
		}
		requests
	});
	(url, server)
}

fn response(status: u16, body: &str) -> (u16, String, String) {
	(status, "Retry-After-Ms: 0\r\n".into(), body.into())
}

#[tokio::test]
async fn retries_identical_request_after_the_server_delay() {
	let (url, server) = fixture(vec![
		(
			429,
			"Retry-After-Ms: 722\r\n".into(),
			r#"{"error":{"code":"rate_limit_exceeded"}}"#.into(),
		),
		response(200, r#"{"data":[1,2],"usage":{"total_tokens":3}}"#),
	])
	.await;
	let result = send_json(
		"fixture",
		super::super::http_client()
			.post(url)
			.bearer_auth("fixture-key")
			.json(&json!({"input":["private text"]})),
	)
	.await
	.unwrap();
	assert_eq!(result["usage"]["total_tokens"], 3);
	let requests = server.await.unwrap();
	assert_eq!(requests.len(), 2);
	assert_eq!(requests[0].0, requests[1].0);
	assert!(requests[1].1.duration_since(requests[0].1) >= Duration::from_millis(722));
}

#[tokio::test]
async fn retries_server_failure_and_stops_after_five_requests() {
	let (url, server) = fixture(vec![response(503, "busy"); 5]).await;
	let error = send_json(
		"fixture",
		super::super::http_client().post(url).json(&json!({"input":["x"]})),
	)
	.await
	.unwrap_err();
	assert!(
		matches!(error.downcast_ref::<ProviderError>(), Some(ProviderError::ApiError {status:503, message, ..}) if message == "busy")
	);
	assert_eq!(server.await.unwrap().len(), 5);
}

#[tokio::test]
async fn returns_permanent_errors_without_retrying() {
	for (status, body) in [
		(400, "bad input"),
		(401, "bad key"),
		(403, "forbidden"),
		(429, r#"{"error":{"code":"insufficient_quota"}}"#),
	] {
		let (url, server) = fixture(vec![response(status, body)]).await;
		let error = send_json(
			"fixture",
			super::super::http_client().post(url).json(&json!({"input":["x"]})),
		)
		.await
		.unwrap_err();
		assert!(
			matches!(error.downcast_ref::<ProviderError>(), Some(ProviderError::ApiError {status:s, message, ..}) if *s == status && message == body)
		);
		assert_eq!(server.await.unwrap().len(), 1);
	}
}

#[tokio::test]
async fn does_not_retry_before_a_delay_exceeding_the_budget() {
	let (url, server) = fixture(vec![(429, "Retry-After: 301\r\n".into(), "busy".into())]).await;
	let result = tokio::time::timeout(
		Duration::from_secs(2),
		send_json(
			"fixture",
			super::super::http_client().post(url).json(&json!({"input":["x"]})),
		),
	)
	.await
	.unwrap();
	assert!(matches!(
		result.unwrap_err().downcast_ref::<ProviderError>(),
		Some(ProviderError::ApiError { status: 429, .. })
	));
	assert_eq!(server.await.unwrap().len(), 1);
}

#[tokio::test]
async fn malformed_success_is_a_parsing_error_without_retry() {
	let (url, server) = fixture(vec![response(200, "invalid JSON")]).await;
	let error = send_json(
		"fixture",
		super::super::http_client().post(url).json(&json!({"input":["x"]})),
	)
	.await
	.unwrap_err();
	assert!(matches!(
		error.downcast_ref::<ProviderError>(),
		Some(ProviderError::ResponseParsingError(_))
	));
	assert_eq!(server.await.unwrap().len(), 1);
}

#[tokio::test]
async fn retries_a_connection_closed_before_the_response() {
	let (url, server) = fixture(vec![response(0, ""), response(200, "{}")]).await;
	let result = send_json(
		"fixture",
		super::super::http_client().post(url).json(&json!({"input":["x"]})),
	)
	.await;
	assert!(result.is_ok(), "{result:?}");
	assert_eq!(server.await.unwrap().len(), 2);
}

#[tokio::test]
async fn retries_a_truncated_response_body() {
	let (url, server) = fixture(vec![response(1, ""), response(200, "{}")]).await;
	let result = send_json(
		"fixture",
		super::super::http_client().post(url).json(&json!({"input":["x"]})),
	)
	.await;
	assert!(result.is_ok(), "{result:?}");
	assert_eq!(server.await.unwrap().len(), 2);
}
