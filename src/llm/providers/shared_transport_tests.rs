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

use super as shared;
use crate::llm::retry::{retry_provider_request, RetryObserver};
use std::{sync::Arc, time::Duration};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn read_request(stream: &mut TcpStream) {
	let mut headers = Vec::new();
	while !headers.ends_with(b"\r\n\r\n") {
		assert!(headers.len() < 8192);
		headers.push(stream.read_u8().await.unwrap());
	}
}

async fn reply(stream: &mut TcpStream) {
	stream
		.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
		.await
		.unwrap();
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn transport_retry_preserves_other_hosts_pooled_connections() {
	tokio::time::timeout(Duration::from_secs(5), async {
		let healthy = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let healthy_url = format!("http://{}", healthy.local_addr().unwrap());
		let healthy_server = tokio::spawn(async move {
			let (mut stream, _) = healthy.accept().await.unwrap();
			for _ in 0..2 {
				read_request(&mut stream).await;
				reply(&mut stream).await;
			}
		});
		let response = shared::send_and_read(shared::http_client().get(&healthy_url), None, None)
			.await
			.unwrap();
		assert_eq!(response.body, "OK");

		let failing = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let failing_url = format!("http://{}", failing.local_addr().unwrap());
		let failing_server = tokio::spawn(async move {
			let (mut stream, _) = failing.accept().await.unwrap();
			read_request(&mut stream).await;
			drop(stream);
			let (mut stream, _) = failing.accept().await.unwrap();
			read_request(&mut stream).await;
			reply(&mut stream).await;
		});
		let response = retry_provider_request(
			|| {
				Box::pin(shared::send_and_read(
					shared::http_client().get(&failing_url),
					None,
					None,
				))
			},
			1,
			Duration::ZERO,
			None,
			None,
			None,
		)
		.await
		.unwrap();
		assert_eq!(response.body, "OK");
		let response = shared::send_and_read(shared::http_client().get(&healthy_url), None, None)
			.await
			.unwrap();
		assert_eq!(response.body, "OK");
		failing_server.await.unwrap();
		healthy_server.await.unwrap();
	})
	.await
	.unwrap();
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn request_timeout_keeps_its_category_after_credential_redaction() {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}/private-request-key", listener.local_addr().unwrap());
	let params = crate::llm::types::ChatCompletionParams::new(&[], "model", 1.0, 1.0, 0, 10)
		.with_api_key("private-request-key".to_owned());
	let events = Arc::new(std::sync::Mutex::new(Vec::new()));
	let sink = events.clone();
	let observer = RetryObserver::new(move |event| sink.lock().unwrap().push(event.clone()));
	let server = tokio::spawn(async move {
		let mut streams = Vec::new();
		for _ in 0..2 {
			let (mut stream, _) = listener.accept().await.unwrap();
			read_request(&mut stream).await;
			streams.push(stream);
		}
		std::future::pending::<()>().await;
	});
	let result = retry_provider_request(
		|| {
			Box::pin(shared::send_and_read(
				shared::http_client().get(&url),
				Some(Duration::from_millis(100)),
				None,
			))
		},
		1,
		Duration::ZERO,
		None,
		params.api_key.as_ref(),
		Some(&observer),
	)
	.await;
	server.abort();
	assert!(server.await.unwrap_err().is_cancelled());
	let error = match result {
		Ok(_) => panic!("the peer never responds"),
		Err(error) => error,
	};
	assert_eq!(
		shared::transport_error_kind(&error),
		Some(shared::TransportErrorKind::Timeout)
	);
	assert!(!format!("{error:#?}").contains("private-request-key"));
	assert!(error.to_string().contains("[redacted]"));
	let events = events.lock().unwrap();
	assert_eq!(events.len(), 1);
	assert_eq!(events[0].reason, "request timed out");
}

#[tokio::test]
#[serial_test::serial(retry_log)]
async fn truncated_response_body_is_retried() {
	tokio::time::timeout(Duration::from_secs(5), async {
		let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
		let url = format!("http://{}", listener.local_addr().unwrap());
		let server = tokio::spawn(async move {
			let (mut stream, _) = listener.accept().await.unwrap();
			read_request(&mut stream).await;
			stream
				.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial")
				.await
				.unwrap();
			drop(stream);
			let (mut stream, _) = listener.accept().await.unwrap();
			read_request(&mut stream).await;
			reply(&mut stream).await;
		});
		let response = retry_provider_request(
			|| Box::pin(shared::send_and_read(shared::http_client().get(&url), None, None)),
			1,
			Duration::ZERO,
			None,
			None,
			None,
		)
		.await
		.unwrap();
		assert_eq!(response.body, "OK");
		server.await.unwrap();
	})
	.await
	.unwrap();
}
