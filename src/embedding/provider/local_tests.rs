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

#[tokio::test]
async fn test_empty_model_rejected() {
	// Rejected before any network probe, so this is safe offline.
	assert!(LocalEmbeddingProvider::new("").await.is_err());
}

static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn test_api_url_default() {
	let _guard = ENV_LOCK.lock().await;
	std::env::remove_var(LOCAL_EMBED_API_URL_ENV);
	assert_eq!(
		LocalEmbeddingProvider::api_url(),
		"http://localhost:11434/v1/embeddings"
	);
}

#[tokio::test]
async fn retries_a_rate_limited_embedding_request_without_replaying_success() {
	use serde_json::Value;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};
	let _guard = ENV_LOCK.lock().await;
	let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
	let url = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
	let previous = std::env::var_os(LOCAL_EMBED_API_URL_ENV);
	std::env::set_var(LOCAL_EMBED_API_URL_ENV, &url);
	let server = tokio::spawn(async move {
		let mut inputs = Vec::new();
		for status in [200, 429, 200] {
			let (mut stream, _) = listener.accept().await.unwrap();
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
						inputs.push(
							serde_json::from_slice::<Value>(&request[offset + 4..offset + 4 + len]).unwrap()["input"]
								.clone(),
						);
						break;
					}
				}
			}
			let body = if status == 429 {
				r#"{"error":{"code":"rate_limit_exceeded","message":"Please try again in 1ms."}}"#
			} else {
				r#"{"data":[{"index":0,"embedding":[1.0,0.0]}],"usage":{"total_tokens":1}}"#
			};
			let response = format!("HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nRetry-After-Ms: 1\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",body.len());
			stream.write_all(response.as_bytes()).await.unwrap();
		}
		inputs
	});
	let provider = LocalEmbeddingProvider {
		model_name: "fixture".into(),
		dimension: 2,
	};
	let first = provider
		.generate_embeddings_batch(vec!["first".into()], InputType::None)
		.await;
	let second = provider
		.generate_embeddings_batch(vec!["second".into()], InputType::None)
		.await;
	match previous {
		Some(value) => std::env::set_var(LOCAL_EMBED_API_URL_ENV, value),
		None => std::env::remove_var(LOCAL_EMBED_API_URL_ENV),
	}
	assert!(first.is_ok(), "{first:?}");
	assert!(second.is_ok(), "{second:?}");
	assert_eq!(
		server.await.unwrap(),
		vec![json!(["first"]), json!(["second"]), json!(["second"])]
	);
}
