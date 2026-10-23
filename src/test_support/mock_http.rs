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

//! Minimal local HTTP mock used by provider unit tests.
//!
//! Serves canned responses on a loopback listener so provider HTTP paths can
//! be exercised without touching the network. Providers point at the mock
//! through their `<PROVIDER>_API_URL` (or base-URL) environment overrides;
//! set those with [`EnvGuard`](super::env::EnvGuard) so they never leak.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// A single-request-scoped loopback HTTP server.
pub struct MockHttp {
	/// Full URL (including path) the server responds on.
	pub url: String,
	requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

/// What the mock saw: request line plus the decoded JSON body (when parseable).
#[derive(Clone, Debug)]
pub struct RecordedRequest {
	pub method: String,
	pub path: String,
	pub body: String,
}

impl MockHttp {
	/// Start a server replying to every request with `status` and `body`.
	///
	/// Tests that need different replies per call pass a Vec<(status, body)> —
	/// replies are consumed in order and the last one repeats.
	pub fn start(replies: Vec<(u16, String)>) -> Self {
		let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
		let port = listener.local_addr().expect("local addr").port();
		let requests: Arc<Mutex<Vec<RecordedRequest>>> = Arc::new(Mutex::new(Vec::new()));
		let req_slot = Arc::clone(&requests);

		std::thread::spawn(move || {
			// Hard cap so a runaway client cannot spin the thread forever.
			for (idx, stream) in listener.incoming().flatten().enumerate() {
				if idx >= 64 {
					break;
				}
				let reply = replies.get(idx.min(replies.len().saturating_sub(1))).cloned();
				if handle_connection(stream, &req_slot, reply).is_err() {
					break;
				}
			}
		});

		Self {
			url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
			requests,
		}
	}

	/// Origin (scheme + host + port) without the canned chat-completions path,
	/// for providers that build their own paths from a base URL.
	pub fn base_url(&self) -> String {
		self.url
			.split_once("/v1/chat/completions")
			.map_or_else(|| self.url.clone(), |(base, _)| base.to_string())
	}

	/// Requests received so far.
	pub fn requests(&self) -> Vec<RecordedRequest> {
		self.requests.lock().map(|r| r.clone()).unwrap_or_default()
	}
}

fn handle_connection(
	stream: TcpStream,
	requests: &Arc<Mutex<Vec<RecordedRequest>>>,
	reply: Option<(u16, String)>,
) -> std::io::Result<()> {
	let mut reader = BufReader::new(stream.try_clone()?);
	let mut request_line = String::new();
	reader.read_line(&mut request_line)?;

	let mut content_length = 0usize;
	loop {
		let mut line = String::new();
		if reader.read_line(&mut line)? == 0 {
			return Ok(());
		}
		let trimmed = line.trim_end();
		if trimmed.is_empty() {
			break;
		}
		if let Some(value) = trimmed.to_ascii_lowercase().strip_prefix("content-length:") {
			content_length = value.trim().parse().unwrap_or(0);
		}
	}

	let mut body = vec![0u8; content_length];
	if content_length > 0 {
		reader.read_exact(&mut body)?;
	}

	let mut parts = request_line.split_whitespace();
	let method = parts.next().unwrap_or_default().to_string();
	let path = parts.next().unwrap_or_default().to_string();
	if let Ok(mut guard) = requests.lock() {
		guard.push(RecordedRequest {
			method,
			path,
			body: String::from_utf8_lossy(&body).to_string(),
		});
	}

	let (status, response_body) = reply.unwrap_or((200, String::from("{}")));
	let mut stream = stream;
	write!(
		stream,
		"HTTP/1.1 {status} MOCK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
		response_body.len(),
		response_body
	)?;
	stream.flush()?;
	Ok(())
}

/// Standard OpenAI-compatible chat completion body for a plain text answer.
pub fn chat_completion_body(content: &str) -> String {
	format!(
		r#"{{"id":"mock-1","object":"chat.completion","choices":[{{"index":0,"message":{{"role":"assistant","content":{content}}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}}}"#,
		content = serde_json::json!(content)
	)
}

/// Chat completion body with one tool call and reasoning: 5 prompt tokens and
/// 20 completion tokens, of which `reasoning` are thinking.
#[cfg(feature = "llm")]
pub fn tool_call_body(call_id: &str, name: &str, arguments: &serde_json::Value, reasoning: u64) -> String {
	serde_json::json!({
		"id": "mock-2",
		"object": "chat.completion",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"reasoning_content": "pondering",
				"tool_calls": [{
					"id": call_id,
					"type": "function",
					"function": {"name": name, "arguments": arguments.to_string()}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {
			"prompt_tokens": 5,
			"completion_tokens": 20,
			"total_tokens": 25,
			"completion_tokens_details": {"reasoning_tokens": reasoning}
		}
	})
	.to_string()
}

#[cfg(test)]
#[path = "mock_http_tests.rs"]
mod tests;
