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

//! Internal SSE accumulation. Callers still receive one complete ProviderResponse.

use super::shared::{self, CapturedResponse};
use anyhow::{bail, ensure, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Duration;

// Bound untrusted framing independently of the final model output.
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
const MAX_STREAM_DATA_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(super) enum Format {
	Chat,
	Messages,
	Responses,
}

#[derive(Debug, thiserror::Error)]
#[error("{0:?} stream ended before its terminal event")]
pub(crate) struct IncompleteStream(Format);

#[derive(Debug, thiserror::Error)]
#[error("{0:#}")]
pub(crate) struct TerminalStreamError(anyhow::Error);

#[derive(Debug, thiserror::Error)]
#[error("Transient provider stream error: {0}")]
struct TransientStreamError(Value);

fn protocol_error(error: anyhow::Error) -> anyhow::Error {
	if error.is::<IncompleteStream>() || error.is::<TransientStreamError>() {
		error
	} else {
		TerminalStreamError(error).into()
	}
}

fn upstream_error(error: &Value) -> anyhow::Error {
	let code = error.get("code").or_else(|| error.get("type"));
	let transient = code.is_some_and(|code| {
		code.as_u64()
			.is_some_and(|status| status == 429 || (500..600).contains(&status))
			|| matches!(
				code.as_str(),
				Some(
					"overloaded_error"
						| "api_error" | "server_error"
						| "internal_error" | "rate_limit_error"
						| "rate_limit_exceeded"
				)
			)
	});
	if transient {
		TransientStreamError(error.clone()).into()
	} else {
		anyhow::anyhow!("Provider stream error: {error}")
	}
}

/// Consume SSE within the caller's retry/cancellation boundary. Some compatible
/// endpoints return a complete JSON response even when streaming was requested;
/// select by Content-Type, never by catching a stream parser failure.
pub(super) async fn send_and_read(
	request: reqwest::RequestBuilder,
	timeout: Option<Duration>,
	extra_headers: Option<&HashMap<String, String>>,
	format: Format,
) -> Result<CapturedResponse> {
	let request = shared::apply_extra_headers(
		request.header(reqwest::header::ACCEPT, "text/event-stream"),
		extra_headers,
	);
	let mut response = shared::apply_request_timeout(request, timeout).send().await?;
	let status = response.status();
	let headers = response.headers().clone();
	let is_sse = headers
		.get(reqwest::header::CONTENT_TYPE)
		.and_then(|value| value.to_str().ok())
		.is_some_and(|value| {
			value
				.split(';')
				.next()
				.is_some_and(|mime| mime.trim().eq_ignore_ascii_case("text/event-stream"))
		});
	let body = if status.is_success() && is_sse {
		let mut decoder = Decoder::default();
		let mut accumulator = Accumulator::new(format);
		while let Some(chunk) = response.chunk().await? {
			decoder
				.push(&chunk, |data| accumulator.push(data))
				.map_err(protocol_error)?;
		}
		if decoder.finish().is_err() {
			return Err(IncompleteStream(format).into());
		}
		serde_json::to_string(&accumulator.finish().map_err(protocol_error)?)?
	} else {
		response.text().await?
	};
	Ok(CapturedResponse { status, headers, body })
}

/// SSE line endings may be LF, CRLF or CR, split at any byte (including UTF-8).
#[derive(Default)]
struct Decoder {
	line: Vec<u8>,
	data: String,
	after_cr: bool,
	lines_seen: bool,
}

impl Decoder {
	fn push(&mut self, bytes: &[u8], mut emit: impl FnMut(&str) -> Result<()>) -> Result<()> {
		for &byte in bytes {
			if self.after_cr {
				self.after_cr = false;
				if byte == b'\n' {
					continue;
				}
			}
			if byte == b'\r' || byte == b'\n' {
				self.after_cr = byte == b'\r';
				let first_line = !self.lines_seen;
				self.lines_seen = true;
				let line = std::str::from_utf8(&self.line).context("Invalid UTF-8 in SSE line")?;
				let line = if first_line {
					line.trim_start_matches('\u{feff}')
				} else {
					line
				};
				if line.is_empty() {
					if !self.data.is_empty() {
						self.data.pop(); // SSE joins data lines with LF, excluding the last LF.
						emit(&self.data)?;
						self.data.clear();
					}
				} else if let Some(value) = line.strip_prefix("data:") {
					self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
					self.data.push('\n');
				} else if line == "data" {
					self.data.push('\n');
				}
				self.line.clear();
			} else {
				self.line.push(byte);
			}
			ensure!(
				self.line.len() + self.data.len() <= MAX_EVENT_BYTES,
				"SSE event exceeds {MAX_EVENT_BYTES} bytes"
			);
		}
		Ok(())
	}

	fn finish(&self) -> Result<()> {
		ensure!(
			self.line.is_empty() && self.data.is_empty(),
			"Truncated SSE event at end of response"
		);
		Ok(())
	}
}

struct Accumulator {
	format: Format,
	response: Value,
	choices: BTreeMap<u64, Value>,
	blocks: BTreeMap<u64, Value>,
	inputs: BTreeMap<u64, String>,
	open_blocks: BTreeSet<u64>,
	finished: bool,
	data_bytes: usize,
}

impl Accumulator {
	fn new(format: Format) -> Self {
		Self {
			format,
			response: json!({}),
			choices: BTreeMap::new(),
			blocks: BTreeMap::new(),
			inputs: BTreeMap::new(),
			open_blocks: BTreeSet::new(),
			finished: false,
			data_bytes: 0,
		}
	}

	fn push(&mut self, data: &str) -> Result<()> {
		if data.is_empty() {
			return Ok(());
		}
		ensure!(!self.finished, "SSE data after terminal event");
		self.data_bytes += data.len();
		ensure!(
			self.data_bytes <= MAX_STREAM_DATA_BYTES,
			"SSE data exceeds {MAX_STREAM_DATA_BYTES} bytes"
		);
		if matches!(self.format, Format::Chat) && data == "[DONE]" {
			self.finished = true;
			return Ok(());
		}
		let event: Value = serde_json::from_str(data).context("Malformed SSE JSON event")?;
		ensure!(event.is_object(), "SSE event must be an object");
		if let Some(error) = event.get("error").filter(|error| !error.is_null()) {
			return Err(upstream_error(error));
		}
		match self.format {
			Format::Chat => self.chat(event),
			Format::Messages => self.messages(event),
			Format::Responses => self.responses(event),
		}
	}

	fn chat(&mut self, event: Value) -> Result<()> {
		for (key, value) in object(&event, "chat chunk")? {
			if key != "choices" && key != "object" && !value.is_null() {
				merge_snapshot(&mut self.response[key], value);
			}
		}
		let choices = event["choices"].as_array().context("Chat chunk is missing choices")?;
		for choice in choices {
			let index = index(choice)?;
			let target = self
				.choices
				.entry(index)
				.or_insert_with(|| json!({"index": index, "message": {}}));
			if let Some(delta) = choice.get("delta").filter(|delta| !delta.is_null()) {
				merge_delta(&mut target["message"], delta)?;
			}
			for (key, value) in object(choice, "chat choice")? {
				if key != "delta" && !value.is_null() {
					merge_snapshot(&mut target[key], value);
				}
			}
		}
		Ok(())
	}

	fn messages(&mut self, event: Value) -> Result<()> {
		match event["type"].as_str().context("Messages event is missing type")? {
			"message_start" => {
				ensure!(self.response.get("type").is_none(), "Duplicate message_start");
				object(&event["message"], "message_start.message")?;
				self.response = event["message"].clone();
			}
			"content_block_start" => {
				let index = index(&event)?;
				object(&event["content_block"], "content_block_start.content_block")?;
				ensure!(
					self.blocks.insert(index, event["content_block"].clone()).is_none(),
					"Duplicate content block {index}"
				);
				self.open_blocks.insert(index);
			}
			"content_block_delta" => {
				let index = index(&event)?;
				ensure!(
					self.open_blocks.contains(&index),
					"Delta for closed or unopened content block {index}"
				);
				let block = self
					.blocks
					.get_mut(&index)
					.context("Delta for unopened content block")?;
				let delta = &event["delta"];
				match delta["type"].as_str().context("Content block delta is missing type")? {
					"text_delta" => append(&mut block["text"], &delta["text"], "text")?,
					"thinking_delta" => append(&mut block["thinking"], &delta["thinking"], "thinking")?,
					"signature_delta" => append(&mut block["signature"], &delta["signature"], "signature")?,
					"input_json_delta" => self
						.inputs
						.entry(index)
						.or_default()
						.push_str(delta["partial_json"].as_str().context("Missing partial_json")?),
					"citations_delta" => {
						if block.get("citations").is_none() {
							block["citations"] = json!([]);
						}
						block["citations"]
							.as_array_mut()
							.context("Invalid citations")?
							.push(delta.get("citation").context("Missing citation")?.clone());
					}
					other => bail!("Unsupported content block delta: {other}"),
				}
			}
			"content_block_stop" => {
				let index = index(&event)?;
				ensure!(
					self.open_blocks.remove(&index),
					"Stop for closed or unopened content block {index}"
				);
				let block = self.blocks.get_mut(&index).context("Stop for unopened content block")?;
				if let Some(input) = self.inputs.remove(&index) {
					block["input"] = serde_json::from_str(&input).context("Invalid streamed tool input JSON")?;
				}
			}
			"message_delta" => {
				for (key, value) in object(&event["delta"], "message_delta.delta")? {
					self.response[key] = value.clone();
				}
				for (key, value) in object(&event, "message_delta")? {
					if key != "type" && key != "delta" {
						merge_snapshot(&mut self.response[key], value);
					}
				}
			}
			"message_stop" => {
				ensure!(self.response["type"] == "message", "message_stop without message_start");
				ensure!(
					self.open_blocks.is_empty(),
					"message_stop before content blocks finished"
				);
				ensure!(
					self.response["stop_reason"].is_string(),
					"message_stop without stop_reason"
				);
				self.response["content"] = json!(ordered(std::mem::take(&mut self.blocks), "content block")?);
				self.finished = true;
			}
			// Anthropic permits new event types; only known delta types mutate content.
			_ => {}
		}
		Ok(())
	}

	fn responses(&mut self, event: Value) -> Result<()> {
		match event["type"].as_str().context("Responses event is missing type")? {
			"response.completed" | "response.incomplete" => {
				object(&event["response"], "terminal response")?;
				self.response = event["response"].clone();
				self.finished = true;
			}
			"response.failed" => return Err(upstream_error(&event["response"]["error"])),
			"error" => return Err(upstream_error(&event)),
			_ => {}
		}
		Ok(())
	}

	fn finish(mut self) -> Result<Value> {
		if !self.finished {
			return Err(IncompleteStream(self.format).into());
		}
		if matches!(self.format, Format::Chat) {
			ensure!(!self.choices.is_empty(), "Chat stream has no choices");
			let mut choices = ordered(self.choices, "choice")?;
			for choice in &mut choices {
				ensure!(
					choice["finish_reason"].is_string(),
					"Chat stream is missing finish_reason"
				);
				ensure!(choice["finish_reason"] != "error", "Chat stream finished with an error");
				validate_tools(&choice["message"])?;
			}
			self.response["choices"] = json!(choices);
			self.response["object"] = json!("chat.completion");
			// Groq's legacy stream usage envelope.
			if self.response.get("usage").is_none() {
				if let Some(usage) = self.response["x_groq"].get("usage").cloned() {
					self.response["usage"] = usage;
				}
			}
		}
		Ok(self.response)
	}
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
	value
		.as_object()
		.with_context(|| format!("Missing or invalid {name} object"))
}

fn index(value: &Value) -> Result<u64> {
	value["index"].as_u64().context("Missing or invalid stream index")
}

fn ordered(values: BTreeMap<u64, Value>, name: &str) -> Result<Vec<Value>> {
	values
		.into_iter()
		.enumerate()
		.map(|(expected, (index, value))| {
			ensure!(index == expected as u64, "Missing {name} index {expected}");
			Ok(value)
		})
		.collect()
}

fn append(target: &mut Value, delta: &Value, name: &str) -> Result<()> {
	let text = delta.as_str().with_context(|| format!("Invalid {name} delta"))?;
	if target.is_null() {
		*target = json!("");
	}
	target.as_str().with_context(|| format!("Invalid accumulated {name}"))?;
	if let Value::String(target) = target {
		target.push_str(text);
	}
	Ok(())
}

/// Usage counters and metadata are snapshots, never additive deltas.
fn merge_snapshot(target: &mut Value, value: &Value) {
	if let (Value::Object(target), Value::Object(value)) = (&mut *target, value) {
		for (key, value) in value {
			if !value.is_null() {
				merge_snapshot(target.entry(key).or_insert(Value::Null), value);
			}
		}
	} else {
		*target = value.clone();
	}
}

fn merge_delta(target: &mut Value, delta: &Value) -> Result<()> {
	ensure!(target.is_object(), "Invalid accumulated delta object");
	for (key, value) in object(delta, "delta")? {
		if value.is_null() {
			continue;
		}
		match key.as_str() {
			"content" | "reasoning" | "reasoning_content" | "thinking" | "refusal" | "arguments" | "name" | "text"
			| "summary" | "data" | "signature" => append(&mut target[key], value, key)?,
			"function" => {
				if target[key].is_null() {
					target[key] = json!({});
				}
				merge_delta(&mut target[key], value)?;
			}
			"tool_calls" | "reasoning_details" => {
				if target[key].is_null() {
					target[key] = json!([]);
				}
				let target = target[key]
					.as_array_mut()
					.context("Invalid accumulated indexed deltas")?;
				for fragment in value.as_array().context("Invalid indexed deltas")? {
					let index = index(fragment)?;
					let position = target.iter().position(|item| item["index"].as_u64() == Some(index));
					let position = match position {
						Some(position) => position,
						None => {
							target.push(json!({"index": index}));
							target.len() - 1
						}
					};
					merge_delta(&mut target[position], fragment)?;
				}
				target.sort_by_key(|item| item["index"].as_u64());
			}
			_ => merge_snapshot(&mut target[key], value),
		}
	}
	Ok(())
}

fn validate_tools(message: &Value) -> Result<()> {
	if let Some(calls) = message.get("tool_calls") {
		for (expected, call) in calls.as_array().context("Invalid tool_calls")?.iter().enumerate() {
			ensure!(index(call)? == expected as u64, "Missing tool call index {expected}");
			ensure!(
				call["id"].as_str().is_some_and(|id| !id.is_empty()),
				"Streamed tool call is missing id"
			);
			ensure!(
				call["function"]["name"].as_str().is_some_and(|name| !name.is_empty()),
				"Streamed tool call is missing name"
			);
			let arguments = call["function"]["arguments"]
				.as_str()
				.context("Streamed tool call is missing arguments")?;
			serde_json::from_str::<Value>(arguments).context("Invalid streamed tool arguments JSON")?;
		}
	}
	Ok(())
}

#[cfg(test)]
#[path = "streaming_tests.rs"]
mod tests;
