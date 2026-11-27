// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Shared request and response layer for endpoints that speak the Anthropic
//! Messages protocol without being Anthropic: MiniMax, Z.ai, DeepSeek, Moonshot,
//! Alibaba, Isoquant and local servers.
//!
//! The split mirrors `openai_compat`: [`chat_completion`] wraps one raw turn
//! in the schema enforcement, and a provider states what differs on its
//! endpoint in [`AnthropicCompatConfig`] (thinking knobs, media, cost). Streams
//! are assembled by the existing Messages accumulator in `streaming`. The
//! native Claude provider keeps its own implementation of threads, betas and
//! thinking binding.

use super::shared;
use crate::errors::config_error;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
	ChatCompletionParams, ImageAttachment, ImageData, Message, OutputFormat, ProviderExchange, ProviderResponse,
	ReasoningEffort, ThinkingBlock, TokenUsage, ToolCall, VideoAttachment, VideoData,
};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

/// The version every endpoint here documents for its Messages API. The betas
/// the native provider sends are Anthropic's own and are left out.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Image input an endpoint takes for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Images {
	/// The model has no image input.
	None,
	/// Inline data only.
	Base64,
	/// Inline data, or a URL the endpoint downloads.
	Base64AndUrl,
}

/// Usage counters as the Messages API reports them: `input_tokens` leaves out
/// cache reads and writes, `output_tokens` includes thinking.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct MessagesUsage {
	pub input_tokens: u64,
	pub cache_write_5m_tokens: u64,
	pub cache_write_1h_tokens: u64,
	pub cache_read_tokens: u64,
	pub output_tokens: u64,
}

impl MessagesUsage {
	pub fn cache_write_tokens(&self) -> u64 {
		self.cache_write_5m_tokens + self.cache_write_1h_tokens
	}
}

/// What differs between the Messages endpoints.
#[derive(Clone, Copy)]
pub(crate) struct AnthropicCompatConfig {
	pub provider_name: &'static str,
	/// Image input the endpoint takes for a model; anything else is refused.
	pub images: fn(&str) -> Images,
	/// Whether a model takes video blocks; anything else is refused.
	pub videos: fn(&str) -> bool,
	/// The endpoint constrains its output to a requested schema. Declared only
	/// from a live probe; otherwise the schema goes through the forced-tool
	/// path and the answer is validated locally.
	pub enforces_response_schema: bool,
	/// Writes the endpoint's thinking knobs for a requested effort.
	pub thinking: fn(model: &str, effort: Option<ReasoningEffort>, request: &mut Value),
	/// Cost in USD of one response. `None` leaves the cost to the caller.
	pub cost: fn(model: &str, usage: &MessagesUsage) -> Option<f64>,
	/// The top-level prompt sent for a conversation with no system turn.
	pub default_system: &'static str,
}

/// The wire protocol an endpoint speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ApiMode {
	/// Anthropic Messages.
	Anthropic,
	/// OpenAI Chat Completions.
	OpenAi,
}

/// An explicit `*_API_MODE` value.
pub(crate) fn parse_api_mode(mode_env: &str, raw: &str) -> Result<ApiMode> {
	match raw.trim() {
		value if value.eq_ignore_ascii_case("anthropic") => Ok(ApiMode::Anthropic),
		value if value.eq_ignore_ascii_case("openai") => Ok(ApiMode::OpenAi),
		value => Err(config_error(&format!("{mode_env} must be `anthropic` or `openai`, got `{value}`")).into()),
	}
}

/// The protocol a URL's own path names: a terminal `/messages` or
/// `/chat/completions`. The query and the host say nothing about it, and a
/// path naming neither leaves the choice to the provider.
pub(crate) fn url_api_mode(api_url: &str) -> Option<ApiMode> {
	let url = reqwest::Url::parse(api_url).ok()?;
	let path = url.path().trim_end_matches('/');
	if path.ends_with("/messages") {
		Some(ApiMode::Anthropic)
	} else if path.ends_with("/chat/completions") {
		Some(ApiMode::OpenAi)
	} else {
		None
	}
}

pub(crate) async fn chat_completion(
	config: AnthropicCompatConfig,
	api_key: String,
	api_url: String,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	chat_completion_as(config, Auth::Bearer, api_key, api_url, params).await
}

/// [`chat_completion`] with the key in `x-api-key`, the header Anthropic's own
/// Messages contract names, and no `Authorization` header. For an endpoint that
/// documents only that one.
pub(crate) async fn chat_completion_with_x_api_key(
	config: AnthropicCompatConfig,
	api_key: String,
	api_url: String,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	chat_completion_as(config, Auth::ApiKey, api_key, api_url, params).await
}

async fn chat_completion_as(
	config: AnthropicCompatConfig,
	auth: Auth,
	api_key: String,
	api_url: String,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	crate::llm::schema_enforcement::chat_completion_enforced(
		&MessagesTransport {
			config,
			auth,
			api_key,
			api_url,
		},
		json_object_as_object_schema(params),
	)
	.await
}

/// Messages has no `json_object` response mode, and this layer sends no prompt
/// of its own for it: an `OutputFormat::Json` request used to go out as an
/// ordinary chat turn and come back as prose, silently dropping the guarantee
/// the caller asked for. It is promoted to the narrowest schema that says the
/// same thing — any JSON object — so it rides the schema path: native
/// `output_config.format` where the endpoint enforces a schema, the forced tool
/// call plus local object validation everywhere else. Only a JSON request is
/// touched; ordinary chat keeps its exact body and prompt.
fn json_object_as_object_schema(mut params: ChatCompletionParams) -> ChatCompletionParams {
	let Some(request) = params.response_format.as_mut() else {
		return params;
	};
	if matches!(request.format, OutputFormat::Json) {
		request.format = OutputFormat::JsonSchema;
		request.schema = Some(json!({ "type": "object" }));
	}
	params
}

/// The header an endpoint reads its key from.
#[derive(Debug, Clone, Copy)]
enum Auth {
	/// `Authorization: Bearer`, what every endpoint here reads.
	Bearer,
	/// `x-api-key`, the header Anthropic's own Messages contract names.
	ApiKey,
}

struct MessagesTransport {
	config: AnthropicCompatConfig,
	auth: Auth,
	api_key: String,
	api_url: String,
}

#[async_trait::async_trait]
impl AiProvider for MessagesTransport {
	fn name(&self) -> &str {
		self.config.provider_name
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		Ok(self.api_key.clone())
	}

	fn enforces_response_schema(&self, _model: &str) -> bool {
		self.config.enforces_response_schema
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		chat_completion_raw(
			self.config,
			self.auth,
			self.api_key.clone(),
			self.api_url.clone(),
			params,
		)
		.await
	}
}

async fn chat_completion_raw(
	config: AnthropicCompatConfig,
	auth: Auth,
	api_key: String,
	api_url: String,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	let messages = convert_messages(&params.messages, config, &params.model)?;

	let mut request_body = json!({
		"model": params.model,
		"messages": messages,
		"stream": true,
	});
	// 0 is the provider default, which these endpoints take as an omitted field.
	if params.max_tokens > 0 {
		request_body["max_tokens"] = json!(params.max_tokens);
	}

	// Every system turn makes up the one top-level prompt, so a system message
	// appended late (schema enforcement adds one) is not lost. Each turn is
	// sent verbatim as its own block with its own cache breakpoint; a blank
	// turn adds no block.
	let system_turns: Vec<&Message> = params
		.messages
		.iter()
		.filter(|message| message.role == "system")
		.collect();
	let system_blocks: Vec<Value> = system_turns
		.iter()
		.filter(|message| !message.content.trim().is_empty())
		.map(|message| {
			let mut block = json!({ "type": "text", "text": message.content });
			if let Some(cache_control) = shared::maybe_ephemeral_cache_control(message.cached) {
				block["cache_control"] = cache_control;
			}
			block
		})
		.collect();
	let system = if system_turns.is_empty() {
		// A conversation with no system turn gets the provider's own fallback
		// prompt, where it has one.
		(!config.default_system.is_empty()).then(|| json!(config.default_system))
	} else if system_blocks.is_empty() {
		// Blank turns only: an empty prompt is the caller's, not a missing one.
		Some(json!(""))
	} else if system_blocks.len() == 1 && system_blocks[0].get("cache_control").is_none() {
		// One plain turn goes as the string every endpoint takes.
		Some(system_blocks[0]["text"].clone())
	} else {
		Some(json!(system_blocks))
	};
	if let Some(system) = system {
		request_body["system"] = system;
	}

	if let Some(tools) = params.tools.as_ref().filter(|tools| !tools.is_empty()) {
		// Sorted by name, so the tool block is a stable cache prefix.
		let mut sorted_tools = tools.clone();
		sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));
		request_body["tools"] = json!(sorted_tools
			.iter()
			.map(|function| {
				let mut tool = json!({
					"name": function.name,
					"description": function.description,
					"input_schema": function.parameters,
				});
				if let Some(cache_control) = &function.cache_control {
					tool["cache_control"] = cache_control.clone();
				}
				tool
			})
			.collect::<Vec<_>>());
	}

	if config.enforces_response_schema {
		if let Some(schema) = requested_schema(&params) {
			request_body["output_config"]["format"] = json!({ "type": "json_schema", "schema": schema });
		}
	}

	(config.thinking)(&params.model, params.reasoning_effort, &mut request_body);

	execute_request(config, auth, api_key, api_url, request_body, params).await
}

fn requested_schema(params: &ChatCompletionParams) -> Option<Value> {
	params
		.response_format
		.as_ref()
		.filter(|format| matches!(format.format, OutputFormat::JsonSchema))
		.and_then(|format| format.schema.clone())
}

async fn execute_request(
	config: AnthropicCompatConfig,
	auth: Auth,
	api_key: String,
	api_url: String,
	request_body: Value,
	params: ChatCompletionParams,
) -> Result<ProviderResponse> {
	let start_time = std::time::Instant::now();
	let request_timeout = params.request_timeout;
	let stream_timeout = params.stream_timeout;

	let response = retry::retry_provider_request(
		|| {
			let api_key = api_key.clone();
			let api_url = api_url.clone();
			let request_body = request_body.clone();
			let provider_name = config.provider_name.to_string();
			let extra_headers = params.extra_headers.clone();

			Box::pin(async move {
				let request = shared::http_client()
					.post(&api_url)
					.header(reqwest::header::CONTENT_TYPE, "application/json")
					.header("anthropic-version", ANTHROPIC_VERSION)
					.body(request_body.to_string());
				// A local server takes no key at all. The others read it from the
				// header their contract names, and from no other.
				let request = if api_key.is_empty() {
					request
				} else {
					match auth {
						Auth::Bearer => request.header(
							reqwest::header::AUTHORIZATION,
							shared::sensitive_header(&format!("Bearer {api_key}"))?,
						),
						Auth::ApiKey => request.header("x-api-key", shared::sensitive_header(&api_key)?),
					}
				};

				let captured = super::streaming::send_and_read(
					request,
					request_timeout,
					stream_timeout,
					extra_headers.as_ref(),
					super::streaming::Format::Messages,
				)
				.await?;

				// Return Err for retryable HTTP errors so the retry loop catches them
				if retry::is_retryable_status(captured.status.as_u16()) {
					return Err(retry::retryable_status_error(
						&captured,
						format!("{provider_name} API error {}: {}", captured.status, captured.body),
					));
				}

				Ok(captured)
			})
		},
		params.max_retries,
		params.retry_timeout,
		params.cancellation_token.as_ref(),
		params.api_key.as_ref(),
		params.retry_observer.as_ref(),
	)
	.await?;

	let request_time_ms = start_time.elapsed().as_millis() as u64;

	if !response.status.is_success() {
		bail!(
			"{} API error {}: {}",
			config.provider_name,
			response.status,
			response.body
		);
	}

	parse_response(
		&response.body,
		request_body,
		config,
		request_time_ms,
		params.api_key.as_ref(),
	)
}

fn parse_response(
	body: &str,
	request_body: Value,
	config: AnthropicCompatConfig,
	request_time_ms: u64,
	api_key: Option<&crate::llm::types::PerCallApiKey>,
) -> Result<ProviderResponse> {
	let mut response_json: Value = shared::parse_response(body, api_key)?;
	let blocks = response_json["content"]
		.as_array()
		.context("Messages response has no content array")?
		.clone();
	let finish_reason = response_json["stop_reason"].as_str().map(str::to_owned);
	let id = response_json["id"].as_str().map(str::to_owned);

	let mut content_parts = Vec::new();
	let mut thinking_parts = Vec::new();
	let mut tool_calls = Vec::new();
	for block in &blocks {
		match block["type"].as_str() {
			Some("text") => {
				if let Some(text) = block["text"].as_str() {
					content_parts.push(text);
				}
			}
			Some("thinking") => {
				if let Some(thinking) = block["thinking"].as_str().filter(|thinking| !thinking.is_empty()) {
					thinking_parts.push(thinking);
				}
			}
			Some("tool_use") => {
				// The call cannot be answered or replayed without these, so a
				// broken block fails the whole response before anything is
				// returned — even beside valid text or other tool calls.
				let id = tool_use_field(block, "id", config.provider_name)?;
				let name = tool_use_field(block, "name", config.provider_name)?;
				let input = block["input"]
					.as_object()
					.context("tool_use block has no input object")?;
				tool_calls.push(ToolCall {
					id: id.to_owned(),
					name: name.to_owned(),
					arguments: Value::Object(input.clone()),
				});
			}
			// `redacted_thinking` carries no text; any other block is not an answer.
			_ => {}
		}
	}
	let content = content_parts.join("\n");

	let usage = response_json.get("usage").filter(|usage| usage.is_object());
	let reported = |field: &str| usage.and_then(|usage| usage[field].as_u64()).unwrap_or(0);
	// With the per-TTL split each write is billed at its own rate; without one
	// every write is a 5m write, the API's default TTL.
	let (cache_write_5m_tokens, cache_write_1h_tokens) = match usage.filter(|usage| usage["cache_creation"].is_object())
	{
		Some(usage) => (
			usage["cache_creation"]["ephemeral_5m_input_tokens"]
				.as_u64()
				.unwrap_or(0),
			usage["cache_creation"]["ephemeral_1h_input_tokens"]
				.as_u64()
				.unwrap_or(0),
		),
		None => (reported("cache_creation_input_tokens"), 0),
	};
	let messages_usage = MessagesUsage {
		input_tokens: reported("input_tokens"),
		cache_write_5m_tokens,
		cache_write_1h_tokens,
		cache_read_tokens: reported("cache_read_input_tokens"),
		output_tokens: reported("output_tokens"),
	};

	// Thinking is billed inside `output_tokens`. DeepSeek reports its size
	// (`output_tokens_details.thinking_tokens`); where the API has no such
	// counter it is estimated from the emitted text. Either way it is carved
	// out of output, not added to it.
	let thinking_content = thinking_parts.join("\n\n");
	let reasoning_tokens = usage
		.and_then(|usage| usage["output_tokens_details"]["thinking_tokens"].as_u64())
		.unwrap_or((thinking_content.len() / 4) as u64);
	let thinking = (!thinking_content.is_empty()).then_some(ThinkingBlock {
		content: thinking_content,
		tokens: reasoning_tokens,
	});

	let cost = (config.cost)(request_body["model"].as_str().unwrap_or_default(), &messages_usage);
	let cache_write_tokens = messages_usage.cache_write_tokens();
	let usage = usage.map(|_| {
		let (output_tokens, reasoning_tokens) =
			TokenUsage::split_output(messages_usage.output_tokens, reasoning_tokens);
		TokenUsage {
			input_tokens: messages_usage.input_tokens,
			cache_read_tokens: messages_usage.cache_read_tokens,
			cache_write_tokens,
			output_tokens,
			reasoning_tokens,
			// input_tokens is already cache-free, so cache reads and writes are
			// their own terms in the total.
			total_tokens: messages_usage.input_tokens
				+ messages_usage.cache_read_tokens
				+ cache_write_tokens
				+ messages_usage.output_tokens,
			cost,
			request_time_ms: Some(request_time_ms),
		}
	});

	shared::set_response_tool_calls(&mut response_json, &tool_calls, None);
	// Thinking blocks go back unchanged on the next turn (the endpoint checks
	// their signature), so the response's blocks are kept verbatim for replay.
	if blocks.iter().any(|block| block["type"] == "thinking") {
		response_json["output_items"] = json!({ "provider": config.provider_name, "items": blocks });
	}

	let structured_output = shared::parse_structured_output_from_text(&content);

	Ok(ProviderResponse {
		content,
		thinking,
		exchange: ProviderExchange::new(request_body, response_json, usage, config.provider_name),
		tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
		finish_reason,
		structured_output,
		id,
	})
}

/// A `tool_use` block's `id` or `name`, when it is a non-blank string.
fn tool_use_field<'a>(block: &'a Value, field: &str, provider: &str) -> Result<&'a str> {
	block[field]
		.as_str()
		.filter(|value| !value.trim().is_empty())
		.with_context(|| format!("{provider} returned a tool_use block without a string {field}"))
}

/// Convert the conversation to Messages turns. Media the endpoint does not take
/// is an error: an endpoint that ignores it would answer as if it never came.
pub(crate) fn convert_messages(messages: &[Message], config: AnthropicCompatConfig, model: &str) -> Result<Vec<Value>> {
	let mut result: Vec<Value> = Vec::new();
	let mut index = 0;

	while index < messages.len() {
		let message = &messages[index];

		// System turns are the top-level prompt.
		if message.role == "system" {
			index += 1;
			continue;
		}

		if message.role == "assistant" {
			if let Some(content) = replayed_assistant_content(message, config.provider_name) {
				result.push(json!({ "role": "assistant", "content": content }));
				index += 1;
				continue;
			}
		}

		match message.role.as_str() {
			"tool" => {
				// The results of one call batch form a single user turn, and
				// the user text that follows them joins it after the results.
				let mut content = Vec::new();
				while index < messages.len() && messages[index].role == "tool" {
					let tool_message = &messages[index];
					content.push(tool_result_block(tool_message, config, model)?);
					index += 1;
				}
				if index < messages.len() && messages[index].role == "user" {
					append_regular_content(&messages[index], &mut content, config, model)?;
					index += 1;
				}
				result.push(json!({ "role": "user", "content": content }));
			}
			"assistant" if message.tool_calls.is_some() => {
				let mut content = Vec::new();
				append_text(message, &mut content);
				for call in shared::parse_generic_tool_calls_lossy(message.tool_calls.as_ref(), config.provider_name) {
					content
						.push(json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments }));
				}
				result.push(json!({ "role": "assistant", "content": content }));
				index += 1;
			}
			_ => {
				let mut content = Vec::new();
				append_regular_content(message, &mut content, config, model)?;
				// A turn with no blocks is invalid for these endpoints.
				if !content.is_empty() {
					result.push(json!({ "role": message.role, "content": content }));
				}
				index += 1;
			}
		}
	}

	Ok(result)
}

fn append_regular_content(
	message: &Message,
	content: &mut Vec<Value>,
	config: AnthropicCompatConfig,
	model: &str,
) -> Result<()> {
	// An empty text block is rejected, and a tool-only answer leaves content empty.
	append_text(message, content);
	if let Some(images) = &message.images {
		for image in images {
			content.push(image_block(image, config, model)?);
		}
	}
	if let Some(videos) = &message.videos {
		for video in videos {
			if !(config.videos)(model) {
				return Err(unsupported_media(config.provider_name, model, "video input"));
			}
			content.push(json!({ "type": "video", "source": video_source(video) }));
		}
	}
	Ok(())
}

fn append_text(message: &Message, content: &mut Vec<Value>) {
	if message.content.trim().is_empty() {
		return;
	}
	let mut block = json!({ "type": "text", "text": message.content });
	if let Some(cache_control) = shared::maybe_ephemeral_cache_control(message.cached) {
		block["cache_control"] = cache_control;
	}
	content.push(block);
}

fn image_block(image: &ImageAttachment, config: AnthropicCompatConfig, model: &str) -> Result<Value> {
	check_image(image, config, model)?;
	Ok(image_body(image))
}

/// One `tool_result` block: its text, then the media the endpoint takes. Media
/// it does not take is refused, never dropped: an endpoint that ignored it
/// would answer as though the caller's attachment had never arrived.
fn tool_result_block(message: &Message, config: AnthropicCompatConfig, model: &str) -> Result<Value> {
	let images = message.images.as_deref().unwrap_or_default();
	let videos = message.videos.as_deref().unwrap_or_default();
	// Every attachment is checked before any is built, so a refused request
	// copies no payload.
	for image in images {
		check_image(image, config, model)?;
	}
	if !videos.is_empty() && !(config.videos)(model) {
		return Err(unsupported_media(config.provider_name, model, "video input"));
	}

	let content = if images.is_empty() && videos.is_empty() {
		json!(message.content)
	} else {
		let mut blocks = Vec::new();
		if !message.content.trim().is_empty() {
			blocks.push(json!({ "type": "text", "text": message.content }));
		}
		blocks.extend(images.iter().map(image_body));
		for video in videos {
			blocks.push(json!({ "type": "video", "source": video_source(video) }));
		}
		Value::Array(blocks)
	};

	let mut block = json!({
		"type": "tool_result",
		"tool_use_id": message.tool_call_id.as_deref().unwrap_or_default(),
		"content": content,
	});
	if let Some(cache_control) = shared::maybe_ephemeral_cache_control(message.cached) {
		block["cache_control"] = cache_control;
	}
	Ok(block)
}

/// Whether the endpoint takes this image, checked without building its block.
fn check_image(image: &ImageAttachment, config: AnthropicCompatConfig, model: &str) -> Result<()> {
	match (&image.data, (config.images)(model)) {
		(ImageData::Base64(_), Images::Base64 | Images::Base64AndUrl) => Ok(()),
		(ImageData::Url(_), Images::Base64AndUrl) => Ok(()),
		(ImageData::Url(_), _) => Err(unsupported_media(
			config.provider_name,
			model,
			"URL images (base64 only)",
		)),
		(_, Images::None) => Err(unsupported_media(config.provider_name, model, "image input")),
	}
}

/// The image block itself; the caller has already checked the endpoint takes it.
fn image_body(image: &ImageAttachment) -> Value {
	let source = match &image.data {
		ImageData::Base64(data) => json!({ "type": "base64", "media_type": image.media_type, "data": data }),
		ImageData::Url(url) => json!({ "type": "url", "url": url }),
	};
	json!({ "type": "image", "source": source })
}

fn video_source(video: &VideoAttachment) -> Value {
	match &video.data {
		VideoData::Base64(data) => json!({ "type": "base64", "media_type": video.media_type, "data": data }),
		VideoData::Url(url) => json!({ "type": "url", "url": url }),
	}
}

fn unsupported_media(provider: &str, model: &str, kind: &str) -> anyhow::Error {
	anyhow::anyhow!("{provider} model {model} does not accept {kind} on its Anthropic-compatible endpoint")
}

/// The assistant turn's captured content blocks, sent back as they came. They
/// stand for the message only while they come from this provider and their
/// `tool_use` blocks are exactly its tool calls; otherwise the turn is rebuilt
/// from the stored fields.
fn replayed_assistant_content(message: &Message, provider_name: &str) -> Option<Vec<Value>> {
	let captured = message.provider_items.as_ref()?;
	if captured["provider"] != provider_name {
		return None;
	}
	let items = captured["items"].as_array()?;
	let calls = match message.tool_calls.as_ref() {
		Some(tool_calls) => shared::parse_generic_tool_calls_strict(tool_calls, provider_name).ok()?,
		None => Vec::new(),
	};
	let mut message_call_ids: Vec<&str> = calls.iter().map(|call| call.id.as_str()).collect();
	let mut item_call_ids: Vec<&str> = items
		.iter()
		.filter(|item| item["type"] == "tool_use")
		.map(|item| item["id"].as_str().unwrap_or_default())
		.collect();
	message_call_ids.sort_unstable();
	item_call_ids.sort_unstable();
	if item_call_ids != message_call_ids {
		return None;
	}

	let mut items = items.clone();
	if let Some(cache_control) = shared::maybe_ephemeral_cache_control(message.cached) {
		// The breakpoint goes on the last text block, where a rebuilt turn has it.
		if let Some(text) = items.iter_mut().rev().find(|item| item["type"] == "text") {
			text["cache_control"] = cache_control;
		}
	}
	Some(items)
}

#[cfg(test)]
#[path = "anthropic_compat_tests.rs"]
mod tests;
