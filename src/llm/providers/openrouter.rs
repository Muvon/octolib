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

//! OpenRouter provider implementation

use super::shared;
use crate::errors::ToolCallError;
use crate::llm::retry;
use crate::llm::traits::{AiProvider, KeepalivePolicy};
use crate::llm::types::{
	ChatCompletionParams, Message, ProviderExchange, ProviderResponse, ReasoningEffort, SamplingSupport, ThinkingBlock,
	TokenUsage, ToolCall,
};
use crate::llm::utils::normalize_model_name;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use tokio::sync::OnceCell;

/// OpenRouter provider (uses OpenAI-compatible API)
#[derive(Debug, Clone)]
pub struct OpenRouterProvider;

impl Default for OpenRouterProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl OpenRouterProvider {
	pub fn new() -> Self {
		Self
	}
}

const DEFAULT_REPETITION_PENALTY: f64 = 1.1;

/// Cheapest valid completion size; OpenRouter has no zero-output primitive for cache pre-warm.
const CACHE_PREWARM_MAX_TOKENS: u32 = 1;

const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const OPENROUTER_API_URL_ENV: &str = "OPENROUTER_API_URL";
const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// Routes whose upstream decodes against a `json_schema`, by model-id prefix.
/// Schema requests also send `provider.require_parameters`, so they only reach
/// endpoints whose catalog entry lists `structured_outputs`; every row here
/// had such endpoints (`/api/v1/models/<id>/endpoints`, 2026-10-26). That
/// flag is the endpoint's own claim and is not enough on its own: Inception's
/// mercury-3 and Alibaba's qwen3.8-omni-flash both advertise it, yet answered a
/// two-field object schema with the JSON array the prompt asked for. Every row
/// returned the object instead (live, 2026-10-26). A route not listed is not
/// trusted with a schema.
const SCHEMA_ENFORCING_ROUTES: &[&str] = &[
	"openai/",
	"anthropic/claude-",
	"google/gemini-",
	"x-ai/grok-",
	"meta/muse-",
	"xiaomi/mimo-",
	"moonshotai/kimi-k3.1",
	"deepseek/deepseek-v4.1-flash",
	"qwen/qwen3.7-plus",
];

#[derive(Debug, Deserialize)]
struct CatalogResponse {
	data: Vec<CatalogModel>,
}

#[derive(Debug, Deserialize)]
struct CatalogModel {
	id: String,
	#[serde(default)]
	supported_parameters: Option<Vec<String>>,
	#[serde(default, deserialize_with = "deserialize_catalog_reasoning")]
	reasoning: Option<CatalogReasoning>,
}

#[derive(Debug, Deserialize)]
struct CatalogReasoning {
	mandatory: bool,
	#[serde(default, deserialize_with = "deserialize_supported_efforts")]
	supported_efforts: Vec<String>,
}

/// An omitted `supported_efforts` means the model exposes no effort selection,
/// but `null` means "all gateway effort values are accepted" (OpenRouter
/// reasoning-tokens guide, Oct 2026), so it expands to the whole ladder.
fn deserialize_supported_efforts<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	Ok(Option::<Vec<String>>::deserialize(deserializer)?
		.unwrap_or_else(|| EFFORT_LADDER.iter().map(|level| level.to_string()).collect()))
}

fn deserialize_catalog_reasoning<'de, D>(deserializer: D) -> std::result::Result<Option<CatalogReasoning>, D::Error>
where
	D: serde::Deserializer<'de>,
{
	let value = Option::<serde_json::Value>::deserialize(deserializer)?;
	Ok(value.and_then(|value| serde_json::from_value(value).ok()))
}

static CATALOG: OnceCell<Vec<CatalogModel>> = OnceCell::const_new();

async fn fetch_catalog(
	api_key: &str,
	api_url: &str,
	per_call_api_key: Option<&crate::llm::types::PerCallApiKey>,
) -> Result<Vec<CatalogModel>> {
	let api_base = api_url
		.trim_end_matches('/')
		.strip_suffix("/chat/completions")
		.ok_or_else(|| anyhow::anyhow!("OpenRouter API URL must end in /chat/completions"))?;
	let response = shared::http_client()
		.get(format!("{api_base}/models"))
		.bearer_auth(api_key)
		.send()
		.await
		.map_err(|error| shared::redact_error(error.into(), per_call_api_key))
		.context("Failed to fetch OpenRouter model catalog")?;
	if !response.status().is_success() {
		let status = response.status();
		let text = response
			.text()
			.await
			.map_err(|error| shared::redact_error(error.into(), per_call_api_key))
			.context("Failed to read OpenRouter model catalog error")?;
		let text = shared::redact_text(&text, per_call_api_key);
		return Err(anyhow::anyhow!("OpenRouter models API error {status}: {text}"));
	}
	let catalog: CatalogResponse = response
		.json()
		.await
		.map_err(|error| shared::redact_error(error.into(), per_call_api_key))
		.context("Failed to parse OpenRouter model catalog")?;
	Ok(catalog.data)
}

fn catalog_model<'a>(catalog: &'a [CatalogModel], model: &str) -> Option<&'a CatalogModel> {
	let model = model.split_once(':').map_or(model, |(id, _)| id);
	let normalized = normalize_model_name(model);
	catalog
		.iter()
		.find(|entry| normalize_model_name(&entry.id) == normalized)
}

fn supports_parameter(catalog: Option<&[CatalogModel]>, model: &str, parameter: &str) -> Option<bool> {
	let supported = catalog_model(catalog?, model)?.supported_parameters.as_ref()?;
	Some(supported.iter().any(|name| name == parameter))
}

fn sampling_support(catalog: Option<&[CatalogModel]>, model: &str) -> SamplingSupport {
	// Unknown support must not turn optional sampling into a routing requirement.
	let supported = |name| supports_parameter(catalog, model, name) == Some(true);
	SamplingSupport {
		temperature: supported("temperature"),
		top_p: supported("top_p"),
		top_k: supported("top_k"),
	}
}

const EFFORT_LADDER: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

#[derive(Debug, PartialEq, Eq)]
enum ReasoningSelection {
	Disabled,
	Omit,
	Effort(&'static str),
}

fn select_effort(entry: &CatalogModel, effort: ReasoningEffort) -> Option<ReasoningSelection> {
	let reasoning = entry.reasoning.as_ref()?;
	let supported = |name: &str| reasoning.supported_efforts.iter().any(|supported| supported == name);

	if effort == ReasoningEffort::None {
		if !reasoning.mandatory {
			return Some(ReasoningSelection::Disabled);
		}
		if reasoning.supported_efforts.is_empty() {
			return Some(ReasoningSelection::Omit);
		}
		return EFFORT_LADDER[1..]
			.iter()
			.copied()
			.find(|name| supported(name))
			.map(ReasoningSelection::Effort);
	}

	let requested = match effort {
		ReasoningEffort::None => unreachable!("handled above"),
		ReasoningEffort::Low => "low",
		ReasoningEffort::Medium => "medium",
		ReasoningEffort::High => "high",
		ReasoningEffort::XHigh => "xhigh",
		ReasoningEffort::Max => "max",
	};
	let index = EFFORT_LADDER.iter().position(|level| *level == requested)?;
	if supported(requested) {
		return Some(ReasoningSelection::Effort(requested));
	}
	// Nearest supported level on the ladder, ties going up: a three-level model
	// (low/high/max) is asked "medium" as its middle level, not its lowest.
	// "none" is never a substitute for a requested reasoning level.
	EFFORT_LADDER
		.iter()
		.enumerate()
		.skip(1)
		.filter(|(_, name)| supported(name))
		.min_by_key(|(position, _)| (position.abs_diff(index), std::cmp::Reverse(*position)))
		.map(|(_, name)| ReasoningSelection::Effort(name))
}

fn catalog_reasoning_selection(
	catalog: &[CatalogModel],
	model: &str,
	effort: ReasoningEffort,
) -> Option<ReasoningSelection> {
	catalog_model(catalog, model).and_then(|entry| select_effort(entry, effort))
}

fn apply_reasoning_effort(
	request_body: &mut serde_json::Value,
	catalog: Option<&[CatalogModel]>,
	model: &str,
	effort: ReasoningEffort,
) {
	match catalog.and_then(|catalog| catalog_reasoning_selection(catalog, model, effort)) {
		Some(ReasoningSelection::Disabled) => {
			request_body["reasoning"] = serde_json::json!({ "enabled": false });
		}
		Some(ReasoningSelection::Omit) => {}
		Some(ReasoningSelection::Effort(value)) => {
			request_body["reasoning_effort"] = serde_json::json!(value);
		}
		None if effort == ReasoningEffort::None => {
			request_body["reasoning"] = serde_json::json!({ "enabled": false });
		}
		None => {
			let value = match effort {
				ReasoningEffort::None => unreachable!("handled above"),
				ReasoningEffort::Low => "low",
				ReasoningEffort::Medium => "medium",
				ReasoningEffort::High => "high",
				ReasoningEffort::XHigh | ReasoningEffort::Max => "high",
			};
			request_body["reasoning_effort"] = serde_json::json!(value);
		}
	}
}

#[async_trait::async_trait]
impl AiProvider for OpenRouterProvider {
	fn name(&self) -> &str {
		"openrouter"
	}

	fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
		sampling_support(CATALOG.get().map(Vec::as_slice), model)
	}

	fn supports_model(&self, model: &str) -> bool {
		// OpenRouter supports many models from different providers (case-insensitive)
		// Accept models with provider prefixes (anthropic/, openai/, meta/, google/, etc.)
		// or direct model names
		let normalized = normalize_model_name(model);
		normalized.starts_with("anthropic/")
			|| normalized.starts_with("openai/")
			|| normalized.starts_with("meta/")
			|| normalized.starts_with("google/")
			|| normalized.starts_with("mistral/")
			|| normalized.starts_with("cohere/")
			|| normalized.contains("claude")
			|| normalized.contains("gpt-")
			|| normalized.contains("llama")
			|| normalized.contains("gemini")
			|| normalized.contains("mistral")
			|| !model.is_empty() // Accept any non-empty model string as fallback
	}

	fn get_api_key(&self) -> Result<String> {
		env::var(OPENROUTER_API_KEY_ENV)
			.map_err(|_| crate::errors::missing_api_key("openrouter", OPENROUTER_API_KEY_ENV).into())
	}

	fn supports_caching(&self, model: &str) -> bool {
		// OpenRouter supports caching for Anthropic models (case-insensitive)
		let normalized = normalize_model_name(model);
		normalized.starts_with("anthropic") || normalized.starts_with("claude")
	}

	fn keepalive_policy(&self, model: &str, use_long_cache: bool) -> Option<KeepalivePolicy> {
		// OpenRouter passes `cache_control` straight through to Anthropic
		// upstream for Claude routes, so the same refresh-on-read TTL semantics
		// apply (5m default / 1h with extended-cache-ttl beta). We deliberately
		// do not enable keepalive for non-Anthropic OpenRouter routes — other
		// upstreams' cache mechanics are not pingable through OR's API.
		let normalized = normalize_model_name(model);
		if !(normalized.starts_with("anthropic") || normalized.starts_with("claude")) {
			return None;
		}
		let ttl_secs = if use_long_cache { 3600 } else { 300 };
		Some(KeepalivePolicy {
			interval: std::time::Duration::from_secs(ttl_secs * 9 / 10),
		})
	}

	fn supports_vision(&self, model: &str) -> bool {
		// Try reference properties first for accurate model-level detection.
		// The registry handles provider-prefixed names such as `openai/gpt-4o`.
		if let Some(caps) = crate::llm::reference_models::get_reference_capabilities(model) {
			return caps.vision;
		}
		// Unknown model — default true (aggregator, let API handle)
		true
	}

	fn supports_video(&self, model: &str) -> bool {
		if let Some(caps) = crate::llm::reference_models::get_reference_capabilities(model) {
			return caps.video;
		}
		false
	}

	fn get_max_input_tokens(&self, model: &str) -> usize {
		if let Some(caps) = crate::llm::reference_models::get_reference_capabilities(model) {
			return caps.max_input_tokens;
		}
		// OpenRouter-specific known families not in reference table
		let normalized = normalize_model_name(model);
		if normalized.contains("claude") {
			return 200_000;
		}
		if normalized.contains("gpt-4o") || normalized.contains("gpt-4-turbo") {
			return 128_000;
		}
		if normalized.starts_with("o1") || normalized.starts_with("o3") {
			return 200_000;
		}
		262_144 // Default for unlisted OpenRouter models
	}

	fn supports_structured_output(&self, model: &str) -> bool {
		// Try reference capabilities; default true for OpenRouter aggregator
		crate::llm::reference_models::get_reference_capabilities(model)
			.map(|c| c.structured_output)
			.unwrap_or(true)
	}

	fn enforces_response_schema(&self, model: &str) -> bool {
		let normalized = normalize_model_name(model);
		SCHEMA_ENFORCING_ROUTES
			.iter()
			.any(|route| normalized.starts_with(route))
	}

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		// OpenRouter proxies to underlying providers
		// Try to detect provider from model name and delegate to their pricing
		let normalized = normalize_model_name(model);

		// Anthropic models (claude)
		if normalized.starts_with("anthropic/") || normalized.contains("claude") {
			// Delegate to Anthropic provider pricing
			let anthropic = crate::llm::providers::AnthropicProvider::new();
			let pricing = anthropic.get_model_pricing(model)?;
			// Fast mode is a request option on the Claude API, but OpenRouter
			// sells it as a separate `-fast` route. It doubles every rate:
			// <https://platform.claude.com/docs/en/about-claude/pricing#fast-mode-pricing>
			if normalized.ends_with("-fast") {
				return Some(crate::llm::types::ModelPricing::new(
					pricing.input_price_per_1m * 2.0,
					pricing.output_price_per_1m * 2.0,
					pricing.cache_write_price_per_1m * 2.0,
					pricing.cache_read_price_per_1m * 2.0,
				));
			}
			return Some(pricing);
		}

		// OpenAI models (gpt)
		if normalized.starts_with("openai/") || normalized.contains("gpt-") {
			let openai = crate::llm::providers::OpenAiProvider::new();
			return openai.get_model_pricing(model);
		}

		// DeepSeek models
		if normalized.starts_with("deepseek") {
			let deepseek = crate::llm::providers::DeepSeekProvider::new();
			return deepseek.get_model_pricing(model);
		}

		// Google models (gemini)
		if normalized.starts_with("google/") || normalized.contains("gemini") {
			let google = crate::llm::providers::GoogleVertexProvider::new();
			return google.get_model_pricing(model);
		}

		// Every other family (xiaomi/, x-ai/, qwen/, meta/, inception/, ...): the
		// reference table's baseline price, so cost estimates made before a call
		// see a real rate instead of none. Billing still reads the response's
		// own usage.cost.
		crate::llm::reference_models::get_reference_pricing(model)
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let api_url = env::var(OPENROUTER_API_URL_ENV).unwrap_or_else(|_| OPENROUTER_API_URL.to_string());
		let catalog_key = api_key.clone();
		let per_call_api_key = params.api_key.clone();
		let catalog_url = api_url.clone();
		// A failed fetch is retried next call; unknown optional support is omitted.
		let _ = CATALOG
			.get_or_try_init(
				|| async move { fetch_catalog(&catalog_key, &catalog_url, per_call_api_key.as_ref()).await },
			)
			.await;
		let requested_schema = params.response_format.as_ref().and_then(|format| format.schema.clone());

		let request_body = build_request_body(&params, CATALOG.get().map(Vec::as_slice))?;

		let response = execute_openrouter_request(
			api_key,
			api_url,
			request_body,
			params.max_retries,
			params.retry_timeout,
			params.request_timeout,
			params.stream_timeout,
			params.cancellation_token.as_ref(),
			params.extra_headers.clone(),
			params.api_key.as_ref(),
			params.retry_observer.as_ref(),
		)
		.await?;

		if let Some(schema) = requested_schema {
			crate::llm::schema_enforcement::validate_response(response, &schema, "openrouter", params.api_key.as_ref())
		} else {
			Ok(response)
		}
	}
}

fn build_request_body(params: &ChatCompletionParams, catalog: Option<&[CatalogModel]>) -> Result<serde_json::Value> {
	// Convert messages to OpenRouter format (same as OpenAI)
	let messages = convert_messages(&params.messages)?;

	// Apply sampling parameters based on model support
	let sampling = sampling_support(catalog, &params.model).effective(params.temperature, params.top_p, params.top_k);

	// Create the request body
	let mut request_body = serde_json::json!({
		"stream": true,
		"model": params.model,
		"messages": messages,
		"usage": {
			"include": true  // Always enable usage tracking for all requests
		},
		"provider": {
			"order": [
				"Anthropic",
				"OpenAI",
				"Amazon Bedrock",
				"Azure",
				"Cloudflare",
				"Google Vertex",
				"xAI",
			],
			"allow_fallbacks": true,
		},
	});
	// Unsupported optional parameters reject every endpoint under strict routing.
	if supports_parameter(catalog, &params.model, "repetition_penalty") == Some(true) {
		request_body["repetition_penalty"] = serde_json::json!(DEFAULT_REPETITION_PENALTY);
	}
	if let Some(temp) = sampling.temperature {
		request_body["temperature"] = serde_json::json!(temp);
	}
	if let Some(top_p) = sampling.top_p {
		request_body["top_p"] = serde_json::json!(top_p);
	}
	if let Some(top_k) = sampling.top_k {
		request_body["top_k"] = serde_json::json!(top_k);
	}

	// Add max_tokens if specified (0 means don't include it in request). OpenRouter has
	// no zero-output primitive, so a cache pre-warm asks for the cheapest valid completion.
	if params.cache_prewarm {
		request_body["max_tokens"] = serde_json::json!(CACHE_PREWARM_MAX_TOKENS);
	} else if params.max_tokens > 0 {
		request_body["max_tokens"] = serde_json::json!(params.max_tokens);
	}

	if let Some(effort) = params.reasoning_effort {
		apply_reasoning_effort(&mut request_body, catalog, &params.model, effort);
	}

	// Add tools if available (OpenRouter supports OpenAI-compatible tools)
	if let Some(tools) = &params.tools {
		if !tools.is_empty() {
			// Sort tools by name for consistent ordering
			let mut sorted_tools = tools.clone();
			sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

			let openai_tools = sorted_tools
				.iter()
				.map(|f| {
					serde_json::json!({
						"type": "function",
						"function": {
							"name": f.name,
							"description": f.description,
							"parameters": f.parameters
						}
					})
				})
				.collect::<Vec<_>>();

			request_body["tools"] = serde_json::json!(openai_tools);
			request_body["tool_choice"] = serde_json::json!("auto");
			// Explicit: OpenRouter proxies to many backends; forward the flag
			// so providers that default to single-call honor parallel batching.
			request_body["parallel_tool_calls"] = serde_json::json!(true);
		}
	}

	// Add structured output format if specified (OpenRouter supports OpenAI-compatible format)
	if let Some(response_format) = &params.response_format {
		match &response_format.format {
			crate::llm::types::OutputFormat::Json => {
				request_body["response_format"] = serde_json::json!({
					"type": "json_object"
				});
			}
			crate::llm::types::OutputFormat::JsonSchema => {
				if let Some(schema) = &response_format.schema {
					// Strict structured outputs need additionalProperties:false on
					// every nested object; OpenRouter forwards strict to OpenAI/Azure.
					// No-op unless mode is Strict.
					let schema = crate::llm::utils::normalize_strict_schema(schema, response_format.mode);

					let mut format_obj = serde_json::json!({
						"type": "json_schema",
						"json_schema": {
							"name": "response",
							"schema": schema
						}
					});

					// Add strict mode if specified
					if matches!(response_format.mode, crate::llm::types::ResponseMode::Strict) {
						format_obj["json_schema"]["strict"] = serde_json::json!(true);
					}

					request_body["response_format"] = format_obj;
					// Without it OpenRouter still routes to the model when no
					// endpoint takes `response_format` and drops the schema
					// (docs: guides/routing/provider-selection, Oct 2026).
					request_body["provider"]["require_parameters"] = serde_json::json!(true);
				}
			}
		}
	}

	if request_body["provider"]["require_parameters"] == true {
		// These are hints, unlike the schema, tools and output-token limit.
		// Requiring an unknown/unsupported hint can exclude every schema endpoint.
		for parameter in ["reasoning", "reasoning_effort", "tool_choice", "parallel_tool_calls"] {
			if supports_parameter(catalog, &params.model, parameter) != Some(true) {
				if let Some(object) = request_body.as_object_mut() {
					object.remove(parameter);
				}
			}
		}
	}
	Ok(request_body)
}

// Reuse OpenAI structures since OpenRouter is compatible
#[derive(Serialize, Deserialize, Debug)]
struct OpenRouterMessage {
	role: String,
	content: serde_json::Value,
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_call_id: Option<String>, // For tool messages: the ID of the tool call
	#[serde(skip_serializing_if = "Option::is_none")]
	name: Option<String>, // For tool messages: the name of the tool
	#[serde(skip_serializing_if = "Option::is_none")]
	tool_calls: Option<serde_json::Value>, // For assistant messages: array of tool calls
	#[serde(skip_serializing_if = "Option::is_none")]
	reasoning_details: Option<serde_json::Value>, // For Gemini thought signatures preservation
}

#[derive(Deserialize, Debug)]
struct OpenRouterResponse {
	id: String,
	choices: Vec<OpenRouterChoice>,
	usage: OpenRouterUsage,
}

#[derive(Deserialize, Debug)]
struct OpenRouterChoice {
	message: OpenRouterResponseMessage,
	finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenRouterResponseMessage {
	content: Option<String>,
	tool_calls: Option<Vec<OpenRouterToolCall>>,
	reasoning_details: Option<serde_json::Value>, // Gemini thought signatures
}

#[derive(Deserialize, Debug)]
struct OpenRouterToolCall {
	id: String,
	#[serde(rename = "type")]
	tool_type: String,
	function: OpenRouterFunction,
}

#[derive(Deserialize, Debug)]
struct OpenRouterFunction {
	name: String,
	arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenRouterUsage {
	#[serde(default)]
	input_tokens: Option<u64>,
	#[serde(default)]
	prompt_tokens: Option<u64>,
	#[serde(default)]
	completion_tokens: Option<u64>,
	#[serde(default)]
	output_tokens: Option<u64>,
	#[serde(default)]
	total_tokens: Option<u64>,
	#[serde(default)]
	prompt_tokens_details: Option<OpenRouterPromptTokensDetails>,
	#[serde(default)]
	completion_tokens_details: Option<OpenRouterCompletionTokensDetails>,
	#[serde(default)]
	cost: Option<f64>, // OpenRouter returns cost directly in usage object
}

#[derive(Deserialize, Debug)]
struct OpenRouterPromptTokensDetails {
	#[serde(default)]
	cached_tokens: u64,
}

#[derive(Deserialize, Debug)]
struct OpenRouterCompletionTokensDetails {
	#[serde(default)]
	reasoning_tokens: u64,
}

// Convert messages to OpenRouter format (same as OpenAI)
fn convert_messages(messages: &[Message]) -> Result<Vec<OpenRouterMessage>, ToolCallError> {
	let messages = shared::chat_completion_messages(messages);
	let messages = messages.as_ref();
	let mut result = Vec::new();

	for message in messages {
		match message.role.as_str() {
			"tool" => {
				// Tool messages in OpenRouter format - MUST include tool_call_id and name
				let tool_call_id = message.tool_call_id.clone();
				let name = message.name.clone();

				let content = if message.cached {
					let mut text_content = serde_json::json!({
						"type": "text",
						"text": message.content
					});
					text_content["cache_control"] = shared::ephemeral_cache_control();
					serde_json::json!([text_content])
				} else {
					serde_json::json!(message.content)
				};

				result.push(OpenRouterMessage {
					role: message.role.clone(),
					content,
					tool_call_id,
					name,
					tool_calls: None,
					reasoning_details: None,
				});
			}
			"assistant" if message.tool_calls.is_some() => {
				// Assistant message with tool calls - convert from unified GenericToolCall format
				let mut content_parts = Vec::new();

				// Add text content if not empty
				if !message.content.trim().is_empty() {
					let mut text_content = serde_json::json!({
						"type": "text",
						"text": message.content
					});

					if message.cached {
						text_content["cache_control"] = shared::ephemeral_cache_control();
					}

					content_parts.push(text_content);
				}

				let content = if content_parts.len() == 1 && !message.cached {
					content_parts[0]["text"].clone()
				} else if content_parts.is_empty() {
					serde_json::Value::Null
				} else {
					serde_json::json!(content_parts)
				};

				// Convert unified GenericToolCall format to OpenRouter format
				let Some(tool_calls_value) = message.tool_calls.as_ref() else {
					return Err(ToolCallError::MissingField {
						field: "tool_calls".to_string(),
					});
				};
				let generic_calls = shared::parse_generic_tool_calls_strict(tool_calls_value, "openrouter")?;

				// Extract reasoning_details from first tool call's meta (Gemini thought signatures)
				let reasoning_details = generic_calls
					.first()
					.and_then(|call| call.meta.as_ref())
					.and_then(|meta| meta.get("reasoning_details"))
					.cloned();

				// Convert GenericToolCall to OpenRouter format
				let openrouter_calls: Vec<serde_json::Value> = generic_calls
					.into_iter()
					.map(|call| {
						serde_json::json!({
							"id": call.id,
							"type": "function",
							"function": {
								"name": call.name,
								"arguments": shared::arguments_to_json_string(&call.arguments)
							}
						})
					})
					.collect();

				let tool_calls = Some(serde_json::Value::Array(openrouter_calls));

				result.push(OpenRouterMessage {
					role: message.role.clone(),
					content,
					tool_call_id: None,
					name: None,
					tool_calls,
					reasoning_details, // Add reasoning_details at message level
				});
			}
			_ => {
				// Handle other message types with cache support
				let mut content_parts = vec![{
					let mut text_content = serde_json::json!({
						"type": "text",
						"text": message.content
					});

					// Add cache_control if needed
					if message.cached {
						text_content["cache_control"] = shared::ephemeral_cache_control();
					}

					text_content
				}];

				// Add images if present
				if let Some(images) = &message.images {
					for image in images {
						if let crate::llm::types::ImageData::Base64(data) = &image.data {
							content_parts.push(serde_json::json!({
								"type": "image_url",
								"image_url": {
									"url": format!("data:{};base64,{}", image.media_type, data)
								}
							}));
						}
					}
				}

				// Add videos if present
				if let Some(videos) = &message.videos {
					for video in videos {
						match &video.data {
							crate::llm::types::VideoData::Base64(data) => {
								content_parts.push(serde_json::json!({
									"type": "video_url",
									"video_url": {
										"url": format!("data:{};base64,{}", video.media_type, data)
									}
								}));
							}
							crate::llm::types::VideoData::Url(url) => {
								content_parts.push(serde_json::json!({
									"type": "video_url",
									"video_url": {
										"url": url
									}
								}));
							}
						}
					}
				}

				let content = if content_parts.len() == 1 && !message.cached {
					content_parts[0]["text"].clone()
				} else {
					serde_json::json!(content_parts)
				};

				result.push(OpenRouterMessage {
					role: message.role.clone(),
					content,
					tool_call_id: None,
					name: None,
					tool_calls: None,
					reasoning_details: None,
				});
			}
		}
	}

	Ok(result)
}

// Execute OpenRouter HTTP request
#[allow(clippy::too_many_arguments)]
async fn execute_openrouter_request(
	api_key: String,
	api_url: String,
	request_body: serde_json::Value,
	max_retries: u32,
	base_timeout: std::time::Duration,
	request_timeout: Option<std::time::Duration>,
	stream_timeout: Option<std::time::Duration>,
	cancellation_token: Option<&tokio::sync::watch::Receiver<bool>>,
	extra_headers: Option<std::collections::HashMap<String, String>>,
	per_call_api_key: Option<&crate::llm::types::PerCallApiKey>,
	retry_observer: Option<&crate::llm::retry::RetryObserver>,
) -> Result<ProviderResponse> {
	let start_time = std::time::Instant::now();

	let response = retry::retry_provider_request(
		|| {
			let client = shared::http_client();
			let extra_headers = extra_headers.clone();
			let api_key = api_key.clone();
			let api_url = api_url.clone();
			let request_body = request_body.clone();
			let openrouter_app_title = std::env::var("OPENROUTER_APP_TITLE").unwrap_or_else(|_| "octolib".to_string());
			let openrouter_http_referer = std::env::var("OPENROUTER_HTTP_REFERER")
				.unwrap_or_else(|_| "https://octomind.run/product/octolib".to_string());
			// Category leaderboards (e.g. "cli-agent") describe the calling app, so no default.
			let openrouter_app_categories = std::env::var("OPENROUTER_APP_CATEGORIES").ok();

			Box::pin(async move {
				let mut req = client
					.post(&api_url)
					.header("Content-Type", "application/json")
					.bearer_auth(&api_key)
					.header("HTTP-Referer", openrouter_http_referer)
					.header("X-Title", openrouter_app_title)
					.json(&request_body);
				if let Some(categories) = openrouter_app_categories {
					req = req.header("X-OpenRouter-Categories", categories);
				}

				let captured = super::streaming::send_and_read(
					req,
					request_timeout,
					stream_timeout,
					extra_headers.as_ref(),
					super::streaming::Format::Chat,
				)
				.await?;

				// Return Err for retryable HTTP errors so the retry loop catches them
				if retry::is_retryable_status(captured.status.as_u16()) {
					return Err(retry::retryable_status_error(
						&captured,
						format!("OpenRouter API error {}: {}", captured.status, captured.body),
					));
				}

				Ok(captured)
			})
		},
		max_retries,
		base_timeout,
		cancellation_token,
		per_call_api_key,
		retry_observer,
	)
	.await?;

	let request_time_ms = start_time.elapsed().as_millis() as u64;

	if !response.status.is_success() {
		return Err(anyhow::anyhow!(
			"OpenRouter API error {}: {}",
			response.status,
			response.body
		));
	}

	let openrouter_response: OpenRouterResponse = shared::parse_response(&response.body, per_call_api_key)?;

	let choice = openrouter_response
		.choices
		.into_iter()
		.next()
		.ok_or_else(|| anyhow::anyhow!("No choices in OpenRouter response"))?;

	let content = choice.message.content.unwrap_or_default();

	// Extract reasoning_details as thinking (for Gemini and other providers)
	let reasoning_details = &choice.message.reasoning_details;

	// Calculate thinking content and extract tokens
	let thinking = match reasoning_details.as_ref() {
		Some(rd) => {
			// Extract text content from reasoning_details array
			let thinking_text = rd
				.as_array()
				.and_then(|arr| {
					let texts: Vec<String> = arr
						.iter()
						.filter_map(|item| item.get("text").and_then(|t| t.as_str().map(|s| s.to_string())))
						.collect();
					if texts.is_empty() {
						None
					} else {
						Some(texts)
					}
				})
				.map(|texts| texts.join("\n\n"))
				.unwrap_or_else(|| rd.to_string());

			// Estimate reasoning tokens from content length (4 chars per token)
			let estimated = (thinking_text.len() / 4) as u64;

			Some(ThinkingBlock {
				content: thinking_text,
				tokens: estimated,
			})
		}
		None => None,
	};

	// Convert tool calls if present
	let tool_calls: Option<Vec<ToolCall>> = choice.message.tool_calls.map(|calls| {
		calls
			.into_iter()
			.filter_map(|call| {
				// Validate tool type - OpenRouter should only have "function" type
				if call.tool_type != "function" {
					tracing::warn!(
						"Unexpected tool type '{}' from OpenRouter API",
						shared::redact_text(&call.tool_type, per_call_api_key)
					);
					return None;
				}

				let arguments = shared::parse_tool_call_arguments_lossy(&call.function.arguments);

				Some(ToolCall {
					id: call.id,
					name: call.function.name,
					arguments,
				})
			})
			.collect()
	});

	// Prefer usage reasoning tokens if present; fallback to estimation from reasoning_details
	let reasoning_tokens = openrouter_response
		.usage
		.completion_tokens_details
		.as_ref()
		.map(|d| d.reasoning_tokens)
		.filter(|v| *v > 0)
		.or_else(|| thinking.as_ref().map(|t| t.tokens))
		.unwrap_or(0);

	let input_tokens_raw = openrouter_response
		.usage
		.input_tokens
		.or(openrouter_response.usage.prompt_tokens)
		.unwrap_or(0);
	let output_tokens = openrouter_response
		.usage
		.completion_tokens
		.or(openrouter_response.usage.output_tokens)
		.unwrap_or(0);
	let cache_read_tokens = openrouter_response
		.usage
		.prompt_tokens_details
		.as_ref()
		.map(|d| d.cached_tokens)
		.unwrap_or(0);
	let total_tokens = openrouter_response
		.usage
		.total_tokens
		.unwrap_or(input_tokens_raw.saturating_add(output_tokens));
	let input_tokens_clean = input_tokens_raw.saturating_sub(cache_read_tokens);
	let (output_tokens, reasoning_tokens) = TokenUsage::split_output(output_tokens, reasoning_tokens);

	// Octolib semantic: input_tokens excludes cache reads
	let usage = TokenUsage {
		input_tokens: input_tokens_clean,
		cache_read_tokens,
		cache_write_tokens: 0,
		output_tokens,
		reasoning_tokens,
		total_tokens,
		cost: openrouter_response.usage.cost, // OpenRouter returns cost directly in usage object
		request_time_ms: Some(request_time_ms),
	};

	// Create response JSON and store tool_calls in unified format
	let mut response_json: serde_json::Value = shared::parse_response(&response.body, per_call_api_key)?;

	// Store tool_calls in unified GenericToolCall format for conversation history
	// Extract reasoning_details from response for Gemini thought signatures
	if let Some(ref tc) = tool_calls {
		let reasoning_details = choice.message.reasoning_details.clone();

		let reasoning_meta = reasoning_details.as_ref().map(|rd| {
			let mut meta_map = serde_json::Map::new();
			meta_map.insert("reasoning_details".to_string(), rd.clone());
			meta_map
		});
		shared::set_response_tool_calls(&mut response_json, tc, reasoning_meta.as_ref());
	}

	let exchange = ProviderExchange::new(request_body, response_json, Some(usage), "openrouter");

	// Try to parse structured output if it was requested
	let structured_output = shared::parse_structured_output_from_text(&content);

	Ok(ProviderResponse {
		content,
		thinking, // Add thinking from reasoning_details
		exchange,
		tool_calls,
		finish_reason: choice.finish_reason,
		structured_output,
		id: Some(openrouter_response.id),
	})
}

#[cfg(test)]
#[path = "openrouter_tests.rs"]
mod tests;
