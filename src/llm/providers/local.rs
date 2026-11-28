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

//! Local provider for OpenAI-compatible local model servers.
//!
//! ## Supported Local Servers
//! - **Ollama**: Default `http://localhost:11434/v1/chat/completions`
//! - **LM Studio**: `http://localhost:1234/v1/chat/completions`
//! - **LocalAI**: `http://localhost:8080/v1/chat/completions`
//! - Any other OpenAI-compatible local server
//!
//! ## Configuration
//! - `LOCAL_API_URL`: API endpoint (default: Ollama local), sent as it is
//! - `LOCAL_API_KEY`: Optional API key
//! - `LOCAL_API_MODE`: `anthropic` or `openai`, the protocol `LOCAL_API_URL`
//!   speaks; with no URL, the Ollama endpoint of that protocol
//! - `LOCAL_ANTHROPIC_MODELS`: comma-separated model IDs that speak the Anthropic
//!   Messages protocol. The list is the strongest statement: a listed model
//!   speaks Messages whatever `LOCAL_API_MODE` or the URL's path say, to
//!   `LOCAL_API_URL` as it is, or to Ollama's Messages endpoint when no URL
//!   is set. Unlisted models follow the mode, then the URL's path, then Chat
//!   Completions.

use super::anthropic_compat::{self, parse_api_mode, AnthropicCompatConfig, ApiMode, Images};
use super::shared::CacheMarker;
use crate::llm::providers::openai_compat::{
	chat_completion as openai_compat_chat_completion, get_optional_api_key, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse, ReasoningEffort};
use anyhow::Result;
use serde_json::{json, Value};
use std::env;

/// Local provider for local model servers
#[derive(Debug, Clone)]
pub struct LocalProvider;

impl Default for LocalProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl LocalProvider {
	pub fn new() -> Self {
		Self
	}
}

pub(crate) const LOCAL_API_KEY_ENV: &str = "LOCAL_API_KEY";
const LOCAL_API_URL_ENV: &str = "LOCAL_API_URL";
const LOCAL_API_URL: &str = "http://localhost:11434/v1/chat/completions";
/// Ollama's Anthropic-compatible endpoint (docs.ollama.com/api/messages).
const LOCAL_MESSAGES_URL: &str = "http://localhost:11434/v1/messages";
const LOCAL_API_MODE_ENV: &str = "LOCAL_API_MODE";
const LOCAL_ANTHROPIC_MODELS_ENV: &str = "LOCAL_ANTHROPIC_MODELS";

/// Whether `allowlist` (`LOCAL_ANTHROPIC_MODELS`) names `model`. Entries are
/// matched whole, so `qwen3` does not opt in `qwen3:8b`.
fn is_anthropic_model(allowlist: &str, model: &str) -> bool {
	!model.is_empty() && allowlist.split(',').any(|entry| entry.trim() == model)
}

/// The protocol and URL for one request, with `LOCAL_ANTHROPIC_MODELS` read
/// from the environment.
fn select_local_endpoint(mode: Option<&str>, api_url: Option<&str>, model: &str) -> Result<(ApiMode, String)> {
	// An empty variable is an unset one.
	let mode = mode
		.filter(|raw| !raw.is_empty())
		.map(|raw| parse_api_mode(LOCAL_API_MODE_ENV, raw))
		.transpose()?;
	let api_url = api_url.filter(|api_url| !api_url.is_empty());
	// A listed model speaks Messages over the mode and over the URL's path.
	let mode = if is_anthropic_model(&env::var(LOCAL_ANTHROPIC_MODELS_ENV).unwrap_or_default(), model) {
		ApiMode::Anthropic
	} else {
		// A URL that names no endpoint was always served as Chat Completions.
		mode.or_else(|| api_url.and_then(anthropic_compat::url_api_mode))
			.unwrap_or(ApiMode::OpenAi)
	};
	let api_url = api_url.map_or_else(
		|| {
			match mode {
				ApiMode::Anthropic => LOCAL_MESSAGES_URL,
				ApiMode::OpenAi => LOCAL_API_URL,
			}
			.to_owned()
		},
		str::to_owned,
	);
	Ok((mode, api_url))
}

/// Ollama's Anthropic-compatible endpoint takes `thinking.type`
/// (`enabled`/`disabled`) and `output_config.effort`, whose values are
/// model-defined (https://docs.ollama.com/api/anthropic-compatibility). No
/// local server was available to probe it, so this follows the doc only.
/// `XHigh` and `Max` send `high`, the ladder the Chat route sends for `local`.
fn write_local_thinking(_model: &str, effort: Option<ReasoningEffort>, request: &mut Value) {
	let effort = match effort {
		None => return,
		Some(ReasoningEffort::None) => {
			request["thinking"] = json!({ "type": "disabled" });
			return;
		}
		Some(ReasoningEffort::Low) => "low",
		Some(ReasoningEffort::Medium) => "medium",
		Some(ReasoningEffort::High | ReasoningEffort::XHigh | ReasoningEffort::Max) => "high",
	};
	// Index assignment keeps an `output_config.format` already in the request.
	request["output_config"]["effort"] = json!(effort);
}

/// Local servers are not known in advance, so media follows what the model
/// tables say the model takes and the server is left to judge the rest.
const ANTHROPIC_COMPAT: AnthropicCompatConfig = AnthropicCompatConfig {
	provider_name: "local",
	images: |model| {
		if LocalProvider.supports_vision(model) {
			// Ollama's Messages endpoint takes inline data only: its docs list URL
			// images as "not supported" (docs.ollama.com/api/anthropic-compatibility)
			// and ollama.com/v1/messages answered `400 invalid image source type:
			// url. Only base64 images are supported.` live on 2026-11-28.
			Images::Base64
		} else {
			Images::None
		}
	},
	videos: |model| LocalProvider.supports_video(model),
	enforces_response_schema: false,
	thinking: write_local_thinking,
	cost: |model, usage| {
		crate::llm::reference_models::calculate_reference_cost(
			model,
			usage.input_tokens,
			usage.cache_read_tokens,
			usage.output_tokens,
		)
	},
	default_system: "",
	requires_max_tokens: true,
	cache_marker: Some(CacheMarker::Ephemeral),
};

#[async_trait::async_trait]
impl AiProvider for LocalProvider {
	fn name(&self) -> &str {
		"local"
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.is_empty()
	}

	fn get_api_key(&self) -> Result<String> {
		Ok(get_optional_api_key(LOCAL_API_KEY_ENV))
	}

	fn supports_caching(&self, _model: &str) -> bool {
		false
	}

	// supports_vision, supports_video, supports_structured_output, get_max_input_tokens
	// are resolved via reference capabilities (trait defaults)

	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		// Try reference pricing for cloud-equivalent cost estimation
		crate::llm::reference_models::get_reference_pricing(model)
	}

	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
		let api_key = super::shared::resolve_api_key(self, &params)?;
		let (mode, api_url) = select_local_endpoint(
			env::var(LOCAL_API_MODE_ENV).ok().as_deref(),
			env::var(LOCAL_API_URL_ENV).ok().as_deref(),
			&params.model,
		)?;
		if mode == ApiMode::Anthropic {
			return anthropic_compat::chat_completion(ANTHROPIC_COMPAT, api_key, api_url, params).await;
		}
		let model = params.model.clone();

		let mut response = openai_compat_chat_completion(
			OpenAiCompatConfig {
				provider_name: "local",
				usage_fallback_cost: None,
				use_response_cost: false,
				enforces_response_schema: false,
				supports_required_tool_choice: false,
				supports_parallel_tool_calls: true,
				cache_marker: None,
			},
			api_key,
			api_url,
			params,
		)
		.await?;

		// Fill cost from reference pricing for cloud-equivalent estimation
		if let Some(ref mut usage) = response.exchange.usage {
			if usage.cost.is_none() {
				usage.cost = crate::llm::reference_models::calculate_reference_cost(
					&model,
					usage.input_tokens,
					usage.cache_read_tokens,
					usage.billable_output_tokens(),
				);
			}
		}

		Ok(response)
	}
}

#[cfg(test)]
#[path = "local_http_tests.rs"]
mod http_tests;
#[cfg(test)]
#[path = "local_tests.rs"]
mod tests;
