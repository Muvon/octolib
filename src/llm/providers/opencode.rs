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

//! OpenCode Zen and OpenCode Go provider implementations.
//!
//! Both are multi-provider proxies from opencode.ai exposing OpenAI-compatible
//! endpoints:
//! - Zen (pay-as-you-go): `https://opencode.ai/zen/v1/chat/completions`
//! - Go (subscription): `https://opencode.ai/zen/go/v1/chat/completions`
//!
//! They serve cross-provider catalogues (Claude, GPT, Gemini, Grok, DeepSeek,
//! GLM, Kimi, Qwen, MiniMax…) with flat model IDs like `claude-opus-5`,
//! `gpt-5.5`, `kimi-k2.7-code`. Catalogues change over time, so any non-empty
//! model ID is accepted and validated by the API itself.
//!
//! Go is subscription-billed. Zen returns the actual request cost as a top-level
//! `cost` value rather than inside the OpenAI usage object.
//!
//! Tool calls work on both (verified live). Structured output: Go honors
//! json_schema `response_format` through the router (verified); on Zen it is
//! per-model — some upstreams reject json_schema — so capability resolution
//! stays on reference capabilities. Go reports `cached_tokens` from automatic
//! upstream prefix caching.
//!
//! Sources: <https://opencode.ai/docs/zen> and <https://opencode.ai/docs/go/>
//!
//! Configuration:
//! - `OPENCODE_API_KEY`: Required API key, shared by both providers (one key
//!   from opencode.ai/auth covers Zen and Go — matches the models.dev registry)
//! - `OPENCODE_ZEN_API_URL` / `OPENCODE_GO_API_URL`: Optional endpoint overrides

use crate::llm::providers::openai_compat::{
    chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderResponse, ReasoningEffort, SamplingSupport, TokenUsage,
};
use crate::llm::utils::contains_ignore_ascii_case;
use anyhow::Result;
use std::env;

const OPENCODE_API_KEY_ENV: &str = "OPENCODE_API_KEY";

const OPENCODE_ZEN_API_URL_ENV: &str = "OPENCODE_ZEN_API_URL";
const OPENCODE_ZEN_API_URL: &str = "https://opencode.ai/zen/v1/chat/completions";

const OPENCODE_GO_API_URL_ENV: &str = "OPENCODE_GO_API_URL";
const OPENCODE_GO_API_URL: &str = "https://opencode.ai/zen/go/v1/chat/completions";

fn get_opencode_api_key(provider_label: &str) -> Result<String> {
    env::var(OPENCODE_API_KEY_ENV).map_err(|_| {
        anyhow::anyhow!(
            "{} API key not found in environment variable: {}",
            provider_label,
            OPENCODE_API_KEY_ENV
        )
    })
}

/// Upstream sampling restrictions for models routed through the proxy.
///
/// The router forwards temperature/top_p verbatim to the upstream vendor
/// (verified live: Kimi K2.7/K3 reject them with "only 1 is allowed" /
/// "only 0.95 is allowed"), so mirror the per-vendor rules already encoded
/// in the sibling providers. Unknown families pass everything through.
fn sampling_support(model: &str) -> SamplingSupport {
    let model = model.to_ascii_lowercase();
    if model.starts_with("claude") {
        crate::llm::providers::anthropic::AnthropicProvider::new().supported_sampling_params(&model)
    } else if model.starts_with("gpt") {
        crate::llm::providers::openai::OpenAiProvider::new().supported_sampling_params(&model)
    } else if model.starts_with("kimi") {
        crate::llm::providers::moonshot::MoonshotProvider::new().supported_sampling_params(&model)
    } else {
        SamplingSupport::ALL
    }
}

/// Kimi reasoning-effort rules for the opencode routers.
///
/// The router forwards `reasoning_effort` verbatim to Moonshot, which accepts
/// the field only on `kimi-k3` and only as `"low"` / `"high"` / `"max"`
/// (verified live: Go returns 400 `Kimi request field reasoning_effort must
/// be one of: low, high, max` on any other value). Other Kimi models don't
/// support the field at all. Mirror the native moonshot provider's
/// `k3_reasoning_effort`: floor intermediate levels to the nearest supported
/// lower effort, drop the field elsewhere.
/// Source: platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model
fn adjust_reasoning_effort(
    model: &str,
    effort: Option<ReasoningEffort>,
) -> Option<ReasoningEffort> {
    if !model.to_ascii_lowercase().starts_with("kimi") {
        return effort;
    }
    if !contains_ignore_ascii_case(model, "kimi-k3") {
        return None;
    }
    match effort {
        Some(ReasoningEffort::Low) | Some(ReasoningEffort::Medium) => Some(ReasoningEffort::Low),
        Some(ReasoningEffort::High) | Some(ReasoningEffort::XHigh) => Some(ReasoningEffort::High),
        Some(ReasoningEffort::Max) => Some(ReasoningEffort::Max),
        None => None,
    }
}

/// Resolve a cloud-equivalent request cost for an OpenCode response.
///
/// Go is subscription-backed and currently returns zero in the OpenAI-shaped
/// usage cost fields. Zero is not a usable per-request price for a known paid
/// model, so fall back to the reference table in that case. Zen's top-level
/// `cost` remains authoritative because it is the actual pay-as-you-go charge.
fn resolve_opencode_cost(
    provider_name: &str,
    model: &str,
    zen_reported_cost: Option<f64>,
    usage: &TokenUsage,
) -> Option<f64> {
    let response_cost = if provider_name == "opencode-zen" {
        zen_reported_cost.or(usage.cost)
    } else {
        usage.cost.filter(|cost| cost.is_finite() && *cost > 0.0)
    };

    response_cost.or_else(|| {
        crate::llm::reference_models::get_reference_pricing(model).map(|pricing| {
            pricing.calculate_cost(
                usage.input_tokens,
                usage.cache_write_tokens,
                usage.cache_read_tokens,
                usage.billable_output_tokens(),
            )
        })
    })
}

/// Remove replay-only placeholders that Kimi rejects as invalid messages.
///
/// Some clients persist an empty final assistant record after a tool round.
/// It carries no conversation state, but OpenCode forwards it to Kimi, which
/// rejects the entire request. Keep every assistant message that carries text,
/// tool calls, media, or thinking; only a structurally empty placeholder is
/// omitted.
fn remove_empty_kimi_assistant_messages(model: &str, messages: &mut Vec<Message>) {
    if !model.to_ascii_lowercase().starts_with("kimi") {
        return;
    }

    messages.retain(|message| {
        let empty_tool_calls = message
            .tool_calls
            .as_ref()
            .is_none_or(|calls| calls.as_array().is_some_and(Vec::is_empty));
        let empty_images = message.images.as_ref().is_none_or(Vec::is_empty);
        let empty_videos = message.videos.as_ref().is_none_or(Vec::is_empty);
        let empty_thinking = message
            .thinking
            .as_ref()
            .is_none_or(|thinking| thinking.content.trim().is_empty());
        let structurally_empty = message.role == "assistant"
            && message.content.trim().is_empty()
            && empty_tool_calls
            && empty_images
            && empty_videos
            && empty_thinking;

        !structurally_empty
    });
}

async fn opencode_chat_completion(
    provider_name: &'static str,
    api_key: String,
    api_url: String,
    mut params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let model = params.model.clone();

    // Kimi models reject unsupported reasoning_effort values — normalize
    // before the generic openai_compat passthrough serializes the field.
    params.reasoning_effort = adjust_reasoning_effort(&model, params.reasoning_effort);
    remove_empty_kimi_assistant_messages(&model, &mut params.messages);

    let mut response = openai_compat_chat_completion(
        OpenAiCompatConfig {
            provider_name,
            usage_fallback_cost: None,
            use_response_cost: true,
            enforces_response_schema: true,
            supports_required_tool_choice: false,
        },
        sampling_support(&model),
        api_key,
        api_url,
        params,
    )
    .await?;

    let reported_cost = if provider_name == "opencode-zen" {
        response.exchange.response.get("cost").and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        })
    } else {
        None
    };

    if let Some(ref mut usage) = response.exchange.usage {
        usage.cost = resolve_opencode_cost(provider_name, &model, reported_cost, usage);
    }

    Ok(response)
}

/// OpenCode Zen provider (pay-as-you-go)
#[derive(Debug, Clone, Default)]
pub struct OpenCodeZenProvider;

impl OpenCodeZenProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AiProvider for OpenCodeZenProvider {
    fn name(&self) -> &str {
        "opencode-zen"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        sampling_support(model)
    }

    fn get_api_key(&self) -> Result<String> {
        get_opencode_api_key("OpenCode Zen")
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(OPENCODE_ZEN_API_URL_ENV, OPENCODE_ZEN_API_URL);
        opencode_chat_completion("opencode-zen", api_key, api_url, params).await
    }
}

/// OpenCode Go provider (subscription)
#[derive(Debug, Clone, Default)]
pub struct OpenCodeGoProvider;

impl OpenCodeGoProvider {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl AiProvider for OpenCodeGoProvider {
    fn name(&self) -> &str {
        "opencode-go"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn supports_caching(&self, _model: &str) -> bool {
        // Automatic upstream prompt-prefix caching — `cached_tokens` observed
        // in usage on repeated prefixes (no opt-in param).
        true
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // Verified live: json_schema response_format honored through the router.
        true
    }

    fn supported_sampling_params(&self, model: &str) -> SamplingSupport {
        sampling_support(model)
    }

    fn get_api_key(&self) -> Result<String> {
        get_opencode_api_key("OpenCode Go")
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(OPENCODE_GO_API_URL_ENV, OPENCODE_GO_API_URL);
        opencode_chat_completion("opencode-go", api_key, api_url, params).await
    }
}

#[cfg(test)]
#[path = "opencode_tests.rs"]
mod tests;
