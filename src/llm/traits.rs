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

//! AI Provider trait definition

use crate::llm::types::{ChatCompletionParams, EffectiveSamplingParams, ProviderResponse, SamplingSupport, ToolChoice};
use anyhow::Result;
use std::time::Duration;

/// How a provider keeps its prompt cache warm during idle periods.
///
/// Returned from [`AiProvider::keepalive_policy`]. If `None`, the caller
/// should not attempt to ping — the provider either manages cache internally
/// (no client-controllable refresh) or has no cache primitive worth pinging.
///
/// Currently the only refresh strategy is "send the cached prefix again with
/// a minimal `max_tokens` value" — the act of reading the cache resets its
/// TTL on Anthropic-style providers. Other strategies (e.g. patching a Gemini
/// `cachedContent` resource) can be added later as enum variants.
#[derive(Debug, Clone)]
pub struct KeepalivePolicy {
	/// How often to fire the ping. Should be < cache TTL by a comfortable
	/// margin (typically TTL × 0.9) to absorb network latency and scheduler
	/// jitter. The caller respects this verbatim — providers must already
	/// account for their own TTL semantics.
	pub interval: Duration,
}

/// Share of a cache lifetime held back before the ping, for network latency
/// and scheduler jitter: one tenth.
const PING_MARGIN_DIVISOR: u32 = 10;

impl KeepalivePolicy {
	/// Ping one tenth of `lifetime` before the cache would expire.
	pub fn before_expiry(lifetime: Duration) -> Self {
		Self {
			interval: lifetime - lifetime / PING_MARGIN_DIVISOR,
		}
	}
}

/// Trait that all AI providers must implement
#[async_trait::async_trait]
pub trait AiProvider: Send + Sync {
	/// Get the provider name (e.g., "openrouter", "openai", "anthropic")
	fn name(&self) -> &str;

	/// Check if the provider supports the given model
	fn supports_model(&self, model: &str) -> bool;

	/// Send a chat completion request
	async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse>;

	/// Send a completion with an explicit tool selection policy.
	///
	/// Providers override this when their wire API supports forced tool choice.
	/// The default accepts `Auto` and rejects stronger policies explicitly.
	async fn chat_completion_with_tool_choice(
		&self,
		params: ChatCompletionParams,
		tool_choice: ToolChoice,
	) -> Result<ProviderResponse> {
		match tool_choice {
			ToolChoice::Auto => self.chat_completion(params).await,
			_ => Err(anyhow::anyhow!(
				"Provider '{}' does not support the requested tool choice",
				self.name()
			)),
		}
	}

	/// Get API key for this provider from environment variables
	/// Each provider should implement this to check their specific environment variable
	fn get_api_key(&self) -> Result<String>;

	/// Which sampling parameters this model supports.
	///
	/// Returns `SamplingSupport` — a boolean mask declaring which parameters the model accepts.
	/// Use `SamplingSupport::ALL` (default), `SamplingSupport::NONE`, or construct custom masks.
	///
	/// The `effective_sampling_params()` helper merges this with user-requested values.
	fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
		SamplingSupport::default()
	}

	/// Whether this provider can force at least one tool call for the model.
	/// Used by the schema fallback to distinguish hard tool selection from
	/// prompt-only guidance.
	fn supports_required_tool_choice(&self, _model: &str) -> bool {
		false
	}

	/// Compute effective sampling parameters by merging user-requested values
	/// with what the model actually supports.
	///
	/// Returns `EffectiveSamplingParams` where supported parameters carry the user's value
	/// and unsupported parameters are `None` (to be omitted from API requests).
	fn effective_sampling_params(&self, params: &ChatCompletionParams) -> EffectiveSamplingParams {
		self.supported_sampling_params(&params.model)
			.effective(params.temperature, params.top_p, params.top_k)
	}

	/// Check if the provider/model supports caching
	fn supports_caching(&self, _model: &str) -> bool {
		false
	}

	/// Whether this provider sends stored assistant `thinking` back to the
	/// model on later requests for `model`, so it occupies context the model
	/// reads and bills. Most providers drop historical reasoning on the wire;
	/// the ones that replay it (Z.AI Preserved Thinking, DeepSeek on tool
	/// requests, Kimi K2.6+/K3 on Moonshot and Ollama) say so here, so callers
	/// sizing a prompt can count reasoning only where it is actually sent.
	fn replays_thinking(&self, _model: &str) -> bool {
		false
	}

	/// How often (and whether) to ping this provider to keep its prompt cache warm.
	///
	/// `use_long_cache` mirrors octomind's `use_long_system_cache` flag — providers
	/// that support multiple TTLs (e.g. Anthropic's 5m vs 1h) use it to derive the
	/// correct ping interval. Default: `None` (do not ping).
	///
	/// Implementors must return `None` when:
	/// - the provider/model does not support caching at all,
	/// - the cache is fully managed server-side with no observable refresh primitive,
	/// - or pinging would cost more than letting the cache expire.
	///
	/// # Per-provider summary (verified Mar–May 2026)
	///
	/// - **Anthropic** ✓ — explicit `cache_control` ephemeral; TTL resets on read;
	///   5m default, 1h opt-in. Returns `Some`.
	/// - **OpenRouter** ✓ — passes `cache_control` through to upstream, working only
	///   for Anthropic/Claude routes. Returns `Some` when model is Claude-family.
	/// - **Octohub** ✓ — passes `cache_control` and `cache_prewarm` through to the
	///   lane serving the chain; the server pre-warms only lanes whose provider
	///   has a policy and answers the rest with a zero-read reply. Returns `Some`.
	/// - **Ollama** ✓ — automatic prefix cache lives as long as the model stays
	///   loaded; each finished request restarts the `keep_alive` timer, 5m default,
	///   1h opt-in. Returns `Some`.
	/// - **OpenAI** ✗ — GPT-5.6 supports explicit write breakpoints, but exposes
	///   no client-controlled refresh/keepalive primitive. A duplicate request
	///   can incur another cache-write charge rather than extending a TTL. `None`.
	/// - **Google Gemini** ✗ via chat completion — implicit cache only here; the
	///   explicit `cachedContent` resource lives behind a separate REST endpoint
	///   and is refreshed by `PATCH cachedContents/{id}` with a new `expireTime`,
	///   not by sending messages. Out of scope for the SendCachedPrefix model.
	///   `None`.
	/// - **Amazon Bedrock** ✗ today — current octolib impl uses the OpenAI-compat
	///   endpoint and does not yet inject `cache_control`. When that lands, swap
	///   to `Some` matching the Anthropic policy for Claude models.
	/// - **DeepSeek / Groq / Moonshot Kimi K2** ✗ — automatic disk/prefix caching
	///   with no client-controllable refresh primitive. `None`.
	/// - **Cloudflare / Together / Fireworks / Cerebras / Nvidia / BytePlus /
	///   Alibaba / Minimax / Featherless / Hetzner / OpenCode Zen & Go /
	///   Local / Zai / CLI providers** ✗
	///   — no caching primitive worth pinging. `None`.
	fn keepalive_policy(&self, _model: &str, _use_long_cache: bool) -> Option<KeepalivePolicy> {
		None
	}

	/// How long an idle prompt cache for `model` is known to survive: the time
	/// since its last read or write after which a request may no longer hit it.
	/// A lower bound, not a deadline — the entry may outlive it. `None` when no
	/// lifetime is known; callers then make no timing assumption.
	///
	/// `use_long_cache` selects the longer TTL on providers that offer one.
	///
	/// # Per-provider sources (checked 2026-11-07)
	///
	/// - **Anthropic, OpenRouter Claude routes** — documented TTL, reset on every
	///   read: 5m, or 1h with `use_long_cache`.
	/// - **OctoHub** — the TTL of the `cache_control` markers it passes to the lane.
	/// - **Ollama** — the `keep_alive` octolib sends: 5m or 1h.
	/// - **Moonshot** — measured 4m: docs say 5m, but most Kimi models missed at 5m.
	/// - **Alibaba** — measured per model: 30m on deepseek-v4.1-flash and qwen3.8-max,
	///   10m on glm-5.3 and qwen3.8-flash, `None` on qwen3.7-plus (about 60s); an assumed
	///   15m elsewhere.
	/// - **Google AI Studio** — measured 5m on gemini-4-argon, gemini-3.8-flash and
	///   gemini-3.6-flash; `None` on other Gemini models.
	/// - **Isoquant** — measured 60m on deepseek-v4.1-flash-1017-fast; `None` on
	///   glm-5.3-flash, which missed at 1m.
	/// - **Reflection** — measured 10m on Beam.
	/// - **Tinker** — measured: 60m idle still hit on gpt-oss, Inkling and
	///   Inkling-2-Small; the true limit is unknown.
	/// - **OpenAI GPT-5.6 and later** — documented "Minimum lifetime 30 minutes".
	/// - **Z.ai, Xiaomi** — measured: 60m idle still hit; the true limit is unknown.
	/// - **Everyone else** — `None`: docs state only "typically" figures or upper
	///   bounds (OpenAI before 5.6, Groq, DeepSeek, Together, Fireworks, Vertex),
	///   nothing at all, or misses appear within seconds (MiniMax) or minutes
	///   (Inception mercury-3).
	fn cache_lifetime(&self, _model: &str, _use_long_cache: bool) -> Option<Duration> {
		None
	}
	/// Get maximum input tokens for a model (actual context window size)
	/// This is what we can send to the API - the provider handles output limits internally
	fn get_max_input_tokens(&self, model: &str) -> usize {
		crate::llm::reference_models::get_reference_capabilities(model)
			.map(|c| c.max_input_tokens)
			.unwrap_or(262_144)
	}

	/// Check if the provider/model supports vision capabilities
	fn supports_vision(&self, model: &str) -> bool {
		crate::llm::reference_models::get_reference_capabilities(model)
			.map(|c| c.vision)
			.unwrap_or(false)
	}

	/// Check if the provider/model supports video capabilities
	fn supports_video(&self, model: &str) -> bool {
		crate::llm::reference_models::get_reference_capabilities(model)
			.map(|c| c.video)
			.unwrap_or(false)
	}

	/// Check if the provider supports structured output
	fn supports_structured_output(&self, model: &str) -> bool {
		crate::llm::reference_models::get_reference_capabilities(model)
			.map(|c| c.structured_output)
			.unwrap_or(false)
	}

	/// Whether the provider guarantees the response conforms to a supplied
	/// JSON schema (constrained/strict decoding), as opposed to merely
	/// emitting valid-but-arbitrary JSON (`json_object` mode).
	///
	/// Callers that deserialize the response into a fixed shape should route
	/// only schema-enforcing providers through the JSON path, and fall back
	/// to a tolerant format (e.g. tag-delimited text) for the rest — a
	/// non-enforcing provider can return valid JSON whose shape does not
	/// match the schema, which would fail typed deserialization.
	///
	/// Defaults to the provider's structured-output support. Providers or
	/// routes that only support JSON mode, or known proxy routes that ignore
	/// the supplied schema, must override this to return false.
	fn enforces_response_schema(&self, model: &str) -> bool {
		self.supports_structured_output(model)
	}

	/// Get pricing information for a model
	/// Returns None if pricing is not available or model is not recognized.
	/// Default falls back to reference pricing for well-known open models.
	fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
		crate::llm::reference_models::get_reference_pricing(model)
	}
}
