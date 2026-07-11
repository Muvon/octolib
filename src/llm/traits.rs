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

use crate::llm::types::{
    ChatCompletionParams, EffectiveSamplingParams, ProviderResponse, SamplingSupport,
};
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

/// Trait that all AI providers must implement
#[async_trait::async_trait]
pub trait AiProvider: Send + Sync {
    /// Get the provider name (e.g., "openrouter", "openai", "anthropic")
    fn name(&self) -> &str;

    /// Check if the provider supports the given model
    fn supports_model(&self, model: &str) -> bool;

    /// Send a chat completion request
    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse>;

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

    /// Compute effective sampling parameters by merging user-requested values
    /// with what the model actually supports.
    ///
    /// Returns `EffectiveSamplingParams` where supported parameters carry the user's value
    /// and unsupported parameters are `None` (to be omitted from API requests).
    fn effective_sampling_params(&self, params: &ChatCompletionParams) -> EffectiveSamplingParams {
        self.supported_sampling_params(&params.model).effective(
            params.temperature,
            params.top_p,
            params.top_k,
        )
    }

    /// Check if the provider/model supports caching
    fn supports_caching(&self, _model: &str) -> bool {
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
    ///   Minimax / Featherless / Octohub / Local / Ollama / Zai / CLI providers** ✗
    ///   — no caching primitive worth pinging. `None`.
    fn keepalive_policy(&self, _model: &str, _use_long_cache: bool) -> Option<KeepalivePolicy> {
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
