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
//! Neither reports per-token cost in usage (Go is subscription-billed; Zen
//! returns a top-level `cost` string not part of the OpenAI usage shape), so
//! cost estimates fall back to reference pricing for known models.
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
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
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

async fn opencode_chat_completion(
    provider_name: &'static str,
    api_key: String,
    api_url: String,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let model = params.model.clone();

    let mut response = openai_compat_chat_completion(
        OpenAiCompatConfig {
            provider_name,
            usage_fallback_cost: None,
            use_response_cost: true,
        },
        api_key,
        api_url,
        params,
    )
    .await?;

    // Derive cost from reference pricing if not returned in the response
    if let Some(ref mut usage) = response.exchange.usage {
        if usage.cost.is_none() {
            if let Some(pricing) = crate::llm::reference_models::get_reference_pricing(&model) {
                usage.cost = Some(pricing.calculate_cost(
                    usage.input_tokens,
                    usage.cache_write_tokens,
                    usage.cache_read_tokens,
                    usage.output_tokens,
                ));
            }
        }
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
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let zen = OpenCodeZenProvider::new();
        assert!(zen.supports_model("claude-opus-5"));
        assert!(zen.supports_model("gpt-5.5"));
        assert!(zen.supports_model("deepseek-v4-flash-free"));
        assert!(!zen.supports_model(""));

        let go = OpenCodeGoProvider::new();
        assert!(go.supports_model("kimi-k2.7-code"));
        assert!(go.supports_model("glm-5.2"));
        assert!(!go.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let zen = OpenCodeZenProvider::new();
        assert_eq!(zen.name(), "opencode-zen");
        assert!(!zen.supports_caching("any-model"));

        let go = OpenCodeGoProvider::new();
        assert_eq!(go.name(), "opencode-go");
        assert!(go.supports_caching("any-model"));
        assert!(go.supports_structured_output("any-model"));
    }
}
