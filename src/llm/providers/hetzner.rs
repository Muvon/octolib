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

//! Hetzner provider implementation.
//!
//! Uses Hetzner's Inference API (Experiments Platform), an OpenAI-compatible
//! endpoint at: `https://inference.hetzner.com/api/v1/chat/completions`
//!
//! Serves a small curated set of open-weight models (DeepSeek, GLM, Kimi,
//! Qwen). The catalogue is fixed and known (see `MODELS`), so unknown model
//! IDs are rejected — update the table when Hetzner adds models
//! (`GET /v1/models`).
//!
//! The API is **free of charge** while in experimental status, so cost is
//! reported as $0. Rate limits per API key: 4M input / 100k output tokens
//! and 10 requests per 60s.
//!
//! Source: <https://docs.hetzner.com/experiments/inference>
//!
//! Configuration:
//! - `HETZNER_API_KEY`: Required API key
//! - `HETZNER_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse};
use anyhow::Result;
use std::env;

/// Hetzner provider
#[derive(Debug, Clone, Default)]
pub struct HetznerProvider;

impl HetznerProvider {
    pub fn new() -> Self {
        Self
    }
}

const HETZNER_API_KEY_ENV: &str = "HETZNER_API_KEY";
const HETZNER_API_URL_ENV: &str = "HETZNER_API_URL";
const HETZNER_API_URL: &str = "https://inference.hetzner.com/api/v1/chat/completions";

/// (model id, vision, max input tokens) — from the Hetzner models table and
/// `GET /v1/models` (`max_model_len`).
const MODELS: &[(&str, bool, usize)] = &[
    ("DeepSeek-V4-Flash-0731", false, 512_000),
    ("GLM-5.2-NVFP4", false, 512_000),
    ("Kimi-K2.7-Code", true, 262_144),
    ("Qwen/Qwen3.6-35B-A3B-FP8", true, 262_144),
];

/// Case-insensitive lookup: the API itself rejects IDs with wrong case
/// ("model use not permitted"), so requests are canonicalized to the
/// table's exact ID.
fn find_model(model: &str) -> Option<&'static (&'static str, bool, usize)> {
    MODELS
        .iter()
        .find(|(id, _, _)| id.eq_ignore_ascii_case(model))
}

#[async_trait::async_trait]
impl AiProvider for HetznerProvider {
    fn name(&self) -> &str {
        "hetzner"
    }

    fn supports_model(&self, model: &str) -> bool {
        find_model(model).is_some()
    }

    fn supports_vision(&self, model: &str) -> bool {
        find_model(model)
            .map(|(_, vision, _)| *vision)
            .unwrap_or(false)
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        find_model(model).map(|(_, _, max)| *max).unwrap_or(262_144)
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(HETZNER_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Hetzner API key not found in environment variable: {}",
                HETZNER_API_KEY_ENV
            )
        })
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // Verified live: json_schema response_format is honored (vLLM guided
        // decoding) even though the Hetzner docs don't document it.
        true
    }

    fn enforces_response_schema(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, _model: &str) -> Option<ModelPricing> {
        // Free while in experimental status — no per-token billing.
        Some(ModelPricing {
            input_price_per_1m: 0.0,
            output_price_per_1m: 0.0,
            cache_write_price_per_1m: 0.0,
            cache_read_price_per_1m: 0.0,
        })
    }

    async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(HETZNER_API_URL_ENV, HETZNER_API_URL);

        if let Some((canonical, _, _)) = find_model(&params.model) {
            params.model = canonical.to_string();
        }
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "hetzner",
                usage_fallback_cost: None,
                use_response_cost: false,
                enforces_response_schema: true,
                supports_required_tool_choice: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        // Cost derives from the pricing table and the returned token counts,
        // so it stays correct when the (currently zero) prices change.
        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                if let Some(pricing) = self.get_model_pricing(&model) {
                    usage.cost = Some(pricing.calculate_cost(
                        usage.input_tokens,
                        usage.cache_write_tokens,
                        usage.cache_read_tokens,
                        usage.billable_output_tokens(),
                    ));
                }
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "hetzner_tests.rs"]
mod tests;
