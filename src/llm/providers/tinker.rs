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

//! Tinker provider implementation (Thinking Machines).
//!
//! Uses Tinker's OpenAI-compatible inference endpoint (beta) at:
//! `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions`
//!
//! Serves the Inkling family plus open-weight checkpoints (Nemotron, GLM,
//! Kimi, Qwen, GPT-OSS, DeepSeek) and any sampler checkpoint saved during
//! training (`tinker://...` paths). Model IDs contain colons
//! (`thinkingmachines/Inkling:peft:262144`); the factory splits only on the
//! first colon, so they pass through intact.
//! Short names `inkling` and `inkling-small` resolve to their full
//! `thinkingmachines/Inkling` and `thinkingmachines/Inkling-Small` IDs.
//! Sampling: temperature and top_p are forwarded; top_k is not supported
//! by this OpenAI-compatible adapter.
//!
//! Reasoning: the server separates chain-of-thought into `reasoning_content`
//! by default (`separate_reasoning` defaults to true since June 2026), which
//! the shared compat layer already parses. `reasoning_effort` accepts the
//! full OpenAI ladder including "xhigh" (0.99).
//!
//! The endpoint is a beta meant for testing and low-traffic use, not
//! production. Vision and structured output are not documented for it, so
//! both are reported as unsupported; caching is not claimed either — the
//! prefill-cache discount is priced in the table, but cached-token reporting
//! in responses is unverified.
//!
//! Source: <https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/>
//! Pricing: <https://tinker-docs.thinkingmachines.ai/tinker/models/>
//!
//! Configuration:
//! - `TINKER_API_KEY`: Required API key
//! - `TINKER_API_URL`: Optional endpoint override

use crate::llm::providers::openai_compat::{
    chat_completion_with_sampling as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, SamplingSupport};
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Tinker provider
#[derive(Debug, Clone, Default)]
pub struct TinkerProvider;

impl TinkerProvider {
    pub fn new() -> Self {
        Self
    }
}

const TINKER_API_KEY_ENV: &str = "TINKER_API_KEY";
const TINKER_API_URL_ENV: &str = "TINKER_API_URL";
const TINKER_API_URL: &str =
    "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions";

/// (model, input, output, cache_write, cache_read) per 1M tokens, from
/// <https://tinker-docs.thinkingmachines.ai/tinker/models/> (Sep 2026,
/// limited-time 50% discount prices — re-verify when the promotion ends).
/// Cached input is Tinker's prefill-cache read price; there is no separate
/// cache-write fee, so cache_write = input.
///
/// First match wins on substring matching, so longer IDs come first:
/// serverless (`:sampling-nvfp4`) before `:peft:` variants before base IDs,
/// and Inkling-Small before Inkling (the base Inkling ID is a substring of
/// every Inkling-Small ID).
const PRICING: &[PricingTuple] = &[
    // Serverless inference (beta)
    (
        "thinkingmachines/inkling-small:peft:262144:sampling-nvfp4",
        0.30,
        1.20,
        0.30,
        0.06,
    ),
    (
        "thinkingmachines/inkling:peft:262144:sampling-nvfp4",
        1.00,
        4.05,
        1.00,
        0.17,
    ),
    // Inkling family — 256K extended-context variants
    (
        "thinkingmachines/inkling-small:peft:262144",
        1.16,
        2.89,
        1.16,
        0.232,
    ),
    (
        "thinkingmachines/inkling:peft:262144",
        3.74,
        9.36,
        3.74,
        0.748,
    ),
    ("thinkingmachines/inkling-small", 0.58, 1.44, 0.58, 0.116),
    ("thinkingmachines/inkling", 1.87, 4.68, 1.87, 0.374),
    // NVIDIA Nemotron
    (
        "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16:peft:262144",
        0.26,
        0.66,
        0.26,
        0.052,
    ),
    (
        "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16",
        0.195,
        0.495,
        0.195,
        0.039,
    ),
    (
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16:peft:262144",
        3.32,
        8.30,
        3.32,
        0.664,
    ),
    (
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16",
        2.49,
        6.225,
        2.49,
        0.498,
    ),
    (
        "nvidia/nvidia-nemotron-3-super-120b-a12b-bf16:peft:262144",
        0.76,
        1.92,
        0.76,
        0.152,
    ),
    (
        "nvidia/nvidia-nemotron-3-super-120b-a12b-bf16",
        0.57,
        1.44,
        0.57,
        0.114,
    ),
    (
        "nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16",
        0.195,
        0.495,
        0.195,
        0.039,
    ),
    // GLM / Kimi
    ("zai-org/glm-5.3:peft:262144", 4.86, 12.15, 4.86, 0.972),
    ("moonshotai/kimi-k2.6:peft:131072", 5.15, 12.81, 5.15, 1.03),
    ("moonshotai/kimi-k2.6", 2.205, 5.49, 2.205, 0.441),
    // Qwen
    ("qwen/qwen3.8-27b:peft:262144", 2.48, 7.46, 2.48, 0.496),
    ("qwen/qwen3.8-27b", 1.86, 5.595, 1.86, 0.372),
    ("qwen/qwen3.6-35b-a3b", 0.54, 1.335, 0.54, 0.108),
    ("qwen/qwen3.6-27b", 1.86, 5.595, 1.86, 0.372),
    (
        "qwen/qwen3.5-397b-a17b:peft:262144",
        4.00,
        10.00,
        4.00,
        0.80,
    ),
    ("qwen/qwen3.5-397b-a17b", 3.00, 7.50, 3.00, 0.60),
    ("qwen/qwen3.5-35b-a3b-base", 0.54, 1.335, 0.54, 0.108),
    ("qwen/qwen3.5-9b-base", 0.66, 1.995, 0.66, 0.132),
    ("qwen/qwen3.5-9b", 0.66, 1.995, 0.66, 0.132),
    ("qwen/qwen3.5-4b", 0.33, 1.00, 0.33, 0.066),
    ("qwen/qwen3-8b", 0.195, 0.60, 0.195, 0.039),
    // OpenAI GPT-OSS
    ("openai/gpt-oss-120b:peft:131072", 0.78, 1.94, 0.78, 0.156),
    ("openai/gpt-oss-120b", 0.33, 0.84, 0.33, 0.066),
    ("openai/gpt-oss-20b", 0.18, 0.45, 0.18, 0.036),
    // DeepSeek
    ("deepseek-ai/deepseek-v3.1", 1.695, 4.215, 1.695, 0.339),
];

/// (model id, context window) from the same pricing page. Same
/// specific-first ordering as `PRICING`.
const CONTEXTS: &[(&str, usize)] = &[
    (
        "thinkingmachines/inkling-small:peft:262144:sampling-nvfp4",
        262_144,
    ),
    (
        "thinkingmachines/inkling:peft:262144:sampling-nvfp4",
        262_144,
    ),
    ("thinkingmachines/inkling-small:peft:262144", 262_144),
    ("thinkingmachines/inkling:peft:262144", 262_144),
    ("thinkingmachines/inkling-small", 65_536),
    ("thinkingmachines/inkling", 65_536),
    (
        "nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16:peft:262144",
        262_144,
    ),
    ("nvidia/nvidia-nemotron-3.5-lightning-30b-a3b-bf16", 65_536),
    (
        "nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16:peft:262144",
        262_144,
    ),
    ("nvidia/nvidia-nemotron-3-ultra-550b-a55b-bf16", 65_536),
    (
        "nvidia/nvidia-nemotron-3-super-120b-a12b-bf16:peft:262144",
        262_144,
    ),
    ("nvidia/nvidia-nemotron-3-super-120b-a12b-bf16", 65_536),
    ("nvidia/nvidia-nemotron-3-nano-30b-a3b-bf16", 65_536),
    ("zai-org/glm-5.3:peft:262144", 262_144),
    ("moonshotai/kimi-k2.6:peft:131072", 131_072),
    ("moonshotai/kimi-k2.6", 32_768),
    ("qwen/qwen3.8-27b:peft:262144", 262_144),
    ("qwen/qwen3.8-27b", 65_536),
    ("qwen/qwen3.6-35b-a3b", 65_536),
    ("qwen/qwen3.6-27b", 65_536),
    ("qwen/qwen3.5-397b-a17b:peft:262144", 262_144),
    ("qwen/qwen3.5-397b-a17b", 65_536),
    ("qwen/qwen3.5-35b-a3b-base", 65_536),
    ("qwen/qwen3.5-9b-base", 65_536),
    ("qwen/qwen3.5-9b", 65_536),
    ("qwen/qwen3.5-4b", 65_536),
    ("qwen/qwen3-8b", 32_768),
    ("openai/gpt-oss-120b:peft:131072", 131_072),
    ("openai/gpt-oss-120b", 32_768),
    ("openai/gpt-oss-20b", 32_768),
    ("deepseek-ai/deepseek-v3.1", 32_768),
];

fn resolve_model(model: &str) -> &str {
    if model.eq_ignore_ascii_case("inkling") {
        "thinkingmachines/Inkling"
    } else if model.eq_ignore_ascii_case("inkling-small") {
        "thinkingmachines/Inkling-Small"
    } else {
        // Preserve fully qualified IDs and case-sensitive checkpoint paths.
        model
    }
}

fn tinker_model_pricing(model: &str) -> Option<ModelPricing> {
    let (input, output, cache_write, cache_read) =
        get_model_pricing(resolve_model(model), PRICING)?;
    Some(ModelPricing::new(input, output, cache_write, cache_read))
}

fn tinker_model_context(model: &str) -> Option<usize> {
    let normalized = normalize_model_name(resolve_model(model));
    CONTEXTS
        .iter()
        .find(|(name, _)| normalized.contains(&normalize_model_name(name)))
        .map(|(_, context)| *context)
}

#[async_trait::async_trait]
impl AiProvider for TinkerProvider {
    fn name(&self) -> &str {
        "tinker"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Any non-empty ID: a known Tinker model or a sampler checkpoint
        // path (`tinker://...`) saved during training.
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(TINKER_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Tinker API key not found in environment variable: {}",
                TINKER_API_KEY_ENV
            )
        })
    }

    fn supports_vision(&self, _model: &str) -> bool {
        // Image input through the OpenAI-compat endpoint is not documented
        // (only the SDK renderers handle images). Don't advertise what we
        // can't verify.
        false
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // response_format is not documented for this endpoint.
        false
    }

    fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
        tinker_model_pricing(model)
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        tinker_model_context(model).unwrap_or(262_144)
    }

    fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
        SamplingSupport::TEMPERATURE_AND_TOP_P
    }

    async fn chat_completion(&self, mut params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(TINKER_API_URL_ENV, TINKER_API_URL);
        params.model = resolve_model(&params.model).to_string();
        let model = params.model.clone();

        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "tinker",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: false,
                supports_required_tool_choice: false,
            },
            self.supported_sampling_params(&model),
            api_key,
            api_url,
            params,
        )
        .await?;

        // Tinker does not report cost in the response — derive it from the
        // Tinker pricing table. Unknown checkpoint paths have no entry, so
        // their cost stays None rather than guessing a model family.
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
#[path = "tinker_tests.rs"]
mod tests;
