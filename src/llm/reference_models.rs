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

//! Reference model properties for well-known models across providers.
//!
//! This module is the single source of truth for model-level facts that are
//! independent of a concrete provider route: capabilities, context windows,
//! and baseline cloud-equivalent pricing. Provider-specific pricing tables
//! still live with their providers when those rates are authoritative.

use crate::llm::types::ModelPricing;
use crate::llm::utils::{normalize_model_name, sanitize_model_name};

/// Capabilities for a well-known model, looked up from the reference table.
#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub vision: bool,
    pub video: bool,
    pub structured_output: bool,
    pub max_input_tokens: usize,
}

/// All known reference properties for a matching model pattern.
#[derive(Debug, Clone, Copy)]
pub struct ModelProperties {
    pub capability_pattern: Option<&'static str>,
    pub pricing_pattern: Option<&'static str>,
    pub capabilities: Option<ModelCapabilities>,
    pub pricing: Option<ModelPricing>,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceModelEntry {
    pattern: &'static str,
    capabilities: Option<ModelCapabilities>,
    pricing: Option<ModelPricing>,
}

const fn caps(
    vision: bool,
    video: bool,
    structured_output: bool,
    max_input_tokens: usize,
) -> Option<ModelCapabilities> {
    Some(ModelCapabilities {
        vision,
        video,
        structured_output,
        max_input_tokens,
    })
}

const fn pricing(
    input_price_per_1m: f64,
    output_price_per_1m: f64,
    cache_write_price_per_1m: f64,
    cache_read_price_per_1m: f64,
) -> Option<ModelPricing> {
    Some(ModelPricing {
        input_price_per_1m,
        output_price_per_1m,
        cache_write_price_per_1m,
        cache_read_price_per_1m,
    })
}

/// Unified reference table. Entries are sorted by pattern specificity so
/// substring matching resolves aliases such as `gpt-4o-mini` before `gpt-4o`.
const REFERENCE_MODELS: &[ReferenceModelEntry] = &[
    ReferenceModelEntry {
        pattern: "gpt-5.6-terra",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(2.50, 15.00, 3.125, 0.25),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-luna",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(1.00, 6.00, 1.25, 0.10),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-sol",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(5.00, 30.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "gpt-4o-mini-realtime-preview",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.60, 2.40, 0.60, 0.30),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.7-code-highspeed",
        capabilities: caps(true, true, true, 256_000),
        pricing: pricing(1.90, 8.00, 1.90, 0.38),
    },
    ReferenceModelEntry {
        pattern: "gpt-4o-realtime-preview",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(5.00, 20.00, 5.00, 2.50),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.7-highspeed",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.75, 0.06),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.5-highspeed",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.75, 0.03),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.5-lightning",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.75, 0.03),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.1-lightning",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.30, 2.40, 0.30, 0.03),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.6-code-preview",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-thinking-turbo",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(1.15, 8.00, 1.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gpt-3.5-turbo-16k-0613",
        capabilities: caps(false, false, true, 16_384),
        pricing: pricing(3.00, 4.00, 3.00, 3.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-3.5-turbo-instruct",
        capabilities: caps(false, false, true, 4_096),
        pricing: pricing(1.50, 2.00, 1.50, 1.50),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.1-flash-lite",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.25, 1.50, 0.25, 0.025),
    },
    ReferenceModelEntry {
        pattern: "gemini-2.5-flash-lite",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.10, 0.40, 0.10, 0.01),
    },
    ReferenceModelEntry {
        pattern: "o4-mini-deep-research",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.00, 4.00, 1.00, 0.25),
    },
    ReferenceModelEntry {
        pattern: "llama-3.2-90b-vision",
        capabilities: caps(true, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "llama-3.2-11b-vision",
        capabilities: caps(true, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "minimax-m3-highspeed",
        capabilities: caps(true, true, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.75, 0.06),
    },
    ReferenceModelEntry {
        pattern: "bytedance-seed-code",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "glm-4-32b-0414-128k",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.10, 0.10, 0.00, 0.01),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.3-chat-latest",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.75, 14.00, 1.75, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.2-chat-latest",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.75, 14.00, 1.75, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.1-chat-latest",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "dola-seed-2.0-lite",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.25, 2.00, 0.25, 0.05),
    },
    ReferenceModelEntry {
        pattern: "dola-seed-2.0-code",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-coder-32b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.1-codex-mini",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(0.25, 2.00, 0.25, 0.025),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-coder-flash",
        capabilities: None,
        pricing: pricing(0.30, 1.50, 0.30, 0.03),
    },
    ReferenceModelEntry {
        pattern: "dola-seed-2.0-pro",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-coder-480b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(2.00, 2.00, 2.00, 2.00),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v4-flash",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(0.14, 0.28, 0.14, 0.028),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-4-6",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-4-5",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "claude-3-7-sonnet",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "claude-3-5-sonnet",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.1-codex-max",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "gpt-5-chat-latest",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "codex-mini-latest",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.50, 6.00, 1.50, 0.375),
    },
    ReferenceModelEntry {
        pattern: "gpt-realtime-mini",
        capabilities: caps(false, false, true, 32_000),
        pricing: pricing(0.60, 2.40, 0.60, 0.06),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-coder-next",
        capabilities: None,
        pricing: pricing(0.11, 0.80, 0.11, 0.11),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-coder-plus",
        capabilities: None,
        pricing: pricing(1.00, 5.00, 1.00, 0.10),
    },
    ReferenceModelEntry {
        pattern: "llama-4-maverick",
        capabilities: caps(true, false, true, 1_048_576),
        pricing: pricing(0.17, 0.60, 0.17, 0.17),
    },
    ReferenceModelEntry {
        pattern: "mistral-medium-3",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.40, 2.00, 0.40, 0.10),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.5-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(1.50, 9.00, 1.50, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gemini-2.5-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.30, 2.50, 0.30, 0.03),
    },
    ReferenceModelEntry {
        pattern: "gemini-2.0-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.10, 0.40, 0.10, 0.025),
    },
    ReferenceModelEntry {
        pattern: "phi-4-multimodal",
        capabilities: caps(true, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-thinking",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "moonshot-v1-128k",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(2.00, 5.00, 2.00, 2.00),
    },
    ReferenceModelEntry {
        pattern: "claude-haiku-4-5",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(1.00, 5.00, 1.25, 0.10),
    },
    ReferenceModelEntry {
        pattern: "claude-3-5-haiku",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(0.80, 4.00, 1.00, 0.08),
    },
    ReferenceModelEntry {
        pattern: "gpt-realtime-1.5",
        capabilities: caps(false, false, true, 32_000),
        pricing: pricing(4.00, 16.00, 4.00, 0.40),
    },
    ReferenceModelEntry {
        pattern: "o3-deep-research",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(5.00, 20.00, 5.00, 1.25),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-vl-72b",
        capabilities: caps(true, true, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "deepseek-v4-pro",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(1.74, 3.48, 1.74, 0.145),
    },
    ReferenceModelEntry {
        pattern: "mistral-large-3",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.50, 1.50, 0.50, 0.125),
    },
    ReferenceModelEntry {
        pattern: "glm-4.6v-flashx",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(0.04, 0.40, 0.00, 0.004),
    },
    ReferenceModelEntry {
        pattern: "moonshot-v1-32k",
        capabilities: caps(false, false, true, 32_768),
        pricing: pricing(1.00, 3.00, 1.00, 1.00),
    },
    ReferenceModelEntry {
        pattern: "claude-mythos-5",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(10.00, 50.00, 12.50, 1.00),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-7",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-6",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-5",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-1",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(15.00, 75.00, 18.75, 1.50),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-4",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "claude-3-sonnet",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.3-instant",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(1.75, 14.00, 0.175, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-4.5-preview",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(75.00, 150.00, 75.00, 75.00),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-8",
        capabilities: None,
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-5",
        capabilities: None,
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        pattern: "seed-1-6-flash",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.075, 0.30, 0.075, 0.015),
    },
    ReferenceModelEntry {
        pattern: "glm-4-7-251222",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(0.60, 2.20, 0.60, 0.11),
    },
    ReferenceModelEntry {
        pattern: "llama-3.1-405b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(3.00, 3.00, 3.00, 3.00),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-vl-7b",
        capabilities: caps(true, true, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-vl-3b",
        capabilities: caps(true, true, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "mistral-medium",
        capabilities: caps(false, false, true, 32_768),
        pricing: pricing(2.70, 8.10, 2.70, 2.70),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.1-pro",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(2.00, 12.00, 2.00, 0.20),
    },
    ReferenceModelEntry {
        pattern: "gemini-3-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.50, 3.00, 0.50, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gemini-2.5-pro",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "glm-4.7-flashx",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(0.07, 0.40, 0.00, 0.01),
    },
    ReferenceModelEntry {
        pattern: "glm-4.6v-flash",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(0.00, 0.00, 0.00, 0.00),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.7-code",
        capabilities: caps(true, true, true, 256_000),
        pricing: pricing(0.95, 4.00, 0.95, 0.19),
    },
    ReferenceModelEntry {
        // Alias without the "k" (self-hosted / gateway deployments name it this
        // way) — the sanitizer can't bridge a real letter difference, so an
        // explicit twin keeps these calls priced.
        pattern: "kimi-2.7-code",
        capabilities: caps(true, true, true, 256_000),
        pricing: pricing(0.95, 4.00, 0.95, 0.19),
    },
    ReferenceModelEntry {
        pattern: "moonshot-v1-8k",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.20, 2.00, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "command-r-plus",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(2.50, 10.00, 2.50, 2.50),
    },
    ReferenceModelEntry {
        pattern: "claude-fable-5",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(10.00, 50.00, 12.50, 1.00),
    },
    ReferenceModelEntry {
        pattern: "claude-haiku-4",
        capabilities: caps(true, false, false, 200_000),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "claude-3-haiku",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(0.25, 1.25, 0.30, 0.03),
    },
    ReferenceModelEntry {
        pattern: "gpt-audio-mini",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.15, 0.60, 0.15, 0.015),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.6-flash",
        capabilities: None,
        pricing: pricing(0.25, 1.50, 0.25, 0.025),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-flash",
        capabilities: None,
        pricing: pricing(0.10, 0.40, 0.10, 0.01),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-vl-plus",
        capabilities: None,
        pricing: pricing(0.20, 1.60, 0.20, 0.02),
    },
    ReferenceModelEntry {
        pattern: "seed-2-0-code",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "seed-2-0-lite",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.25, 2.00, 0.25, 0.05),
    },
    ReferenceModelEntry {
        pattern: "seed-2-0-mini",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.10, 0.40, 0.10, 0.02),
    },
    ReferenceModelEntry {
        pattern: "llama-4-scout",
        capabilities: caps(true, false, true, 524_288),
        pricing: pricing(0.08, 0.30, 0.08, 0.08),
    },
    ReferenceModelEntry {
        pattern: "llama-3.3-70b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.60, 0.60, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "llama-3.1-70b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.60, 0.60, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-397b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.60, 3.60, 0.60, 0.35),
    },
    ReferenceModelEntry {
        pattern: "mistral-large",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(2.00, 6.00, 2.00, 2.00),
    },
    ReferenceModelEntry {
        pattern: "mistral-small",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.10, 0.30, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "mixtral-8x22b",
        capabilities: caps(false, false, true, 65_536),
        pricing: pricing(0.90, 0.90, 0.90, 0.90),
    },
    ReferenceModelEntry {
        pattern: "grok-4.1-fast",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.50, 0.20, 0.05),
    },
    ReferenceModelEntry {
        pattern: "glm-5.1-turbo",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.40, 4.40, 0.00, 0.26),
    },
    ReferenceModelEntry {
        pattern: "glm-4.7-flash",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(0.00, 0.00, 0.00, 0.00),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5-flash",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.00, 0.00, 0.00, 0.00),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-turbo",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(1.15, 8.00, 1.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(15.00, 75.00, 18.75, 1.50),
    },
    ReferenceModelEntry {
        pattern: "claude-3-opus",
        capabilities: caps(true, false, false, 200_000),
        pricing: pricing(15.00, 75.00, 18.75, 1.50),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.3-codex",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.75, 14.00, 1.75, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.2-codex",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.75, 14.00, 1.75, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.1-codex",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "gpt-audio-1.5",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(2.50, 10.00, 2.50, 0.25),
    },
    ReferenceModelEntry {
        pattern: "gpt-3.5-turbo",
        capabilities: caps(false, false, true, 16_385),
        pricing: pricing(0.50, 1.50, 0.50, 0.50),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.7-plus",
        capabilities: None,
        pricing: pricing(0.32, 1.28, 0.32, 0.03),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.6-plus",
        capabilities: None,
        pricing: pricing(0.50, 3.00, 0.50, 0.05),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-plus",
        capabilities: None,
        pricing: pricing(0.40, 2.40, 0.40, 0.04),
    },
    ReferenceModelEntry {
        pattern: "seed-2-0-pro",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "llama-3.2-3b",
        capabilities: caps(false, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "llama-3.2-1b",
        capabilities: caps(false, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "llama-3.1-8b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.7-max",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(1.25, 3.75, 1.25, 0.13),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-72b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.60, 0.60, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-32b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "mixtral-8x7b",
        capabilities: caps(false, false, true, 32_768),
        pricing: pricing(0.24, 0.24, 0.24, 0.24),
    },
    ReferenceModelEntry {
        pattern: "gemma-3n-e4b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.02, 0.04, 0.02, 0.02),
    },
    ReferenceModelEntry {
        pattern: "gemini-3-pro",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(2.00, 12.00, 2.00, 0.20),
    },
    ReferenceModelEntry {
        pattern: "glm-5v-turbo",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(1.20, 4.00, 0.00, 0.24),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5-airx",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(1.10, 4.50, 0.00, 0.22),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.7",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.30, 1.20, 0.375, 0.06),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.5",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.30, 1.20, 0.375, 0.03),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.1",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.27, 0.95, 0.27, 0.027),
    },
    ReferenceModelEntry {
        pattern: "phi-3-vision",
        capabilities: caps(true, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-0915",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-0905",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2-0711",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.4-mini",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(0.75, 4.50, 0.75, 0.075),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.4-nano",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(0.20, 1.25, 0.20, 0.02),
    },
    ReferenceModelEntry {
        pattern: "gpt-4.1-mini",
        capabilities: caps(false, false, true, 1_047_576),
        pricing: pricing(0.40, 1.60, 0.40, 0.10),
    },
    ReferenceModelEntry {
        pattern: "gpt-4.1-nano",
        capabilities: caps(false, false, true, 1_047_576),
        pricing: pricing(0.10, 0.40, 0.10, 0.025),
    },
    ReferenceModelEntry {
        pattern: "gpt-realtime",
        capabilities: caps(false, false, true, 32_000),
        pricing: pricing(4.00, 16.00, 4.00, 0.40),
    },
    ReferenceModelEntry {
        pattern: "gpt-oss-120b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.35, 0.75, 0.35, 0.35),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.6-max",
        capabilities: None,
        pricing: pricing(1.30, 7.80, 1.30, 0.13),
    },
    ReferenceModelEntry {
        pattern: "llama-3-70b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.60, 0.60, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-9b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.17, 0.25, 0.17, 0.17),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-235b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.60, 1.20, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-vl",
        capabilities: caps(true, true, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "qwen-2.5-7b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v4",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(0.14, 0.28, 0.14, 0.028),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v3",
        capabilities: caps(false, false, true, 65_536),
        pricing: pricing(0.28, 0.42, 0.28, 0.028),
    },
    ReferenceModelEntry {
        pattern: "deepseek-r1",
        capabilities: caps(false, false, true, 65_536),
        pricing: pricing(0.28, 0.42, 0.28, 0.028),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v2",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.14, 0.28, 0.14, 0.014),
    },
    ReferenceModelEntry {
        pattern: "grok-4-fast",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.50, 0.20, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gemma-4-31b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "gemma-4-26b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "gemma-4-e4b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gemma-4-e2b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.02, 0.02, 0.02, 0.02),
    },
    ReferenceModelEntry {
        pattern: "gemma-3-27b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "gemma-3-12b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "gemma-2-27b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "glm-5-turbo",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(1.20, 4.00, 0.00, 0.24),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5-air",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.20, 1.10, 0.00, 0.03),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.5-pro",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(30.00, 180.00, 30.00, 30.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.4-pro",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(30.00, 180.00, 30.00, 30.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.2-pro",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(21.00, 168.00, 21.00, 21.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-5-codex",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "gpt-4o-mini",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(0.15, 0.60, 0.15, 0.075),
    },
    ReferenceModelEntry {
        pattern: "gpt-4-turbo",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(10.00, 30.00, 10.00, 10.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-oss-20b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.03, 0.10, 0.03, 0.03),
    },
    ReferenceModelEntry {
        pattern: "llama-3-8b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-32b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        pattern: "qwen-2-72b",
        capabilities: caps(false, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "mistral-7b",
        capabilities: caps(false, false, false, 32_768),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gemma-3-4b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gemma-2-9b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "minimax-m3",
        capabilities: caps(true, true, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.75, 0.06),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.255, 1.00, 0.255, 0.0255),
    },
    ReferenceModelEntry {
        pattern: "gpt-5-mini",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(0.25, 2.00, 0.25, 0.025),
    },
    ReferenceModelEntry {
        pattern: "gpt-5-nano",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(0.05, 0.40, 0.05, 0.005),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-max",
        capabilities: None,
        pricing: pricing(1.20, 6.00, 1.20, 0.12),
    },
    ReferenceModelEntry {
        pattern: "qwen-flash",
        capabilities: None,
        pricing: pricing(0.05, 0.40, 0.05, 0.005),
    },
    ReferenceModelEntry {
        pattern: "qwen-turbo",
        capabilities: None,
        pricing: pricing(0.05, 0.20, 0.05, 0.005),
    },
    ReferenceModelEntry {
        pattern: "moondream",
        capabilities: caps(true, false, false, 8_192),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "qwen-3-8b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.05, 0.05, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "qwen-2-vl",
        capabilities: caps(true, true, true, 32_768),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "codestral",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.30, 0.90, 0.30, 0.30),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5-x",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(2.20, 8.90, 0.00, 0.45),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.6",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        // Alias twin of kimi-k2.6 (see kimi-2.7-code above).
        pattern: "kimi-2.6",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.5",
        capabilities: caps(true, false, true, 256_000),
        pricing: pricing(0.60, 3.00, 0.60, 0.10),
    },
    ReferenceModelEntry {
        pattern: "command-r",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.15, 0.60, 0.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gpt-5-pro",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(15.00, 120.00, 15.00, 15.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-audio",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(2.50, 10.00, 2.50, 2.50),
    },
    ReferenceModelEntry {
        pattern: "gpt-4-32k",
        capabilities: caps(false, false, true, 32_768),
        pricing: pricing(60.00, 120.00, 60.00, 60.00),
    },
    ReferenceModelEntry {
        pattern: "starcoder",
        capabilities: caps(false, false, false, 8_192),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "qwen-plus",
        capabilities: None,
        pricing: pricing(0.40, 1.20, 0.40, 0.04),
    },
    ReferenceModelEntry {
        pattern: "bakllava",
        capabilities: caps(true, false, false, 4_096),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "seed-1-8",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.25, 2.00, 0.25, 0.05),
    },
    ReferenceModelEntry {
        pattern: "seed-1-6",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.25, 2.00, 0.25, 0.05),
    },
    ReferenceModelEntry {
        pattern: "glm-4.6v",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(0.30, 0.90, 0.00, 0.05),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5v",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.60, 1.80, 0.00, 0.11),
    },
    ReferenceModelEntry {
        pattern: "qwen-max",
        capabilities: None,
        pricing: pricing(1.60, 6.40, 1.60, 0.16),
    },
    ReferenceModelEntry {
        pattern: "pixtral",
        capabilities: caps(true, false, true, 131_072),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "glm-5.2",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.40, 4.40, 0.00, 0.26),
    },
    ReferenceModelEntry {
        pattern: "glm-5.1",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.40, 4.40, 0.00, 0.26),
    },
    ReferenceModelEntry {
        pattern: "glm-4.7",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(0.60, 2.20, 0.60, 0.11),
    },
    ReferenceModelEntry {
        pattern: "glm-4.6",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.60, 2.20, 0.00, 0.11),
    },
    ReferenceModelEntry {
        pattern: "glm-4.5",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.60, 2.20, 0.00, 0.11),
    },
    ReferenceModelEntry {
        pattern: "glm-ocr",
        capabilities: caps(true, false, false, 128_000),
        pricing: pricing(0.03, 0.03, 0.00, 0.00),
    },
    ReferenceModelEntry {
        // Thinking Machines Inkling (open-weights, no first-party inference API;
        // served via partners — pricing from Together, verified July 17, 2026).
        pattern: "inkling",
        capabilities: caps(true, false, true, 524_288),
        pricing: pricing(1.00, 4.05, 0.00, 0.17),
    },
    ReferenceModelEntry {
        pattern: "kimi-k3",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(3.00, 15.00, 3.00, 0.30),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.60, 2.50, 0.60, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.5",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(5.00, 30.00, 5.00, 0.50),
    },
    ReferenceModelEntry {
        // OpenAI routes the bare alias to gpt-5.6-sol.
        pattern: "gpt-5.6",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(5.00, 30.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.4",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(2.50, 15.00, 2.50, 0.25),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.3",
        capabilities: caps(false, false, true, 400_000),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "gpt-5.2",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.75, 14.00, 1.75, 0.175),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.1",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "gpt-4.1",
        capabilities: caps(false, false, true, 1_047_576),
        pricing: pricing(2.00, 8.00, 2.00, 0.50),
    },
    ReferenceModelEntry {
        pattern: "o4-mini",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.10, 4.40, 1.10, 0.275),
    },
    ReferenceModelEntry {
        pattern: "o3-mini",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(1.10, 4.40, 1.10, 0.55),
    },
    ReferenceModelEntry {
        pattern: "o1-mini",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(1.10, 4.40, 1.10, 0.55),
    },
    ReferenceModelEntry {
        pattern: "codegen",
        capabilities: caps(false, false, false, 2_048),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "grok-4",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(3.00, 15.00, 3.00, 0.75),
    },
    ReferenceModelEntry {
        pattern: "grok-3",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(3.00, 15.00, 3.00, 0.75),
    },
    ReferenceModelEntry {
        pattern: "glm-5v",
        capabilities: caps(true, false, true, 128_000),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "m2-her",
        capabilities: caps(false, false, false, 128_000),
        pricing: pricing(0.30, 1.20, 0.0, 0.0),
    },
    ReferenceModelEntry {
        pattern: "gpt-4o",
        capabilities: caps(true, false, true, 128_000),
        pricing: pricing(2.50, 10.00, 2.50, 1.25),
    },
    ReferenceModelEntry {
        pattern: "o3-pro",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(20.00, 80.00, 20.00, 20.00),
    },
    ReferenceModelEntry {
        pattern: "o1-pro",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(150.00, 600.00, 150.00, 150.00),
    },
    ReferenceModelEntry {
        pattern: "llava",
        capabilities: caps(true, false, false, 4_096),
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "glm-5",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(1.00, 3.20, 0.00, 0.20),
    },
    ReferenceModelEntry {
        pattern: "glm-4",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.60, 2.20, 0.60, 0.06),
    },
    ReferenceModelEntry {
        pattern: "phi-4",
        capabilities: caps(false, false, true, 16_384),
        pricing: pricing(0.07, 0.14, 0.07, 0.07),
    },
    ReferenceModelEntry {
        pattern: "phi-3",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.05, 0.10, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "gpt-5",
        capabilities: caps(false, false, true, 400_000),
        pricing: pricing(1.25, 10.00, 1.25, 0.125),
    },
    ReferenceModelEntry {
        pattern: "gpt-4",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(30.00, 60.00, 30.00, 30.00),
    },
    ReferenceModelEntry {
        pattern: "dbrx",
        capabilities: caps(false, false, true, 32_768),
        pricing: pricing(0.75, 0.75, 0.75, 0.75),
    },
    ReferenceModelEntry {
        pattern: "o3",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(2.00, 8.00, 2.00, 0.50),
    },
    ReferenceModelEntry {
        pattern: "o1",
        capabilities: caps(false, false, true, 200_000),
        pricing: pricing(15.00, 60.00, 15.00, 7.50),
    },
];

fn matches_model(normalized: &str, pattern: &str) -> bool {
    let sanitized_pattern = sanitize_model_name(pattern);
    normalized.contains(&sanitized_pattern)
}

fn normalized_model(model: &str) -> String {
    sanitize_model_name(&normalize_model_name(model))
}

/// Look up all known reference properties for a model by fuzzy name matching.
pub fn get_reference_model_properties(model: &str) -> Option<ModelProperties> {
    let normalized = normalized_model(model);
    let capability_entry = REFERENCE_MODELS
        .iter()
        .find(|entry| entry.capabilities.is_some() && matches_model(&normalized, entry.pattern));
    let pricing_entry = REFERENCE_MODELS
        .iter()
        .find(|entry| entry.pricing.is_some() && matches_model(&normalized, entry.pattern));

    if capability_entry.is_none() && pricing_entry.is_none() {
        return None;
    }

    Some(ModelProperties {
        capability_pattern: capability_entry.map(|entry| entry.pattern),
        pricing_pattern: pricing_entry.map(|entry| entry.pattern),
        capabilities: capability_entry.and_then(|entry| entry.capabilities),
        pricing: pricing_entry.and_then(|entry| entry.pricing),
    })
}

/// Look up reference capabilities for a model by fuzzy name matching.
pub fn get_reference_capabilities(model: &str) -> Option<ModelCapabilities> {
    let normalized = normalized_model(model);
    REFERENCE_MODELS
        .iter()
        .find(|entry| entry.capabilities.is_some() && matches_model(&normalized, entry.pattern))
        .and_then(|entry| entry.capabilities)
}

/// Look up baseline cloud-equivalent pricing for a model by fuzzy name matching.
pub fn get_reference_pricing(model: &str) -> Option<ModelPricing> {
    let normalized = normalized_model(model);
    REFERENCE_MODELS
        .iter()
        .find(|entry| entry.pricing.is_some() && matches_model(&normalized, entry.pattern))
        .and_then(|entry| entry.pricing)
}

/// Calculate cost using reference pricing.
pub fn calculate_reference_cost(model: &str, input_tokens: u64, output_tokens: u64) -> Option<f64> {
    let pricing = get_reference_pricing(model)?;
    Some(pricing.calculate_cost(input_tokens, 0, 0, output_tokens))
}

/// Schema-enforcement policy for proxy/aggregator routes.
///
/// If the model is known and the reference table says it cannot produce
/// structured output, report false. Unknown proxy models stay optimistic:
/// the proxy is the only layer that can know its current route inventory.
pub fn proxy_route_enforces_response_schema(model: &str) -> bool {
    get_reference_capabilities(model)
        .map(|caps| caps.structured_output)
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same_capabilities(left: ModelCapabilities, right: ModelCapabilities) {
        assert_eq!(left.vision, right.vision);
        assert_eq!(left.video, right.video);
        assert_eq!(left.structured_output, right.structured_output);
        assert_eq!(left.max_input_tokens, right.max_input_tokens);
    }

    fn assert_same_pricing(left: ModelPricing, right: ModelPricing) {
        assert_eq!(left.input_price_per_1m, right.input_price_per_1m);
        assert_eq!(left.output_price_per_1m, right.output_price_per_1m);
        assert_eq!(
            left.cache_write_price_per_1m,
            right.cache_write_price_per_1m
        );
        assert_eq!(left.cache_read_price_per_1m, right.cache_read_price_per_1m);
    }

    #[test]
    fn unified_properties_can_return_capabilities_and_pricing() {
        let props = get_reference_model_properties("llama3.1:8b").unwrap();
        assert!(props.capabilities.unwrap().structured_output);
        assert_eq!(props.pricing.unwrap().input_price_per_1m, 0.10);
    }

    #[test]
    fn pricing_only_entries_do_not_imply_capabilities() {
        let props = get_reference_model_properties("qwen-plus-latest").unwrap();
        assert!(props.pricing.is_some());
        assert_eq!(props.pricing_pattern, Some("qwen-plus"));
        assert_eq!(props.capability_pattern, None);
        assert!(get_reference_capabilities("qwen-plus-latest").is_none());
    }

    #[test]
    fn unified_properties_merge_independent_best_matches() {
        let props = get_reference_model_properties("claude-opus-4-8").unwrap();
        assert_eq!(props.pricing_pattern, Some("claude-opus-4-8"));
        assert_eq!(props.capability_pattern, Some("claude-opus-4"));
        assert_eq!(
            props.capabilities.unwrap().max_input_tokens,
            get_reference_capabilities("claude-opus-4-8")
                .unwrap()
                .max_input_tokens
        );
        assert_eq!(
            props.pricing.unwrap().input_price_per_1m,
            get_reference_pricing("claude-opus-4-8")
                .unwrap()
                .input_price_per_1m
        );
    }

    #[test]
    fn proxy_policy_uses_known_structured_output_and_keeps_unknowns_optimistic() {
        assert!(proxy_route_enforces_response_schema("deepseek-v4-pro"));
        assert!(!proxy_route_enforces_response_schema("mistral-7b"));
        assert!(proxy_route_enforces_response_schema(
            "unknown/provider-model"
        ));
    }

    #[test]
    fn every_capability_entry_is_reachable() {
        for entry in REFERENCE_MODELS {
            if let Some(expected) = entry.capabilities {
                let actual = get_reference_capabilities(entry.pattern)
                    .unwrap_or_else(|| panic!("missing capabilities for {}", entry.pattern));
                assert_same_capabilities(actual, expected);

                let props = get_reference_model_properties(entry.pattern)
                    .unwrap_or_else(|| panic!("missing properties for {}", entry.pattern));
                assert_eq!(props.capability_pattern, Some(entry.pattern));
                assert_same_capabilities(props.capabilities.unwrap(), expected);
            }
        }
    }

    #[test]
    fn every_pricing_entry_is_reachable() {
        for entry in REFERENCE_MODELS {
            if let Some(expected) = entry.pricing {
                let actual = get_reference_pricing(entry.pattern)
                    .unwrap_or_else(|| panic!("missing pricing for {}", entry.pattern));
                assert_same_pricing(actual, expected);

                let props = get_reference_model_properties(entry.pattern)
                    .unwrap_or_else(|| panic!("missing properties for {}", entry.pattern));
                assert_eq!(props.pricing_pattern, Some(entry.pattern));
                assert_same_pricing(props.pricing.unwrap(), expected);
            }
        }
    }
}
