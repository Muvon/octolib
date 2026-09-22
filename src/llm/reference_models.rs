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
        // Amazon Nova 2 Lite via US geo cross-region routing (`us.` model ID
        // prefix): bills 10% above the global tier per the AWS Price List API.
        pattern: "us.amazon.nova-2-lite",
        capabilities: caps(true, true, false, 1_000_000),
        pricing: pricing(0.33, 2.75, 0.00, 0.0825),
    },
    ReferenceModelEntry {
        // Amazon Nova 2 Lite (GA Dec 2025): 1M-context multimodal model on
        // Bedrock; text/image/video input, no structured outputs, free cache
        // writes with $0.075 cache reads (global cross-region tier; the `us.`
        // geo tier above is 10% higher).
        pattern: "nova-2-lite",
        capabilities: caps(true, true, false, 1_000_000),
        pricing: pricing(0.30, 2.50, 0.00, 0.075),
    },
    ReferenceModelEntry {
        // Amazon Nova 2 Pro (Preview): text/image/video/audio input on
        // Bedrock, global cross-region Standard tier (us-east-1). No context
        // window or cache-read rate is published for the preview.
        pattern: "nova-2-pro",
        capabilities: None,
        pricing: pricing(1.25, 10.00, 1.25, 1.25),
    },
    ReferenceModelEntry {
        // Amazon Nova 2 Omni (Preview): text-token rate; audio input bills
        // $1.00 and image output $40.00 per 1M. No published context window.
        pattern: "nova-2-omni",
        capabilities: None,
        pricing: pricing(0.30, 2.50, 0.30, 0.30),
    },
    ReferenceModelEntry {
        // Amazon Nova 2 Sonic: speech-to-speech model with 1M context; text
        // tokens bill at this rate, speech tokens at $3.00/$12.00 per 1M.
        pattern: "nova-2-sonic",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.33, 2.75, 0.33, 0.33),
    },
    ReferenceModelEntry {
        // Amazon Nova Premier: 1M-context multimodal reasoning model; legacy
        // lifecycle (EOL 2026-09-14) but still served and billable.
        pattern: "nova-premier",
        capabilities: caps(true, true, false, 1_000_000),
        pricing: pricing(2.50, 12.50, 0.00, 0.625),
    },
    ReferenceModelEntry {
        // Amazon Nova Pro: 300K-context multimodal model; no structured
        // outputs, free cache writes with $0.20 cache reads.
        pattern: "nova-pro",
        capabilities: caps(true, true, false, 300_000),
        pricing: pricing(0.80, 3.20, 0.00, 0.20),
    },
    ReferenceModelEntry {
        // Amazon Nova Lite: 300K-context multimodal model; no structured
        // outputs, free cache writes with $0.015 cache reads.
        pattern: "nova-lite",
        capabilities: caps(true, true, false, 300_000),
        pricing: pricing(0.06, 0.24, 0.00, 0.015),
    },
    ReferenceModelEntry {
        // Amazon Nova Micro: text-only 128K-context model; no structured
        // outputs, free cache writes with $0.00875 cache reads.
        pattern: "nova-micro",
        capabilities: caps(false, false, false, 128_000),
        pricing: pricing(0.035, 0.14, 0.00, 0.00875),
    },
    ReferenceModelEntry {
        // NVIDIA Nemotron 3 Nano Omni 30B A3B: text/image/video/audio input,
        // 256K context; only a free OpenRouter route exists, so no rate.
        pattern: "nemotron-3-nano-omni-30b-a3b",
        capabilities: caps(true, true, false, 256_000),
        pricing: None,
    },
    ReferenceModelEntry {
        // Bedrock spells Nemotron 3 Super as `nvidia.nemotron-super-3-120b`;
        // us-east-1 on-demand rate, 256K context.
        pattern: "nemotron-super-3-120b",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.15, 0.65, 0.15, 0.15),
    },
    ReferenceModelEntry {
        // NVIDIA Nemotron 3.5 Content Safety: text+image classifier, 131K context.
        pattern: "nemotron-3.5-content-safety",
        capabilities: caps(true, false, false, 131_072),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        // Bedrock Qwen3 Next 80B A3B (`qwen.qwen3-next-80b-a3b`): us-east-1
        // Standard tier, 256K context.
        pattern: "qwen3-next-80b-a3b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.15, 1.20, 0.15, 0.15),
    },
    ReferenceModelEntry {
        // Bedrock Qwen3 VL 235B A22B: text+image input, 256K context.
        pattern: "qwen3-vl-235b-a22b",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.53, 2.66, 0.53, 0.53),
    },
    ReferenceModelEntry {
        // Bedrock Qwen3 Coder 30B A3B: us-east-1 Standard tier, 256K context.
        pattern: "qwen3-coder-30b-a3b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.15, 0.60, 0.15, 0.15),
    },
    ReferenceModelEntry {
        // Mistral Small 4 (Mar 2026): 256K context, text+image input. The API
        // names it `mistral-small-2603`, so both spellings precede the generic
        // `mistral-small` row.
        pattern: "mistral-small-2603",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.15, 0.60, 0.15, 0.015),
    },
    ReferenceModelEntry {
        pattern: "mistral-small-4",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.15, 0.60, 0.15, 0.015),
    },
    ReferenceModelEntry {
        // Ministral 3 (Dec 2025) on Bedrock (`mistral.ministral-3-<size>-instruct`):
        // 128K context on that route, same flat rate as the Mistral API.
        pattern: "ministral-3-14b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.20, 0.20, 0.20, 0.20),
    },
    ReferenceModelEntry {
        pattern: "ministral-3-8b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.15, 0.15, 0.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "ministral-3-3b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.10, 0.10, 0.10, 0.10),
    },
    ReferenceModelEntry {
        // Ministral 3 on the Mistral API and OpenRouter (`ministral-14b-2512`):
        // 256K context, cached input at 10% of the flat rate.
        pattern: "ministral-14b",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.20, 0.20, 0.20, 0.02),
    },
    ReferenceModelEntry {
        pattern: "ministral-8b",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.15, 0.15, 0.15, 0.015),
    },
    ReferenceModelEntry {
        pattern: "ministral-3b",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.10, 0.10, 0.10, 0.01),
    },
    ReferenceModelEntry {
        // Bedrock Devstral 2 123B (`mistral.devstral-2-123b`): us-east-1
        // on-demand rate, 256K context.
        pattern: "devstral-2-123b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.40, 2.00, 0.40, 0.40),
    },
    ReferenceModelEntry {
        // Same Devstral 2 weights under the Mistral API / OpenRouter spelling
        // (`devstral-2512`), which also publishes a cache-read rate.
        pattern: "devstral-2512",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.40, 2.00, 0.40, 0.04),
    },
    ReferenceModelEntry {
        // Magistral Small 1.2 (`mistral.magistral-small-2509` on Bedrock):
        // text+image input, 128K context, no structured outputs.
        pattern: "magistral-small",
        capabilities: caps(true, false, false, 131_072),
        pricing: pricing(0.50, 1.50, 0.50, 0.50),
    },
    ReferenceModelEntry {
        // Bedrock gpt-oss-safeguard (`openai.gpt-oss-safeguard-120b`): safety
        // reasoning models, us-east-1 on-demand rate, 128K context. Structured
        // outputs are confirmed only for the 20B route.
        pattern: "gpt-oss-safeguard-120b",
        capabilities: caps(false, false, false, 131_072),
        pricing: pricing(0.15, 0.60, 0.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "gpt-oss-safeguard-20b",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.07, 0.20, 0.07, 0.07),
    },
    ReferenceModelEntry {
        // Writer Palmyra Vision 7B on Bedrock: text+image input, 4K context.
        pattern: "palmyra-vision-7b",
        capabilities: caps(true, false, false, 4_096),
        pricing: pricing(0.15, 0.60, 0.15, 0.15),
    },
    ReferenceModelEntry {
        // AI21 Jamba 1.5 Large on Bedrock (`ai21.jamba-1-5-large-v1:0`):
        // text-only, 256K context.
        pattern: "jamba-1-5-large",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(2.00, 8.00, 2.00, 2.00),
    },
    ReferenceModelEntry {
        // Xiaomi MiMo v2.6 Pro UltraSpeed bills 10x the Pro rate, so it must
        // precede it here.
        pattern: "mimo-v2.6-pro-ultraspeed",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(4.35, 8.70, 4.35, 0.036),
    },
    ReferenceModelEntry {
        // Xiaomi MiMo v2.6 Pro / Flash (Sep 2026): 1M context,
        // text/image/video/audio input on OpenRouter.
        pattern: "mimo-v2.6-pro",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.435, 0.87, 0.435, 0.0036),
    },
    ReferenceModelEntry {
        pattern: "mimo-v2.6-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.14, 0.28, 0.14, 0.0028),
    },
    ReferenceModelEntry {
        // Sakana Fugu Max / Ultra: 1M context, text+image input; Ultra v2
        // shares the Ultra rate card and resolves through the same row.
        pattern: "fugu-max",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(2.00, 6.00, 2.00, 0.25),
    },
    ReferenceModelEntry {
        pattern: "fugu-ultra",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(5.00, 30.00, 5.00, 0.50),
    },
    ReferenceModelEntry {
        // Tencent Hunyuan 3 preview is priced above the release, so it must
        // precede the bare `hy3` row.
        pattern: "hy3-preview",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.18, 0.60, 0.18, 0.06),
    },
    ReferenceModelEntry {
        // Tencent Hunyuan 3: 262K-context text model.
        pattern: "hy3",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.132, 0.528, 0.132, 0.033),
    },
    ReferenceModelEntry {
        // Tencent Hunyuan MT2 translation models: 8K context, text-only.
        pattern: "hy-mt2-1.8b",
        capabilities: caps(false, false, false, 8_192),
        pricing: pricing(0.044, 0.177, 0.044, 0.044),
    },
    ReferenceModelEntry {
        pattern: "hy-mt2-7b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.074, 0.295, 0.074, 0.074),
    },
    ReferenceModelEntry {
        pattern: "hy-mt2-30b-a3b",
        capabilities: caps(false, false, true, 8_192),
        pricing: pricing(0.074, 0.295, 0.074, 0.074),
    },
    ReferenceModelEntry {
        // Nex AGI Nex N2.5 Pro / Mini: 262K context, text+image input.
        pattern: "nex-n2.5-pro",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.075, 0.25, 0.075, 0.015),
    },
    ReferenceModelEntry {
        pattern: "nex-n2.5-mini",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.025, 0.10, 0.025, 0.0025),
    },
    ReferenceModelEntry {
        // StepFun Step 3.7 Flash: 262K context, text/image/video input.
        pattern: "step-3.7-flash",
        capabilities: caps(true, true, true, 262_144),
        pricing: pricing(0.20, 1.15, 0.20, 0.04),
    },
    ReferenceModelEntry {
        // StepFun Step 3.5 Flash: 262K-context text model, no structured outputs.
        pattern: "step-3.5-flash",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.10, 0.30, 0.10, 0.10),
    },
    ReferenceModelEntry {
        // AionLabs Aion 3.0 Mini is priced below the base model, so it must
        // precede it here.
        pattern: "aion-3.0-mini",
        capabilities: caps(false, false, false, 131_072),
        pricing: pricing(0.70, 1.40, 0.70, 0.18),
    },
    ReferenceModelEntry {
        // AionLabs Aion 3.0: 131K-context text model, no structured outputs.
        pattern: "aion-3.0",
        capabilities: caps(false, false, false, 131_072),
        pricing: pricing(3.00, 6.00, 3.00, 0.75),
    },
    ReferenceModelEntry {
        pattern: "aion-2.0",
        capabilities: caps(false, false, false, 131_072),
        pricing: pricing(0.80, 1.60, 0.80, 0.20),
    },
    ReferenceModelEntry {
        // Unbiased Pareto: vendor-qualified so the variable-priced
        // `openrouter/pareto-code` route does not inherit this rate.
        pattern: "unbiased/pareto",
        capabilities: caps(true, false, false, 262_144),
        pricing: pricing(2.50, 7.50, 2.50, 0.25),
    },
    ReferenceModelEntry {
        // Prism ML Ternary Bonsai 2 27B: 262K context, text+image input.
        pattern: "ternary-bonsai-2-27b",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.075, 0.50, 0.075, 0.075),
    },
    ReferenceModelEntry {
        // Inference.net Schematron v2 extraction models: 128K context.
        pattern: "schematron-v2-small",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.05, 0.23, 0.05, 0.05),
    },
    ReferenceModelEntry {
        pattern: "schematron-v2-turbo",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.03, 0.15, 0.03, 0.03),
    },
    ReferenceModelEntry {
        // Perceptron MK1: 32K context, text/image/video input.
        pattern: "perceptron-mk1",
        capabilities: caps(true, true, true, 32_768),
        pricing: pricing(0.15, 1.50, 0.15, 0.15),
    },
    ReferenceModelEntry {
        pattern: "nemotron-3.5-lightning-30b-a3b",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.05, 0.20, 0.05, 0.01),
    },
    ReferenceModelEntry {
        pattern: "nemotron-3-ultra-550b-a55b",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.60, 0.12),
    },
    ReferenceModelEntry {
        // NVIDIA Nemotron 3 Super 120B: 262K-context text model on the NVIDIA
        // API and OpenRouter; no structured outputs, no prompt caching.
        pattern: "nemotron-3-super-120b-a12b",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.085, 0.40, 0.085, 0.085),
    },
    ReferenceModelEntry {
        // Aggregator routes (OpenRouter, OpenCode Zen) expose Nemotron 3.5
        // Lightning under the short ID at 262K context and a higher rate than
        // the NVIDIA-hosted `-30b-a3b` route above.
        pattern: "nemotron-3.5-lightning",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.08, 0.20, 0.08, 0.04),
    },
    ReferenceModelEntry {
        // NVIDIA Nemotron 3 Nano 30B A3B on OpenRouter: 262K-context text
        // model. Sits after every more specific Nemotron row.
        pattern: "nemotron-3-nano-30b-a3b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.05, 0.20, 0.05, 0.03),
    },
    ReferenceModelEntry {
        // Contributor tiers trade training rights for a much lower rate, so
        // they must precede their base versions in this substring-matched table.
        pattern: "muse-spark-1.3-contributor",
        capabilities: caps(true, true, false, 1_048_576),
        pricing: pricing(0.10, 0.20, 0.10, 0.002),
    },
    ReferenceModelEntry {
        pattern: "muse-spark-1.2-contributor",
        capabilities: caps(true, true, false, 1_048_576),
        pricing: pricing(0.10, 0.20, 0.10, 0.002),
    },
    ReferenceModelEntry {
        // Meta Muse Spark 1.3 (Sep 2026): current flagship on the Meta Model
        // API, OpenCode Zen and OpenRouter; 1.1/1.2/1.3 share one rate card.
        pattern: "muse-spark-1.3",
        capabilities: caps(true, true, false, 1_048_576),
        pricing: pricing(1.25, 4.25, 1.25, 0.15),
    },
    ReferenceModelEntry {
        // Meta Muse Spark 1.2 (Aug 2026): closed flagship on the Meta Model API
        // and OpenRouter; text/image/video/audio input, 1M context.
        pattern: "muse-spark-1.2",
        capabilities: caps(true, true, false, 1_048_576),
        pricing: pricing(1.25, 4.25, 1.25, 0.15),
    },
    ReferenceModelEntry {
        pattern: "muse-spark-1.1",
        capabilities: caps(true, true, false, 1_048_576),
        pricing: pricing(1.25, 4.25, 1.25, 0.15),
    },
    ReferenceModelEntry {
        // Meta Muse Glimmer 30B (Aug 2026): open-weight agentic model hosted on
        // Together and OpenRouter; text+image input, 128K context.
        pattern: "muse-glimmer",
        capabilities: caps(true, false, false, 131_072),
        pricing: pricing(0.30, 1.10, 0.30, 0.04),
    },
    ReferenceModelEntry {
        // Fireworks' Qwen 3.8 serverless path is text-only with 262K context.
        pattern: "qwen3p8-2p4t-a95b",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(2.00, 6.00, 2.00, 0.25),
    },
    ReferenceModelEntry {
        // Fireworks' licensed Qwen 3.7 Plus route accepts text and images.
        pattern: "qwen3p7-plus",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.40, 1.60, 0.40, 0.08),
    },
    ReferenceModelEntry {
        // Same Qwen 3.8 2.4T weights on aggregator routes, which spell the ID
        // with dots (`qwen/qwen3.8-2.4t-a95b`) and serve 1M context.
        pattern: "qwen-3.8-2.4t",
        capabilities: caps(false, false, true, 1_048_576),
        pricing: pricing(2.00, 6.00, 2.00, 0.25),
    },
    ReferenceModelEntry {
        // Qwen 3.6 35B A3B: open-weight multimodal MoE served free by Hetzner
        // and priced on OpenRouter; 262K context, text/image/video input.
        pattern: "qwen-3.6-35b-a3b",
        capabilities: caps(true, true, true, 262_144),
        pricing: pricing(0.10, 0.90, 0.10, 0.05),
    },
    ReferenceModelEntry {
        // Inception Mercury 2.5 (Sep 2026): diffusion LLM, 260K context.
        pattern: "mercury-2.5",
        capabilities: caps(false, false, true, 260_000),
        pricing: pricing(0.04, 0.15, 0.04, 0.004),
    },
    ReferenceModelEntry {
        // Inception Mercury 2 (Mar 2026): diffusion LLM, 128K context.
        pattern: "mercury-2",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.25, 0.75, 0.25, 0.025),
    },
    ReferenceModelEntry {
        // Tencent Hunyuan 4 preview (Aug 2026): 1M-context text model.
        pattern: "hy4-preview",
        capabilities: caps(false, false, true, 1_048_576),
        pricing: pricing(0.834, 2.501, 0.834, 0.042),
    },
    ReferenceModelEntry {
        // IBM Granite 4.2 8B (Aug 2026): 131K-context text model.
        pattern: "granite-4.2",
        capabilities: caps(false, false, true, 131_072),
        pricing: pricing(0.06, 0.25, 0.06, 0.015),
    },
    ReferenceModelEntry {
        // InclusionAI Ling 3.0 Flash finance variant — priced above the base
        // model, so it must precede it here.
        pattern: "ling-3.0-flash-fin",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.06, 0.18, 0.06, 0.012),
    },
    ReferenceModelEntry {
        // InclusionAI Ling 3.0 Flash: 262K-context MoE, no structured outputs.
        pattern: "ling-3.0-flash",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.021, 0.063, 0.021, 0.0042),
    },
    ReferenceModelEntry {
        // Sakana Namazu (Aug 2026): 262K-context model with image/file input.
        pattern: "sakana-namazu",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.95, 4.00, 0.95, 0.15),
    },
    ReferenceModelEntry {
        // Upstage Solar Pro 4 (Aug 2026): 512K-context text model.
        pattern: "solar-pro4",
        capabilities: caps(false, false, true, 524_288),
        pricing: pricing(0.03, 0.12, 0.03, 0.006),
    },
    ReferenceModelEntry {
        // Poolside Laguna S 2.1 (Jul 2026): 1M-context coding model.
        pattern: "laguna-s-2.1",
        capabilities: caps(false, false, false, 1_048_576),
        pricing: pricing(0.09, 0.18, 0.09, 0.009),
    },
    ReferenceModelEntry {
        pattern: "laguna-xs-2.1",
        capabilities: caps(false, false, false, 262_144),
        pricing: pricing(0.06, 0.12, 0.06, 0.03),
    },
    ReferenceModelEntry {
        // Meituan LongCat 2.0 (Jul 2026): 1M-context text model.
        pattern: "longcat-2.0",
        capabilities: caps(false, false, false, 1_048_576),
        pricing: pricing(0.30, 1.20, 0.30, 0.006),
    },
    ReferenceModelEntry {
        // Kwaipilot KAT Coder Pro v2.5 (Jul 2026): 262K-context coding model.
        pattern: "kat-coder-pro-v2.5",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(0.74, 2.96, 0.74, 0.15),
    },
    ReferenceModelEntry {
        // Xiaomi MiMo v2.5 Pro is text-only and priced above the multimodal
        // base model, so it must precede it here.
        pattern: "mimo-v2.5-pro",
        capabilities: caps(false, false, true, 1_050_000),
        pricing: pricing(0.435, 0.87, 0.435, 0.0036),
    },
    ReferenceModelEntry {
        // Xiaomi MiMo v2.5: 1.05M context, text/image/video/audio input;
        // served free on OpenCode Zen and priced on OpenRouter.
        pattern: "mimo-v2.5",
        capabilities: caps(true, true, true, 1_050_000),
        pricing: pricing(0.14, 0.28, 0.14, 0.0028),
    },
    ReferenceModelEntry {
        // OpenAI GPT-6 Astra: 1.05M-context flagship, text/image input.
        pattern: "gpt-6-astra",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(10.00, 50.00, 12.50, 1.00),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-terra",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(2.00, 12.00, 2.50, 0.20),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-luna",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(0.20, 1.20, 0.25, 0.02),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-sol",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(4.00, 20.00, 5.00, 0.40),
    },
    ReferenceModelEntry {
        pattern: "gpt-5.6-cyber",
        capabilities: caps(true, false, true, 400_000),
        pricing: pricing(12.50, 75.00, 15.625, 1.25),
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
        pricing: pricing(0.60, 2.40, 0.375, 0.06),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.5-highspeed",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.375, 0.03),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.5-lightning",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.375, 0.03),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2.1-lightning",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.60, 2.40, 0.375, 0.03),
    },
    ReferenceModelEntry {
        pattern: "kimi-k2.6-code-preview",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.95, 4.00, 0.95, 0.16),
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
        pattern: "gemini-3.1-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.50, 3.00, 0.50, 0.05),
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
        pricing: pricing(0.30, 1.20, 0.0, 0.06),
    },
    ReferenceModelEntry {
        pattern: "bytedance-seed-code",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.10),
    },
    ReferenceModelEntry {
        pattern: "glm-4-32b-0414-128k",
        capabilities: caps(false, false, true, 128_000),
        pricing: pricing(0.10, 0.10, 0.00, 0.00),
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
        // DeepSeek-V4.1-Flash, served as `deepseek-flash` (2026-09-10):
        // peak rates as the static baseline — off-peak is half.
        pattern: "deepseek-flash",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(0.3, 1.2, 0.3, 0.006),
    },
    ReferenceModelEntry {
        // The same V4.1-Flash weights under their model-card name, which is how
        // ollama (`deepseek-v4.1-flash:cloud`) and Alibaba spell it. It must sit
        // ahead of the generic `deepseek-v4` row: that spelling contains it too,
        // and would bill V4.1 at V4 Flash's 0.44/1.32 as a text-only model.
        pattern: "deepseek-v4.1-flash",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(0.3, 1.2, 0.3, 0.006),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v4-flash-vision-exp",
        capabilities: caps(true, false, true, 1_000_000),
        // Experimental multimodal route (2026-08-21): text-identical to
        // v4-flash and billed at the same peak baseline.
        pricing: pricing(0.44, 1.32, 0.44, 0.014),
    },
    ReferenceModelEntry {
        pattern: "deepseek-v4-flash",
        capabilities: caps(false, false, true, 1_000_000),
        // 2026-08-16 peak/off-peak revision (matches providers/deepseek.rs):
        // peak rates as the static baseline — off-peak is half, so peak never
        // undercounts spend through the ollama/together lanes.
        pricing: pricing(0.44, 1.32, 0.44, 0.014),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-4-6",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(3.00, 15.00, 3.75, 0.30),
    },
    ReferenceModelEntry {
        // 1M context, but unlike 4.6+ the >200K tier is billed at a premium
        // ($6/$22.50) that a single pricing row cannot express.
        pattern: "claude-sonnet-4-5",
        capabilities: caps(true, false, false, 1_000_000),
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
        pattern: "gemini-3.8-flash",
        capabilities: caps(true, true, true, 1_048_576),
        // Introductory pricing through Dec 31, 2026 (matches providers/google_vertex.rs)
        pricing: pricing(0.75, 3.75, 0.75, 0.075),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.7-flash",
        capabilities: caps(true, true, true, 1_048_576),
        // Introductory pricing through Dec 31, 2026 (matches providers/google_vertex.rs)
        pricing: pricing(0.75, 3.75, 0.75, 0.075),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.6-flash",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.75, 3.75, 0.75, 0.075),
    },
    ReferenceModelEntry {
        pattern: "gemini-3.5-flash-lite",
        capabilities: caps(true, true, true, 1_048_576),
        pricing: pricing(0.30, 2.50, 0.30, 0.03),
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
        pattern: "gpt-realtime-2.1-mini",
        capabilities: caps(false, false, true, 32_000),
        pricing: pricing(0.60, 2.40, 0.60, 0.06),
    },
    ReferenceModelEntry {
        pattern: "gpt-realtime-2.1",
        capabilities: caps(false, false, true, 32_000),
        pricing: pricing(4.00, 24.00, 4.00, 0.40),
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
        // 2026-08-16 peak/off-peak revision (matches providers/deepseek.rs) —
        // peak baseline; off-peak is half.
        pricing: pricing(1.32, 3.96, 1.32, 0.044),
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
        // Cache hits on the 5.1 pair are 0.025x input ($0.25), not the 0.1x
        // every other Claude uses.
        pattern: "claude-mythos-5-1",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(10.00, 50.00, 12.50, 0.25),
    },
    ReferenceModelEntry {
        pattern: "claude-mythos-5",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(10.00, 50.00, 12.50, 1.00),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-5",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-7",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-opus-4-6",
        capabilities: caps(true, false, false, 1_000_000),
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
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(5.00, 25.00, 6.25, 0.50),
    },
    ReferenceModelEntry {
        pattern: "claude-sonnet-5",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(2.00, 10.00, 2.50, 0.20),
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
        // Cache hits on the 5.1 pair are 0.025x input ($0.25), not the 0.1x
        // every other Claude uses.
        pattern: "claude-fable-5-1",
        capabilities: caps(true, false, false, 1_000_000),
        pricing: pricing(10.00, 50.00, 12.50, 0.25),
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
        // Structured output verified against the live openrouter route
        // (2026-08-23): the catalogue advertises structured_outputs=false for
        // this model, but a strict json_schema request is honoured. The probe,
        // not the catalogue flag, decides — an entry missing here falls back to
        // "enforces", which would be right by luck rather than by measurement.
        pattern: "qwen-3.7-flash",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.03, 0.13, 0.03, 0.003),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.6-flash",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.25, 1.50, 0.25, 0.025),
    },
    ReferenceModelEntry {
        // Groq preview route: image/text input, JSON object mode, 131K context.
        pattern: "qwen-3.6-27b",
        capabilities: caps(true, false, true, 131_072),
        pricing: pricing(0.60, 3.00, 0.60, 0.60),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-flash",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.10, 0.40, 0.10, 0.01),
    },
    ReferenceModelEntry {
        pattern: "qwen-3-vl-plus",
        capabilities: caps(true, true, true, 262_144),
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
        pattern: "grok-4.20-multi-agent",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-4.20",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-build-latest",
        capabilities: caps(true, false, true, 500_000),
        pricing: pricing(2.00, 6.00, 2.00, 0.30),
    },
    ReferenceModelEntry {
        pattern: "grok-build-0.1",
        capabilities: caps(true, false, true, 256_000),
        pricing: pricing(1.00, 2.00, 1.00, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-code-fast",
        capabilities: caps(true, false, true, 256_000),
        pricing: pricing(1.00, 2.00, 1.00, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-4.5",
        capabilities: caps(true, false, true, 500_000),
        pricing: pricing(2.00, 6.00, 2.00, 0.30),
    },
    ReferenceModelEntry {
        pattern: "grok-4.7",
        capabilities: caps(true, false, true, 500_000),
        pricing: pricing(2.00, 6.00, 2.00, 0.50),
    },
    ReferenceModelEntry {
        pattern: "grok-4.6",
        capabilities: caps(true, false, true, 500_000),
        pricing: pricing(2.00, 6.00, 2.00, 0.50),
    },
    ReferenceModelEntry {
        pattern: "grok-4.3",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-latest",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-4.1-fast",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
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
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.32, 1.28, 0.32, 0.03),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.6-plus",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.50, 3.00, 0.50, 0.05),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.5-plus",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.40, 2.40, 0.40, 0.04),
    },
    ReferenceModelEntry {
        // Seed 2.1 Turbo (Aug 2026): multimodal, 262K context; no cache rates
        // published, so cache columns mirror the input price.
        pattern: "seed-2-1-turbo",
        capabilities: caps(true, false, true, 262_144),
        pricing: pricing(0.50, 2.50, 0.50, 0.50),
    },
    ReferenceModelEntry {
        // Seed 2.1 Pro: ByteDance publishes no USD card — ¥6/¥30/¥1.2 cache-hit
        // converted at ~7.1 CNY/USD. Context unpublished; 256K carried from
        // Seed 2.0 Pro by trackers.
        pattern: "seed-2-1-pro",
        capabilities: caps(true, false, true, 256_000),
        pricing: pricing(0.85, 4.15, 0.85, 0.17),
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
        // Qwen3.8-Flash (Aug 2026): multimodal MoE (125B total / 6B active),
        // 1M context, structured output honoured on the OpenRouter route.
        // Baseline = Model Studio list price; implicit cache hits 20% of input.
        pattern: "qwen-3.8-flash",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.113, 0.382, 0.113, 0.0226),
    },
    ReferenceModelEntry {
        // Qwen3.8-27B (Aug 2026 open weights): dense vision-language, 262K
        // native context (Groq serves 131K). Baseline from OpenRouter.
        pattern: "qwen-3.8-27b",
        capabilities: caps(true, true, true, 262_144),
        pricing: pricing(0.35, 2.75, 0.35, 0.35),
    },
    ReferenceModelEntry {
        pattern: "qwen-3.8-max",
        capabilities: caps(true, true, true, 1_000_000),
        // Priced by providers/alibaba.rs — no third-party host yet.
        pricing: None,
    },
    ReferenceModelEntry {
        pattern: "qwen-3.7-max",
        capabilities: caps(false, false, true, 262_144),
        pricing: pricing(2.50, 7.50, 2.50, 0.25),
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
        // Generic fallback rides flash peak rates (off-peak is half).
        pricing: pricing(0.44, 1.32, 0.44, 0.014),
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
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
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
        capabilities: caps(true, false, true, 1_050_000),
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
        pricing: pricing(0.30, 1.20, 0.0, 0.06),
    },
    ReferenceModelEntry {
        pattern: "minimax-m2",
        capabilities: caps(false, false, false, 1_000_000),
        pricing: pricing(0.30, 1.20, 0.375, 0.03),
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
        pricing: pricing(0.95, 4.00, 0.95, 0.16),
    },
    ReferenceModelEntry {
        // Alias twin of kimi-k2.6 (see kimi-2.7-code above).
        pattern: "kimi-2.6",
        capabilities: caps(false, false, true, 256_000),
        pricing: pricing(0.95, 4.00, 0.95, 0.16),
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
        // Native image/video input and 1M context. Must precede glm-5.3.
        pattern: "glm-5.3-flash",
        capabilities: caps(true, true, true, 1_000_000),
        pricing: pricing(0.15, 0.50, 0.00, 0.03),
    },
    ReferenceModelEntry {
        // GLM-5.3 is text-only with a 1M context window.
        pattern: "glm-5.3",
        capabilities: caps(false, false, true, 1_000_000),
        pricing: pricing(1.40, 4.40, 0.00, 0.26),
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
        // Inkling Small (Jul 29, 2026) — same partner-served setup as Inkling
        // below; pricing from Together, verified Aug 3, 2026. Must stay above
        // the bare "inkling" entry (substring match, first entry wins).
        pattern: "inkling-small",
        capabilities: caps(true, false, true, 1_048_576),
        pricing: pricing(0.50, 1.20, 0.00, 0.10),
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
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(5.00, 30.00, 5.00, 0.50),
    },
    ReferenceModelEntry {
        // OpenAI routes the bare alias to gpt-5.6-sol.
        pattern: "gpt-5.6",
        capabilities: caps(true, false, true, 1_050_000),
        pricing: pricing(4.00, 20.00, 5.00, 0.40),
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
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
    },
    ReferenceModelEntry {
        pattern: "grok-3",
        capabilities: caps(true, false, true, 1_000_000),
        pricing: pricing(1.25, 2.50, 1.25, 0.20),
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

// A bare pattern such as `o3` must not match inside `nano-3-30b`, so the hit
// has to sit on a separator boundary at both ends.
fn matches_model(normalized: &str, pattern: &str) -> bool {
    let sanitized_pattern = sanitize_model_name(pattern);
    normalized
        .match_indices(sanitized_pattern.as_str())
        .any(|(start, hit)| {
            let before = normalized[..start].chars().next_back();
            let after = normalized[start + hit.len()..].chars().next();
            before.is_none_or(|c| !c.is_ascii_alphanumeric())
                && after.is_none_or(|c| !c.is_ascii_alphanumeric())
        })
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
pub fn calculate_reference_cost(
    model: &str,
    input_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let pricing = get_reference_pricing(model)?;
    Some(pricing.calculate_cost(input_tokens, 0, cache_read_tokens, output_tokens))
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
#[path = "reference_models_tests.rs"]
mod tests;
