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

//! Reference pricing for media models — the media counterpart of
//! `llm::reference_models`, so a caller reads a cost off
//! [`MediaUsage`](super::types::MediaUsage) without having to know a rate up front.
//!
//! Two things make media pricing structurally different from LLM pricing, and
//! both are why this is its own table rather than a column on that one:
//!
//! 1. **The billing unit varies per model.** LLM rates are always per 1M
//!    tokens; a media model bills per image, per video-second, per character,
//!    or per compute-second. An entry therefore carries the unit alongside the
//!    rate, and resolves to a [`CostEstimate`] — the same rate snapshot a
//!    caller can pass explicitly.
//! 2. **The same model costs different amounts on different hosts.** `flux` is
//!    billed per image on Replicate and per GPU-second on fal, and `veo` is
//!    reachable through three providers at three prices, so entries are keyed
//!    by provider first and model pattern second. A bare model-name table would
//!    cross-match and misbill.
//!
//! A table rate is an ESTIMATE by construction: it is our record of a published
//! list price, not an amount the upstream billed us. It therefore only ever
//! reaches `MediaUsage::estimated_cost`, never `provider_reported_cost`, and
//! an explicit caller-supplied rate always wins over it. An unknown model
//! yields `None` — no default rate, no zero, no guess.
//!
//! Only units an adapter can actually observe are priced. `Megapixels` is
//! deliberately absent: no adapter reports output dimensions today, so a
//! per-megapixel rate could never be applied and would read as coverage the
//! table does not have.

use super::types::{CostEstimate, UsageUnit};
use crate::utils::naming::{normalize_model_name, sanitize_model_name};

/// Reference rate for a matched media model, plus the pattern that matched —
/// callers surface the pattern when explaining where a price came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaModelPricing {
    pub provider: &'static str,
    pub pattern: &'static str,
    pub unit: UsageUnit,
    pub usd_per_unit: f64,
}

#[derive(Debug, Clone, Copy)]
struct MediaPricingEntry {
    provider: &'static str,
    pattern: &'static str,
    unit: UsageUnit,
    usd_per_unit: f64,
}

const fn entry(
    provider: &'static str,
    pattern: &'static str,
    unit: UsageUnit,
    usd_per_unit: f64,
) -> MediaPricingEntry {
    MediaPricingEntry {
        provider,
        pattern,
        unit,
        usd_per_unit,
    }
}

/// Reference media pricing, USD per unit.
///
/// EVERY rate here is an `estimate` and must be confirmed against the
/// provider's official pricing page before it bills a customer — same
/// convention and same caveat as the embedding table in [`crate::embedding`].
/// These are raw provider list prices; margin is applied downstream.
///
/// Ordering is load-bearing: lookup is a first-match substring scan within one
/// provider, so a more specific pattern MUST precede a less specific one
/// (`flux-1.1-pro-ultra` before `flux-1.1-pro`). Patterns are sanitized the
/// same way model names are, so they must be written with the model's own
/// separators — `gen4_turbo`, not `gen4-turbo`.
const REFERENCE_MEDIA_MODELS: &[MediaPricingEntry] = &[
    // ── ElevenLabs (estimate — verify at elevenlabs.io/pricing) ──
    // Billed against a subscription character quota; these are the
    // pay-as-you-go per-character equivalents. Scribe bills input audio, which
    // is not known until the upload completes, so it stays unpriced at submit.
    entry("elevenlabs", "flash", UsageUnit::Characters, 0.000_03),
    entry("elevenlabs", "turbo", UsageUnit::Characters, 0.000_03),
    entry(
        "elevenlabs",
        "multilingual",
        UsageUnit::Characters,
        0.000_06,
    ),
    // ── fal (estimate — verify per endpoint at fal.ai/pricing) ──
    // fal prices per endpoint, not per family. Video endpoints bill the
    // requested duration; everything else falls back to GPU wall-clock, which
    // is the only quantity fal's queue metrics report.
    entry("fal", "veo", UsageUnit::VideoSeconds, 0.20),
    entry("fal", "kling", UsageUnit::VideoSeconds, 0.09),
    entry("fal", "minimax", UsageUnit::VideoSeconds, 0.07),
    entry("fal", "fal-ai", UsageUnit::ComputeSeconds, 0.000_56),
    // ── Replicate (estimate — verify at replicate.com/pricing) ──
    // Official models carry a flat per-output price; community models bill
    // GPU-seconds, which the trailing per-owner entries cannot express — those
    // resolve to nothing and stay unpriced rather than guess a GPU class.
    entry("replicate", "flux-1.1-pro-ultra", UsageUnit::Images, 0.055),
    entry("replicate", "flux-1.1-pro", UsageUnit::Images, 0.04),
    entry("replicate", "flux-schnell", UsageUnit::Images, 0.003),
    entry("replicate", "flux-dev", UsageUnit::Images, 0.025),
    entry("replicate", "veo", UsageUnit::VideoSeconds, 0.50),
    entry("replicate", "kling", UsageUnit::VideoSeconds, 0.09),
    // ── Runway (estimate — verify at runwayml.com/pricing) ──
    // Runway sells credits; these are one second of generated video converted
    // at the standard plan's credit price. The video models are listed by name
    // rather than as a `gen4` prefix: that prefix would also catch gen4_image,
    // and stamping a per-second video rate onto an image job is worse than
    // leaving it unpriced.
    entry("runway", "gen4_turbo", UsageUnit::VideoSeconds, 0.05),
    entry("runway", "gen3a_turbo", UsageUnit::VideoSeconds, 0.05),
    entry("runway", "gen4_aleph", UsageUnit::VideoSeconds, 0.15),
];

fn matches_model(normalized: &str, pattern: &str) -> bool {
    normalized.contains(&sanitize_model_name(pattern))
}

fn normalized_model(model: &str) -> String {
    sanitize_model_name(&normalize_model_name(model))
}

/// Look up the reference rate for a media model. `provider` is the octolib
/// adapter name (`fal`, `replicate`, …) and is matched exactly, because the
/// same model bills differently on different hosts.
pub fn get_reference_pricing(provider: &str, model: &str) -> Option<MediaModelPricing> {
    let normalized = normalized_model(model);
    REFERENCE_MEDIA_MODELS
        .iter()
        .find(|entry| {
            entry.provider.eq_ignore_ascii_case(provider)
                && matches_model(&normalized, entry.pattern)
        })
        .map(|entry| MediaModelPricing {
            provider: entry.provider,
            pattern: entry.pattern,
            unit: entry.unit,
            usd_per_unit: entry.usd_per_unit,
        })
}

/// The reference rate as a [`CostEstimate`], ready to hand to an adapter's
/// existing estimate path.
///
/// `known_quantity` is the billable amount the caller can already account for
/// at submit time — video seconds requested, characters of input text. It is
/// attached only when its unit is the one this model bills on; anything else,
/// including `None`, leaves the quantity open so the adapter's usage builder
/// fills it from the response metrics, exactly as it does for a
/// caller-supplied rate.
///
/// Returning a `CostEstimate` rather than a bespoke type is what keeps this
/// module additive: the rate rides into [`JobHandle`](super::types::JobHandle)
/// at submit and is frozen there, so a job resumed after a restart is priced
/// at the rate it was submitted under and never re-reads a table that has
/// since changed.
pub fn reference_cost_estimate(
    provider: &str,
    model: &str,
    known_quantity: Option<(UsageUnit, f64)>,
) -> Option<CostEstimate> {
    let pricing = get_reference_pricing(provider, model)?;
    Some(CostEstimate {
        unit: pricing.unit,
        usd_per_unit: pricing.usd_per_unit,
        quantity: known_quantity
            .filter(|(unit, _)| *unit == pricing.unit)
            .map(|(_, quantity)| quantity)
            .filter(|value| value.is_finite() && *value >= 0.0),
    })
}

#[cfg(test)]
#[path = "reference_pricing_tests.rs"]
mod tests;
