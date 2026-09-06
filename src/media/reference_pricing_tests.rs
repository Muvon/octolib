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

use super::*;

/// The model ids these providers actually take, so the table is exercised
/// against real shapes rather than the tidy names in the patterns.
#[test]
fn resolves_real_world_media_model_id_shapes() {
    assert!(get_reference_pricing("fal", "fal-ai/flux/dev").is_some());
    assert!(get_reference_pricing("replicate", "black-forest-labs/flux-1.1-pro").is_some());
    assert!(get_reference_pricing("runway", "gen4_turbo").is_some());
    assert!(get_reference_pricing("elevenlabs", "eleven_flash_v2_5").is_some());
    assert!(get_reference_pricing("elevenlabs", "eleven_multilingual_v2").is_some());
}

/// `sanitize_model_name` inserts a dash at every letter/digit boundary but
/// leaves underscores alone, so `gen4_turbo` normalizes to `gen-4_turbo`. A
/// pattern written `gen4-turbo` would sanitize to `gen-4-turbo` and silently
/// never match — this test is the guard against writing patterns that way.
#[test]
fn patterns_survive_the_shared_sanitizer() {
    let turbo = get_reference_pricing("runway", "gen4_turbo").unwrap();
    assert_eq!(turbo.pattern, "gen4_turbo");
    let aleph = get_reference_pricing("runway", "gen4_aleph").unwrap();
    assert_eq!(aleph.pattern, "gen4_aleph");
    // gen4_image is an image model; a per-second video rate must not reach it.
    assert!(get_reference_pricing("runway", "gen4_image").is_none());
    // gen3a_turbo must not be captured by the gen4 entries.
    let legacy = get_reference_pricing("runway", "gen3a_turbo").unwrap();
    assert_eq!(legacy.pattern, "gen3a_turbo");
}

#[test]
fn provider_scopes_the_lookup() {
    // Same family, different host, different billing unit — the whole reason
    // the table is keyed by provider first.
    let fal = get_reference_pricing("fal", "fal-ai/veo/3.1").unwrap();
    let replicate = get_reference_pricing("replicate", "google/veo-3.1").unwrap();
    assert_eq!(fal.unit, UsageUnit::VideoSeconds);
    assert_eq!(replicate.unit, UsageUnit::VideoSeconds);
    assert_ne!(fal.usd_per_unit, replicate.usd_per_unit);

    // A model priced on one host is not silently priced on another.
    assert!(get_reference_pricing("runway", "google/veo-3.1").is_none());
    assert!(get_reference_pricing("openrouter", "google/veo-3.1").is_none());
}

#[test]
fn provider_match_is_case_insensitive_but_not_fuzzy() {
    assert!(get_reference_pricing("FAL", "fal-ai/flux/dev").is_some());
    assert!(get_reference_pricing("fal-ai", "fal-ai/flux/dev").is_none());
}

#[test]
fn unknown_model_is_unpriced() {
    assert!(get_reference_pricing("fal", "").is_none());
    assert!(get_reference_pricing("replicate", "some-owner/never-heard-of-it").is_none());
    assert!(reference_cost_estimate("replicate", "some-owner/never-heard-of-it", None).is_none());
}

#[test]
fn specific_patterns_beat_broader_ones() {
    let ultra = get_reference_pricing("replicate", "black-forest-labs/flux-1.1-pro-ultra").unwrap();
    assert_eq!(ultra.pattern, "flux-1.1-pro-ultra");
    let pro = get_reference_pricing("replicate", "black-forest-labs/flux-1.1-pro").unwrap();
    assert_eq!(pro.pattern, "flux-1.1-pro");

    // fal's trailing catch-all must not swallow the named video endpoints.
    let veo = get_reference_pricing("fal", "fal-ai/veo/3.1").unwrap();
    assert_eq!(veo.pattern, "veo");
    let other = get_reference_pricing("fal", "fal-ai/some-unlisted-endpoint").unwrap();
    assert_eq!(other.pattern, "fal-ai");

    // Runway has no catch-all at all, so an unlisted model stays unpriced
    // rather than inheriting a sibling's billing unit.
    assert!(get_reference_pricing("runway", "gen5_something").is_none());
}

#[test]
fn estimate_carries_the_units_billable_quantity() {
    // 8 seconds of gen4_turbo at $0.05/s — the adapter multiplies this out.
    let estimate =
        reference_cost_estimate("runway", "gen4_turbo", Some((UsageUnit::VideoSeconds, 8.0)))
            .unwrap();
    assert_eq!(estimate.unit, UsageUnit::VideoSeconds);
    assert_eq!(estimate.quantity, Some(8.0));
    assert!((estimate.quantity.unwrap() * estimate.usd_per_unit - 0.40).abs() < 1e-9);
}

/// A quantity in the wrong unit is ignored rather than multiplied by the
/// model's rate — an image count is not a GPU-second count. The slot stays
/// open so the adapter's `rate.quantity.or(<metric>)` fallback fills it.
#[test]
fn estimate_leaves_provider_reported_quantities_open() {
    let estimate =
        reference_cost_estimate("fal", "fal-ai/flux/dev", Some((UsageUnit::Images, 2.0))).unwrap();
    assert_eq!(estimate.unit, UsageUnit::ComputeSeconds);
    assert_eq!(estimate.quantity, None);
}

/// A nonsense quantity must not become a nonsense charge; the adapters' own
/// validators reject non-finite rates, and this is the same guard for the
/// amount. A missing quantity must likewise stay open rather than bill zero.
#[test]
fn estimate_rejects_an_unusable_quantity() {
    let nan = reference_cost_estimate(
        "runway",
        "gen4_turbo",
        Some((UsageUnit::VideoSeconds, f64::NAN)),
    )
    .unwrap();
    assert_eq!(nan.quantity, None);
    let negative = reference_cost_estimate(
        "runway",
        "gen4_turbo",
        Some((UsageUnit::VideoSeconds, -1.0)),
    )
    .unwrap();
    assert_eq!(negative.quantity, None);
    // An unknown video duration must leave the slot open, not charge 0 seconds.
    let unknown = reference_cost_estimate("runway", "gen4_turbo", None).unwrap();
    assert_eq!(unknown.quantity, None);
}

/// Ordering is load-bearing: lookup is a first-match substring scan, so a
/// broad pattern above a narrow one silently shadows it. This is the media
/// counterpart of `every_pricing_entry_is_reachable` on the LLM table, keyed
/// by (provider, pattern) because the media table is provider-scoped.
#[test]
fn every_entry_is_reachable() {
    for entry in REFERENCE_MEDIA_MODELS {
        let resolved = get_reference_pricing(entry.provider, entry.pattern)
            .unwrap_or_else(|| panic!("{}:{} resolves to nothing", entry.provider, entry.pattern));
        assert_eq!(
            resolved.pattern, entry.pattern,
            "{}:{} is shadowed by the earlier, less specific pattern {}",
            entry.provider, entry.pattern, resolved.pattern
        );
    }
}

/// Only units an adapter can actually observe or supply may carry a rate —
/// otherwise the table advertises coverage that can never be applied.
#[test]
fn every_rate_is_usable() {
    for entry in REFERENCE_MEDIA_MODELS {
        assert!(
            entry.usd_per_unit.is_finite() && entry.usd_per_unit >= 0.0,
            "{}:{} has an unusable rate",
            entry.provider,
            entry.pattern
        );
        assert!(
            matches!(
                entry.unit,
                UsageUnit::Images
                    | UsageUnit::ComputeSeconds
                    | UsageUnit::Characters
                    | UsageUnit::VideoSeconds
                    | UsageUnit::AudioSeconds
            ),
            "{}:{} prices a unit no adapter reports",
            entry.provider,
            entry.pattern
        );
    }
}
