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
use serde_json::json;

#[test]
fn minimax_rates_match_each_model_and_billing_unit() {
	for (model, unit, rate) in [
		("MiniMax-H3-Max", UsageUnit::VideoSeconds, 0.08),
		("MiniMax-H3-Highspeed", UsageUnit::VideoSeconds, 0.06),
		("MiniMax-H3-Lite", UsageUnit::VideoSeconds, 0.04),
		("MiniMax-H3", UsageUnit::VideoSeconds, 0.13),
		("image-01", UsageUnit::Images, 0.0035),
		("speech-2.8-hd", UsageUnit::Characters, 0.0001),
		("speech-2.8-turbo", UsageUnit::Characters, 0.00006),
		("speech-02-hd", UsageUnit::Characters, 0.0001),
		("speech-02-turbo", UsageUnit::Characters, 0.00006),
	] {
		let pricing = get_reference_pricing("minimax", model).unwrap();
		assert_eq!(pricing.pattern, model.to_ascii_lowercase());
		assert_eq!(pricing.unit, unit);
		assert_eq!(pricing.usd_per_unit, rate);
	}
}

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
	assert!(reference_cost_estimate("replicate", "some-owner/never-heard-of-it", None, None).is_none());
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
	let estimate = reference_cost_estimate("runway", "gen4_turbo", None, Some((UsageUnit::VideoSeconds, 8.0))).unwrap();
	assert_eq!(estimate.unit, UsageUnit::VideoSeconds);
	assert_eq!(estimate.quantity, Some(8.0));
	assert!((estimate.quantity.unwrap() * estimate.usd_per_unit - 0.40).abs() < 1e-9);
}

/// A quantity in the wrong unit is ignored rather than multiplied by the
/// model's rate — an image count is not a GPU-second count. The slot stays
/// open so the adapter's `rate.quantity.or(<metric>)` fallback fills it.
#[test]
fn estimate_leaves_provider_reported_quantities_open() {
	let estimate = reference_cost_estimate("fal", "fal-ai/flux/dev", None, Some((UsageUnit::Images, 2.0))).unwrap();
	assert_eq!(estimate.unit, UsageUnit::ComputeSeconds);
	assert_eq!(estimate.quantity, None);
}

/// A nonsense quantity must not become a nonsense charge; the adapters' own
/// validators reject non-finite rates, and this is the same guard for the
/// amount. A missing quantity must likewise stay open rather than bill zero.
#[test]
fn estimate_rejects_an_unusable_quantity() {
	let nan = reference_cost_estimate("runway", "gen4_turbo", None, Some((UsageUnit::VideoSeconds, f64::NAN))).unwrap();
	assert_eq!(nan.quantity, None);
	let negative =
		reference_cost_estimate("runway", "gen4_turbo", None, Some((UsageUnit::VideoSeconds, -1.0))).unwrap();
	assert_eq!(negative.quantity, None);
	// An unknown video duration must leave the slot open, not charge 0 seconds.
	let unknown = reference_cost_estimate("runway", "gen4_turbo", None, None).unwrap();
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
		let pricing = get_reference_pricing(entry.provider, entry.pattern).unwrap();
		assert!(
			pricing.usd_per_unit.is_finite() && pricing.usd_per_unit >= 0.0,
			"{}:{} has an unusable rate",
			entry.provider,
			entry.pattern
		);
		assert!(
			matches!(
				pricing.unit,
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

fn price(provider: &str, model: &str, body: Value) -> CostEstimate {
	reference_cost_estimate(provider, model, Some(body.as_object().unwrap()), None)
		.unwrap_or_else(|| panic!("{provider}:{model} has no price"))
}

fn assert_usd(actual: f64, expected: f64) {
	assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
}

/// One unit of the estimate: a whole clip or text, or one image.
fn job_cost(estimate: &CostEstimate) -> f64 {
	estimate.usd_per_unit * estimate.quantity.unwrap_or(1.0)
}

/// Every media lane octohub routes, copied from octomind-api's
/// docker/containers/octohub/octohub.toml `[media_models]` (2026-10-27). A lane
/// added there without a price here fails this test, not a customer's bill.
const ROUTED_MEDIA_LANES: &[(&str, &str)] = &[
	("fal", "fal-ai/nano-banana-2"),
	("fal", "fal-ai/nano-banana-pro"),
	("fal", "fal-ai/flux-2-pro"),
	("fal", "fal-ai/flux-2-flex"),
	("fal", "fal-ai/flux-2-max"),
	("fal", "fal-ai/flux-2/klein/4b"),
	("fal", "fal-ai/gpt-image-2"),
	("fal", "fal-ai/ideogram/v3"),
	("fal", "google/nano-banana-2.1"),
	("fal", "blackforestlabs/flux-3/text-to-image"),
	("fal", "ideogram/v4"),
	("fal", "fal-ai/recraft/v4.1/text-to-image"),
	("fal", "bytedance/seedance-2.0/text-to-video"),
	("fal", "bytedance/seedance-2.0/fast/text-to-video"),
	("fal", "bytedance/seedance-2.0/mini/text-to-video"),
	("fal", "bytedance/seedance-2.5/text-to-video"),
	("fal", "fal-ai/kling-video/v2/master/text-to-video"),
	("fal", "minimax/h3/text-to-video"),
	("fal", "minimax/h3-max/text-to-video"),
	("fal", "fal-ai/kling-video/v3/standard/text-to-video"),
	("fal", "fal-ai/elevenlabs/music"),
	("fal", "fal-ai/minimax/speech-02-hd"),
	("fal", "fal-ai/minimax/speech-2.8-hd"),
	("fal", "fal-ai/minimax/speech-2.8-turbo"),
	("fal", "fal-ai/mmaudio-v2"),
	("openrouter", "black-forest-labs/flux.2-pro"),
	("openrouter", "black-forest-labs/flux.2-flex"),
	("openrouter", "black-forest-labs/flux.2-max"),
	("openrouter", "black-forest-labs/flux.2-klein-4b"),
	("higgsfield", "higgsfield-ai/soul/v2/standard"),
	("higgsfield", "ideogram/v4.0"),
	("higgsfield", "recraft/v4.1/text-to-image"),
	("higgsfield", "bytedance/seedance-2.0/text-to-video"),
	("higgsfield", "minimax/h3/text-to-video"),
	("higgsfield", "kling-video/v3.0/std/text-to-video"),
	("higgsfield", "alibaba/wan-3.0/text-to-video"),
	("minimax", "image-01"),
	("minimax", "MiniMax-H3"),
	("minimax", "MiniMax-H3-Max"),
	("minimax", "MiniMax-H3-Lite"),
	("minimax", "MiniMax-H3-Highspeed"),
	("minimax", "speech-02-hd"),
	("minimax", "speech-2.8-hd"),
	("minimax", "speech-2.8-turbo"),
];

/// Every routed lane carries a positive price for its default request, and a
/// request whose cost-moving fields cannot be read (a length the model picks,
/// an unknown tier, an `auto` size) is priced at least as high, never at zero.
/// fal, Higgsfield and OpenRouter bill by parameters, so each of their lanes
/// needs a per-request rule; MiniMax bills per model, from the flat table.
#[test]
fn every_routed_media_lane_has_a_positive_worst_case_price() {
	let unreadable = json!({
		"resolution": "unlisted-tier",
		"duration": "auto",
		"image_size": "auto",
		"size": "auto",
		"quality": "auto",
		"rendering_speed": "unlisted",
		"thinking_level": "unlisted"
	});
	for &(provider, model) in ROUTED_MEDIA_LANES {
		let own = get_reference_pricing(provider, model).map(|pricing| pricing.pattern);
		assert!(
			own.is_some_and(|pattern| pattern.eq_ignore_ascii_case(model)),
			"{provider}:{model} has no price of its own (matched {own:?})"
		);
		let default = job_cost(&price(provider, model, json!({})));
		let worst = job_cost(&price(provider, model, unreadable.clone()));
		assert!(
			default.is_finite() && default > 0.0,
			"{provider}:{model} default costs {default}"
		);
		assert!(
			worst.is_finite() && worst >= default,
			"{provider}:{model} worst case {worst} < default {default}"
		);
	}
}

/// The 2026-10-27 incident: a 1080p Seedance clip sent through provider options
/// is priced at 1080p, and a length the model picks is priced at the longest.
#[test]
fn seedance_prices_the_tier_and_duration_the_request_sets() {
	const SEEDANCE: &str = "bytedance/seedance-2.0/text-to-video";
	let default = price("fal", SEEDANCE, json!({"duration": "5"}));
	assert_eq!((default.unit, default.quantity), (UsageUnit::VideoSeconds, Some(5.0)));
	assert_usd(default.usd_per_unit, 0.3034);
	assert_usd(
		price("fal", SEEDANCE, json!({"duration": "8", "resolution": "1080p"})).usd_per_unit,
		0.682,
	);
	assert_eq!(
		price("fal", SEEDANCE, json!({})).quantity,
		Some(15.0),
		"fal's default is \"auto\""
	);
	assert_eq!(price("fal", SEEDANCE, json!({"duration": "auto"})).quantity, Some(15.0));
	// 480p has no figure of its own; 720p's rate bounds it.
	assert_usd(
		price("fal", SEEDANCE, json!({"duration": "5", "resolution": "480p"})).usd_per_unit,
		0.3034,
	);
	// 4k is ByteDance's token meter: (h * w * duration * 24) / 1024 at $0.008 per 1,000.
	let four_k = 3840.0 * 2160.0 * 24.0 / 1024.0 * 0.008 / 1000.0;
	assert_usd(
		price("fal", SEEDANCE, json!({"duration": "5", "resolution": "4k"})).usd_per_unit,
		four_k,
	);
	assert_usd(
		price("fal", SEEDANCE, json!({"duration": "5", "resolution": "8k"})).usd_per_unit,
		four_k,
	);
	let longest = price(
		"fal",
		"bytedance/seedance-2.5/text-to-video",
		json!({"resolution": "1080p"}),
	);
	assert_usd(longest.usd_per_unit, 1.164);
	assert_eq!(longest.quantity, Some(30.0));
	// Higgsfield defaults to 5 s at 720p on the same meter.
	let higgsfield_default = price("higgsfield", SEEDANCE, json!({}));
	assert_eq!(higgsfield_default.quantity, Some(5.0));
	assert_usd(higgsfield_default.usd_per_unit, 0.3034);
}

#[test]
fn video_rates_follow_audio_resolution_and_overrun() {
	let kling = "fal-ai/kling-video/v3/standard/text-to-video";
	assert_usd(price("fal", kling, json!({})).usd_per_unit, 0.126);
	assert_usd(
		price("fal", kling, json!({"generate_audio": false})).usd_per_unit,
		0.084,
	);
	let shots = price(
		"fal",
		kling,
		json!({"multi_prompt": [{"prompt": "a", "duration": "5"}]}),
	);
	assert_eq!(shots.quantity, Some(15.0));
	assert_eq!(
		price(
			"fal",
			"fal-ai/kling-video/v2/master/text-to-video",
			json!({"duration": "10"})
		)
		.quantity,
		Some(10.0)
	);
	assert_usd(price("fal", "minimax/h3/text-to-video", json!({})).usd_per_unit, 0.13);
	let max = price(
		"fal",
		"minimax/h3-max/text-to-video",
		json!({"duration": 5, "resolution": "1080P"}),
	);
	assert_usd(max.usd_per_unit, 0.16);
	assert_usd(max.quantity.unwrap(), 5.7);
	assert_usd(
		job_cost(&price("higgsfield", "alibaba/wan-3.0/text-to-video", json!({}))),
		1.0,
	);
	assert_eq!(
		price(
			"higgsfield",
			"alibaba/wan-3.0/text-to-video",
			json!({"duration": "auto"})
		)
		.quantity,
		Some(30.0)
	);
}

/// fal's and OpenRouter's own worked examples, recomputed from the requested
/// size, and FLUX.2's 4 MP maximum when the size cannot be read.
#[test]
fn megapixel_images_round_up_the_requested_size() {
	// The default landscape_4_3 is 1024x768: one megapixel.
	assert_usd(price("fal", "fal-ai/flux-2-pro", json!({})).usd_per_unit, 0.03);
	let full_hd = json!({"image_size": {"width": 1920, "height": 1080}});
	assert_usd(price("fal", "fal-ai/flux-2-pro", full_hd.clone()).usd_per_unit, 0.045);
	assert_usd(price("fal", "fal-ai/flux-2-flex", full_hd).usd_per_unit, 0.10);
	assert_usd(
		price("fal", "fal-ai/flux-2-flex", json!({"image_size": "square"})).usd_per_unit,
		0.05,
	);
	assert_usd(
		price("fal", "fal-ai/flux-2-pro", json!({"image_size": "auto"})).usd_per_unit,
		0.03 + 3.0 * 0.015,
	);
	assert_usd(
		price(
			"openrouter",
			"black-forest-labs/flux.2-pro",
			json!({"size": "1920x1080"}),
		)
		.usd_per_unit,
		0.045,
	);
	assert_usd(
		price("openrouter", "black-forest-labs/flux.2-klein-4b", json!({})).usd_per_unit,
		0.014 + 3.0 * 0.001,
	);
	// Ideogram 4: "a 2048 x 2048 image will cost ... $0.10" in QUALITY mode.
	let quality = json!({
		"expansion_model": "None",
		"rendering_speed": "QUALITY",
		"image_size": {"width": 2048, "height": 2048}
	});
	assert_usd(price("fal", "ideogram/v4", quality).usd_per_unit, 0.10);
	// Prompt expansion is on by default and carries its fee.
	assert_usd(price("fal", "ideogram/v4", json!({})).usd_per_unit, 0.015 + 0.01);
	// FLUX 3 tiers in 1024² megapixels at $0.048.
	assert_usd(
		price("fal", "blackforestlabs/flux-3/text-to-image", json!({})).usd_per_unit,
		0.048,
	);
	assert_usd(
		price(
			"fal",
			"blackforestlabs/flux-3/text-to-image",
			json!({"resolution": "2k"}),
		)
		.usd_per_unit,
		0.192,
	);
	assert_usd(
		price(
			"fal",
			"blackforestlabs/flux-3/text-to-image",
			json!({"resolution": "8k"}),
		)
		.usd_per_unit,
		0.768,
	);
}

#[test]
fn tiered_images_add_their_published_surcharges() {
	let image = price("fal", "fal-ai/nano-banana-2", json!({}));
	assert_eq!((image.unit, image.quantity), (UsageUnit::Images, None));
	assert_usd(image.usd_per_unit, 0.08);
	let loaded = json!({"resolution": "4K", "enable_web_search": true, "thinking_level": "high"});
	assert_usd(price("fal", "fal-ai/nano-banana-2", loaded).usd_per_unit, 0.177);
	assert_usd(
		price("fal", "fal-ai/nano-banana-2", json!({"enable_image_search": true})).usd_per_unit,
		0.095,
	);
	assert_usd(
		price("fal", "fal-ai/nano-banana-pro", json!({"resolution": "4K"})).usd_per_unit,
		0.30,
	);
	assert_usd(
		price("fal", "google/nano-banana-2.1", json!({"thinking_level": "high"})).usd_per_unit,
		0.040 + 32_768.0 * 8.823 / 1_000_000.0,
	);
	assert_usd(
		price("fal", "fal-ai/gpt-image-2", json!({"prompt": "cat"})).usd_per_unit,
		0.145 + 3.0 * 5.0 / 1_000_000.0,
	);
	assert_usd(
		price(
			"fal",
			"fal-ai/gpt-image-2",
			json!({"quality": "low", "image_size": "square_hd"}),
		)
		.usd_per_unit,
		0.006,
	);
	// A size the table does not list costs as much as the dearest listed one.
	assert_usd(
		price("fal", "fal-ai/gpt-image-2", json!({"image_size": "portrait_16_9"})).usd_per_unit,
		0.401,
	);
	assert_usd(
		price(
			"higgsfield",
			"higgsfield-ai/soul/v2/standard",
			json!({"resolution": "1080p"}),
		)
		.usd_per_unit,
		0.0057,
	);
}

#[test]
fn audio_bills_started_minutes_and_the_billed_text() {
	let music = |body: Value| job_cost(&price("fal", "fal-ai/elevenlabs/music", body));
	assert_usd(music(json!({"music_length_ms": 30_000})), 0.60);
	assert_usd(music(json!({"music_length_ms": 61_000})), 1.20);
	// Without a length the model picks one: the schema's 600,000 ms maximum.
	assert_usd(music(json!({"prompt": "jazz"})), 6.00);
	let speech = price("fal", "fal-ai/minimax/speech-02-hd", json!({"text": "hello"}));
	assert_eq!(speech.quantity, Some(5.0));
	assert_usd(speech.usd_per_unit, 0.0001);
	// The 2.8 endpoints bill `prompt`; octolib's portable `text` stands in for it.
	assert_eq!(
		price("fal", "fal-ai/minimax/speech-2.8-hd", json!({"text": "hello"})).quantity,
		Some(5.0)
	);
	assert_eq!(
		price("fal", "fal-ai/minimax/speech-2.8-hd", json!({})).quantity,
		Some(10_000.0)
	);
	assert_eq!(
		price("fal", "fal-ai/mmaudio-v2", json!({"duration": 2.5})).quantity,
		Some(3.0)
	);
}

/// A route without a rule of its own falls to its family's row, or to nothing.
#[test]
fn sibling_routes_do_not_borrow_a_rule() {
	assert_eq!(
		get_reference_pricing("fal", "fal-ai/whisper").unwrap().pattern,
		"fal-ai"
	);
	assert!(get_reference_pricing("replicate", "black-forest-labs/flux.2-pro").is_none());
}
