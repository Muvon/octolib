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
//! Only units an adapter can actually observe are priced. A model whose price
//! moves with its own parameters (resolution tier, pixel size, rendering mode,
//! duration) has a per-request rule instead of a rate: it reads the exact body
//! the adapter sends and, where that body leaves the cost open, takes the
//! endpoint's worst case.

use super::types::{CostEstimate, UsageUnit};
use crate::utils::naming::{normalize_model_name, sanitize_model_name};
use serde_json::{Map, Value};

/// Reference rate for a matched media model, plus the pattern that matched —
/// callers surface the pattern when explaining where a price came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaModelPricing {
	pub provider: &'static str,
	pub pattern: &'static str,
	pub unit: UsageUnit,
	pub usd_per_unit: f64,
}

#[derive(Clone, Copy)]
struct MediaPricingEntry {
	provider: &'static str,
	pattern: &'static str,
	price: Price,
}

#[derive(Clone, Copy)]
enum Price {
	/// One rate per unit, whatever the request sets.
	Flat { unit: UsageUnit, usd_per_unit: f64 },
	/// A price that moves with the request's own parameters, read off the body
	/// the adapter sends (see "Per-request list prices" below).
	Request(fn(&Body) -> CostEstimate),
}

const fn entry(provider: &'static str, pattern: &'static str, unit: UsageUnit, usd_per_unit: f64) -> MediaPricingEntry {
	MediaPricingEntry {
		provider,
		pattern,
		price: Price::Flat { unit, usd_per_unit },
	}
}

const fn rule(provider: &'static str, pattern: &'static str, price: fn(&Body) -> CostEstimate) -> MediaPricingEntry {
	MediaPricingEntry {
		provider,
		pattern,
		price: Price::Request(price),
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
	// ── Cloudflare Workers AI (verified Sep 18, 2026 at the model pages) ──
	// Audio only: images bill per 512x512 output tile plus per step, which
	// one unit cannot express, so the adapter prices those itself. Per-minute
	// rates are stored per second; the duration comes from the response.
	entry("cloudflare", "aura-2", UsageUnit::Characters, 0.000_03),
	entry("cloudflare", "aura-1", UsageUnit::Characters, 0.000_015),
	entry("cloudflare", "melotts", UsageUnit::AudioSeconds, 0.000_205 / 60.0),
	entry(
		"cloudflare",
		"whisper-large-v3-turbo",
		UsageUnit::AudioSeconds,
		0.000_513 / 60.0,
	),
	entry("cloudflare", "whisper", UsageUnit::AudioSeconds, 0.000_453 / 60.0),
	entry("cloudflare", "nova-3", UsageUnit::AudioSeconds, 0.0052 / 60.0),
	// ── ElevenLabs (estimate — verify at elevenlabs.io/pricing) ──
	// Billed against a subscription character quota; these are the
	// pay-as-you-go per-character equivalents. Scribe bills input audio, which
	// is not known until the upload completes, so it stays unpriced at submit.
	entry("elevenlabs", "flash", UsageUnit::Characters, 0.000_03),
	entry("elevenlabs", "turbo", UsageUnit::Characters, 0.000_03),
	entry("elevenlabs", "multilingual", UsageUnit::Characters, 0.000_06),
	// ── fal list prices, read 2026-10-27 at https://fal.ai/models/<endpoint> ──
	// Per-request rules (see "Per-request list prices" below), ahead of fal's
	// family rows so a priced endpoint never reaches them.
	rule("fal", "fal-ai/nano-banana-2", nano_banana_2),
	rule("fal", "fal-ai/nano-banana-pro", nano_banana_pro),
	rule("fal", "google/nano-banana-2.1", nano_banana_2_1),
	// "$0.03 for the first output megapixel, plus $0.015 per additional input
	// or output megapixel, with megapixels rounded up".
	rule("fal", "fal-ai/flux-2-pro", |body| {
		per_image(first_and_extra(body.fal_flux_2_megapixels(), 0.03, 0.015))
	}),
	// "$0.05 per megapixel on both input and output side, rounded up to the
	// nearest megapixel".
	rule("fal", "fal-ai/flux-2-flex", |body| {
		per_image(0.05 * body.fal_flux_2_megapixels())
	}),
	// "The first processed megapixel will cost $0.07. Each additional
	// megapixel will cost $0.03".
	rule("fal", "fal-ai/flux-2-max", |body| {
		per_image(first_and_extra(body.fal_flux_2_megapixels(), 0.07, 0.03))
	}),
	// "$0.005 per megapixels".
	rule("fal", "fal-ai/flux-2/klein/4b", |body| {
		per_image(0.005 * body.fal_flux_2_megapixels())
	}),
	rule("fal", "blackforestlabs/flux-3/text-to-image", fal_flux_3),
	rule("fal", "fal-ai/gpt-image-2", gpt_image_2),
	// "Each image costs $0.03 with TURBO, $0.06 with BALANCED, and $0.09 with
	// QUALITY".
	rule("fal", "fal-ai/ideogram/v3", |body| {
		per_image(body.tier(
			"rendering_speed",
			"BALANCED",
			&[("TURBO", 0.03), ("BALANCED", 0.06), ("QUALITY", 0.09)],
		))
	}),
	rule("fal", "ideogram/v4", fal_ideogram_v4),
	// "Price: $0.035 per images".
	rule("fal", "fal-ai/recraft/v4.1/text-to-image", |_| per_image(0.035)),
	// Seedance's `duration` defaults to "auto": the model picks 4-15 s while it
	// runs and fal's result does not say which, so it is priced at 15 s.
	rule("fal", "bytedance/seedance-2.0/text-to-video", |body| {
		body.video("720p", SEEDANCE_2_PER_SECOND, SEEDANCE_2_MAX_SECS, SEEDANCE_2_MAX_SECS)
	}),
	// "For every second of 720p video you generated, you will be charged
	// $0.2419/second"; 480p (no figure) meters fewer pixels at the same
	// "$0.0112 per 1,000 tokens", so 720p's rate bounds it.
	rule("fal", "bytedance/seedance-2.0/fast/text-to-video", |body| {
		body.video(
			"720p",
			&[("480p", 0.2419), ("720p", 0.2419)],
			SEEDANCE_2_MAX_SECS,
			SEEDANCE_2_MAX_SECS,
		)
	}),
	// "roughly $0.0721/second for 480p and $0.1547/second for 720p".
	rule("fal", "bytedance/seedance-2.0/mini/text-to-video", |body| {
		body.video(
			"720p",
			&[("480p", 0.0721), ("720p", 0.1547)],
			SEEDANCE_2_MAX_SECS,
			SEEDANCE_2_MAX_SECS,
		)
	}),
	// "For 480p, your request will cost roughly $0.2205 per second of generated
	// video, for 720p ... roughly $0.4730 ..., and for 1080p, roughly $1.164".
	rule("fal", "bytedance/seedance-2.5/text-to-video", |body| {
		body.video(
			"720p",
			&[("480p", 0.2205), ("720p", 0.4730), ("1080p", 1.164)],
			SEEDANCE_2_5_MAX_SECS,
			SEEDANCE_2_5_MAX_SECS,
		)
	}),
	// "For 5s video your request will cost $1.40. For every aditional second
	// you will be charged $0.28": a flat $0.28 a second, 5 or 10 s.
	rule("fal", "fal-ai/kling-video/v2/master/text-to-video", |body| {
		per_second(0.28, body.seconds("duration", 5.0, 10.0))
	}),
	rule(
		"fal",
		"fal-ai/kling-video/v3/standard/text-to-video",
		fal_kling_3_standard,
	),
	// "Video costs $0.05 per second at 480p, $0.06 per second at 768p, $0.13 per
	// second at 2K and $0.16 per second at 4K"; 5-15 s.
	rule("fal", "minimax/h3/text-to-video", |body| {
		body.video(
			"2K",
			&[("480P", 0.05), ("768P", 0.06), ("2K", 0.13), ("4K", 0.16)],
			5.0,
			15.0,
		)
	}),
	rule("fal", "minimax/h3-max/text-to-video", fal_h3_max),
	rule("fal", "fal-ai/elevenlabs/music", eleven_music),
	// "$0.1 per 1000 characters"; `text` holds up to 5,000.
	rule("fal", "fal-ai/minimax/speech-02-hd", |body| {
		body.characters("text", 5_000, 0.1 / 1000.0)
	}),
	// "$0.1 per 1000 character"; `prompt` holds up to 10,000.
	rule("fal", "fal-ai/minimax/speech-2.8-hd", |body| {
		body.characters("prompt", 10_000, 0.1 / 1000.0)
	}),
	// "$0.06 per 1000 character".
	rule("fal", "fal-ai/minimax/speech-2.8-turbo", |body| {
		body.characters("prompt", 10_000, 0.06 / 1000.0)
	}),
	// "$0.001 per second" of the requested `duration` (default 8, at most 30),
	// each started second counted whole.
	rule("fal", "fal-ai/mmaudio-v2", |body| CostEstimate {
		unit: UsageUnit::AudioSeconds,
		usd_per_unit: 0.001,
		quantity: Some(body.seconds("duration", 8.0, 30.0).ceil()),
	}),
	// ── fal (estimate — verify per endpoint at fal.ai/pricing) ──
	// fal prices per endpoint, not per family. Video endpoints bill the
	// requested duration; everything else falls back to GPU wall-clock, which
	// is the only quantity fal's queue metrics report.
	entry("fal", "veo", UsageUnit::VideoSeconds, 0.20),
	entry("fal", "kling", UsageUnit::VideoSeconds, 0.09),
	entry("fal", "minimax", UsageUnit::VideoSeconds, 0.07),
	entry("fal", "fal-ai", UsageUnit::ComputeSeconds, 0.000_56),
	// ── Higgsfield, read 2026-10-27 at
	// https://open.higgsfield.ai/models/<endpoint>/playground. Its cards show
	// discounted rates; these are the standard rates printed beside them. ──
	// "Images cost $0.0032 each at 720p and $0.0057 each at 1080p".
	rule("higgsfield", "higgsfield-ai/soul/v2/standard", |body| {
		per_image(body.tier("resolution", "720p", &[("720p", 0.0032), ("1080p", 0.0057)]))
	}),
	// "Images cost $0.03 each".
	rule("higgsfield", "ideogram/v4.0", |_| per_image(0.03)),
	// "Images cost $0.035 each at 1k", its only resolution.
	rule("higgsfield", "recraft/v4.1/text-to-image", |_| per_image(0.035)),
	// "Billable video tokens = ceil(generated video seconds × output width ×
	// output height × 24 fps / 1024)" at "480p/720p/1080p $0.014, 4K $0.008"
	// per 1,000: the meter and rates of SEEDANCE_2_PER_SECOND. 4-15 s, default 5.
	rule("higgsfield", "bytedance/seedance-2.0/text-to-video", |body| {
		body.video("720p", SEEDANCE_2_PER_SECOND, 5.0, SEEDANCE_2_MAX_SECS)
	}),
	// "standard rates are $0.13 per second at 2K", its only resolution; 5-15 s.
	rule("higgsfield", "minimax/h3/text-to-video", |body| {
		body.video("2K", &[("2K", 0.13)], 5.0, 15.0)
	}),
	// "standard rates are $0.084 per second", one flat rate: the catalog's price range
	// is a single value and no sound surcharge is published (read 2026-10-27); 3-15 s.
	rule("higgsfield", "kling-video/v3.0/std/text-to-video", |body| {
		per_second(0.084, body.seconds("duration", 5.0, 15.0))
	}),
	// "Priced per generated second by resolution: 480p $0.05, 720p $0.10, or
	// 1080p $0.20"; 2-30 s.
	rule("higgsfield", "alibaba/wan-3.0/text-to-video", |body| {
		body.video("1080p", &[("480p", 0.05), ("720p", 0.10), ("1080p", 0.20)], 5.0, 30.0)
	}),
	// ── OpenRouter FLUX.2, read 2026-10-27 at
	// https://openrouter.ai/black-forest-labs/<model>. OpenRouter reports its
	// own charge on the result; these price a response that omits it. ──
	// "The first megapixel is charged $0.03 and then each subsequent MP will be
	// charged $0.015".
	rule("openrouter", "black-forest-labs/flux.2-pro", |body| {
		per_image(first_and_extra(body.openrouter_megapixels(), 0.03, 0.015))
	}),
	// "We charge $0.06 for each megapixel on both input and output side".
	rule("openrouter", "black-forest-labs/flux.2-flex", |body| {
		per_image(0.06 * body.openrouter_megapixels())
	}),
	// "The first generated megapixel is charged $0.07. Each subsequent
	// megapixel is charged $0.03".
	rule("openrouter", "black-forest-labs/flux.2-max", |body| {
		per_image(first_and_extra(body.openrouter_megapixels(), 0.07, 0.03))
	}),
	// "The first generated megapixel is charged $0.014. Each subsequent
	// megapixel is charged $0.001".
	rule("openrouter", "black-forest-labs/flux.2-klein-4b", |body| {
		per_image(first_and_extra(body.openrouter_megapixels(), 0.014, 0.001))
	}),
	// ── MiniMax (top-resolution pay-as-you-go rates, verified Oct 26, 2026) ──
	// https://platform.minimax.io/docs/guides/pricing-paygo
	entry("minimax", "minimax-h3-max", UsageUnit::VideoSeconds, 0.08),
	entry("minimax", "minimax-h3-highspeed", UsageUnit::VideoSeconds, 0.06),
	entry("minimax", "minimax-h3-lite", UsageUnit::VideoSeconds, 0.04),
	entry("minimax", "minimax-h3", UsageUnit::VideoSeconds, 0.13),
	entry("minimax", "image-01", UsageUnit::Images, 0.0035),
	entry("minimax", "speech-2.8-hd", UsageUnit::Characters, 0.0001),
	entry("minimax", "speech-2.8-turbo", UsageUnit::Characters, 0.00006),
	entry("minimax", "speech-02-hd", UsageUnit::Characters, 0.0001),
	entry("minimax", "speech-02-turbo", UsageUnit::Characters, 0.00006),
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

fn find_entry(provider: &str, model: &str) -> Option<&'static MediaPricingEntry> {
	let normalized = normalized_model(model);
	REFERENCE_MEDIA_MODELS
		.iter()
		.find(|entry| entry.provider.eq_ignore_ascii_case(provider) && matches_model(&normalized, entry.pattern))
}

impl MediaPricingEntry {
	fn estimate(&self, body: Option<&Map<String, Value>>, known_quantity: Option<(UsageUnit, f64)>) -> CostEstimate {
		match self.price {
			Price::Flat { unit, usd_per_unit } => CostEstimate {
				unit,
				usd_per_unit,
				quantity: known_quantity
					.filter(|(known_unit, _)| *known_unit == unit)
					.map(|(_, quantity)| quantity)
					.filter(|value| value.is_finite() && *value >= 0.0),
			},
			Price::Request(price) => price(&Body(body.unwrap_or(&Map::new()))),
		}
	}
}

/// Look up the reference rate for a media model. `provider` is the octolib
/// adapter name (`fal`, `replicate`, …) and is matched exactly, because the
/// same model bills differently on different hosts. A model priced per
/// request reports the rate of a request that sets nothing.
pub fn get_reference_pricing(provider: &str, model: &str) -> Option<MediaModelPricing> {
	let entry = find_entry(provider, model)?;
	let estimate = entry.estimate(None, None);
	Some(MediaModelPricing {
		provider: entry.provider,
		pattern: entry.pattern,
		unit: estimate.unit,
		usd_per_unit: estimate.usd_per_unit,
	})
}

/// The reference rate as a [`CostEstimate`], ready to hand to an adapter's
/// existing estimate path.
///
/// `body` is the request the adapter sends, provider options and field maps
/// merged in: a model priced per request reads its price and quantity off it.
/// `known_quantity` is the billable amount the caller can already account for
/// at submit time for a flat rate — video seconds requested, characters of
/// input text. It is attached only when its unit is the one this model bills
/// on; anything else, including `None`, leaves the quantity open so the
/// adapter's usage builder fills it from the response metrics, exactly as it
/// does for a caller-supplied rate.
///
/// Returning a `CostEstimate` rather than a bespoke type is what keeps this
/// module additive: the rate rides into [`JobHandle`](super::types::JobHandle)
/// at submit and is frozen there, so a job resumed after a restart is priced
/// at the rate it was submitted under and never re-reads a table that has
/// since changed.
pub fn reference_cost_estimate(
	provider: &str,
	model: &str,
	body: Option<&Map<String, Value>>,
	known_quantity: Option<(UsageUnit, f64)>,
) -> Option<CostEstimate> {
	find_entry(provider, model).map(|entry| entry.estimate(body, known_quantity))
}

// ── Per-request list prices ──────────────────────────────────────────────
//
// A rule reads the body an adapter is about to send, with provider options and
// field maps already merged in, so the price follows whatever the caller set
// and not only the portable fields. An absent field takes the endpoint's
// documented default, because that is what the provider renders. Where the
// body leaves the cost open (a length the model picks, a tier with no
// published rate, a value the rule does not know) the rule takes the
// endpoint's WORST case from its list pricing: the dearest tier, the longest
// allowed duration, the largest size, every step rounded up. A job is
// overcharged before it is ever undercharged, and never priced at zero.
// Every rate is a LIST price, never a promotion, quoted beside its rule.

/// fal's `image_size` presets, from the "ImageSize" table in fal's docs
/// (https://fal.ai/docs/llms-full.txt, read 2026-10-27).
const FAL_IMAGE_PRESETS: &[(&str, u64, u64)] = &[
	("square_hd", 1024, 1024),
	("square", 512, 512),
	("portrait_4_3", 768, 1024),
	("portrait_16_9", 576, 1024),
	("landscape_4_3", 1024, 768),
	("landscape_16_9", 1024, 576),
];

/// The megapixel these endpoints bill in, rounded up: "1 megapixel is 1024 ×
/// 1024 pixels" (https://fal.ai/models/fal-ai/flux-2-pro, read 2026-10-27).
/// The flex page's worked examples use the same unit; the max and klein pages,
/// and OpenRouter's FLUX.2 pages, state none of their own.
const MEGAPIXEL_PIXELS: u64 = 1024 * 1024;

/// FLUX.2's largest output, the worst case when a request names no size it
/// can be measured by: "up to 4MP output" (https://docs.bfl.ml/flux_2/flux2_overview,
/// read 2026-10-27), 2048 × 2048 in billed megapixels.
const FLUX_2_MAX_SIZE: (u64, u64) = (2048, 2048);

/// The largest `{width, height}` fal's `ImageSize` schema accepts ("maximum":
/// 14142 per side, read 2026-10-27 in each image endpoint's OpenAPI): the size
/// a request is priced at when its `image_size` cannot be read.
const FAL_MAX_IMAGE_SIZE: (u64, u64) = (14_142, 14_142);

/// "If web search is used, an additional $0.015 will be charged" (both fal Nano
/// Banana pages, read 2026-10-27).
const FAL_NANO_BANANA_WEB_SEARCH_USD: f64 = 0.015;

/// "The output can run up to about 0.7 s longer than requested"
/// (https://fal.ai/models/minimax/h3-max/text-to-video, read 2026-10-27). That
/// page never says whether it bills requested or generated seconds, and its
/// siblings disagree (reference-to-video: requested; extend-video: generated),
/// so the overrun is billed as the worst case.
const H3_MAX_OVERRUN_SECS: f64 = 0.7;

/// Seedance 2.0 per second by resolution, the same ByteDance meter on fal and
/// Higgsfield: "(height of output video * width of output video * duration *
/// 24) / 1024" tokens at "$0.014 per 1,000 tokens for 480p, 720p and 1080p" and
/// "$0.008 per 1,000 tokens for 4k" (https://fal.ai/models/bytedance/seedance-2.0/text-to-video,
/// read 2026-10-27; Higgsfield's playground quotes the same rates).
/// - 720p and 1080p: fal's own per-second figures, "$0.3034/second" and
///   "$0.682/second".
/// - 480p: no published figure; it meters fewer pixels than 720p at the same
///   token rate, so 720p's rate bounds it.
/// - 4k: no published figure; 3840 × 2160 × 24 / 1024 = 194,400 tokens a
///   second at $0.008 per 1,000. Neither host documents 4k pixel sizes; 3840 ×
///   2160 is the native 16:9 4k size third parties report (read 2026-10-27).
const SEEDANCE_2_PER_SECOND: &[(&str, f64)] = &[("480p", 0.3034), ("720p", 0.3034), ("1080p", 0.682), ("4k", 1.5552)];

/// Longest clips the endpoints accept, from each one's `duration` schema: the
/// worst case when the model picks the length (`"auto"`).
const SEEDANCE_2_MAX_SECS: f64 = 15.0;
const SEEDANCE_2_5_MAX_SECS: f64 = 30.0;

const TOKENS_PER_MILLION: f64 = 1_000_000.0;
const MILLIS_PER_MINUTE: f64 = 60_000.0;
const SECONDS_PER_MINUTE: f64 = 60.0;

/// "Your request will cost $0.08 per image ... 2K and 4K outputs will be
/// charged at 1.5 times and 2 times the standard rate, respectively. 0.5K
/// (512px) resolution outputs will be charged at 0.75 times the standard rate.
/// If web search is used, an additional $0.015 will be charged. If high thinking
/// is used, an additional $0.002 will be charged." The surcharges are per
/// request; adding them to every image overstates a multi-image request and
/// never understates one. `enable_image_search` has no published price, so it
/// is charged as a second search.
fn nano_banana_2(body: &Body) -> CostEstimate {
	const PER_IMAGE_USD: f64 = 0.08;
	let scale = body.tier(
		"resolution",
		"1K",
		&[("0.5K", 0.75), ("1K", 1.0), ("2K", 1.5), ("4K", 2.0)],
	);
	let searches = ["enable_web_search", "enable_image_search"]
		.into_iter()
		.filter(|field| body.flag(field))
		.count() as f64;
	// fal's default when the field is omitted is `minimal`.
	let thinking = body.tier("thinking_level", "minimal", &[("minimal", 0.0), ("high", 0.002)]);
	per_image(PER_IMAGE_USD * scale + searches * FAL_NANO_BANANA_WEB_SEARCH_USD + thinking)
}

/// "Your request will cost $0.15 per image ... 4K outputs will be charged at
/// double the standard rate. If web search is used, an additional $0.015 will
/// be charged." 2K is the standard rate.
fn nano_banana_pro(body: &Body) -> CostEstimate {
	const PER_IMAGE_USD: f64 = 0.15;
	let scale = body.tier("resolution", "1K", &[("1K", 1.0), ("2K", 1.0), ("4K", 2.0)]);
	let search = if body.flag("enable_web_search") {
		FAL_NANO_BANANA_WEB_SEARCH_USD
	} else {
		0.0
	};
	per_image(PER_IMAGE_USD * scale + search)
}

/// Nano Banana 2.1 bills Google's token counts, which fal does not return. fal
/// publishes one image's cost "from a short prompt (about 100 tokens), with
/// `medium` thinking": 1K $0.040, 2K $0.059, 4K $0.134, with input at "$1.764"
/// and output text and thinking at "$8.823" per 1M tokens. The prompt is added
/// at one token per character, an upper bound; `minimal` thinking costs less
/// than the table. fal: "There are no flat per-image fees or surcharges for
/// resolution, web search, or thinking" — thinking bills as output text and
/// search results as input tokens, counts nobody knows up front (read
/// 2026-10-27). So `high` thinking and web search are each priced at a full
/// `max_output_tokens` budget (32,768, the cap fal's Nano Banana 2 schema
/// documents) at its own rate.
fn nano_banana_2_1(body: &Body) -> CostEstimate {
	const INPUT_USD_PER_TOKEN: f64 = 1.764 / TOKENS_PER_MILLION;
	const THINKING_USD_PER_TOKEN: f64 = 8.823 / TOKENS_PER_MILLION;
	const TOKEN_BUDGET: f64 = 32_768.0;
	let image = body.tier("resolution", "1K", &[("1K", 0.040), ("2K", 0.059), ("4K", 0.134)]);
	let thinking = body.tier(
		"thinking_level",
		"medium",
		&[
			("minimal", 0.0),
			("medium", 0.0),
			("high", TOKEN_BUDGET * THINKING_USD_PER_TOKEN),
		],
	);
	let search = if body.flag("enable_web_search") {
		TOKEN_BUDGET * INPUT_USD_PER_TOKEN
	} else {
		0.0
	};
	let prompt_characters = body.text_length("prompt") + body.text_length("system_prompt");
	per_image(image + thinking + search + prompt_characters as f64 * INPUT_USD_PER_TOKEN)
}

/// FLUX 3 bills "megapixels" at $0.048: "Image pricing varies by resolution.
/// A 1K (1 MP) image costs $0.048" (https://fal.ai/models/blackforestlabs/flux-3/text-to-image,
/// read 2026-10-27). Black Forest Labs names each tier by its pixel count,
/// "768²≈0.6 MP, 1K≈1 MP (1024×1024-equivalent), 1.5K≈2 MP, 2K≈4 MP, 4K≈16 MP"
/// (https://bfl.ai/pricing?category=flux.2, read 2026-10-27), so a tier is
/// priced as its side squared in 1024² megapixels, rounded up.
fn fal_flux_3(body: &Body) -> CostEstimate {
	const PER_MEGAPIXEL_USD: f64 = 0.048;
	let megapixels = body.tier(
		"resolution",
		"1k",
		&[("768sq", 1.0), ("1k", 1.0), ("1.5k", 3.0), ("2k", 4.0), ("4k", 16.0)],
	);
	per_image(PER_MEGAPIXEL_USD * megapixels)
}

/// fal's table for GPT Image 2 "from a technical standpoint": one image by size
/// and quality, `[low, medium, high]`, ascending by area. The endpoint is
/// served as openai/gpt-image-2.
const GPT_IMAGE_2_PER_IMAGE: &[((u64, u64), [f64; 3])] = &[
	((1024, 768), [0.005, 0.037, 0.145]),
	((1024, 1024), [0.006, 0.053, 0.211]),
	((1024, 1536), [0.005, 0.042, 0.165]),
	((1920, 1080), [0.005, 0.040, 0.158]),
	((2560, 1440), [0.007, 0.056, 0.222]),
	((3840, 2160), [0.012, 0.101, 0.401]),
];

/// GPT Image 2 by size and quality, plus the prompt at "Text tokens (per 1M):
/// $5.00 input", one token per character as an upper bound. `quality: auto`
/// lets the model choose, so it is priced as `high`. The table's prices do not
/// grow with area (a square costs more than 1024x1536), so a size it does not
/// list, `image_size: auto` included, is priced at the dearest listed size.
fn gpt_image_2(body: &Body) -> CostEstimate {
	const PROMPT_USD_PER_TOKEN: f64 = 5.0 / TOKENS_PER_MILLION;
	let quality = match body.text("quality", "high") {
		"low" => 0,
		"medium" => 1,
		_ => 2,
	};
	let size = body.fal_image_size("landscape_4_3", FAL_MAX_IMAGE_SIZE);
	let price = GPT_IMAGE_2_PER_IMAGE
		.iter()
		.find(|(listed, _)| *listed == size)
		.map_or_else(
			|| {
				GPT_IMAGE_2_PER_IMAGE
					.iter()
					.map(|(_, rates)| rates[quality])
					.fold(0.0, f64::max)
			},
			|(_, rates)| rates[quality],
		);
	per_image(price + body.text_length("prompt") as f64 * PROMPT_USD_PER_TOKEN)
}

/// "Your request will cost $0.0075 per megapixel in TURBO mode, $0.015 per
/// megapixel in BALANCED mode, or $0.025 per megapixel in QUALITY mode." The
/// schema adds that `expansion_model: None` "skips its fee", a fee neither fal
/// nor Ideogram's API pricing states (read 2026-10-27); it is charged as one more
/// of the endpoint's $0.01 billing units (the page's endpoint billing record).
fn fal_ideogram_v4(body: &Body) -> CostEstimate {
	const EXPANSION_FEE_USD: f64 = 0.01;
	let per_megapixel = body.tier(
		"rendering_speed",
		"BALANCED",
		&[("TURBO", 0.0075), ("BALANCED", 0.015), ("QUALITY", 0.025)],
	);
	let (width, height) = body.fal_image_size("square_hd", FAL_MAX_IMAGE_SIZE);
	let expansion = if body.text("expansion_model", "Medium") == "None" {
		0.0
	} else {
		EXPANSION_FEE_USD
	};
	per_image(per_megapixel * megapixels(width, height) + expansion)
}

/// "For every second of video you generated, you will be charged $0.084 (audio
/// off) or $0.126 (audio on)"; `generate_audio` defaults on, 3-15 s. A
/// `multi_prompt` sets its own shot durations, so it is priced at 15 s.
fn fal_kling_3_standard(body: &Body) -> CostEstimate {
	const MAX_SECS: f64 = 15.0;
	let per_second_usd = if body.flag_or("generate_audio", true) {
		0.126
	} else {
		0.084
	};
	let seconds = if body.get("multi_prompt").is_some() {
		MAX_SECS
	} else {
		body.seconds("duration", 5.0, MAX_SECS)
	};
	per_second(per_second_usd, seconds)
}

/// The page shows promotional rates "35% off for a limited time. The discount
/// ends October 31, after which 480p is $0.05/second, 768p is $0.08/second, and
/// 1080p is $0.16/second" — the list rates used here; up to 15 s.
fn fal_h3_max(body: &Body) -> CostEstimate {
	let per_second_usd = body.tier("resolution", "768P", &[("480P", 0.05), ("768P", 0.08), ("1080P", 0.16)]);
	per_second(
		per_second_usd,
		body.seconds("duration", 5.0, 15.0) + H3_MAX_OVERRUN_SECS,
	)
}

/// "Your request will cost $0.6 per output audio minute. The audio will be
/// rounded up to the closest minute." Without `music_length_ms` the model picks
/// the length and fal's result does not report it, so it is priced at the
/// schema's 600,000 ms maximum.
fn eleven_music(body: &Body) -> CostEstimate {
	const PER_MINUTE_USD: f64 = 0.60;
	const MAX_MILLIS: f64 = 600_000.0;
	let millis = body
		.get("music_length_ms")
		.and_then(Value::as_f64)
		.filter(|millis| millis.is_finite() && *millis > 0.0)
		.unwrap_or(MAX_MILLIS);
	let minutes = (millis / MILLIS_PER_MINUTE).ceil();
	CostEstimate {
		unit: UsageUnit::AudioSeconds,
		usd_per_unit: PER_MINUTE_USD / SECONDS_PER_MINUTE,
		quantity: Some(minutes * SECONDS_PER_MINUTE),
	}
}

fn per_image(usd_per_image: f64) -> CostEstimate {
	// The count is left open: the adapter takes it from the images produced.
	CostEstimate {
		unit: UsageUnit::Images,
		usd_per_unit: usd_per_image,
		quantity: None,
	}
}

fn per_second(usd_per_second: f64, seconds: f64) -> CostEstimate {
	CostEstimate {
		unit: UsageUnit::VideoSeconds,
		usd_per_unit: usd_per_second,
		quantity: Some(seconds),
	}
}

/// One rate for the first megapixel, another for each one after it.
fn first_and_extra(megapixels: f64, first_usd: f64, extra_usd: f64) -> f64 {
	first_usd + extra_usd * (megapixels - 1.0).max(0.0)
}

/// Megapixels as billed, rounded up.
fn megapixels(width: u64, height: u64) -> f64 {
	width.saturating_mul(height).div_ceil(MEGAPIXEL_PIXELS) as f64
}

/// A request body, read the way the provider reads it.
struct Body<'a>(&'a Map<String, Value>);

impl<'a> Body<'a> {
	/// A field the caller set. `null` reads as unset, so the default applies.
	fn get(&self, field: &str) -> Option<&'a Value> {
		self.0.get(field).filter(|value| !value.is_null())
	}

	fn text(&self, field: &str, default: &'static str) -> &'a str {
		self.get(field).and_then(Value::as_str).unwrap_or(default)
	}

	/// A flag the caller turned on; anything else reads as off.
	fn flag(&self, field: &str) -> bool {
		self.flag_or(field, false)
	}

	fn flag_or(&self, field: &str, default: bool) -> bool {
		self.get(field).and_then(Value::as_bool).unwrap_or(default)
	}

	/// The rate of the tier the request selects: the default's when the field is
	/// unset, the dearest tier's for a value without a published rate.
	fn tier(&self, field: &str, default: &'static str, rates: &[(&str, f64)]) -> f64 {
		let value = self.text(field, default);
		rates
			.iter()
			.find(|(tier, _)| *tier == value)
			.map(|(_, rate)| *rate)
			.unwrap_or_else(|| rates.iter().map(|(_, rate)| *rate).fold(0.0, f64::max))
	}

	/// A duration in seconds, given as a number or a numeric string (`"5"`):
	/// `default` when unset, `max` when the model picks it (`"auto"`) or the
	/// value cannot be read.
	fn seconds(&self, field: &str, default: f64, max: f64) -> f64 {
		let seconds = match self.get(field) {
			None => return default,
			Some(Value::Number(number)) => number.as_f64(),
			Some(Value::String(text)) => text.trim_end_matches('s').parse().ok(),
			Some(_) => None,
		};
		seconds
			.filter(|seconds| seconds.is_finite() && *seconds > 0.0)
			.unwrap_or(max)
	}

	/// Output width and height from fal's `image_size`: a preset or `{width,
	/// height}`, `worst` for anything else.
	fn fal_image_size(&self, default: &'static str, worst: (u64, u64)) -> (u64, u64) {
		let preset = match self.get("image_size") {
			None => default,
			Some(Value::String(name)) => name.as_str(),
			Some(Value::Object(size)) => {
				let side = |name: &str| size.get(name).and_then(Value::as_u64).filter(|pixels| *pixels > 0);
				return side("width").zip(side("height")).unwrap_or(worst);
			}
			Some(_) => return worst,
		};
		FAL_IMAGE_PRESETS
			.iter()
			.find(|(name, _, _)| *name == preset)
			.map_or(worst, |(_, width, height)| (*width, *height))
	}

	/// FLUX.2 megapixels on fal, from the requested `image_size` (default
	/// `landscape_4_3`); one that cannot be read is priced at the model's 4 MP
	/// maximum.
	fn fal_flux_2_megapixels(&self) -> f64 {
		let (width, height) = self.fal_image_size("landscape_4_3", FLUX_2_MAX_SIZE);
		megapixels(width, height)
	}

	/// FLUX.2 megapixels on OpenRouter, from its `size` (`"1024x768"`). Without
	/// one the model picks the size, so it is priced at its 4 MP maximum.
	fn openrouter_megapixels(&self) -> f64 {
		let (width, height) = self
			.get("size")
			.and_then(Value::as_str)
			.and_then(|size| size.split_once('x'))
			.and_then(|(width, height)| Some((width.trim().parse().ok()?, height.trim().parse().ok()?)))
			.unwrap_or(FLUX_2_MAX_SIZE);
		megapixels(width, height)
	}

	/// The resolution tier's per-second rate for the requested duration.
	fn video(
		&self,
		default_resolution: &'static str,
		rates: &[(&str, f64)],
		default_secs: f64,
		max_secs: f64,
	) -> CostEstimate {
		let per_second_usd = self.tier("resolution", default_resolution, rates);
		per_second(per_second_usd, self.seconds("duration", default_secs, max_secs))
	}

	/// A per-character rate over the text the provider bills: `field`, else the
	/// portable `text` octolib sends, else the field's maximum length.
	fn characters(&self, field: &str, max_characters: usize, usd_per_character: f64) -> CostEstimate {
		let count = [field, "text"]
			.into_iter()
			.map(|name| self.text_length(name))
			.find(|count| *count > 0)
			.unwrap_or(max_characters);
		CostEstimate {
			unit: UsageUnit::Characters,
			usd_per_unit: usd_per_character,
			quantity: Some(count as f64),
		}
	}

	/// Characters in a text field; an absent one counts as empty.
	fn text_length(&self, field: &str) -> usize {
		self.get(field)
			.and_then(Value::as_str)
			.map_or(0, |text| text.chars().count())
	}
}

#[cfg(test)]
#[path = "reference_pricing_tests.rs"]
mod tests;
