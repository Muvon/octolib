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
use crate::llm::utils::is_model_in_pricing_table;
use crate::test_support::env::EnvGuard;

#[test]
fn test_supports_model() {
	let provider = AlibabaProvider::new();
	assert!(provider.supports_model("qwen3.8-max"));
	assert!(provider.supports_model("qwen3-coder-plus"));
	assert!(!provider.supports_model(""));
}

#[test]
fn cache_lifetime_is_the_explicit_ttl_or_the_measured_implicit_cache() {
	let provider = AlibabaProvider::new();
	let minutes = |m: u64| Some(std::time::Duration::from_secs(m * 60));
	// Explicit cache: 5 minutes, or an hour when the long cache meets Qwen4.
	assert_eq!(provider.cache_lifetime("qwen3.8-flash", false), minutes(5));
	assert_eq!(provider.cache_lifetime("qwen3.8-flash", true), minutes(5));
	assert_eq!(provider.cache_lifetime("qwen3.7-plus-2026-05-26", false), minutes(5));
	assert_eq!(provider.cache_lifetime("qwen4-max", false), minutes(5));
	assert_eq!(provider.cache_lifetime("qwen4-max", true), minutes(60));
	// Implicit cache.
	assert_eq!(provider.cache_lifetime("glm-5.3", true), minutes(10));
	assert_eq!(provider.cache_lifetime("deepseek-v4.1-flash", false), minutes(30));
	assert_eq!(provider.cache_lifetime("kimi-k3", false), minutes(15));
	assert_eq!(provider.cache_lifetime("qwen-plus-character", false), minutes(15));
}

#[test]
fn explicit_cache_covers_listed_families_and_their_snapshots() {
	assert!(uses_explicit_cache("qwen3.8-max"));
	assert!(uses_explicit_cache("qwen3.8-max-0902"));
	assert!(uses_explicit_cache("Qwen4-Max"));
	assert!(!uses_explicit_cache("qwen-plus-character"));
	assert!(!uses_explicit_cache("qwen3.8-omni-flash"));
	assert!(!uses_explicit_cache("glm-5.3"));
}

#[test]
fn explicit_cache_bills_its_own_write_and_hit_shares() {
	// qwen3.8-flash input is $0.15 per 1M: a write bills 125%, a hit 10%.
	let five_minutes = calculate_local_usage_cost(
		"qwen3.8-flash",
		0,
		1_000_000,
		1_000_000,
		0,
		Some(CacheMarker::Ephemeral),
	)
	.unwrap();
	assert!((five_minutes - (0.1875 + 0.015)).abs() < 1e-9);
	// qwen4-max input is $1.50 per 1M: a one-hour write bills 200%.
	let hour = calculate_local_usage_cost("qwen4-max", 0, 1_000_000, 0, 0, Some(CacheMarker::EphemeralHour)).unwrap();
	assert!((hour - 3.0).abs() < 1e-9);
	// Without a marker the table's implicit rates stand.
	let implicit = calculate_local_usage_cost("qwen3.8-flash", 0, 1_000_000, 1_000_000, 0, None).unwrap();
	assert!((implicit - (0.15 + 0.016)).abs() < 1e-9);
}

#[test]
#[serial_test::serial]
fn test_default_capabilities() {
	let _env = EnvGuard::new()
		.remove("ALIBABA_API_MODE")
		.remove("ALIBABA_API_URL")
		.remove("ALIBABA_WORKSPACE_ID");
	let provider = AlibabaProvider::new();
	assert_eq!(provider.name(), "alibaba");
	assert!(provider.supports_caching("qwen3.8-max"));
	// Structured output is native for supported Qwen families.
	assert!(provider.supports_structured_output("qwen3.8-max"));
	assert!(provider.enforces_response_schema("qwen3.8-max"));
	assert!(!provider.enforces_response_schema("deepseek-v4-flash-0731"));
	// qwen3.8-max carries a 1M context window via reference capabilities
	assert_eq!(provider.get_max_input_tokens("qwen3.8-max"), 1_000_000);
	// Verified live: 3.8-max/3.7-plus/3.6-flash take images and video,
	// 3.7-max rejects image parts outright
	assert!(provider.supports_vision("qwen3.8-max"));
	assert!(provider.supports_vision("qwen3.7-plus"));
	assert!(provider.supports_vision("qwen3.6-flash"));
	assert!(provider.supports_video("qwen3.8-max"));
	assert!(provider.supports_video("qwen3.6-flash"));
	assert!(!provider.supports_vision("qwen3.7-max"));
}

#[test]
#[serial_test::serial]
fn test_glm_replayed_thinking_counts_toward_context() {
	// On Chat, GLM reads every prior turn's reasoning back on this host, so
	// callers sizing a prompt must count it; DeepSeek reasoning is not sent.
	// The shared Messages layer replays every model's thinking blocks.
	let provider = AlibabaProvider::new();
	let _env = EnvGuard::new()
		.remove("ALIBABA_API_MODE")
		.remove("ALIBABA_API_URL")
		.remove("ALIBABA_WORKSPACE_ID");
	assert!(provider.replays_thinking("glm-5.3"));
	assert!(provider.replays_thinking("glm-5.2-fast-preview"));
	assert!(!provider.replays_thinking("deepseek-v4-flash-0731"));
	assert!(!provider.replays_thinking("qwen3.8-flash"));

	let _env = _env.set("ALIBABA_WORKSPACE_ID", "ws-1234abcd");
	assert!(provider.replays_thinking("deepseek-v4-flash-0731"));
	assert!(provider.replays_thinking("qwen3.8-flash"));
	assert!(provider.replays_thinking("glm-5.3"));
}

#[test]
#[serial_test::serial]
fn test_qwen4_max_uses_its_list_price_and_capabilities() {
	let _env = EnvGuard::new()
		.remove("ALIBABA_API_MODE")
		.remove("ALIBABA_API_URL")
		.remove("ALIBABA_WORKSPACE_ID");
	let provider = AlibabaProvider::new();
	let p = provider.get_model_pricing("qwen4-max").unwrap();
	assert_eq!(p.input_price_per_1m, 1.50);
	assert_eq!(p.output_price_per_1m, 4.50);
	// Implicit cache: writes bill at the input rate.
	assert_eq!(p.cache_write_price_per_1m, 1.50);
	assert_eq!(p.cache_read_price_per_1m, 0.075);

	assert_eq!(provider.get_max_input_tokens("qwen4-max"), 1_000_000);
	assert!(provider.supports_vision("qwen4-max"));
	assert!(provider.supports_video("qwen4-max"));
	assert!(provider.enforces_response_schema("qwen4-max"));
	// Chat Completions keeps earlier turns' reasoning by default.
	assert!(provider.replays_thinking("qwen4-max"));
}

#[test]
fn test_macaron_models_use_their_list_prices() {
	let provider = AlibabaProvider::new();
	for (model, input, output, cache_read) in [
		("mindlab/macaron-v1.1-coding-venti", 1.19, 3.74, 0.22),
		("mindlab/macaron-v1.1-venti", 1.19, 3.74, 0.22),
		("mindlab/macaron-v1-tall", 0.45, 2.60, 0.08),
	] {
		let p = provider.get_model_pricing(model).unwrap();
		assert_eq!(p.input_price_per_1m, input, "{model}");
		assert_eq!(p.output_price_per_1m, output, "{model}");
		assert_eq!(p.cache_read_price_per_1m, cache_read, "{model}");
	}
	assert_eq!(provider.get_max_input_tokens("mindlab/macaron-v1.1-venti"), 1_048_576);
	assert_eq!(provider.get_max_input_tokens("mindlab/macaron-v1-tall"), 262_144);
	// Only the coding variant takes images.
	assert!(provider.supports_vision("mindlab/macaron-v1.1-coding-venti"));
	assert!(!provider.supports_vision("mindlab/macaron-v1.1-venti"));
}

#[test]
fn test_pricing_qwen() {
	let provider = AlibabaProvider::new();

	let p = provider.get_model_pricing("qwen3.8-max").unwrap();
	assert_eq!(p.input_price_per_1m, 2.00);
	assert_eq!(p.output_price_per_1m, 6.00);
	assert_eq!(p.cache_read_price_per_1m, 0.25);

	// Moving alias and dated snapshots share the list price.
	let p = provider.get_model_pricing("qwen3.7-max").unwrap();
	assert_eq!(p.input_price_per_1m, 2.50);
	assert_eq!(p.output_price_per_1m, 7.50);
	assert_eq!(p.cache_read_price_per_1m, 0.50);

	let p = provider.get_model_pricing("qwen3.7-max-2026-06-08").unwrap();
	assert_eq!(p.input_price_per_1m, 2.50);
	assert_eq!(p.output_price_per_1m, 7.50);

	let p = provider.get_model_pricing("qwen3.7-plus").unwrap();
	assert_eq!(p.input_price_per_1m, 0.40);
	assert_eq!(p.output_price_per_1m, 1.60);
	assert_eq!(p.cache_read_price_per_1m, 0.08);

	let p = provider.get_model_pricing("qwen3.6-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.25);
	assert_eq!(p.cache_read_price_per_1m, 0.025);

	// Dated DeepSeek V4 snapshots carry their own busy-hour rate card; the
	// moving aliases keep the flat rate.
	let p = provider.get_model_pricing("deepseek-v4-flash-0731").unwrap();
	assert_eq!(p.input_price_per_1m, 0.44);
	assert_eq!(p.output_price_per_1m, 1.32);
	assert_eq!(p.cache_read_price_per_1m, 0.044);

	let p = provider.get_model_pricing("deepseek-v4-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.20);
	assert_eq!(p.cache_read_price_per_1m, 0.04);

	let p = provider.get_model_pricing("deepseek-v4-pro-0813").unwrap();
	assert_eq!(p.input_price_per_1m, 1.32);
	assert_eq!(p.output_price_per_1m, 3.96);
	assert_eq!(p.cache_read_price_per_1m, 0.132);

	let p = provider.get_model_pricing("deepseek-v4-pro").unwrap();
	assert_eq!(p.input_price_per_1m, 2.40);
	assert_eq!(p.output_price_per_1m, 4.80);
	assert_eq!(p.cache_read_price_per_1m, 0.20);

	// V4.1 Flash: busy rate, and the fixed busy-hour cache-hit price
	let p = provider.get_model_pricing("deepseek-v4.1-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.30);
	assert_eq!(p.output_price_per_1m, 1.20);
	assert_eq!(p.cache_read_price_per_1m, 0.006);

	// Unversioned aliases must not shadow more specific entries
	let p = provider.get_model_pricing("qwen-plus-latest").unwrap();
	assert_eq!(p.input_price_per_1m, 0.40);
	assert_eq!(p.output_price_per_1m, 1.20);
}

#[test]
fn test_pricing_added_models() {
	let provider = AlibabaProvider::new();

	// Tiered models resolve to their lowest tier.
	let p = provider.get_model_pricing("qwen3.7-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.03);
	assert_eq!(p.output_price_per_1m, 0.13);
	assert_eq!(p.cache_read_price_per_1m, 0.006);

	let p = provider.get_model_pricing("qwen3.6-max-preview").unwrap();
	assert_eq!(p.input_price_per_1m, 1.30);
	assert_eq!(p.output_price_per_1m, 7.80);
	assert_eq!(p.cache_read_price_per_1m, 0.13);

	let p = provider.get_model_pricing("qwen3.5-plus").unwrap();
	assert_eq!(p.input_price_per_1m, 0.40);
	assert_eq!(p.output_price_per_1m, 2.40);
	assert_eq!(p.cache_read_price_per_1m, 0.04);

	let p = provider.get_model_pricing("qwen3-vl-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.05);
	assert_eq!(p.output_price_per_1m, 0.40);
	assert_eq!(p.cache_read_price_per_1m, 0.01);

	let p = provider.get_model_pricing("qwen-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.05);
	assert_eq!(p.output_price_per_1m, 0.40);
	assert_eq!(p.cache_read_price_per_1m, 0.01);

	// Families that share a suffix with the legacy qwen-max/qwen-plus aliases
	// must resolve to their own row.
	let p = provider.get_model_pricing("qwen3-max-2026-01-23").unwrap();
	assert_eq!(p.input_price_per_1m, 1.20);
	assert_eq!(p.output_price_per_1m, 6.00);
	assert_eq!(p.cache_read_price_per_1m, 0.24);

	let p = provider.get_model_pricing("qwen-vl-max").unwrap();
	assert_eq!(p.input_price_per_1m, 0.80);
	assert_eq!(p.output_price_per_1m, 3.20);
	assert_eq!(p.cache_read_price_per_1m, 0.16);

	// Open-weight checkpoints
	let p = provider.get_model_pricing("qwen3.8-2.4t-a95b").unwrap();
	assert_eq!(p.input_price_per_1m, 2.00);
	assert_eq!(p.output_price_per_1m, 6.00);
	assert_eq!(p.cache_read_price_per_1m, 0.25);

	let p = provider.get_model_pricing("qwen3.8-27b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.50);
	assert_eq!(p.output_price_per_1m, 3.00);
	assert_eq!(p.cache_read_price_per_1m, 0.10);

	// No context caching: hits bill at the input rate.
	let p = provider.get_model_pricing("qwen3.6-27b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.60);
	assert_eq!(p.output_price_per_1m, 3.60);
	assert_eq!(p.cache_read_price_per_1m, 0.60);

	let p = provider.get_model_pricing("qwen3.6-35b-a3b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.375);
	assert_eq!(p.output_price_per_1m, 2.25);
	assert_eq!(p.cache_read_price_per_1m, 0.375);

	let p = provider.get_model_pricing("qwen3.5-397b-a17b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.60);
	assert_eq!(p.output_price_per_1m, 3.60);

	let p = provider.get_model_pricing("qwen3.5-122b-a10b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.40);
	assert_eq!(p.output_price_per_1m, 3.20);

	let p = provider.get_model_pricing("qwen3.5-35b-a3b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.25);
	assert_eq!(p.output_price_per_1m, 2.00);

	let p = provider.get_model_pricing("qwen3.5-27b").unwrap();
	assert_eq!(p.input_price_per_1m, 0.30);
	assert_eq!(p.output_price_per_1m, 2.40);

	// Older open-weight checkpoints must not fall through to the reference table.
	for (model, input, output) in [
		("qwen3-235b-a22b-thinking-2507", 0.23, 2.30),
		("qwen3-235b-a22b-instruct-2507", 0.23, 0.92),
		("qwen3-235b-a22b", 0.70, 2.80),
		("qwen3-32b", 0.16, 0.64),
		("qwen3-8b", 0.18, 0.70),
		("qwen3-vl-235b-a22b-thinking", 0.40, 4.00),
		("qwen3-vl-235b-a22b-instruct", 0.40, 1.60),
		("qwen3-coder-next", 0.30, 1.50),
		("qwen3-coder-480b-a35b-instruct", 1.50, 7.50),
		("qwen3-coder-30b-a3b-instruct", 0.45, 2.25),
		("qwen-plus-character", 0.50, 1.40),
		("qwen-plus-character-ja", 0.50, 1.40),
		("deepseek-v3.2", 0.57, 1.71),
		("glm-5.1", 1.40, 4.40),
	] {
		let p = provider.get_model_pricing(model).unwrap();
		assert_eq!(p.input_price_per_1m, input, "{model}");
		assert_eq!(p.output_price_per_1m, output, "{model}");
	}

	// Explicit-cache-only models bill hits at 10% of input.
	let p = provider.get_model_pricing("qwen3.6-plus").unwrap();
	assert_eq!(p.cache_read_price_per_1m, 0.05);
	let p = provider.get_model_pricing("qwen3.5-flash").unwrap();
	assert_eq!(p.cache_read_price_per_1m, 0.01);

	// Third-party
	let p = provider.get_model_pricing("glm-5.3").unwrap();
	assert_eq!(p.input_price_per_1m, 1.40);
	assert_eq!(p.output_price_per_1m, 4.40);
	assert_eq!(p.cache_read_price_per_1m, 0.28);

	// Prime bills 2x and must not fall through to glm-5.3.
	let p = provider.get_model_pricing("glm-5.3-prime").unwrap();
	assert_eq!(p.input_price_per_1m, 2.80);
	assert_eq!(p.output_price_per_1m, 8.80);
	assert_eq!(p.cache_read_price_per_1m, 0.56);

	// The fast preview bills 2x and must not fall through to glm-5.2.
	let p = provider.get_model_pricing("glm-5.2-fast-preview").unwrap();
	assert_eq!(p.input_price_per_1m, 2.80);
	assert_eq!(p.output_price_per_1m, 8.80);
	assert_eq!(p.cache_read_price_per_1m, 0.56);
	let p = provider.get_model_pricing("glm-5.2").unwrap();
	assert_eq!(p.input_price_per_1m, 1.40);
	assert_eq!(p.cache_read_price_per_1m, 0.28);
	let p = provider.get_model_pricing("glm-5.1").unwrap();
	assert_eq!(p.cache_read_price_per_1m, 0.26);

	// K3.1 Pro is cheaper than K3 and must not fall through to kimi-k3.
	let p = provider.get_model_pricing("kimi-k3.1-pro").unwrap();
	assert_eq!(p.input_price_per_1m, 2.00);
	assert_eq!(p.output_price_per_1m, 10.00);
	assert_eq!(p.cache_read_price_per_1m, 0.20);

	let p = provider.get_model_pricing("kimi-k3").unwrap();
	assert_eq!(p.input_price_per_1m, 3.00);
	assert_eq!(p.output_price_per_1m, 15.00);
	assert_eq!(p.cache_read_price_per_1m, 0.60);

	let p = provider.get_model_pricing("kimi-k2.7-code").unwrap();
	assert_eq!(p.input_price_per_1m, 0.95);
	assert_eq!(p.output_price_per_1m, 4.00);
	assert_eq!(p.cache_read_price_per_1m, 0.19);
}

#[test]
fn test_pricing_falls_back_to_reference() {
	let provider = AlibabaProvider::new();
	assert!(!is_model_in_pricing_table("kimi-k2.5", PRICING));
	let p = provider.get_model_pricing("kimi-k2.5").unwrap();
	assert!(p.input_price_per_1m > 0.0);
}

#[test]
fn test_qwen3_8_flash_pricing() {
	let provider = AlibabaProvider::new();
	let p = provider.get_model_pricing("qwen3.8-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.15);
	assert_eq!(p.output_price_per_1m, 0.47);
	assert_eq!(p.cache_write_price_per_1m, 0.15);
	assert_eq!(p.cache_read_price_per_1m, 0.016);
}

#[test]
fn test_qwen3_8_omni_flash_pricing() {
	let provider = AlibabaProvider::new();
	let p = provider.get_model_pricing("qwen3.8-omni-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.15);
	assert_eq!(p.output_price_per_1m, 0.47);
	assert_eq!(p.cache_write_price_per_1m, 0.15);
	assert_eq!(p.cache_read_price_per_1m, 0.016);
}

#[test]
fn test_qwen3_5_omni_flash_pricing() {
	let provider = AlibabaProvider::new();
	// International list price: text/image/video input $0.40, text output
	// $2.20. Context caching is unsupported, so hits bill at the input rate.
	let p = provider.get_model_pricing("qwen3.5-omni-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.40);
	assert_eq!(p.output_price_per_1m, 2.20);
	assert_eq!(p.cache_write_price_per_1m, 0.40);
	assert_eq!(p.cache_read_price_per_1m, 0.40);
}

#[test]
fn test_qwen3_omni_flash_pricing() {
	let provider = AlibabaProvider::new();
	// International list price: text input $0.43, text output after text-only
	// input $1.66. Not on the implicit-cache list, so hits bill at the input rate.
	let p = provider.get_model_pricing("qwen3-omni-flash").unwrap();
	assert_eq!(p.input_price_per_1m, 0.43);
	assert_eq!(p.output_price_per_1m, 1.66);
	assert_eq!(p.cache_write_price_per_1m, 0.43);
	assert_eq!(p.cache_read_price_per_1m, 0.43);
}

#[test]
fn test_qwen3_5_omni_plus_pricing() {
	let provider = AlibabaProvider::new();
	// International list price: text/image/video input $1.40, text output
	// $8.30. Context caching is unsupported, so hits bill at the input rate.
	// https://www.alibabacloud.com/help/en/model-studio/qwen3-5-omni-plus
	let p = provider.get_model_pricing("qwen3.5-omni-plus").unwrap();
	assert_eq!(p.input_price_per_1m, 1.40);
	assert_eq!(p.output_price_per_1m, 8.30);
	assert_eq!(p.cache_write_price_per_1m, 1.40);
	assert_eq!(p.cache_read_price_per_1m, 1.40);

	// The dated snapshot carries the moving alias's list rates.
	let p = provider.get_model_pricing("qwen3.5-omni-plus-2026-03-15").unwrap();
	assert_eq!(p.input_price_per_1m, 1.40);
	assert_eq!(p.output_price_per_1m, 8.30);
}

/// qwen-long is a text-generation model with no published international USD
/// rate, so it is declared without a price and resolves its context window
/// from the reference table.
#[test]
fn test_qwen_long_resolves_without_a_published_price() {
	let provider = AlibabaProvider::new();
	assert!(provider.supports_model("qwen-long"));
	assert!(!is_model_in_pricing_table("qwen-long", PRICING));
	assert!(provider.get_model_pricing("qwen-long").is_none());
	// HTTP requests carry at most 1M tokens of the 10M-token window.
	assert_eq!(provider.get_max_input_tokens("qwen-long"), 1_000_000);
	assert!(!provider.supports_vision("qwen-long"));
}

#[test]
fn test_qwen3_8_max_0902_snapshot_pricing() {
	let provider = AlibabaProvider::new();
	// The dated snapshot carries the moving alias's list rates.
	let p = provider.get_model_pricing("qwen3.8-max-0902").unwrap();
	assert_eq!(p.input_price_per_1m, 2.00);
	assert_eq!(p.output_price_per_1m, 6.00);
	assert_eq!(p.cache_read_price_per_1m, 0.25);
}

#[test]
fn test_cost_calculation() {
	let provider = AlibabaProvider::new();
	let pricing = provider.get_model_pricing("qwen3.8-max").unwrap();

	// 1M input + 500K output, no cache
	let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
	let expected = 2.00 + 0.5 * 6.00; // $5.00
	assert!((cost - expected).abs() < 0.001);

	// Cache hits must be cheaper than fresh input
	let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
	assert!(cost_cached < cost);

	let long = calculate_local_usage_cost("qwen3.7-plus", 256_001, 0, 0, 100_000, None).unwrap();
	let expected_long = 256_001.0 / 1_000_000.0 * 1.20 + 0.1 * 4.80;
	assert!((long - expected_long).abs() < 0.001);

	let boundary = calculate_local_usage_cost("qwen3.7-plus", 256_000, 0, 0, 100_000, None).unwrap();
	let expected_boundary = 0.256 * 0.40 + 0.1 * 1.60;
	assert!((boundary - expected_boundary).abs() < 0.001);

	// Dated snapshots follow the same list-price tiers as the moving alias.
	let snapshot_long = calculate_local_usage_cost("qwen3.7-plus-2026-05-26", 256_001, 0, 0, 100_000, None).unwrap();
	assert!((snapshot_long - expected_long).abs() < 0.001);
}

#[test]
fn the_default_endpoint_follows_the_mode() {
	// (ALIBABA_API_MODE, ALIBABA_API_URL, the protocol and URL that follow)
	for (mode, api_url, want_mode, want_url) in [
		// With nothing configured the default is Chat Completions.
		(None, None, ApiMode::OpenAi, ALIBABA_API_URL),
		(Some(""), None, ApiMode::OpenAi, ALIBABA_API_URL),
		(Some(" OpenAI "), None, ApiMode::OpenAi, ALIBABA_API_URL),
		// A configured Messages base is taken at its word, exactly as given:
		// the documented pay-as-you-go workspace shape, and a plan base.
		(
			None,
			Some(WORKSPACE_MESSAGES_URL),
			ApiMode::Anthropic,
			WORKSPACE_MESSAGES_URL,
		),
		(None, Some(PLAN_MESSAGES_URL), ApiMode::Anthropic, PLAN_MESSAGES_URL),
		// A URL naming no endpoint was always served by Chat Completions.
		(
			None,
			Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
			ApiMode::OpenAi,
			"https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
		),
		// The explicit mode beats the URL's path, and the URL is not rewritten.
		(
			Some("openai"),
			Some(WORKSPACE_MESSAGES_URL),
			ApiMode::OpenAi,
			WORKSPACE_MESSAGES_URL,
		),
		(
			Some("anthropic"),
			Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"),
			ApiMode::Anthropic,
			"https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
		),
	] {
		assert_eq!(
			select_endpoint(mode, api_url, None).expect("selected"),
			(want_mode, want_url.to_owned()),
			"mode {mode:?}, url {api_url:?}"
		);
	}
}

/// Model Studio's documented pay-as-you-go Messages base (per workspace).
const WORKSPACE_MESSAGES_URL: &str = "https://ws-1234abcd.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages";
/// A Token Plan / Coding Plan base (also `/apps/anthropic/v1/messages`), used
/// here only as an example of an explicitly configured endpoint.
const PLAN_MESSAGES_URL: &str = "https://plan.example/apps/anthropic/v1/messages";

/// A workspace identity selects the official regional pay-as-you-go Messages
/// base, unchanged in every part but the host label.
#[test]
fn a_configured_workspace_selects_the_payg_messages_default() {
	assert_eq!(
		select_endpoint(None, None, Some("ws-1234abcd")).expect("selected"),
		(ApiMode::Anthropic, WORKSPACE_MESSAGES_URL.to_owned())
	);
	// No mode needed, and an empty identity is an unset one.
	assert_eq!(
		select_endpoint(Some(""), Some(""), Some("ws-1234abcd")).expect("selected"),
		(ApiMode::Anthropic, WORKSPACE_MESSAGES_URL.to_owned())
	);
	// A URL is still the most specific configuration and is not rewritten.
	assert_eq!(
		select_endpoint(None, Some(PLAN_MESSAGES_URL), Some("ws-1234abcd")).expect("selected"),
		(ApiMode::Anthropic, PLAN_MESSAGES_URL.to_owned())
	);
	// An explicit Chat mode keeps the Chat route, workspace or not.
	assert_eq!(
		select_endpoint(Some("openai"), None, Some("ws-1234abcd")).expect("selected"),
		(ApiMode::OpenAi, ALIBABA_API_URL.to_owned())
	);
}

/// Nothing that is not a single DNS label reaches the host, so no configured
/// value can add a domain, a path or a port.
#[test]
fn workspace_ids_that_are_not_a_single_dns_label_are_refused() {
	for workspace in [
		"",
		"ws.attacker.example",
		"ws/../other",
		"ws:8443",
		"-ws",
		"ws-",
		"WS-UPPER",
		"ws?x=1",
		"ws#frag",
		"ws id",
		"ws@host",
		"ws%2e",
		"ws\\host",
	] {
		let error = payg_messages_url(workspace).expect_err("not a DNS label");
		assert!(
			error.to_string().contains("single DNS label"),
			"unexpected error for {workspace:?}: {error}"
		);
	}
	let long = "a".repeat(64);
	assert!(payg_messages_url(&long).is_err(), "64 characters is not a label");
	assert!(payg_messages_url(&"a".repeat(63)).is_ok(), "63 is the longest label");
}

/// Without an identity or a URL the safe existing Chat default stands, and an
/// explicit Messages mode is refused instead of pointed at a guessed host.
#[test]
fn no_workspace_and_no_url_keep_the_chat_default() {
	assert_eq!(
		ALIBABA_API_URL,
		"https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
	);
	assert_eq!(
		select_endpoint(None, None, None).expect("selected"),
		(ApiMode::OpenAi, ALIBABA_API_URL.to_owned())
	);
	let error = select_endpoint(Some("anthropic"), None, None).expect_err("no derivable host");
	assert!(
		error.to_string().contains("ALIBABA_API_URL") && error.to_string().contains("ALIBABA_WORKSPACE_ID"),
		"the error names both variables that resolve it: {error}"
	);
	// The legacy generic path is never substituted for a missing URL.
	assert!(!ALIBABA_API_URL.contains("claude-code-proxy"));
}

/// The Chat-only live probe and the Chat-only video field stay on the Chat
/// route: a Messages turn must not advertise either.
#[test]
#[serial_test::serial]
fn schema_and_video_claims_follow_the_route() {
	let provider = AlibabaProvider::new();
	let _env = EnvGuard::new()
		.remove("ALIBABA_API_MODE")
		.remove("ALIBABA_API_URL")
		.remove("ALIBABA_WORKSPACE_ID");
	assert!(provider.enforces_response_schema("qwen3.8-max"));
	assert!(provider.supports_video("qwen3.8-max"));
	assert!(provider.supports_vision("qwen3.8-max"));

	let _env = _env.set("ALIBABA_WORKSPACE_ID", "ws-1234abcd");
	assert!(
		!provider.enforces_response_schema("qwen3.8-max"),
		"Messages takes the forced-tool path: {enabled:?}",
		enabled = provider.enforces_response_schema("qwen3.8-max")
	);
	assert!(!provider.supports_video("qwen3.8-max"));
	assert!(!provider.supports_video("qwen3.6-flash"));
	// Images and the context window are the same on both routes.
	assert!(provider.supports_vision("qwen3.8-max"));
	assert_eq!(provider.get_max_input_tokens("qwen3.8-max"), 1_000_000);
}

/// `output_config.effort` is documented per model family on the Messages
/// contract; an effort floors to the nearest documented value and a family the
/// contract does not list gets no knob at all.
#[test]
fn messages_effort_follows_the_documented_family_tiers() {
	use ReasoningEffort::*;
	let cases: [(&str, [&str; 5], &str); 5] = [
		("qwen4-max", ["low", "high", "high", "xhigh", "xhigh"], "qwen4-max"),
		("qwen3.8-max", ["low", "medium", "xhigh", "xhigh", "xhigh"], "qwen3.8"),
		("glm-5.3", ["low", "high", "high", "max", "max"], "glm"),
		("deepseek-v4-pro", ["high", "high", "high", "max", "max"], "deepseek"),
		("kimi-k3.1-pro", ["low", "medium", "high", "high", "max"], "kimi"),
	];
	for (model, tiers, label) in cases {
		for (effort, want) in [
			(None, tiers[0]),
			(Low, tiers[0]),
			(Medium, tiers[1]),
			(High, tiers[2]),
			(XHigh, tiers[3]),
			(Max, tiers[4]),
		] {
			assert_eq!(
				messages_effort(model, effort),
				Some(want),
				"{label} {model} at {effort:?}"
			);
		}
	}
	assert_eq!(messages_effort("qwen3-coder-plus", High), Option::None);
	assert_eq!(messages_effort("unknown-model", Max), Option::None);
}

/// The knob lands beside a requested schema in the same `output_config` object.
#[test]
fn the_thinking_hook_merges_with_a_requested_schema() {
	let mut body = serde_json::json!({
		"output_config": { "format": { "type": "json_schema", "schema": { "type": "object" } } }
	});
	(ANTHROPIC_COMPAT.thinking)("qwen4-max", Some(ReasoningEffort::Max), &mut body);
	assert_eq!(body["output_config"]["effort"], "xhigh");
	assert_eq!(body["output_config"]["format"]["type"], "json_schema");

	let mut untouched = serde_json::json!({});
	(ANTHROPIC_COMPAT.thinking)("qwen3-coder-plus", Some(ReasoningEffort::High), &mut untouched);
	assert!(
		untouched.get("output_config").is_none(),
		"a family with no documented tier sends nothing: {untouched}"
	);
}
#[test]
fn an_unknown_mode_is_refused_before_any_request() {
	let error = select_endpoint(Some("claude"), None, Some("ws-1234abcd")).expect_err("refused");
	assert!(
		error.to_string().contains("ALIBABA_API_MODE must be"),
		"unexpected error: {error}"
	);
}

#[test]
fn messages_images_follow_the_reference_capabilities() {
	assert_eq!((ANTHROPIC_COMPAT.images)("qwen3-vl-plus"), Images::Base64);
	assert_eq!(
		(ANTHROPIC_COMPAT.images)("deepseek-v4-pro"),
		Images::None,
		"a known text-only model refuses an image instead of dropping it"
	);
	assert_eq!(
		(ANTHROPIC_COMPAT.images)("custom-model-not-in-the-table"),
		Images::Base64,
		"an unknown model is not refused on a guess"
	);
}

const MESSAGES_BODY: &str = r#"{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"hi from messages"}],"stop_reason":"end_turn","usage":{"input_tokens":1000000,"output_tokens":1000000}}"#;
const CHAT_BODY: &str = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"qwen4-max","choices":[{"index":0,"message":{"role":"assistant","content":"chat answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000000,"completion_tokens":1000000,"total_tokens":2000000}}"#;
/// qwen4-max at 1M input and 1M output: $1.50 + $4.50.
const QWEN4_MAX_MILLION_COST: f64 = 6.00;

fn png_message() -> crate::llm::types::Message {
	use crate::llm::types::{ImageAttachment, ImageData, Message, SourceType};
	Message::user("what is this?").with_images(vec![ImageAttachment {
		data: ImageData::Base64("aW1hZ2U=".into()),
		media_type: "image/png".into(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
	}])
}

#[tokio::test]
#[serial_test::serial]
async fn messages_mode_speaks_messages_on_the_configured_url() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_BODY.to_string())]);
	let path = "/api/v2/apps/claude-code-proxy/v1/messages";
	let _env = EnvGuard::new()
		.set("ALIBABA_API_URL", format!("{}{path}", mock.base_url()))
		.set("ALIBABA_API_MODE", "");

	let params = ChatCompletionParams::new(&[png_message()], "qwen4-max", 100).with_api_key("mock-key".to_owned());
	let response = AlibabaProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	assert_eq!(response.content, "hi from messages");
	let request = &mock.requests()[0];
	assert_eq!(request.method, "POST");
	assert_eq!(request.path, path, "the URL is not rewritten");
	assert_eq!(request.header("authorization"), Some("Bearer mock-key"));
	assert!(request.header("anthropic-version").is_some(), "Messages header");

	let sent: serde_json::Value = serde_json::from_str(&request.body).expect("request json");
	assert_eq!(sent["model"], "qwen4-max");
	assert_eq!(sent["stream"], true);
	assert_eq!(sent["messages"][0]["content"][1]["type"], "image");
	assert_eq!(sent["messages"][0]["content"][1]["source"]["data"], "aW1hZ2U=");
	for field in ["stream_options", "output_config", "thinking"] {
		assert!(sent.get(field).is_none(), "{field} must not be sent");
	}

	let usage = response.exchange.usage.expect("usage");
	assert_eq!((usage.input_tokens, usage.output_tokens), (1_000_000, 1_000_000));
	assert_eq!(usage.cost, Some(QWEN4_MAX_MILLION_COST));
}

/// Every model gets the cache marker; one without the explicit cache, priced
/// by the reference table only, keeps the implicit rates.
#[tokio::test]
#[serial_test::serial]
async fn messages_mark_every_model_and_price_reference_only_models() {
	use crate::llm::types::{FunctionDefinition, Message};

	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(
			"ALIBABA_API_URL",
			format!("{}/apps/anthropic/v1/messages", mock.base_url()),
		)
		.set("ALIBABA_API_MODE", "");

	// Priced by the reference table only.
	let model = "kimi-k2.5";
	assert!(!is_model_in_pricing_table(model, PRICING));

	let mut message = Message::user("hi");
	message.cached = true;
	let tool = FunctionDefinition {
		name: "lookup".to_owned(),
		description: "Look something up".to_owned(),
		parameters: serde_json::json!({ "type": "object", "properties": {} }),
		cache_control: Some(serde_json::json!({ "type": "ephemeral" })),
		freeform: None,
	};
	let params = ChatCompletionParams::new(&[message], model, 100)
		.with_api_key("k".to_owned())
		.with_tools(vec![tool]);
	let response = AlibabaProvider::new()
		.chat_completion(params)
		.await
		.expect("messages completion");

	let sent: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	let marker = serde_json::json!({ "type": "ephemeral" });
	assert_eq!(sent["messages"][0]["content"][0]["cache_control"], marker);
	assert_eq!(sent["tools"][0]["cache_control"], marker);

	let usage = response.exchange.usage.expect("usage");
	assert!(usage.cost.is_some_and(|cost| cost > 0.0), "reference-priced cost");
}

#[tokio::test]
#[serial_test::serial]
async fn openai_mode_keeps_chat_completions_and_its_pricing() {
	use crate::llm::types::Message;

	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, CHAT_BODY.to_string())]);
	let path = "/compatible-mode/v1/chat/completions";
	let _env = EnvGuard::new()
		.set("ALIBABA_API_URL", format!("{}{path}", mock.base_url()))
		.set("ALIBABA_API_MODE", "openai");

	let params = ChatCompletionParams::new(&[Message::user("hi")], "qwen4-max", 100).with_api_key("k".to_owned());
	let response = AlibabaProvider::new()
		.chat_completion(params)
		.await
		.expect("chat completion");

	assert_eq!(response.content, "chat answer");
	assert_eq!(mock.requests()[0].path, path, "the URL is unchanged");
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(usage.cost, Some(QWEN4_MAX_MILLION_COST));
}

/// A Messages write and hit on a model with the explicit cache, as the Token
/// Plan reported them on 2026-11-28.
const MESSAGES_EXPLICIT_BODY: &str = r#"{"id":"msg_2","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":0,"cache_creation_input_tokens":1000000,"cache_read_input_tokens":1000000,"output_tokens":0}}"#;
/// The same on Chat Completions, where `prompt_tokens` includes both.
const CHAT_EXPLICIT_BODY: &str = r#"{"id":"chatcmpl-2","object":"chat.completion","created":1,"model":"qwen4-max","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2000000,"completion_tokens":0,"total_tokens":2000000,"prompt_tokens_details":{"cached_tokens":1000000,"cache_creation_input_tokens":1000000,"cache_type":"ephemeral"}}}"#;

fn cached_user(text: &str) -> crate::llm::types::Message {
	let mut message = crate::llm::types::Message::user(text);
	message.cached = true;
	message
}

#[tokio::test]
#[serial_test::serial]
async fn messages_bill_an_explicit_cache_at_its_own_shares() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_EXPLICIT_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(
			"ALIBABA_API_URL",
			format!("{}/apps/anthropic/v1/messages", mock.base_url()),
		)
		.set("ALIBABA_API_MODE", "");

	let params = ChatCompletionParams::new(&[cached_user("hi")], "qwen3.8-flash", 100).with_api_key("k".to_owned());
	let response = AlibabaProvider::new()
		.chat_completion(params)
		.await
		.expect("completion");

	let sent: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(
		sent["messages"][0]["content"][0]["cache_control"],
		serde_json::json!({ "type": "ephemeral" })
	);
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(usage.cache_write_tokens, usage.cache_read_tokens),
		(1_000_000, 1_000_000)
	);
	let cost = usage.cost.expect("cost");
	assert!((cost - (0.1875 + 0.015)).abs() < 1e-9, "cost {cost}");
}

#[tokio::test]
#[serial_test::serial]
async fn chat_sends_the_hour_marker_for_the_qwen4_long_cache() {
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, CHAT_EXPLICIT_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(
			"ALIBABA_API_URL",
			format!("{}/compatible-mode/v1/chat/completions", mock.base_url()),
		)
		.set("ALIBABA_API_MODE", "");

	let params = ChatCompletionParams::new(&[cached_user("hi")], "qwen4-max", 100)
		.with_api_key("k".to_owned())
		.with_long_cache(true);
	let response = AlibabaProvider::new()
		.chat_completion(params)
		.await
		.expect("completion");

	let sent: serde_json::Value = serde_json::from_str(&mock.requests()[0].body).expect("request json");
	assert_eq!(
		sent["messages"][0]["content"],
		serde_json::json!([{ "type": "text", "text": "hi", "cache_control": { "type": "ephemeral", "ttl": "1h" } }])
	);
	let usage = response.exchange.usage.expect("usage");
	assert_eq!(
		(usage.input_tokens, usage.cache_write_tokens, usage.cache_read_tokens),
		(0, 1_000_000, 1_000_000)
	);
	// qwen4-max input is $1.50 per 1M: a one-hour write at 200%, a hit at 10%.
	let cost = usage.cost.expect("cost");
	assert!((cost - (3.0 + 0.15)).abs() < 1e-9, "cost {cost}");
}

#[tokio::test]
#[serial_test::serial]
async fn messages_refuse_video_before_any_request() {
	use crate::llm::types::{Message, SourceType, VideoAttachment, VideoData};

	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, MESSAGES_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(
			"ALIBABA_API_URL",
			format!("{}/api/v2/apps/claude-code-proxy/v1/messages", mock.base_url()),
		)
		.set("ALIBABA_API_MODE", "");

	let clip = Message::user("watch").with_videos(vec![VideoAttachment {
		data: VideoData::Base64("dmlkZW8=".into()),
		media_type: "video/mp4".into(),
		source_type: SourceType::Url,
		dimensions: None,
		size_bytes: None,
		duration_secs: None,
	}]);
	let error = AlibabaProvider::new()
		.chat_completion(ChatCompletionParams::new(&[clip], "qwen4-max", 10).with_api_key("k".to_owned()))
		.await
		.expect_err("Messages has no video block");
	assert!(
		error.to_string().contains("does not accept video input"),
		"unexpected error: {error}"
	);
	assert!(mock.requests().is_empty(), "a refused request is not sent");
}

#[test]
fn test_native_schema_capability_is_model_specific() {
	assert!(natively_enforces_response_schema("qwen3.8-flash"));
	assert!(natively_enforces_response_schema("qwen3.8-flash-2026-08-26"));
	assert!(natively_enforces_response_schema("qwen3.7-plus"));
	assert!(natively_enforces_response_schema("qwen3.7-flash"));
	assert!(!natively_enforces_response_schema("deepseek-v4-flash-0731"));
	assert!(!natively_enforces_response_schema("qwen3.6-flash"));
}
