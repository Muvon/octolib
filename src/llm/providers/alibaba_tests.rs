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

#[test]
fn test_supports_model() {
    let provider = AlibabaProvider::new();
    assert!(provider.supports_model("qwen3.8-max"));
    assert!(provider.supports_model("qwen3-coder-plus"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
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

    let p = provider
        .get_model_pricing("qwen3.7-max-2026-06-08")
        .unwrap();
    assert_eq!(p.input_price_per_1m, 2.50);
    assert_eq!(p.output_price_per_1m, 7.50);

    let p = provider.get_model_pricing("qwen3.7-plus").unwrap();
    assert_eq!(p.input_price_per_1m, 0.40);
    assert_eq!(p.output_price_per_1m, 1.60);
    assert_eq!(p.cache_read_price_per_1m, 0.08);

    let p = provider.get_model_pricing("qwen3.6-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.25);
    assert_eq!(p.cache_read_price_per_1m, 0.05);

    // Dated DeepSeek V4 snapshots carry their own busy-hour rate card; the
    // moving aliases keep the flat rate.
    let p = provider
        .get_model_pricing("deepseek-v4-flash-0731")
        .unwrap();
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

    // V4.1 Flash: busy rate, and the 10% cache-hit exception
    let p = provider.get_model_pricing("deepseek-v4.1-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.30);
    assert_eq!(p.output_price_per_1m, 1.20);
    assert_eq!(p.cache_read_price_per_1m, 0.03);

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

    // Third-party
    let p = provider.get_model_pricing("glm-5.3").unwrap();
    assert_eq!(p.input_price_per_1m, 1.40);
    assert_eq!(p.output_price_per_1m, 4.40);
    assert_eq!(p.cache_read_price_per_1m, 0.28);

    let p = provider.get_model_pricing("kimi-k3").unwrap();
    assert_eq!(p.input_price_per_1m, 3.00);
    assert_eq!(p.output_price_per_1m, 15.00);
    assert_eq!(p.cache_read_price_per_1m, 0.30);
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

    let long = calculate_local_usage_cost("qwen3.7-plus", 256_001, 0, 0, 100_000).unwrap();
    let expected_long = 256_001.0 / 1_000_000.0 * 1.20 + 0.1 * 4.80;
    assert!((long - expected_long).abs() < 0.001);

    let boundary = calculate_local_usage_cost("qwen3.7-plus", 256_000, 0, 0, 100_000).unwrap();
    let expected_boundary = 0.256 * 0.40 + 0.1 * 1.60;
    assert!((boundary - expected_boundary).abs() < 0.001);

    // Dated snapshots follow the same list-price tiers as the moving alias.
    let snapshot_long =
        calculate_local_usage_cost("qwen3.7-plus-2026-05-26", 256_001, 0, 0, 100_000).unwrap();
    assert!((snapshot_long - expected_long).abs() < 0.001);
}

#[test]
fn test_native_schema_capability_is_model_specific() {
    assert!(natively_enforces_response_schema("qwen3.8-flash"));
    assert!(natively_enforces_response_schema(
        "qwen3.8-flash-2026-08-26"
    ));
    assert!(natively_enforces_response_schema("qwen3.7-plus"));
    assert!(natively_enforces_response_schema("qwen3.7-flash"));
    assert!(!natively_enforces_response_schema("deepseek-v4-flash-0731"));
    assert!(!natively_enforces_response_schema("qwen3.6-flash"));
}
