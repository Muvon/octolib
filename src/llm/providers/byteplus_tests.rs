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

#[test]
fn test_supports_model() {
    let provider = BytePlusProvider::new();
    assert!(provider.supports_model("seed-2-0-pro-260328"));
    assert!(provider.supports_model("seed-2-0-lite-260228"));
    assert!(provider.supports_model("dola-seed-2.0-pro"));
    assert!(provider.supports_model("glm-4-7-251222"));
    assert!(provider.supports_model("any-model"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = BytePlusProvider::new();
    assert_eq!(provider.name(), "byteplus");
    assert!(provider.supports_caching("any-model"));
    assert!(provider.supports_structured_output("any-model"));
}

#[test]
fn test_pricing_seed_models() {
    let provider = BytePlusProvider::new();

    let p = provider.get_model_pricing("seed-2-0-pro-260328").unwrap();
    assert_eq!(p.input_price_per_1m, 0.50);
    assert_eq!(p.output_price_per_1m, 3.00);
    assert_eq!(p.cache_read_price_per_1m, 0.10);

    let p = provider.get_model_pricing("seed-2-0-mini-260215").unwrap();
    assert_eq!(p.input_price_per_1m, 0.10);
    assert_eq!(p.output_price_per_1m, 0.40);

    let p = provider.get_model_pricing("seed-2-1-turbo-260812").unwrap();
    assert_eq!(p.input_price_per_1m, 0.50);
    assert_eq!(p.output_price_per_1m, 2.50);

    let p = provider.get_model_pricing("seed-1-6-flash-250715").unwrap();
    assert_eq!(p.input_price_per_1m, 0.075);
    assert_eq!(p.output_price_per_1m, 0.30);
}

#[test]
fn test_pricing_coding_plan_aliases() {
    let provider = BytePlusProvider::new();

    let p = provider.get_model_pricing("dola-seed-2.0-pro").unwrap();
    assert_eq!(p.input_price_per_1m, 0.50);
    assert_eq!(p.output_price_per_1m, 3.00);

    let p = provider.get_model_pricing("dola-seed-2.0-lite").unwrap();
    assert_eq!(p.input_price_per_1m, 0.25);

    let p = provider.get_model_pricing("bytedance-seed-code").unwrap();
    assert_eq!(p.input_price_per_1m, 0.50);
}

#[test]
fn test_pricing_falls_back_to_reference() {
    let provider = BytePlusProvider::new();
    let p = provider.get_model_pricing("glm-5.1").unwrap();
    assert!(p.input_price_per_1m > 0.0);
}

#[test]
fn test_cost_calculation() {
    let provider = BytePlusProvider::new();
    let pricing = provider.get_model_pricing("seed-2-0-pro-260328").unwrap();

    // 1M input + 500K output, no cache
    let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
    let expected = 0.50 + 0.5 * 3.00; // $2.00
    assert!((cost - expected).abs() < 0.001);

    // With cache: 500K regular + 500K cache_read + 500K output
    let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
    let expected_cached = 0.25 + 0.5 * 0.10 + 0.5 * 3.00; // $1.80
    assert!((cost_cached - expected_cached).abs() < 0.001);
    assert!(cost_cached < cost);

    // Seed 2.0 prompts above 128K use the documented 2x tier.
    let long = calculate_usage_cost("seed-2-0-pro-260328", 128_001, 0, 0, 100_000).unwrap();
    let expected_long = 128_001.0 / 1_000_000.0 * 1.00 + 0.1 * 6.00;
    assert!((long - expected_long).abs() < 0.001);

    // Exactly 128K remains in the base tier.
    let boundary = calculate_usage_cost("seed-2-0-pro-260328", 128_000, 0, 0, 100_000).unwrap();
    let expected_boundary = 0.128 * 0.50 + 0.1 * 3.00;
    assert!((boundary - expected_boundary).abs() < 0.001);
}
