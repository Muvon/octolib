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
    let provider = MetaProvider::new();
    assert!(provider.supports_model("muse-spark-1.3"));
    assert!(provider.supports_model("muse-spark-1.3-contributor"));
    assert!(provider.supports_model("muse-spark-1.2"));
    assert!(provider.supports_model("muse-spark-1.2-contributor"));
    assert!(provider.supports_model("muse-spark-1.1"));
    // Closed catalog: unknown and legacy models are rejected
    assert!(!provider.supports_model("muse-spark-1.4"));
    assert!(!provider.supports_model("llama-4-maverick"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_capabilities() {
    let provider = MetaProvider::new();
    assert_eq!(provider.name(), "meta");
    for model in ["muse-spark-1.3", "muse-spark-1.1"] {
        assert!(provider.supports_caching(model));
        assert!(provider.supports_vision(model));
        assert!(provider.supports_video(model));
        assert!(provider.supports_structured_output(model));
        assert!(provider.enforces_response_schema(model));
        assert!(!provider.supports_required_tool_choice(model));
        assert_eq!(provider.get_max_input_tokens(model), 1_048_576);
    }
}

#[test]
fn test_pricing_standard_tier() {
    let provider = MetaProvider::new();
    for model in ["muse-spark-1.3", "muse-spark-1.2", "muse-spark-1.1"] {
        let p = provider.get_model_pricing(model).unwrap();
        assert_eq!(p.input_price_per_1m, 1.25);
        assert_eq!(p.output_price_per_1m, 4.25);
        assert_eq!(p.cache_write_price_per_1m, 1.25);
        assert_eq!(p.cache_read_price_per_1m, 0.15);
    }
}

#[test]
fn test_pricing_contributor_tier_wins_over_base_pattern() {
    let provider = MetaProvider::new();
    for model in ["muse-spark-1.3-contributor", "muse-spark-1.2-contributor"] {
        let p = provider.get_model_pricing(model).unwrap();
        assert_eq!(p.input_price_per_1m, 0.10);
        assert_eq!(p.output_price_per_1m, 0.20);
        assert_eq!(p.cache_write_price_per_1m, 0.10);
        assert_eq!(p.cache_read_price_per_1m, 0.002);
    }
}

#[test]
fn test_cost_calculation_cached_input() {
    let provider = MetaProvider::new();
    let pricing = provider.get_model_pricing("muse-spark-1.3").unwrap();

    // 1M input + 500K output (reasoning tokens bill at the output rate)
    let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
    let expected = 1.25 + 0.5 * 4.25;
    assert!((cost - expected).abs() < 0.001);

    // 500K regular + 500K cached input + 500K output → caching saves money
    let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
    let expected_cached = 0.5 * 1.25 + 0.5 * 0.15 + 0.5 * 4.25;
    assert!((cost_cached - expected_cached).abs() < 0.001);
    assert!(cost_cached < cost);
}
