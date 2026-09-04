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
fn test_supports_model_before_cache() {
    let provider = GoogleStudioProvider::new();

    // Before cache is populated, accept any non-empty model
    assert!(provider.supports_model("gemini-2.5-flash"));
    assert!(provider.supports_model("gemini-3.6-flash"));
    assert!(provider.supports_model("gemini-3.7-flash"));
    assert!(provider.supports_model("gemini-3.8-flash"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = GoogleStudioProvider::new();
    assert_eq!(provider.name(), "google-studio");
    assert!(provider.supports_caching("gemini-3.5-flash"));
    assert!(provider.supports_caching("gemini-2.5-pro"));
    assert!(!provider.supports_caching("gemini-2.0-flash"));
    assert!(provider.supports_vision("gemini-3.1-pro"));
    assert!(provider.supports_structured_output("any-model"));
    assert_eq!(provider.get_max_input_tokens("gemini-3.6-flash"), 1_048_576);
}

#[test]
fn test_sampling_params() {
    let provider = GoogleStudioProvider::new();
    assert_eq!(
        provider.supported_sampling_params("gemini-3.8-flash"),
        SamplingSupport::NONE
    );
    assert_eq!(
        provider.supported_sampling_params("gemini-3.7-flash"),
        SamplingSupport::NONE
    );
    assert_eq!(
        provider.supported_sampling_params("gemini-3.6-flash"),
        SamplingSupport::NONE
    );
    assert_eq!(
        provider.supported_sampling_params("gemini-2.5-flash"),
        SamplingSupport::ALL
    );
}

#[test]
fn test_model_pricing() {
    let provider = GoogleStudioProvider::new();

    let p = provider.get_model_pricing("gemini-3.8-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.75);
    assert_eq!(p.output_price_per_1m, 3.75);

    let p = provider.get_model_pricing("gemini-3.7-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.75);
    assert_eq!(p.output_price_per_1m, 3.75);

    // Intro rate applies to 3.6 Flash too, through Dec 31, 2026
    let p = provider.get_model_pricing("gemini-3.6-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.75);
    assert_eq!(p.output_price_per_1m, 3.75);

    // Preview suffixes resolve to the base model's pricing
    let p = provider
        .get_model_pricing("gemini-3.1-pro-preview")
        .unwrap();
    assert_eq!(p.input_price_per_1m, 2.00);
    assert_eq!(p.output_price_per_1m, 12.00);

    let p = provider.get_model_pricing("gemini-2.5-flash-lite").unwrap();
    assert_eq!(p.input_price_per_1m, 0.10);
    assert_eq!(p.output_price_per_1m, 0.40);

    assert!(provider.get_model_pricing("gemma-3-27b").is_none());
}

#[test]
fn test_cost_calculation() {
    // 1M input + 0.5M output on gemini-2.5-flash: $0.30 + 0.5 * $2.50 = $1.55
    let cost = calculate_usage_cost("gemini-2.5-flash", 1_000_000, 0, 0, 500_000).unwrap();
    assert!((cost - 1.55).abs() < 1e-9);
}

#[test]
fn test_pro_long_context_tier() {
    // At or below 200K input, Pro bills the standard tier:
    // 0.2M * $1.25 + 0.1M * $10.00 = $1.25
    let short = calculate_usage_cost("gemini-2.5-pro", 200_000, 0, 0, 100_000).unwrap();
    assert!((short - 1.25).abs() < 1e-9);

    // Above 200K total input: 2x input/cache, 1.5x output.
    // 0.200001M * $2.50 + 0.1M * $15.00 = $2.0000025
    let long = calculate_usage_cost("gemini-2.5-pro", 200_001, 0, 0, 100_000).unwrap();
    assert!((long - 2.0000025).abs() < 1e-9);

    // Cache writes and reads count toward the threshold and get the 2x rate.
    // 0.15M * $4.00 + 0.05M * $4.00 + 0.05M * $0.40 + 0.01M * $18.00 = $1.00
    let cached = calculate_usage_cost("gemini-3.1-pro", 150_000, 50_000, 50_000, 10_000).unwrap();
    assert!((cached - 1.0).abs() < 1e-9);

    // Flash has no long-context tier: 0.5M * $0.75 + 0.01M * $3.75 = $0.4125
    let flash = calculate_usage_cost("gemini-3.7-flash", 500_000, 0, 0, 10_000).unwrap();
    assert!((flash - 0.4125).abs() < 1e-9);
}
