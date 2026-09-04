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
    let provider = GoogleVertexProvider::new();

    // Before cache is populated, accept any non-empty model
    assert!(provider.supports_model("gemini-1.5-pro"));
    assert!(provider.supports_model("gemini-2.0-flash"));
    assert!(provider.supports_model("anything-goes"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_supports_caching() {
    let provider = GoogleVertexProvider::new();
    assert!(provider.supports_caching("gemini-3-flash"));
    assert!(provider.supports_caching("gemini-2.5-pro"));
    assert!(provider.supports_caching("gemini-2.5-flash"));
    assert!(!provider.supports_caching("gemini-2.0-flash"));
    assert!(!provider.supports_caching("gemini-1.5-pro"));
}

#[test]
fn test_model_pricing() {
    let provider = GoogleVertexProvider::new();

    let p = provider.get_model_pricing("gemini-3.1-pro").unwrap();
    assert_eq!(p.input_price_per_1m, 2.00);
    assert_eq!(p.output_price_per_1m, 12.00);

    let p = provider.get_model_pricing("gemini-3.5-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 1.50);
    assert_eq!(p.output_price_per_1m, 9.00);

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

    let p = provider.get_model_pricing("gemini-2.5-flash").unwrap();
    assert_eq!(p.input_price_per_1m, 0.30);
    assert_eq!(p.output_price_per_1m, 2.50);

    // "-lite" variants must not be shadowed by their shorter prefixes
    let p = provider.get_model_pricing("gemini-3.1-flash-lite").unwrap();
    assert_eq!(p.input_price_per_1m, 0.25);
    assert_eq!(p.output_price_per_1m, 1.50);

    let p = provider.get_model_pricing("gemini-3.5-flash-lite").unwrap();
    assert_eq!(p.input_price_per_1m, 0.30);
    assert_eq!(p.output_price_per_1m, 2.50);

    // Unknown models return None (no fallback to zero)
    assert!(provider.get_model_pricing("gemma-3-27b").is_none());
}

#[test]
fn test_max_input_tokens_fallback() {
    let provider = GoogleVertexProvider::new();
    assert_eq!(provider.get_max_input_tokens("gemini-3-flash"), 1_048_576);
    assert_eq!(provider.get_max_input_tokens("gemini-2.5-pro"), 1_048_576);
    assert_eq!(provider.get_max_input_tokens("gemini-1.5-pro"), 1_000_000);
    assert_eq!(provider.get_max_input_tokens("text-bison"), 8_192);
}

#[test]
fn test_sampling_params() {
    let provider = GoogleVertexProvider::new();
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
        provider.supported_sampling_params("gemini-3.5-flash"),
        SamplingSupport::ALL
    );
    assert_eq!(
        provider.supported_sampling_params("gemini-2.5-pro"),
        SamplingSupport::ALL
    );
}
