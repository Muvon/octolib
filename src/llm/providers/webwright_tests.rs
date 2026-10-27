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
	let provider = WebWrightProvider::new();
	assert!(provider.supports_model("wren-r1"));
	assert!(provider.supports_model("wren-r1-fast"));
	// Case-insensitive: requests are canonicalized to the table's ID.
	assert!(provider.supports_model("Wren-R1"));
	assert!(provider.supports_model("WREN-R1-FAST"));
	// The catalogue is fixed, so unknown IDs are rejected rather than served
	// by wren-r1.
	assert!(!provider.supports_model("any-future-model"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_capabilities() {
	let provider = WebWrightProvider::new();
	assert_eq!(provider.name(), "webwright");
	assert!(provider.supports_caching("wren-r1"));
	assert!(!provider.supports_vision("wren-r1"));
	assert!(!provider.supports_video("wren-r1"));
	assert!(!provider.supports_structured_output("wren-r1"));
	assert!(!provider.supports_required_tool_choice("wren-r1"));
}

#[test]
fn test_schema_is_never_enforced() {
	let provider = WebWrightProvider::new();
	// The OpenAI-compatible route has no evidence of constrained decoding.
	assert!(!provider.enforces_response_schema("wren-r1"));
	assert!(!provider.enforces_response_schema("any-model"));
}

#[test]
fn test_published_pricing() {
	let provider = WebWrightProvider::new();
	// Standard tier rates, https://webwright.ai/pricing
	let pricing = provider
		.get_model_pricing("wren-r1")
		.expect("wren-r1 publishes Standard-tier rates");
	assert_eq!(pricing.input_price_per_1m, 0.40);
	assert_eq!(pricing.output_price_per_1m, 1.60);
	assert_eq!(pricing.cache_write_price_per_1m, 0.40);
	assert_eq!(pricing.cache_read_price_per_1m, 0.10);
}

#[test]
fn test_max_input_tokens() {
	let provider = WebWrightProvider::new();
	assert_eq!(provider.get_max_input_tokens("wren-r1"), 200_000);
	assert_eq!(provider.get_max_input_tokens("wren-r1-fast"), 200_000);
	// Unknown models still report the deployed model's window rather than the
	// generic reference default.
	assert_eq!(provider.get_max_input_tokens("any-model"), 200_000);
}

/// wren-r1-fast is the Premium plan chat model, so it carries the Premium rates.
/// https://webwright.ai/models and https://webwright.ai/pricing
#[test]
fn test_wren_r1_fast_published_pricing() {
	let provider = WebWrightProvider::new();
	let pricing = provider
		.get_model_pricing("wren-r1-fast")
		.expect("wren-r1-fast publishes Premium-tier rates");
	assert_eq!(pricing.input_price_per_1m, 0.80);
	assert_eq!(pricing.output_price_per_1m, 3.20);
	assert_eq!(pricing.cache_write_price_per_1m, 0.80);
	assert_eq!(pricing.cache_read_price_per_1m, 0.20);
}
