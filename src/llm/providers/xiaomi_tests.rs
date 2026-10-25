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
	let provider = XiaomiProvider::new();
	assert!(provider.supports_model("mimo-v2.6-pro"));
	assert!(provider.supports_model("mimo-v2.6-flash"));
	assert!(provider.supports_model("mimo-v2.6-pro-ultraspeed"));
	// Retired 2026-10-21; upstream rejects the IDs
	assert!(!provider.supports_model("mimo-v2.5-pro"));
	assert!(!provider.supports_model("mimo-v2.5"));
	assert!(!provider.supports_model(""));

	// Case-insensitive: the provider canonicalizes before sending
	assert!(provider.supports_model("MiMo-V2.6-Pro"));
	assert_eq!(find_model("MiMo-V2.6-Pro").map(|(id, _)| *id), Some("mimo-v2.6-pro"));
}

#[test]
fn test_model_capabilities() {
	let provider = XiaomiProvider::new();
	assert!(provider.supports_vision("mimo-v2.6-flash"));
	assert!(provider.supports_video("mimo-v2.6-flash"));
	assert!(provider.supports_caching("mimo-v2.6-pro"));
	assert!(provider.replays_thinking("mimo-v2.6-pro"));
	assert_eq!(provider.get_max_input_tokens("mimo-v2.6-pro"), 1_048_576);
	assert_eq!(provider.get_max_input_tokens("mimo-v2.6-pro-ultraspeed"), 1_048_576);
}

#[test]
fn test_chat_features() {
	let provider = XiaomiProvider::new();
	// json_object mode only: supported, but the schema is not enforced upstream
	assert!(provider.supports_structured_output("mimo-v2.6-pro"));
	assert!(!provider.enforces_response_schema("mimo-v2.6-pro"));
	assert!(!provider.supports_required_tool_choice("mimo-v2.6-pro"));
	assert_eq!(
		provider.supported_sampling_params("mimo-v2.6-pro"),
		SamplingSupport::TEMPERATURE_AND_TOP_P
	);
}

#[test]
fn test_pricing() {
	let provider = XiaomiProvider::new();
	// Ultraspeed contains "mimo-v2.6-pro" and must not resolve to the Pro row
	let pricing = provider.get_model_pricing("mimo-v2.6-pro-ultraspeed").unwrap();
	assert_eq!(pricing.input_price_per_1m, 4.35);
	assert_eq!(pricing.output_price_per_1m, 8.70);
	assert_eq!(pricing.cache_read_price_per_1m, 0.036);

	let pricing = provider.get_model_pricing("mimo-v2.6-pro").unwrap();
	assert_eq!(pricing.input_price_per_1m, 0.435);
	assert_eq!(pricing.output_price_per_1m, 0.87);
	assert_eq!(pricing.cache_read_price_per_1m, 0.0036);

	let pricing = provider.get_model_pricing("mimo-v2.6-flash").unwrap();
	assert_eq!(pricing.input_price_per_1m, 0.14);
	assert_eq!(pricing.output_price_per_1m, 0.28);
	assert_eq!(pricing.cache_read_price_per_1m, 0.0028);

	// Unknown models are not priced (strict catalogue)
	assert!(provider.get_model_pricing("mimo-v9").is_none());
}
