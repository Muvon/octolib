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

const BEAM: &str = "Beam-1.0-501B-A23B";

#[test]
fn accepts_only_listed_models_in_any_case() {
	let provider = ReflectionProvider::new();
	assert!(provider.supports_model(BEAM));
	assert!(provider.supports_model("beam-1.0-501b-a23b"));
	assert!(!provider.supports_model("Beam-2.0"));
	assert!(!provider.supports_model(""));
}

/// The API answers 404 to any other casing of the id.
#[test]
fn lookup_returns_the_exact_upstream_id() {
	assert_eq!(find_model("beam-1.0-501b-a23b").map(|(id, _)| *id), Some(BEAM));
}

#[test]
fn declares_probed_capabilities() {
	let provider = ReflectionProvider::new();
	assert!(!provider.supports_vision(BEAM));
	assert!(!provider.supports_video(BEAM));
	assert!(provider.supports_caching(BEAM));
	assert!(provider.supports_structured_output(BEAM));
	assert!(provider.enforces_response_schema(BEAM));
	assert!(provider.supports_required_tool_choice(BEAM));
	assert_eq!(provider.get_max_input_tokens(BEAM), 1_048_576);
	assert_eq!(
		provider.supported_sampling_params(BEAM),
		SamplingSupport::TEMPERATURE_AND_TOP_P
	);
}

#[test]
fn prices_beam_and_no_unknown_model() {
	let provider = ReflectionProvider::new();
	let pricing = provider.get_model_pricing(BEAM).unwrap();
	assert_eq!(pricing.input_price_per_1m, 0.125);
	assert_eq!(pricing.output_price_per_1m, 0.40);
	assert_eq!(pricing.cache_write_price_per_1m, 0.125);
	assert_eq!(pricing.cache_read_price_per_1m, 0.015);
	assert!(provider.get_model_pricing("Beam-2.0").is_none());
}
