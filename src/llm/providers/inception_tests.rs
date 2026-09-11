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
    let provider = InceptionProvider::new();
    assert!(provider.supports_model("mercury-2.5"));
    assert!(provider.supports_model("mercury-2"));
    // Mercury Edit 2 is FIM/edit-only; the chat endpoint 404s it
    assert!(!provider.supports_model("mercury-edit-2"));
    assert!(!provider.supports_model("unknown-model"));
    assert!(!provider.supports_model(""));

    // Case-insensitive: the provider canonicalizes before sending
    assert!(provider.supports_model("MERCURY-2.5"));
    assert_eq!(
        find_model("mercury-2.5").map(|(id, _)| *id),
        Some("mercury-2.5")
    );
}

#[test]
fn test_model_capabilities() {
    let provider = InceptionProvider::new();
    // Text-only models (input_modalities: ["text"])
    assert!(!provider.supports_vision("mercury-2.5"));
    assert_eq!(provider.get_max_input_tokens("mercury-2.5"), 260_000);
    assert_eq!(provider.get_max_input_tokens("mercury-2"), 128_000);
    assert_eq!(provider.get_max_input_tokens("unknown-model"), 128_000);
}

#[test]
fn test_chat_features() {
    let provider = InceptionProvider::new();
    // supported_features: ["tools", "json_mode", "structured_outputs"]
    assert!(provider.supports_structured_output("mercury-2.5"));
    assert!(provider.enforces_response_schema("mercury-2.5"));
    assert!(provider.supports_required_tool_choice("mercury-2.5"));
    // Automatic prefix caching with discounted cached-input billing
    assert!(provider.supports_caching("mercury-2.5"));
    // supported_sampling_parameters: ["temperature", "stop"]
    assert_eq!(
        provider.supported_sampling_params("mercury-2.5"),
        SamplingSupport::TEMPERATURE_ONLY
    );
}

#[test]
fn test_pricing() {
    let provider = InceptionProvider::new();
    // Mercury 2.5 launch pricing (80% off): $0.04 in / $0.15 out / $0.004 cached
    let pricing = provider.get_model_pricing("mercury-2.5").unwrap();
    assert_eq!(pricing.input_price_per_1m, 0.04);
    assert_eq!(pricing.output_price_per_1m, 0.15);
    assert_eq!(pricing.cache_read_price_per_1m, 0.004);

    let pricing = provider.get_model_pricing("mercury-2").unwrap();
    assert_eq!(pricing.input_price_per_1m, 0.25);
    assert_eq!(pricing.output_price_per_1m, 0.75);
    assert_eq!(pricing.cache_read_price_per_1m, 0.025);

    // Unknown models are not priced (strict catalogue)
    assert!(provider.get_model_pricing("mercury-9").is_none());
}
