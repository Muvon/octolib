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
	let provider = OllamaProvider::new();
	assert!(provider.supports_model("llama3.2"));
	assert!(provider.supports_model("qwen2.5"));
	assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
	let provider = OllamaProvider::new();
	assert_eq!(provider.name(), "ollama");
	assert!(provider.supports_caching("any-model"));
}

#[test]
fn test_vision_model_specific() {
	let provider = OllamaProvider::new();
	// Vision models detected via reference capabilities
	assert!(provider.supports_vision("llava:latest"));
	assert!(provider.supports_vision("qwen2.5-vl:72b"));
	assert!(provider.supports_vision("gemma3:27b"));
	// Text-only models correctly report no vision
	assert!(!provider.supports_vision("llama3.1:8b"));
	assert!(!provider.supports_vision("mistral:7b"));
	// Unknown models default to false
	assert!(!provider.supports_vision("unknown-model"));
}

#[test]
fn test_video_model_specific() {
	let provider = OllamaProvider::new();
	// Ollama chat API does not support video (only text and images)
	assert!(!provider.supports_video("qwen2.5-vl:72b"));
	assert!(!provider.supports_video("llama3.1:8b"));
	assert!(!provider.supports_video("llava:latest"));
}

#[test]
fn test_structured_output_model_specific() {
	let provider = OllamaProvider::new();
	assert!(provider.supports_structured_output("llama3.1:8b"));
	assert!(provider.supports_structured_output("qwen2.5:72b"));
	assert!(!provider.supports_structured_output("mistral:7b"));
}

#[test]
fn test_schema_is_never_enforced() {
	let provider = OllamaProvider::new();
	// Structured output still works (forced tool), but nothing guarantees the shape.
	assert!(provider.supports_structured_output("llama3.1:8b"));
	assert!(!provider.enforces_response_schema("llama3.1:8b"));
	assert!(!provider.enforces_response_schema("deepseek-v4-pro"));
	assert!(!provider.enforces_response_schema("glm-5.3-flash"));
}

#[test]
fn test_context_window_model_specific() {
	let provider = OllamaProvider::new();
	assert_eq!(provider.get_max_input_tokens("llama3.1:8b"), 131_072);
	assert_eq!(provider.get_max_input_tokens("mistral:7b"), 32_768);
	// Unknown models get reference-capabilities fallback default
	assert_eq!(provider.get_max_input_tokens("unknown-model"), 262_144);
}

#[test]
fn test_ollama_replays_thinking_only_for_preserve_thinking_kimi() {
	let provider = OllamaProvider::new();
	assert!(provider.replays_thinking("kimi-k2.6"));
	assert!(!provider.replays_thinking("minimax-m3"));
	assert!(!provider.replays_thinking("qwen3"));
}

#[test]
fn test_keepalive_pings_at_nine_tenths_of_keep_alive() {
	let provider = OllamaProvider::new();
	let short = provider.keepalive_policy("gpt-oss:120b", false).expect("5m policy");
	let long = provider.keepalive_policy("gpt-oss:120b", true).expect("1h policy");
	assert_eq!(short.interval, std::time::Duration::from_secs(270));
	assert_eq!(long.interval, std::time::Duration::from_secs(3240));
}

#[test]
fn test_pricing_uses_ollama_rates_before_reference_rows() {
	let provider = OllamaProvider::new();

	let gpt_oss = provider.get_model_pricing("gpt-oss:120b").expect("listed model");
	assert_eq!(gpt_oss.input_price_per_1m, 0.15);
	assert_eq!(gpt_oss.output_price_per_1m, 0.60);
	assert_eq!(gpt_oss.cache_read_price_per_1m, 0.014);

	// Catalogue ids carry tags the pricing page omits.
	let deepseek = provider.get_model_pricing("deepseek-v4-pro:0813").expect("tagged id");
	assert_eq!(deepseek.input_price_per_1m, 1.32);

	// The longer id must not fall through to the `kimi-k3` row.
	let kimi = provider.get_model_pricing("kimi-k3.1-pro").expect("listed model");
	assert_eq!(kimi.input_price_per_1m, 2.00);

	// Not on the cloud price list: reference rate for local models.
	let llama = provider.get_model_pricing("llama-3.1-8b").expect("reference row");
	assert_eq!(llama.input_price_per_1m, 0.10);
}
