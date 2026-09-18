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
    assert!(!provider.supports_caching("any-model"));
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
