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
    let provider = LocalProvider::new();

    assert!(provider.supports_model("llama3.2"));
    assert!(provider.supports_model("mistral-7b"));
    assert!(provider.supports_model("gpt4all-j"));
    assert!(provider.supports_model("any-model-name"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_get_api_key_optional() {
    let provider = LocalProvider::new();
    let result = provider.get_api_key();
    assert!(result.is_ok());
}

#[test]
fn test_default_capabilities() {
    let provider = LocalProvider::new();
    assert_eq!(provider.name(), "local");
    assert!(!provider.supports_caching("any-model"));
}

#[test]
fn test_capabilities_model_specific() {
    let provider = LocalProvider::new();
    // Vision models
    assert!(provider.supports_vision("llava:latest"));
    assert!(provider.supports_vision("gemma-3-27b"));
    // Text-only models
    assert!(!provider.supports_vision("llama-3.1-8b"));
    // Structured output
    assert!(provider.supports_structured_output("llama-3.1-8b"));
    assert!(!provider.supports_structured_output("mistral-7b"));
    // Context windows
    assert_eq!(provider.get_max_input_tokens("llama-3.1-8b"), 131_072);
    assert_eq!(provider.get_max_input_tokens("mistral-7b"), 32_768);
}
