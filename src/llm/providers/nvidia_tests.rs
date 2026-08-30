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
    let provider = NvidiaProvider::new();
    assert!(provider.supports_model("nvidia/llama-3.1-nemotron-ultra-253b-v1"));
    assert!(provider.supports_model("deepseek-ai/deepseek-v3.2"));
    assert!(provider.supports_model("minimaxai/minimax-m2.1"));
    assert!(provider.supports_model("meta/llama-3.1-405b-instruct"));
    assert!(provider.supports_model("any-model"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = NvidiaProvider::new();
    assert_eq!(provider.name(), "nvidia");
    assert!(!provider.supports_caching("any-model"));
    assert!(provider.supports_structured_output("any-model"));
}
