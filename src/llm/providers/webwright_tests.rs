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
    assert!(provider.supports_model("any-model"));
    assert!(!provider.supports_model(""));
}

#[test]
fn test_default_capabilities() {
    let provider = WebWrightProvider::new();
    assert_eq!(provider.name(), "webwright");
    assert!(!provider.supports_caching("wren-r1"));
    assert!(!provider.supports_video("wren-r1"));
    assert!(!provider.enforces_response_schema("wren-r1"));
}

#[test]
fn test_api_url_override() {
    let provider = WebWrightProvider::new();
    // The endpoint override is read from WEBWRIGHT_API_URL, defaulting to the
    // hosted OpenAI-compatible route.
    assert_eq!(
        WEBWRIGHT_API_URL,
        "https://webwright.ai/v1/chat/completions"
    );
    assert_eq!(WEBWRIGHT_API_KEY_ENV, "WEBWRIGHT_API_KEY");
    assert_eq!(provider.name(), "webwright");
}
