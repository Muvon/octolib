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
fn test_provider_creation() {
    assert!(LocalRerankerProvider::new("bge-reranker-v2-m3").is_ok());
    assert!(LocalRerankerProvider::new("any-model-name").is_ok());
    assert!(LocalRerankerProvider::new("").is_err());
}

#[test]
fn test_api_url_default() {
    std::env::remove_var(LOCAL_RERANK_API_URL_ENV);
    assert_eq!(
        LocalRerankerProvider::api_url(),
        "http://localhost:8012/v1/rerank"
    );
}

#[test]
fn test_is_model_supported() {
    let provider = LocalRerankerProvider::new("bge-reranker-v2-m3").unwrap();
    assert!(provider.is_model_supported());
}
