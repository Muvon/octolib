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
fn test_cohere_provider_creation() {
    assert!(CohereProvider::new("rerank-v4.0-pro").is_ok());
    assert!(CohereProvider::new("rerank-v4.0-fast").is_ok());
    assert!(CohereProvider::new("rerank-english-v3.0").is_ok());
    assert!(CohereProvider::new("rerank-multilingual-v3.0").is_ok());
    assert!(CohereProvider::new("rerank-v3.5").is_ok());
    assert!(CohereProvider::new("invalid-model").is_err());
    // Removed deprecated v2 models
    assert!(CohereProvider::new("rerank-english-v2.0").is_err());
    assert!(CohereProvider::new("rerank-multilingual-v2.0").is_err());
}

#[test]
fn test_cohere_v4_endpoint_routing() {
    let v4_pro = CohereProvider::new("rerank-v4.0-pro").unwrap();
    assert!(v4_pro.is_v4_model());

    let v4_fast = CohereProvider::new("rerank-v4.0-fast").unwrap();
    assert!(v4_fast.is_v4_model());

    let v3 = CohereProvider::new("rerank-english-v3.0").unwrap();
    assert!(!v3.is_v4_model());
}

#[test]
fn test_cohere_model_validation() {
    let models = [
        "rerank-v4.0-pro",
        "rerank-v4.0-fast",
        "rerank-english-v3.0",
        "rerank-multilingual-v3.0",
        "rerank-v3.5",
    ];
    for model in models {
        let provider = CohereProvider::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
