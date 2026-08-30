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
fn test_jina_provider_creation() {
    assert!(JinaProvider::new("jina-reranker-v3").is_ok());
    assert!(JinaProvider::new("jina-reranker-m0").is_ok());
    assert!(JinaProvider::new("jina-reranker-v2-base-multilingual").is_ok());
    assert!(JinaProvider::new("jina-colbert-v2").is_ok());
    assert!(JinaProvider::new("jina-reranker-v1-base-en").is_err());
    assert!(JinaProvider::new("invalid-model").is_err());
}

#[test]
fn test_jina_model_validation() {
    let models = [
        "jina-reranker-v3",
        "jina-reranker-m0",
        "jina-reranker-v2-base-multilingual",
        "jina-colbert-v2",
    ];
    for model in models {
        let provider = JinaProvider::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
