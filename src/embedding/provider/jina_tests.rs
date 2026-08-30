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
    // Test valid models
    assert!(JinaProviderImpl::new("jina-embeddings-v4").is_ok());
    assert!(JinaProviderImpl::new("jina-embeddings-v3").is_ok());
    assert!(JinaProviderImpl::new("jina-clip-v2").is_ok());
    assert!(JinaProviderImpl::new("jina-colbert-v2").is_ok());
    assert!(JinaProviderImpl::new("jina-code-embeddings-0.5b").is_ok());

    // Test invalid model
    assert!(JinaProviderImpl::new("invalid-model").is_err());
}

#[test]
fn test_jina_model_dimensions() {
    assert_eq!(
        JinaProviderImpl::new("jina-embeddings-v4")
            .unwrap()
            .get_dimension(),
        2048
    );
    assert_eq!(
        JinaProviderImpl::new("jina-embeddings-v3")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        JinaProviderImpl::new("jina-clip-v2")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        JinaProviderImpl::new("jina-embeddings-v2-small-en")
            .unwrap()
            .get_dimension(),
        512
    );
    assert_eq!(
        JinaProviderImpl::new("jina-colbert-v2")
            .unwrap()
            .get_dimension(),
        128
    );
    assert_eq!(
        JinaProviderImpl::new("jina-colbert-v2-96")
            .unwrap()
            .get_dimension(),
        96
    );
    assert_eq!(
        JinaProviderImpl::new("jina-colbert-v2-64")
            .unwrap()
            .get_dimension(),
        64
    );
    assert_eq!(
        JinaProviderImpl::new("jina-code-embeddings-0.5b")
            .unwrap()
            .get_dimension(),
        896
    );
    assert_eq!(
        JinaProviderImpl::new("jina-code-embeddings-1.5b")
            .unwrap()
            .get_dimension(),
        1536
    );
}

#[test]
fn test_jina_model_validation() {
    let models = [
        "jina-embeddings-v5-text-small",
        "jina-embeddings-v5-text-nano",
        "jina-embeddings-v5-omni-small",
        "jina-embeddings-v5-omni-nano",
        "jina-embeddings-v4",
        "jina-embeddings-v3",
        "jina-clip-v2",
        "jina-clip-v1",
        "jina-embeddings-v2-small-en",
        "jina-colbert-v2",
        "jina-colbert-v2-96",
        "jina-colbert-v2-64",
        "jina-code-embeddings-0.5b",
        "jina-code-embeddings-1.5b",
    ];
    for model in models {
        let provider = JinaProviderImpl::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
