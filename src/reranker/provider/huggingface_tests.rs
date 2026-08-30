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
fn test_huggingface_reranker_creation() {
    #[cfg(feature = "huggingface")]
    {
        assert!(HuggingFaceReranker::new("cross-encoder/ms-marco-MiniLM-L-6-v2").is_ok());
        assert!(HuggingFaceReranker::new("BAAI/bge-reranker-base").is_ok());
        assert!(HuggingFaceReranker::new("jinaai/jina-reranker-v2-base-multilingual").is_ok());
        assert!(HuggingFaceReranker::new("").is_err());
    }
}

#[test]
fn test_recommended_models_not_empty() {
    #[cfg(feature = "huggingface")]
    {
        let models = HuggingFaceReranker::recommended_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"cross-encoder/ms-marco-MiniLM-L-6-v2"));
        assert!(models.contains(&"BAAI/bge-reranker-v2-m3"));
        assert!(models.contains(&"jinaai/jina-reranker-v2-base-multilingual"));
    }
}
