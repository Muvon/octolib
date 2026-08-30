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
fn test_fastembed_provider_creation() {
    // Test model name validation - this works without downloading
    let test_cases = vec![
        ("bge-reranker-base", true),
        ("BAAI/bge-reranker-base", true),
        ("bge-reranker-v2-m3", true),
        ("jina-reranker-v1-turbo-en", true),
        ("jina-reranker-v2-base-multilingual", true),
        ("invalid-model", false),
    ];

    for (model, should_be_valid) in &test_cases {
        let result = FastEmbedProvider::map_model_name(model);
        if *should_be_valid {
            assert!(result.is_ok(), "Model '{}' should be valid", model);
        } else {
            assert!(result.is_err(), "Model '{}' should be invalid", model);
        }
    }

    // Try actual provider creation (may require model download)
    match FastEmbedProvider::new("bge-reranker-base") {
        Ok(provider) => {
            assert!(provider.is_model_supported());
        }
        Err(e) => {
            // Model download may be needed - graceful handling
            println!("Provider creation deferred (model download needed): {}", e);
        }
    }
}

#[test]
fn test_list_supported_models() {
    let models = FastEmbedProvider::list_supported_models();
    assert!(!models.is_empty());
    assert!(models.contains(&"bge-reranker-base".to_string()));
}
