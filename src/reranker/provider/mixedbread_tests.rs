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
fn test_mixedbread_provider_creation() {
    assert!(MixedbreadProvider::new("mxbai-rerank-large-v2").is_ok());
    assert!(MixedbreadProvider::new("mxbai-rerank-base-v2").is_ok());
    assert!(MixedbreadProvider::new("mxbai-rerank-large-v1").is_ok());
    assert!(MixedbreadProvider::new("mxbai-rerank-base-v1").is_ok());
    assert!(MixedbreadProvider::new("mxbai-rerank-xsmall-v1").is_ok());
    assert!(MixedbreadProvider::new("invalid-model").is_err());
}

#[test]
fn test_mixedbread_model_validation() {
    let models = [
        "mxbai-rerank-large-v2",
        "mxbai-rerank-base-v2",
        "mxbai-rerank-large-v1",
        "mxbai-rerank-base-v1",
        "mxbai-rerank-xsmall-v1",
    ];
    for model in models {
        let provider = MixedbreadProvider::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
