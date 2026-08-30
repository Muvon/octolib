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
fn test_voyage_provider_creation() {
    // Test valid models
    assert!(VoyageProviderImpl::new("rerank-2.5").is_ok());
    assert!(VoyageProviderImpl::new("rerank-2.5-lite").is_ok());
    assert!(VoyageProviderImpl::new("rerank-2").is_ok());
    assert!(VoyageProviderImpl::new("rerank-2-lite").is_ok());
    assert!(VoyageProviderImpl::new("rerank-1").is_ok());
    assert!(VoyageProviderImpl::new("rerank-lite-1").is_ok());

    // Test invalid model
    assert!(VoyageProviderImpl::new("invalid-model").is_err());
}

#[test]
fn test_voyage_model_validation() {
    let models = [
        "rerank-2.5",
        "rerank-2.5-lite",
        "rerank-2",
        "rerank-2-lite",
        "rerank-1",
        "rerank-lite-1",
    ];
    for model in models {
        let provider = VoyageProviderImpl::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
