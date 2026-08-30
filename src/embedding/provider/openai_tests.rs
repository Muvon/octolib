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
fn test_openai_provider_creation() {
    // Test valid models
    assert!(OpenAIProviderImpl::new("text-embedding-3-small").is_ok());
    assert!(OpenAIProviderImpl::new("text-embedding-3-large").is_ok());
    assert!(OpenAIProviderImpl::new("text-embedding-ada-002").is_ok());

    // Test invalid model
    assert!(OpenAIProviderImpl::new("invalid-model").is_err());
}

#[test]
fn test_model_dimensions() {
    let provider_small = OpenAIProviderImpl::new("text-embedding-3-small").unwrap();
    assert_eq!(provider_small.get_dimension(), 1536);

    let provider_large = OpenAIProviderImpl::new("text-embedding-3-large").unwrap();
    assert_eq!(provider_large.get_dimension(), 3072);

    let provider_ada = OpenAIProviderImpl::new("text-embedding-ada-002").unwrap();
    assert_eq!(provider_ada.get_dimension(), 1536);
}

#[test]
fn test_model_validation() {
    let provider_valid = OpenAIProviderImpl::new("text-embedding-3-small").unwrap();
    assert!(provider_valid.is_model_supported());

    // This would panic if we tried to create an invalid model, so we test indirectly
    let supported_models = [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ];
    for model in supported_models {
        let provider = OpenAIProviderImpl::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
