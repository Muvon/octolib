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
    // Test valid models
    let provider = TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct");
    assert!(provider.is_ok());
    assert_eq!(provider.unwrap().get_dimension(), 1024);

    // Test invalid model
    let invalid = TogetherProviderImpl::new("invalid-model");
    assert!(invalid.is_err());
}

#[test]
fn test_model_dimensions() {
    let provider = TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct").unwrap();
    assert_eq!(provider.get_dimension(), 1024);
}

#[test]
fn test_model_validation() {
    let provider_valid =
        TogetherProviderImpl::new("intfloat/multilingual-e5-large-instruct").unwrap();
    assert!(provider_valid.is_model_supported());

    let provider_invalid = TogetherProviderImpl::new("unknown-model");
    assert!(provider_invalid.is_err());
}
