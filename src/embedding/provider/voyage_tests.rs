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
    assert!(VoyageProviderImpl::new("voyage-4-large").is_ok());
    assert!(VoyageProviderImpl::new("voyage-4").is_ok());
    assert!(VoyageProviderImpl::new("voyage-3.5").is_ok());
    assert!(VoyageProviderImpl::new("voyage-context-3").is_ok());

    // Test invalid model
    assert!(VoyageProviderImpl::new("invalid-model").is_err());
}

#[test]
fn test_voyage_model_dimensions() {
    assert_eq!(
        VoyageProviderImpl::new("voyage-4-large")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-4").unwrap().get_dimension(),
        1024
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-4-lite")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-3.5")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-code-2")
            .unwrap()
            .get_dimension(),
        1536
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-context-3")
            .unwrap()
            .get_dimension(),
        1024
    );
    assert_eq!(
        VoyageProviderImpl::new("voyage-multimodal-3.5")
            .unwrap()
            .get_dimension(),
        1024
    );
}

#[test]
fn test_voyage_model_validation() {
    let models = [
        "voyage-4-large",
        "voyage-4",
        "voyage-4-lite",
        "voyage-3.5",
        "voyage-3.5-lite",
        "voyage-3-large",
        "voyage-code-3",
        "voyage-code-4",
        "voyage-context-3",
        "voyage-multimodal-3.5",
    ];
    for model in models {
        let provider = VoyageProviderImpl::new(model).unwrap();
        assert!(provider.is_model_supported());
    }
}
