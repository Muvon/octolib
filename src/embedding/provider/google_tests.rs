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
fn test_google_provider_creation_valid_models() {
	for model in GoogleProvider::get_supported_models() {
		let provider = GoogleProviderImpl::new(model).unwrap();
		assert_eq!(provider.get_dimension(), 3072);
		assert!(provider.is_model_supported());
	}
}

#[test]
fn test_google_provider_rejects_unknown_model() {
	let err = GoogleProviderImpl::new("gemini-embedding-999")
		.err()
		.expect("unknown model must be rejected");
	assert!(err.to_string().contains("Unsupported Google model"));
	assert!(GoogleProviderImpl::new("").is_err());
}

#[test]
fn test_google_supported_models_list() {
	let models = GoogleProvider::get_supported_models();
	assert_eq!(models, vec!["gemini-embedding-2", "gemini-embedding-001"]);
}
