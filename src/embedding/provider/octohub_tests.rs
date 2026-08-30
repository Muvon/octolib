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
    assert!(OctoHubEmbeddingProvider::new("voyage-3.5").is_ok());
    assert!(OctoHubEmbeddingProvider::new("any-model").is_ok());
    assert!(OctoHubEmbeddingProvider::new("").is_err());
}

#[test]
fn test_api_url_default() {
    // Clear env to test default
    std::env::remove_var(OCTOHUB_API_URL_ENV);
    assert_eq!(
        OctoHubEmbeddingProvider::api_url(),
        "https://hub.octomind.run/v1/embeddings"
    );
}

#[test]
fn test_parse_single() {
    let response = json!([0.1, 0.2, 0.3]);
    let result = OctoHubEmbeddingProvider::parse_single(&response).unwrap();
    assert_eq!(result, vec![0.1_f32, 0.2, 0.3]);
}

#[test]
fn test_parse_batch() {
    let response = json!([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
    let result = OctoHubEmbeddingProvider::parse_batch(&response).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0], vec![0.1_f32, 0.2, 0.3]);
    assert_eq!(result[1], vec![0.4_f32, 0.5, 0.6]);
}
