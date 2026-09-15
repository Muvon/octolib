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

#[tokio::test]
async fn test_empty_model_rejected() {
    // Rejected before any network probe, so this is safe offline.
    assert!(OctoHubEmbeddingProvider::new("").await.is_err());
}

#[tokio::test]
async fn test_probed_dimension_is_reported_and_reused() {
    // Callers size their vector store from get_dimension before the first write;
    // a 0 here made octocode build its tables with a zero-length embedding field.
    let model = "dimension-cache-test-model";
    DIMENSION_CACHE.write().unwrap().insert(
        (OctoHubEmbeddingProvider::api_url(), model.to_string()),
        1536,
    );
    let provider = OctoHubEmbeddingProvider::new(model).await.unwrap();
    assert_eq!(provider.get_dimension(), 1536);
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
