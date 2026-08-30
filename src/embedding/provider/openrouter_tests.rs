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
async fn test_openrouter_provider_dimension_probe() {
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return;
    }
    super::super::tests::refresh_http_client();
    let provider = OpenRouterProviderImpl::new("qwen/qwen3-embedding-8b")
        .await
        .unwrap();
    assert_eq!(provider.get_dimension(), 4096);
    assert!(provider.is_model_supported());
}

#[tokio::test]
async fn test_openrouter_invalid_model_rejected() {
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        return;
    }
    super::super::tests::refresh_http_client();
    let result = OpenRouterProviderImpl::new("not/a-real-model-xyz").await;
    assert!(result.is_err());
}
