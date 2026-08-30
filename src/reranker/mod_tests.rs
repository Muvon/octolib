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
fn test_parse_provider_model() {
    let (provider, model) = parse_provider_model("voyage:rerank-2.5").unwrap();
    assert_eq!(provider, RerankProviderType::Voyage);
    assert_eq!(model, "rerank-2.5");

    let (provider, model) = parse_provider_model("cohere:rerank-english-v3.0").unwrap();
    assert_eq!(provider, RerankProviderType::Cohere);
    assert_eq!(model, "rerank-english-v3.0");

    let (provider, model) = parse_provider_model("jina:jina-reranker-v3").unwrap();
    assert_eq!(provider, RerankProviderType::Jina);
    assert_eq!(model, "jina-reranker-v3");

    #[cfg(feature = "fastembed")]
    {
        let (provider, model) = parse_provider_model("fastembed:bge-reranker-base").unwrap();
        assert_eq!(provider, RerankProviderType::FastEmbed);
        assert_eq!(model, "bge-reranker-base");
    }
}

#[tokio::test]
async fn test_create_provider() {
    // API-based providers - require API keys
    let voyage = create_rerank_provider_from_parts(&RerankProviderType::Voyage, "rerank-2.5").await;
    match voyage {
        Ok(_) => {}
        Err(e) => {
            // Expected if no API key is set
            assert!(
                e.to_string().contains("API key") || e.to_string().contains("VOYAGE_API_KEY"),
                "Expected API key error, got: {}",
                e
            );
        }
    }

    let cohere =
        create_rerank_provider_from_parts(&RerankProviderType::Cohere, "rerank-english-v3.0").await;
    match cohere {
        Ok(_) => {}
        Err(e) => {
            // Expected if no API key is set
            assert!(
                e.to_string().contains("API key") || e.to_string().contains("COHERE_API_KEY"),
                "Expected API key error, got: {}",
                e
            );
        }
    }

    let jina =
        create_rerank_provider_from_parts(&RerankProviderType::Jina, "jina-reranker-v3").await;
    match jina {
        Ok(_) => {}
        Err(e) => {
            // Expected if no API key is set
            assert!(
                e.to_string().contains("API key") || e.to_string().contains("JINA_API_KEY"),
                "Expected API key error, got: {}",
                e
            );
        }
    }

    // FastEmbed - local provider, may require model download
    #[cfg(feature = "fastembed")]
    {
        let fastembed =
            create_rerank_provider_from_parts(&RerankProviderType::FastEmbed, "bge-reranker-base")
                .await;
        match fastembed {
            Ok(provider) => {
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Model download may be needed - graceful handling for CI
                println!(
                    "FastEmbed provider creation skipped (model download needed): {}",
                    e
                );
            }
        }
    }
}

#[tokio::test]
async fn test_invalid_models() {
    let result =
        create_rerank_provider_from_parts(&RerankProviderType::Voyage, "invalid-model").await;
    assert!(result.is_err());

    let result =
        create_rerank_provider_from_parts(&RerankProviderType::Cohere, "invalid-model").await;
    assert!(result.is_err());

    let result =
        create_rerank_provider_from_parts(&RerankProviderType::Jina, "invalid-model").await;
    assert!(result.is_err());

    #[cfg(feature = "fastembed")]
    {
        let result =
            create_rerank_provider_from_parts(&RerankProviderType::FastEmbed, "invalid-model")
                .await;
        assert!(result.is_err());
    }
}
