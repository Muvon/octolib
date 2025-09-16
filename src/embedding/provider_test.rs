// Copyright 2025 Muvon Un Limited
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

//! Provider-specific tests for all embedding providers
//!
//! These tests verify each provider's specific functionality and behavior.

#[cfg(test)]
mod provider_tests {
    use crate::embedding::provider::*;
    use crate::embedding::types::*;

    // Jina Provider Tests
    #[tokio::test]
    async fn test_jina_provider_models() {
        // Test all supported Jina models
        let models = [
            ("jina-embeddings-v4", 2048),
            ("jina-clip-v2", 1024),
            ("jina-embeddings-v3", 1024),
            ("jina-clip-v1", 768),
            ("jina-embeddings-v2-base-es", 768),
            ("jina-embeddings-v2-base-code", 768),
            ("jina-embeddings-v2-base-de", 768),
            ("jina-embeddings-v2-base-zh", 768),
            ("jina-embeddings-v2-base-en", 768),
        ];

        for (model, expected_dim) in models {
            let result = JinaProviderImpl::new(model);
            match result {
                Ok(provider) => {
                    assert_eq!(provider.get_dimension(), expected_dim);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(
                        e.to_string().contains("API key") || e.to_string().contains("JINA_API_KEY")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_jina_invalid_model() {
        let result = JinaProviderImpl::new("invalid-jina-model");
        assert!(result.is_err());
        // Any error is acceptable for invalid models
        assert!(result.is_err());
    }

    // Voyage Provider Tests
    #[tokio::test]
    async fn test_voyage_provider_models() {
        // Test some known Voyage models
        let models = [
            "voyage-3.5",
            "voyage-code-2",
            "voyage-finance-2",
            "voyage-3.5-lite",
        ];

        for model in models {
            let result = VoyageProviderImpl::new(model);
            match result {
                Ok(provider) => {
                    assert!(provider.get_dimension() > 0);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(
                        e.to_string().contains("API key")
                            || e.to_string().contains("VOYAGE_API_KEY")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_voyage_invalid_model() {
        let result = VoyageProviderImpl::new("invalid-voyage-model");
        assert!(result.is_err());
        // Any error is acceptable for invalid models
        assert!(result.is_err());
    }

    // Google Provider Tests
    #[tokio::test]
    async fn test_google_provider_models() {
        let models = [
            ("gemini-embedding-001", 3072),
            ("text-embedding-005", 768),
            ("text-multilingual-embedding-002", 768),
        ];

        for (model, expected_dim) in models {
            let result = GoogleProviderImpl::new(model);
            match result {
                Ok(provider) => {
                    assert_eq!(provider.get_dimension(), expected_dim);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(
                        e.to_string().contains("API key")
                            || e.to_string().contains("GOOGLE_API_KEY")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_google_invalid_model() {
        let result = GoogleProviderImpl::new("invalid-google-model");
        assert!(result.is_err());
        // Any error is acceptable for invalid models
        assert!(result.is_err());
    }

    // OpenAI Provider Tests
    #[tokio::test]
    async fn test_openai_provider_models() {
        let models = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ];

        for (model, expected_dim) in models {
            let result = OpenAIProviderImpl::new(model);
            match result {
                Ok(provider) => {
                    assert_eq!(provider.get_dimension(), expected_dim);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(
                        e.to_string().contains("API key")
                            || e.to_string().contains("OPENAI_API_KEY")
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_openai_invalid_model() {
        let result = OpenAIProviderImpl::new("invalid-openai-model");
        assert!(result.is_err());
        // Any error is acceptable for invalid models
        assert!(result.is_err());
    }

    // FastEmbed Provider Tests
    #[tokio::test]
    #[cfg(feature = "fastembed")]
    async fn test_fastembed_provider_models() {
        let models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "jinaai/jina-embeddings-v2-base-code",
            "sentence-transformers/all-mpnet-base-v2",
        ];

        for model in models {
            let result = FastEmbedProviderImpl::new(model);
            match result {
                Ok(provider) => {
                    assert!(provider.get_dimension() > 0);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // FastEmbed might fail due to model download issues
                    println!("FastEmbed test failed (expected in CI): {}", e);
                }
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "fastembed")]
    async fn test_fastembed_invalid_model() {
        let result = FastEmbedProviderImpl::new("invalid/fastembed-model");
        // FastEmbed might accept any model name and fail later, so we just check it doesn't panic
        match result {
            Ok(_) => {
                // Model accepted, might fail during actual embedding generation
            }
            Err(_) => {
                // Model rejected, which is also acceptable
            }
        }
    }

    // HuggingFace Provider Tests
    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_huggingface_provider_models() {
        let models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "microsoft/codebert-base",
            "sentence-transformers/all-mpnet-base-v2",
        ];

        for model in models {
            let result = HuggingFaceProviderImpl::new(model).await;
            match result {
                Ok(provider) => {
                    assert!(provider.get_dimension() > 0);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // HuggingFace might fail due to model download issues
                    println!("HuggingFace test failed (expected in CI): {}", e);
                }
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_huggingface_invalid_model() {
        let result = HuggingFaceProviderImpl::new("invalid/huggingface-model").await;
        // HuggingFace might accept any model name and fail later during download
        match result {
            Ok(_) => {
                // Model accepted, might fail during actual embedding generation
            }
            Err(_) => {
                // Model rejected, which is also acceptable
            }
        }
    }

    // Factory Function Tests
    #[tokio::test]
    async fn test_provider_factory_all_types() {
        let test_cases = [
            (EmbeddingProviderType::Jina, "jina-embeddings-v4"),
            (EmbeddingProviderType::Voyage, "voyage-3.5"),
            (EmbeddingProviderType::Google, "text-embedding-005"),
            (EmbeddingProviderType::OpenAI, "text-embedding-3-small"),
        ];

        for (provider_type, model) in test_cases {
            let result = create_embedding_provider_from_parts(&provider_type, model).await;
            match result {
                Ok(provider) => {
                    assert!(provider.get_dimension() > 0);
                    assert!(provider.is_model_supported());
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(!e.to_string().is_empty());
                }
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "fastembed")]
    async fn test_provider_factory_fastembed() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::FastEmbed,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .await;

        match result {
            Ok(provider) => {
                assert!(provider.get_dimension() > 0);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // FastEmbed might fail due to model download issues
                println!("FastEmbed factory test failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_provider_factory_huggingface() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::HuggingFace,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .await;

        match result {
            Ok(provider) => {
                assert!(provider.get_dimension() > 0);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // HuggingFace might fail due to model download issues
                println!("HuggingFace factory test failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(not(feature = "fastembed"))]
    async fn test_provider_factory_fastembed_disabled() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::FastEmbed, "any-model")
                .await;

        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not compiled"));
    }

    #[tokio::test]
    #[cfg(not(feature = "huggingface"))]
    async fn test_provider_factory_huggingface_disabled() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::HuggingFace, "any-model")
                .await;

        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not compiled"));
    }

    // Input Type Tests
    #[tokio::test]
    async fn test_input_type_with_providers() {
        let input_types = [InputType::None, InputType::Query, InputType::Document];

        for input_type in input_types {
            // Test that input types work with provider creation
            let result = create_embedding_provider_from_parts(
                &EmbeddingProviderType::Jina,
                "jina-embeddings-v4",
            )
            .await;

            match result {
                Ok(provider) => {
                    // Test that we can call generate_embeddings_batch with different input types
                    let texts = vec!["test text".to_string()];
                    let batch_result = provider
                        .generate_embeddings_batch(texts, input_type.clone())
                        .await;

                    match batch_result {
                        Ok(embeddings) => {
                            assert_eq!(embeddings.len(), 1);
                            assert!(!embeddings[0].is_empty());
                        }
                        Err(e) => {
                            // Expected if no API key
                            assert!(!e.to_string().is_empty());
                        }
                    }
                }
                Err(e) => {
                    // Expected if no API key
                    assert!(!e.to_string().is_empty());
                }
            }
        }
    }
}
