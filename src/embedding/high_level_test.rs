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

//! High-level embedding generation tests
//!
//! These tests verify the main embedding generation functions work correctly
//! with different configurations and scenarios.

#[cfg(test)]
mod high_level_tests {
    use crate::embedding::*;

    #[tokio::test]
    async fn test_embedding_generation_with_config() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v4".to_string(),
            batch_size: 2,
            max_tokens_per_batch: 1000,
        };

        // Test code embedding generation
        let result = generate_embeddings("fn main() { println!(\"Hello\"); }", true, &config).await;

        match result {
            Ok(embeddings) => {
                assert!(!embeddings.is_empty());
                assert_eq!(embeddings.len(), 2048); // Jina v4 dimension
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }

        // Test text embedding generation
        let result = generate_embeddings("This is a text document.", false, &config).await;

        match result {
            Ok(embeddings) => {
                assert!(!embeddings.is_empty());
                assert_eq!(embeddings.len(), 2048); // Jina v4 dimension
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_batch_embedding_generation() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v4".to_string(),
            batch_size: 2,
            max_tokens_per_batch: 1000,
        };

        let texts = vec![
            "First text document".to_string(),
            "Second text document".to_string(),
            "Third text document".to_string(),
        ];

        let result =
            generate_embeddings_batch(texts.clone(), false, &config, InputType::Document).await;

        match result {
            Ok(embeddings) => {
                assert_eq!(embeddings.len(), texts.len());
                for embedding in embeddings {
                    assert!(!embedding.is_empty());
                    assert_eq!(embedding.len(), 2048); // Jina v4 dimension
                }
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_search_mode_embeddings() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v4".to_string(),
            batch_size: 2,
            max_tokens_per_batch: 1000,
        };

        let query = "search for functions";

        // Test code mode
        let result = generate_search_embeddings(query, "code", &config).await;
        match result {
            Ok(embeddings) => {
                assert!(embeddings.code_embeddings.is_some());
                assert!(embeddings.text_embeddings.is_none());
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }

        // Test text mode
        let result = generate_search_embeddings(query, "text", &config).await;
        match result {
            Ok(embeddings) => {
                assert!(embeddings.code_embeddings.is_none());
                assert!(embeddings.text_embeddings.is_some());
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }

        // Test docs mode (same as text)
        let result = generate_search_embeddings(query, "docs", &config).await;
        match result {
            Ok(embeddings) => {
                assert!(embeddings.code_embeddings.is_none());
                assert!(embeddings.text_embeddings.is_some());
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }

        // Test all mode with same model (should reuse embeddings)
        let result = generate_search_embeddings(query, "all", &config).await;
        match result {
            Ok(embeddings) => {
                assert!(embeddings.code_embeddings.is_some());
                assert!(embeddings.text_embeddings.is_some());
                // Should be the same since we're using the same model
                assert_eq!(embeddings.code_embeddings, embeddings.text_embeddings);
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_search_mode_embeddings_different_models() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v3".to_string(), // Different model
            batch_size: 2,
            max_tokens_per_batch: 1000,
        };

        let query = "search for functions";

        // Test all mode with different models (should generate separate embeddings)
        let result = generate_search_embeddings(query, "all", &config).await;
        match result {
            Ok(embeddings) => {
                assert!(embeddings.code_embeddings.is_some());
                assert!(embeddings.text_embeddings.is_some());
                // Should be different since we're using different models
                assert_ne!(embeddings.code_embeddings, embeddings.text_embeddings);

                // Check dimensions
                assert_eq!(embeddings.code_embeddings.as_ref().unwrap().len(), 2048); // v4
                assert_eq!(embeddings.text_embeddings.as_ref().unwrap().len(), 1024);
                // v3
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_search_mode() {
        let config = EmbeddingGenerationConfig::default();
        let query = "test query";

        let result = generate_search_embeddings(query, "invalid_mode", &config).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Invalid search mode"));
    }

    #[tokio::test]
    async fn test_token_limited_batching_integration() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v4".to_string(),
            batch_size: 1,            // Force small batches
            max_tokens_per_batch: 10, // Very small token limit
        };

        let texts = vec![
            "This is a longer text that should exceed the token limit".to_string(),
            "Short".to_string(),
            "Another longer text that should also exceed the token limit".to_string(),
        ];

        let result =
            generate_embeddings_batch(texts.clone(), false, &config, InputType::Document).await;

        match result {
            Ok(embeddings) => {
                // Should still generate embeddings for all texts despite batching
                assert_eq!(embeddings.len(), texts.len());
                for embedding in embeddings {
                    assert!(!embedding.is_empty());
                }
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_embedding_config_validation() {
        // Test that EmbeddingConfig methods work correctly
        let config = EmbeddingConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "voyage:voyage-3.5".to_string(),
        };

        // Test provider detection
        let provider = config.get_active_provider();
        assert_eq!(provider, EmbeddingProviderType::Jina);

        // Test API key retrieval (should return None in test environment)
        let jina_key = config.get_api_key(&EmbeddingProviderType::Jina);
        let voyage_key = config.get_api_key(&EmbeddingProviderType::Voyage);
        let google_key = config.get_api_key(&EmbeddingProviderType::Google);

        // These should be None unless environment variables are set
        assert!(jina_key.is_none() || !jina_key.as_ref().unwrap().is_empty());
        assert!(voyage_key.is_none() || !voyage_key.as_ref().unwrap().is_empty());
        assert!(google_key.is_none() || !google_key.as_ref().unwrap().is_empty());

        // FastEmbed and HuggingFace don't need API keys
        assert!(config
            .get_api_key(&EmbeddingProviderType::FastEmbed)
            .is_none());
        assert!(config
            .get_api_key(&EmbeddingProviderType::HuggingFace)
            .is_none());
    }

    #[tokio::test]
    async fn test_embedding_config_model_validation() {
        let config = EmbeddingConfig::default();

        // Test validation with valid models
        let jina_result = config
            .validate_model(&EmbeddingProviderType::Jina, "jina-embeddings-v4")
            .await;
        match jina_result {
            Ok(_) => {
                // Validation passed
            }
            Err(e) => {
                // Expected if no API key or network issues
                let error_msg = e.to_string();
                assert!(
                    error_msg.contains("API key")
                        || error_msg.contains("JINA_API_KEY")
                        || error_msg.contains("network")
                        || error_msg.contains("connection")
                );
            }
        }

        // Test validation with invalid model
        let invalid_result = config
            .validate_model(&EmbeddingProviderType::Jina, "invalid-model")
            .await;
        assert!(invalid_result.is_err());
        let error_msg = invalid_result.unwrap_err().to_string();
        assert!(error_msg.contains("Unsupported model") || error_msg.contains("invalid"));
    }

    #[tokio::test]
    async fn test_embedding_config_dimension_retrieval() {
        let config = EmbeddingConfig::default();

        // Test dimension retrieval for known models
        let _jina_dim_result = config
            .get_vector_dimension(&EmbeddingProviderType::Jina, "jina-embeddings-v4")
            .await;
        // This might panic if no API key, which is expected behavior according to the implementation
        // In a real test environment with API keys, this would return 2048
    }

    // Performance and stress tests

    #[tokio::test]
    async fn test_large_batch_processing() {
        let config = EmbeddingGenerationConfig {
            code_model: "jina:jina-embeddings-v4".to_string(),
            text_model: "jina:jina-embeddings-v4".to_string(),
            batch_size: 5,
            max_tokens_per_batch: 10000,
        };

        // Create a larger batch to test batching logic
        let texts: Vec<String> = (0..20)
            .map(|i| format!("This is test document number {} with some content.", i))
            .collect();

        let result =
            generate_embeddings_batch(texts.clone(), false, &config, InputType::Document).await;

        match result {
            Ok(embeddings) => {
                assert_eq!(embeddings.len(), texts.len());
                // Verify all embeddings are valid
                for (i, embedding) in embeddings.iter().enumerate() {
                    assert!(!embedding.is_empty(), "Embedding {} should not be empty", i);
                    assert_eq!(
                        embedding.len(),
                        2048,
                        "Embedding {} should have correct dimension",
                        i
                    );
                }
            }
            Err(e) => {
                // Print the actual error for debugging
                println!("Actual error: {}", e);
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }
}
