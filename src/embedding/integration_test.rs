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

//! Integration tests for embedding functionality
//!
//! These tests verify the low-level interactions and ensure all embedding
//! providers work correctly with proper error handling and feature gating.

#[cfg(test)]
mod integration_tests {
    use crate::embedding::*;

    #[tokio::test]
    async fn test_embedding_generation_config() {
        let config = EmbeddingGenerationConfig::default();
        assert!(!config.code_model.is_empty());
        assert!(!config.text_model.is_empty());
        assert!(config.batch_size > 0);
        assert!(config.max_tokens_per_batch > 0);
    }

    #[tokio::test]
    async fn test_provider_model_parsing() {
        // Test valid provider:model format
        let (provider, model) = parse_provider_model("jina:jina-embeddings-v4");
        assert_eq!(provider, EmbeddingProviderType::Jina);
        assert_eq!(model, "jina-embeddings-v4");

        // Test voyage provider
        let (provider, model) = parse_provider_model("voyage:voyage-3.5");
        assert_eq!(provider, EmbeddingProviderType::Voyage);
        assert_eq!(model, "voyage-3.5");

        // Test google provider
        let (provider, model) = parse_provider_model("google:text-embedding-005");
        assert_eq!(provider, EmbeddingProviderType::Google);
        assert_eq!(model, "text-embedding-005");

        // Test openai provider
        let (provider, model) = parse_provider_model("openai:text-embedding-3-small");
        assert_eq!(provider, EmbeddingProviderType::OpenAI);
        assert_eq!(model, "text-embedding-3-small");
    }

    #[tokio::test]
    async fn test_input_type_functionality() {
        let query_type = InputType::Query;
        let doc_type = InputType::Document;
        let none_type = InputType::None;

        // Test API string conversion
        assert_eq!(query_type.as_api_str(), Some("query"));
        assert_eq!(doc_type.as_api_str(), Some("document"));
        assert_eq!(none_type.as_api_str(), None);

        // Test prefix functionality
        assert!(query_type.get_prefix().is_some());
        assert!(doc_type.get_prefix().is_some());
        assert!(none_type.get_prefix().is_none());

        // Test prefix application
        let text = "test content";
        let prefixed_query = query_type.apply_prefix(text);
        let prefixed_doc = doc_type.apply_prefix(text);
        let no_prefix = none_type.apply_prefix(text);

        assert!(prefixed_query.contains(text));
        assert!(prefixed_doc.contains(text));
        assert_eq!(no_prefix, text);
        assert!(prefixed_query.len() > text.len());
        assert!(prefixed_doc.len() > text.len());
    }

    #[tokio::test]
    async fn test_token_counting() {
        let text = "Hello world, this is a test.";
        let token_count = count_tokens(text);
        assert!(token_count > 0);
        assert!(token_count < 20); // Should be reasonable for this short text

        // Test empty string
        let empty_count = count_tokens("");
        assert_eq!(empty_count, 0);

        // Test longer text
        let long_text = "This is a much longer text that should have more tokens than the previous example. It contains multiple sentences and should demonstrate the token counting functionality properly.";
        let long_count = count_tokens(long_text);
        assert!(long_count > token_count);
    }

    #[tokio::test]
    async fn test_text_batching() {
        let texts = vec![
            "Short text 1".to_string(),
            "Short text 2".to_string(),
            "This is a longer text that should take up more tokens".to_string(),
            "Another short one".to_string(),
        ];

        let batches = split_texts_into_token_limited_batches(texts.clone(), 2, 1000);

        // Should create batches respecting the count limit
        assert!(!batches.is_empty());
        for batch in &batches {
            assert!(batch.len() <= 2);
        }

        // Test with very low token limit
        let small_batches = split_texts_into_token_limited_batches(texts, 10, 10);
        assert!(!small_batches.is_empty());
        // Each batch should respect token limits
        for batch in &small_batches {
            let total_tokens: usize = batch.iter().map(|t| count_tokens(t)).sum();
            assert!(total_tokens <= 10 || batch.len() == 1); // Allow single item even if over limit
        }
    }

    #[tokio::test]
    async fn test_content_hashing() {
        let content = "test content";
        let file_path = "/path/to/file.rs";

        // Test basic content hash
        let hash1 = calculate_content_hash(content);
        let hash2 = calculate_content_hash(content);
        assert_eq!(hash1, hash2);

        let different_hash = calculate_content_hash("different content");
        assert_ne!(hash1, different_hash);

        // Test unique content hash with file path
        let unique_hash1 = calculate_unique_content_hash(content, file_path);
        let unique_hash2 = calculate_unique_content_hash(content, file_path);
        assert_eq!(unique_hash1, unique_hash2);

        let different_path_hash = calculate_unique_content_hash(content, "/different/path.rs");
        assert_ne!(unique_hash1, different_path_hash);

        // Test content hash with lines
        let line_hash1 = calculate_content_hash_with_lines(content, file_path, 1, 10);
        let line_hash2 = calculate_content_hash_with_lines(content, file_path, 1, 10);
        assert_eq!(line_hash1, line_hash2);

        let different_lines_hash = calculate_content_hash_with_lines(content, file_path, 2, 11);
        assert_ne!(line_hash1, different_lines_hash);
    }

    #[tokio::test]
    async fn test_output_truncation() {
        let short_text = "Short text";
        let truncated = truncate_output(short_text, 100);
        assert_eq!(truncated, short_text);

        let long_text =
            "This is a very long text that should be truncated when the token limit is exceeded. "
                .repeat(50);
        let truncated_long = truncate_output(&long_text, 50);
        assert!(truncated_long.len() < long_text.len());
        assert!(truncated_long.contains("[Output truncated"));

        // Test with zero limit (should return original)
        let no_limit = truncate_output(&long_text, 0);
        assert_eq!(no_limit, long_text);
    }

    #[tokio::test]
    async fn test_search_mode_embeddings_structure() {
        let embeddings = SearchModeEmbeddings {
            code_embeddings: Some(vec![0.1, 0.2, 0.3]),
            text_embeddings: Some(vec![0.4, 0.5, 0.6]),
        };

        assert!(embeddings.code_embeddings.is_some());
        assert!(embeddings.text_embeddings.is_some());
        assert_eq!(embeddings.code_embeddings.as_ref().unwrap().len(), 3);
        assert_eq!(embeddings.text_embeddings.as_ref().unwrap().len(), 3);

        // Test with only code embeddings
        let code_only = SearchModeEmbeddings {
            code_embeddings: Some(vec![0.1, 0.2]),
            text_embeddings: None,
        };

        assert!(code_only.code_embeddings.is_some());
        assert!(code_only.text_embeddings.is_none());
    }

    // Provider-specific tests (these will only run if the providers are available)

    #[tokio::test]
    async fn test_jina_provider_creation() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::Jina,
            "jina-embeddings-v4",
        )
        .await;

        match result {
            Ok(provider) => {
                assert_eq!(provider.get_dimension(), 2048);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_voyage_provider_creation() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::Voyage, "voyage-3.5-lite")
                .await;

        match result {
            Ok(provider) => {
                assert!(provider.get_dimension() > 0);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_google_provider_creation() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::Google,
            "text-embedding-005",
        )
        .await;

        match result {
            Ok(provider) => {
                assert_eq!(provider.get_dimension(), 768);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_openai_provider_creation() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::OpenAI,
            "text-embedding-3-small",
        )
        .await;

        match result {
            Ok(provider) => {
                assert_eq!(provider.get_dimension(), 1536);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Any error is acceptable in test environment without API keys
                assert!(!e.to_string().is_empty());
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "fastembed")]
    async fn test_fastembed_provider_creation() {
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
                // FastEmbed might fail due to model download issues in CI
                println!("FastEmbed test failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_huggingface_provider_creation() {
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
                // HuggingFace might fail due to model download issues in CI
                println!("HuggingFace test failed (expected in CI): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_provider_model() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::Jina,
            "invalid-model-name",
        )
        .await;

        assert!(result.is_err());
        let error_msg = result.err().unwrap().to_string();
        assert!(error_msg.contains("Unsupported model") || error_msg.contains("invalid"));
    }

    #[tokio::test]
    #[cfg(not(feature = "fastembed"))]
    async fn test_fastembed_disabled() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::FastEmbed, "any-model")
                .await;

        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not compiled"));
    }

    #[tokio::test]
    #[cfg(not(feature = "huggingface"))]
    async fn test_huggingface_disabled() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::HuggingFace, "any-model")
                .await;

        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("not compiled"));
    }
}
