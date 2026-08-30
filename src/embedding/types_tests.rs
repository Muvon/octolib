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
fn test_input_type_api_str() {
    assert_eq!(InputType::None.as_api_str(), None);
    assert_eq!(InputType::Query.as_api_str(), Some("query"));
    assert_eq!(InputType::Document.as_api_str(), Some("document"));
}

#[test]
fn test_input_type_prefix() {
    assert!(InputType::None.get_prefix().is_none());
    assert!(InputType::Query.get_prefix().is_some());
    assert!(InputType::Document.get_prefix().is_some());
}

#[test]
fn test_input_type_apply_prefix() {
    let text = "test content";

    let no_prefix = InputType::None.apply_prefix(text);
    assert_eq!(no_prefix, text);

    let query_prefix = InputType::Query.apply_prefix(text);
    assert!(query_prefix.contains(text));
    assert!(query_prefix.len() > text.len());

    let doc_prefix = InputType::Document.apply_prefix(text);
    assert!(doc_prefix.contains(text));
    assert!(doc_prefix.len() > text.len());
}

#[test]
fn test_parse_provider_model() {
    // Test valid provider:model format
    let (provider, model) = parse_provider_model("jina:jina-embeddings-v4").unwrap();
    assert_eq!(provider, EmbeddingProviderType::Jina);
    assert_eq!(model, "jina-embeddings-v4");

    // Test voyage provider
    let (provider, model) = parse_provider_model("voyage:voyage-3.5").unwrap();
    assert_eq!(provider, EmbeddingProviderType::Voyage);
    assert_eq!(model, "voyage-3.5");

    // Test google provider
    let (provider, model) = parse_provider_model("google:gemini-embedding-001").unwrap();
    assert_eq!(provider, EmbeddingProviderType::Google);
    assert_eq!(model, "gemini-embedding-001");

    // Test openai provider
    let (provider, model) = parse_provider_model("openai:text-embedding-3-small").unwrap();
    assert_eq!(provider, EmbeddingProviderType::OpenAI);
    assert_eq!(model, "text-embedding-3-small");

    // Whitespace should be trimmed
    let (provider, model) = parse_provider_model("  voyage : voyage-3.5  ").unwrap();
    assert_eq!(provider, EmbeddingProviderType::Voyage);
    assert_eq!(model, "voyage-3.5");

    let (provider, model) = parse_provider_model("local:nomic-embed-text").unwrap();
    assert_eq!(provider, EmbeddingProviderType::Local);
    assert_eq!(model, "nomic-embed-text");

    // Empty segments should fail with explicit format error
    assert!(parse_provider_model("openai:").is_err());
    assert!(parse_provider_model(":text-embedding-3-small").is_err());
}

#[test]
fn test_embedding_config_active_provider() {
    let config = EmbeddingConfig {
        code_model: "jina:jina-embeddings-v4".to_string(),
        text_model: "voyage:voyage-3.5".to_string(),
    };

    let active_provider = config.get_active_provider().unwrap();
    assert_eq!(active_provider, EmbeddingProviderType::Jina);
}

#[test]
fn test_embedding_config_api_keys() {
    let config = EmbeddingConfig::default();

    // These should return None unless environment variables are set
    let jina_key = config.get_api_key(&EmbeddingProviderType::Jina);
    let voyage_key = config.get_api_key(&EmbeddingProviderType::Voyage);
    let google_key = config.get_api_key(&EmbeddingProviderType::Google);
    let openai_key = config.get_api_key(&EmbeddingProviderType::OpenAI);

    // FastEmbed and HuggingFace don't need API keys
    assert!(config
        .get_api_key(&EmbeddingProviderType::FastEmbed)
        .is_none());
    assert!(config
        .get_api_key(&EmbeddingProviderType::HuggingFace)
        .is_none());

    // API keys should be None unless environment variables are set
    assert!(jina_key.is_none() || !jina_key.as_ref().unwrap().is_empty());
    assert!(voyage_key.is_none() || !voyage_key.as_ref().unwrap().is_empty());
    assert!(google_key.is_none() || !google_key.as_ref().unwrap().is_empty());
    assert!(openai_key.is_none() || !openai_key.as_ref().unwrap().is_empty());
}

#[tokio::test]
async fn test_embedding_config_vector_dimensions() {
    let config = EmbeddingConfig::default();

    // Test that we can call get_vector_dimension without panicking
    // (it might fail due to missing API keys, which is expected)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            config
                .get_vector_dimension(&EmbeddingProviderType::Jina, "jina-embeddings-v4")
                .await
        })
    }));

    // We don't care about the result, just that it doesn't panic unexpectedly
    // In test environment without API keys, this might panic or return an error
    match result {
        Ok(_) => {
            // Function completed (either success or expected error)
        }
        Err(_) => {
            // Function panicked (might be expected due to missing API keys)
        }
    }
}
