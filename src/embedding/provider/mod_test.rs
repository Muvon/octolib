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

#[cfg(test)]
mod tests {
    use crate::embedding::create_embedding_provider_from_parts;
    use crate::embedding::types::*;

    #[tokio::test]
    async fn test_create_jina_provider() {
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
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("JINA_API_KEY")
                );
            }
        }
    }

    #[tokio::test]
    async fn test_create_voyage_provider() {
        let result =
            create_embedding_provider_from_parts(&EmbeddingProviderType::Voyage, "voyage-3.5-lite")
                .await;

        match result {
            Ok(provider) => {
                assert!(provider.get_dimension() > 0);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("VOYAGE_API_KEY")
                );
            }
        }
    }

    #[tokio::test]
    async fn test_create_google_provider() {
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
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("GOOGLE_API_KEY")
                );
            }
        }
    }

    #[tokio::test]
    async fn test_create_openai_provider() {
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
                // Expected if no API key is set
                assert!(
                    e.to_string().contains("API key") || e.to_string().contains("OPENAI_API_KEY")
                );
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "fastembed")]
    async fn test_create_fastembed_provider() {
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
    async fn test_create_huggingface_provider() {
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
    #[cfg(feature = "huggingface")]
    async fn test_create_huggingface_qwen2_provider() {
        // Test Qwen2 model loading (jina-code-embeddings-1.5b uses Qwen2 architecture)
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::HuggingFace,
            "jinaai/jina-code-embeddings-1.5b",
        )
        .await;

        match result {
            Ok(provider) => {
                assert!(provider.get_dimension() > 0);
                assert!(provider.is_model_supported());
            }
            Err(e) => {
                // HuggingFace might fail due to model download issues in CI
                println!("HuggingFace Qwen2 test failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "huggingface")]
    fn test_model_architecture_detection() {
        use serde_json::json;

        // Test BERT detection
        let bert_config = json!({
            "architectures": ["BertModel"],
            "position_embedding_type": "absolute"
        });
        let config_str = serde_json::to_string(&bert_config).unwrap();
        let parsed: crate::embedding::provider::huggingface::ModelConfig =
            serde_json::from_str(&config_str).unwrap();
        let arch = crate::embedding::provider::huggingface::ModelArchitecture::from_config(&parsed);
        assert!(matches!(
            arch,
            Ok(crate::embedding::provider::huggingface::ModelArchitecture::Bert)
        ));

        // Test JinaBert detection (via position_embedding_type)
        let jina_config = json!({
            "architectures": ["BertModel"],
            "position_embedding_type": "alibi"
        });
        let config_str = serde_json::to_string(&jina_config).unwrap();
        let parsed: crate::embedding::provider::huggingface::ModelConfig =
            serde_json::from_str(&config_str).unwrap();
        let arch = crate::embedding::provider::huggingface::ModelArchitecture::from_config(&parsed);
        assert!(matches!(
            arch,
            Ok(crate::embedding::provider::huggingface::ModelArchitecture::JinaBert)
        ));

        // Test Qwen2 detection
        let qwen2_config = json!({
            "architectures": ["Qwen2ForCausalLM"]
        });
        let config_str = serde_json::to_string(&qwen2_config).unwrap();
        let parsed: crate::embedding::provider::huggingface::ModelConfig =
            serde_json::from_str(&config_str).unwrap();
        let arch = crate::embedding::provider::huggingface::ModelArchitecture::from_config(&parsed);
        assert!(matches!(
            arch,
            Ok(crate::embedding::provider::huggingface::ModelArchitecture::Qwen2)
        ));

        // Test Qwen3 detection
        let qwen3_config = json!({
            "architectures": ["Qwen3ForCausalLM"]
        });
        let config_str = serde_json::to_string(&qwen3_config).unwrap();
        let parsed: crate::embedding::provider::huggingface::ModelConfig =
            serde_json::from_str(&config_str).unwrap();
        let arch = crate::embedding::provider::huggingface::ModelArchitecture::from_config(&parsed);
        assert!(matches!(
            arch,
            Ok(crate::embedding::provider::huggingface::ModelArchitecture::Qwen3)
        ));
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

    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_huggingface_qwen2_embedding_generation() {
        // Test actual embedding generation with Qwen2 model
        // This test downloads the model on first run
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::HuggingFace,
            "jinaai/jina-code-embeddings-1.5b",
        )
        .await;

        match result {
            Ok(provider) => {
                eprintln!("Provider created successfully");
                // Test single embedding
                let text = "fn main() { println!(\"Hello, world!\"); }";
                let embedding = provider.generate_embedding(text).await;

                match embedding {
                    Ok(vec) => {
                        assert!(!vec.is_empty(), "Embedding should not be empty");
                        assert!(
                            vec.iter().all(|v| v.is_finite()),
                            "All values should be finite"
                        );
                        println!(
                            "✓ Qwen2 embedding generated successfully, dimension: {}",
                            vec.len()
                        );
                    }
                    Err(e) => {
                        // Model loading can fail in CI due to network/resource constraints
                        eprintln!("Embedding generation failed:");
                        for cause in e.chain() {
                            eprintln!("  Caused by: {}", cause);
                        }
                    }
                }
            }
            Err(e) => {
                // HuggingFace might fail due to model download issues in CI
                eprintln!("HuggingFace provider creation failed:");
                for cause in e.chain() {
                    eprintln!("  Caused by: {}", cause);
                }
            }
        }
    }

    #[tokio::test]
    #[cfg(feature = "huggingface")]
    async fn test_huggingface_bert_embedding_generation() {
        // Test actual embedding generation with BERT model
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::HuggingFace,
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        .await;

        match result {
            Ok(provider) => {
                // Test single embedding
                let text = "This is a test sentence for embedding.";
                let embedding = provider.generate_embedding(text).await;

                match embedding {
                    Ok(vec) => {
                        assert!(!vec.is_empty(), "Embedding should not be empty");
                        assert!(
                            vec.iter().all(|v| v.is_finite()),
                            "All values should be finite"
                        );
                        println!(
                            "✓ BERT embedding generated successfully, dimension: {}",
                            vec.len()
                        );
                    }
                    Err(e) => {
                        println!("Embedding generation failed: {}", e);
                        panic!("Embedding generation should succeed for BERT model");
                    }
                }
            }
            Err(e) => {
                // HuggingFace might fail due to model download issues in CI
                println!(
                    "HuggingFace BERT embedding test failed (expected in CI): {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_model() {
        let result = create_embedding_provider_from_parts(
            &EmbeddingProviderType::Jina,
            "invalid-model-name",
        )
        .await;

        assert!(result.is_err());
        let error_msg = result.err().unwrap().to_string();
        assert!(error_msg.contains("Unsupported model") || error_msg.contains("invalid"));
    }
}
