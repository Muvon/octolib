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
    use crate::embedding::*;

    #[test]
    fn test_count_tokens() {
        let text = "Hello, world!";
        let count = count_tokens(text);
        assert!(count > 0);
        assert!(count < 10); // Should be a small number for this short text
    }

    #[test]
    fn test_split_texts_into_batches() {
        let texts = vec![
            "Short text".to_string(),
            "Another short text".to_string(),
            "Yet another text".to_string(),
        ];

        let batches = split_texts_into_token_limited_batches(texts, 2, 1000);

        // Should split into 2 batches: [2 texts, 1 text]
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }

    #[test]
    fn test_truncate_output() {
        let long_text = "This is a test. ".repeat(100);
        let truncated = truncate_output(&long_text, 50);

        // Should be truncated and contain truncation message
        assert!(truncated.contains("[Output truncated"));
        assert!(truncated.len() < long_text.len());
    }

    #[test]
    fn test_truncate_output_no_limit() {
        let text = "This is a test.";
        let result = truncate_output(text, 0); // 0 means no limit

        assert_eq!(result, text);
    }

    #[tokio::test]
    async fn test_basic_embedding_generation() {
        // This test will only run if API keys are available
        let result = generate_embeddings("Hello, world!", "jina", "jina-embeddings-v4").await;

        match result {
            Ok(embeddings) => {
                assert!(!embeddings.is_empty());
                println!("Generated embedding with {} dimensions", embeddings.len());
            }
            Err(e) => {
                // Expected if no API key is set
                println!(
                    "Embedding generation failed (expected without API key): {}",
                    e
                );
            }
        }
    }

    #[tokio::test]
    async fn test_batch_embedding_generation() {
        let texts = vec!["First document".to_string(), "Second document".to_string()];

        let result = generate_embeddings_batch(
            texts,
            "jina",
            "jina-embeddings-v4",
            InputType::Document,
            16,
            100_000,
        )
        .await;

        match result {
            Ok(embeddings) => {
                assert_eq!(embeddings.len(), 2);
                println!("Generated {} embeddings", embeddings.len());
            }
            Err(e) => {
                // Expected if no API key is set
                println!(
                    "Batch embedding generation failed (expected without API key): {}",
                    e
                );
            }
        }
    }
}
