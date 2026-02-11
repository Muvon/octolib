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

//! # Embedding Module
//!
//! Self-sufficient embedding generation with multi-provider support.
//! Supports Jina, Voyage, Google, OpenAI, FastEmbed, and HuggingFace providers.

pub mod constants;
#[cfg(test)]
mod mod_test;
pub mod provider;
pub mod types;

use anyhow::Result;
use tiktoken_rs::cl100k_base;

pub use provider::{create_embedding_provider_from_parts, EmbeddingProvider};
pub use types::*;

/// Generate embeddings using specified provider and model
pub async fn generate_embeddings(contents: &str, provider: &str, model: &str) -> Result<Vec<f32>> {
    // Parse provider and model from the string
    let (provider_type, model_name) = parse_provider_model(&format!("{}:{}", provider, model))?;

    let provider_impl = create_embedding_provider_from_parts(&provider_type, &model_name).await?;
    provider_impl.generate_embedding(contents).await
}

/// Count tokens in a text using tiktoken (cl100k_base tokenizer)
pub fn count_tokens(text: &str) -> usize {
    let bpe = cl100k_base().expect("Failed to load cl100k_base tokenizer");
    bpe.encode_with_special_tokens(text).len()
}

/// Truncate output if it exceeds token limit
pub fn truncate_output(output: &str, max_tokens: usize) -> String {
    if max_tokens == 0 {
        return output.to_string();
    }

    let token_count = count_tokens(output);

    if token_count <= max_tokens {
        return output.to_string();
    }

    // Simple truncation - cut at character boundary
    // Estimate roughly where to cut (tokens are ~4 chars average)
    let estimated_chars = max_tokens * 3; // Conservative estimate
    let truncated = if output.len() > estimated_chars {
        &output[..estimated_chars]
    } else {
        output
    };

    // Find last newline to avoid cutting mid-line
    let last_newline = truncated.rfind('\n').unwrap_or(truncated.len());
    let final_truncated = &truncated[..last_newline];

    format!(
        "{}\n\n[Output truncated - {} tokens estimated, max {} allowed. Use more specific queries to reduce output size]",
        final_truncated,
        token_count,
        max_tokens
    )
}

/// Split texts into batches respecting both count and token limits
pub fn split_texts_into_token_limited_batches(
    texts: Vec<String>,
    max_batch_size: usize,
    max_tokens_per_batch: usize,
) -> Vec<Vec<String>> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_token_count = 0;

    for text in texts {
        let text_tokens = count_tokens(&text);

        // If adding this text would exceed either limit, start a new batch
        if !current_batch.is_empty()
            && (current_batch.len() >= max_batch_size
                || current_token_count + text_tokens > max_tokens_per_batch)
        {
            batches.push(current_batch);
            current_batch = Vec::new();
            current_token_count = 0;
        }

        current_batch.push(text);
        current_token_count += text_tokens;
    }

    // Add the last batch if it's not empty
    if !current_batch.is_empty() {
        batches.push(current_batch);
    }

    batches
}

/// Generate batch embeddings using specified provider and model
/// Includes token-aware batching and input_type support
pub async fn generate_embeddings_batch(
    texts: Vec<String>,
    provider: &str,
    model: &str,
    input_type: types::InputType,
    batch_size: usize,
    max_tokens_per_batch: usize,
) -> Result<Vec<Vec<f32>>> {
    // Parse provider and model from the string
    let (provider_type, model_name) = parse_provider_model(&format!("{}:{}", provider, model))?;

    let provider_impl = create_embedding_provider_from_parts(&provider_type, &model_name).await?;

    // Split texts into token-limited batches
    let batches = split_texts_into_token_limited_batches(texts, batch_size, max_tokens_per_batch);

    let mut all_embeddings = Vec::new();

    // Process each batch with input_type
    for batch in batches {
        let batch_embeddings = provider_impl
            .generate_embeddings_batch(batch, input_type.clone())
            .await?;
        all_embeddings.extend(batch_embeddings);
    }

    Ok(all_embeddings)
}
