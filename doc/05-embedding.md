# Embedding Generation Guide

Octolib provides a unified interface for generating embeddings across multiple providers. This guide covers all supported providers, configuration options, and usage patterns.

## 🎯 Supported Providers

| Provider | Models | Features | API Key Required |
|----------|--------|----------|------------------|
| **Jina** | jina-embeddings-v4, jina-clip-v2, jina-embeddings-v3 | High-quality embeddings | ✅ JINA_API_KEY |
| **Voyage** | voyage-3.5, voyage-code-2, voyage-finance-2 | Specialized models | ✅ VOYAGE_API_KEY |
| **Google** | gemini-embedding-001, text-embedding-005 | Google AI embeddings | ✅ GOOGLE_API_KEY |
| **OpenAI** | text-embedding-3-small, text-embedding-3-large | OpenAI embeddings | ✅ OPENAI_API_KEY |
| **FastEmbed** | Local sentence-transformers models | Local processing | ❌ No API key |
| **HuggingFace** | sentence-transformers models | HuggingFace Hub | ❌ No API key |

## 🚀 Quick Start

### Basic Embedding Generation

```rust
use octolib::embedding::generate_embeddings;

async fn basic_example() -> anyhow::Result<()> {
    // Generate a single embedding
    let embedding = generate_embeddings(
        "Hello, world!",
        "voyage",           // provider
        "voyage-3.5-lite"   // model
    ).await?;

    println!("Embedding dimension: {}", embedding.len());
    Ok(())
}
```

### Batch Embedding Generation

```rust
use octolib::embedding::{generate_embeddings_batch, InputType};

async fn batch_example() -> anyhow::Result<()> {
    let texts = vec![
        "First document".to_string(),
        "Second document".to_string(),
        "Third document".to_string(),
    ];

    let embeddings = generate_embeddings_batch(
        texts,
        "jina",                    // provider
        "jina-embeddings-v4",      // model
        InputType::Document,       // input type for better embeddings
        16,                        // batch size
        100_000,                   // max tokens per batch
    ).await?;

    println!("Generated {} embeddings", embeddings.len());
    Ok(())
}
```

## 🔧 Provider Configuration

### Environment Variables

Set the appropriate API key for your chosen provider:

```bash
# Jina AI
export JINA_API_KEY="your-jina-api-key"

# Voyage AI
export VOYAGE_API_KEY="your-voyage-api-key"

# Google AI
export GOOGLE_API_KEY="your-google-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

## 🎯 Input Types

Use `InputType` to optimize embeddings for specific use cases:

```rust
use octolib::embedding::{generate_embeddings_batch, InputType};

// For indexing documents
InputType::Document

// For search queries
InputType::Query

// Default behavior
InputType::None
```

## 🔧 Utility Functions

### Token Counting

```rust
use octolib::embedding::count_tokens;

let text = "This is a sample text for token counting.";
let token_count = count_tokens(text);
println!("Token count: {}", token_count);
```

### Text Batching

```rust
use octolib::embedding::split_texts_into_token_limited_batches;

let texts = vec!["Document 1".to_string(), "Document 2".to_string()];
let batches = split_texts_into_token_limited_batches(texts, 16, 100_000);
```

## 🎯 Best Practices

1. **Choose the Right Provider**
   - **Jina**: High-quality general embeddings, multimodal support
   - **Voyage**: Specialized models for code, finance, etc.
   - **Google**: Strong multilingual support
   - **OpenAI**: Reliable, well-tested embeddings

2. **Use Batch Processing** for multiple texts to improve performance

3. **Monitor Token Usage** to stay within provider limits

4. **Handle Errors Gracefully** with proper error handling

For more examples, see the main [Advanced Guide](04-advanced-guide.md).
