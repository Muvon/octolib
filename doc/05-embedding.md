# Embedding Generation Guide

Octolib provides a unified interface for generating embeddings across multiple providers. This guide covers all supported providers, configuration options, and usage patterns.

## 🎯 Supported Providers

| Provider | Models | Features | API Key Required |
|----------|--------|----------|------------------|
| **Jina** | jina-embeddings-v4 (2048d), jina-embeddings-v3 (1024d), jina-clip-v2 (1024d), jina-colbert-v2 (128d/96d/64d), jina-code-embeddings (1024d) | High-quality embeddings, multimodal, late-interaction, code-specialized | ✅ JINA_API_KEY |
| **Voyage** | voyage-4-large/4/4-lite (1024d, MRL), voyage-3.5 (1024d), voyage-code-3 (1024d), voyage-context-3 (1024d), voyage-multimodal-3.5 (1024d) | Specialized models, MRL support, contextualized chunks | ✅ VOYAGE_API_KEY |
| **Google** | gemini-embedding-001 (3072d), text-embedding-005 (768d), text-multilingual-embedding-002 (768d) | Google AI embeddings, multilingual | ✅ GOOGLE_API_KEY |
| **OpenAI** | text-embedding-3-small (1536d), text-embedding-3-large (3072d), text-embedding-ada-002 (1536d) | OpenAI embeddings, reliable | ✅ OPENAI_API_KEY |
| **FastEmbed** | Local sentence-transformers models | Local processing | ❌ No API key |
| **HuggingFace** | sentence-transformers models | HuggingFace Hub | ❌ No API key |

### 📝 Model Notes

**Jina AI:**
- `jina-embeddings-v4`: 2048d, multimodal (text+images), 32K context
- `jina-embeddings-v3`: 1024d, multilingual, 8K context
- `jina-colbert-v2`: Late-interaction retrieval (128d/96d/64d variants)
- `jina-code-embeddings-0.5b/1.5b`: Code-specialized, 32K context

**Voyage AI:**
- `voyage-4-large/4/4-lite`: Latest v4 series with MRL (Matryoshka Representation Learning) - supports dimension truncation to 2048/1024/512/256
- `voyage-context-3`: Contextualized chunk embeddings with document-level awareness
- `voyage-multimodal-3.5`: Multimodal support (text/images/video)
- All v4 models share the same embedding space (interoperable)

**Google:**
- Note: `text-embedding-004` is deprecated (Jan 14, 2026)

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
   - **Jina**: High-quality general embeddings, multimodal support (v4), late-interaction (colbert-v2), code-specialized models
   - **Voyage**: Specialized models for code/finance/law, v4 series with MRL support, contextualized chunks
   - **Google**: Strong multilingual support, high-dimensional embeddings
   - **OpenAI**: Reliable, well-tested embeddings

2. **Model Selection Tips**
   - Use `jina-embeddings-v4` for multimodal tasks (text + images)
   - Use `jina-colbert-v2` for late-interaction retrieval (higher accuracy)
   - Use `jina-code-embeddings` for code search and retrieval
   - Use `voyage-4-large` for best retrieval quality
   - Use `voyage-4-lite` for optimized latency/cost
   - Use `voyage-context-3` for document-aware chunk embeddings
   - Use `voyage-code-3` for code-related tasks

3. **Use Batch Processing** for multiple texts to improve performance

4. **Monitor Token Usage** to stay within provider limits

5. **Handle Errors Gracefully** with proper error handling

## 🔗 Related Features

For document relevance scoring after retrieval, see the [Reranker documentation](06-reranking.md).

For more examples, see the main [Advanced Guide](04-advanced-guide.md).
