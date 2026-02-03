# Document Reranking Guide

Octolib provides document reranking capabilities to improve search result relevance. Rerankers use cross-encoder models that jointly process query-document pairs for more accurate relevance scoring than embedding-based similarity alone.

## 🎯 What is Reranking?

Reranking is a two-stage retrieval process:

1. **Initial Retrieval**: Use embeddings or lexical search (BM25, TF-IDF) to get candidate documents
2. **Reranking**: Score candidates with a cross-encoder model for precise relevance ordering

Cross-encoders process query and document together, enabling deeper semantic understanding than separate embeddings.

## 🚀 Supported Providers

### API-Based Providers (require API keys)

| Provider | Models | Context Length | Features |
|----------|--------|----------------|----------|
| **Voyage AI** | rerank-2.5, rerank-2.5-lite, rerank-2, rerank-2-lite, rerank-1, rerank-lite-1 | 4K-32K tokens | Multilingual, instruction-following |
| **Cohere** | rerank-english-v3.0, rerank-multilingual-v3.0, rerank-english-v2.0, rerank-multilingual-v2.0 | Up to 4K tokens | Enterprise-grade, multilingual |
| **Jina AI** | jina-reranker-v3, jina-reranker-v2-base-multilingual, jina-reranker-v1-base-en, jina-colbert-v2 | 1K-131K tokens | Automatic chunking, multilingual |

### Local Providers (no API keys, requires features)

| Provider | Models | Features |
|----------|--------|----------|
| **FastEmbed** | bge-reranker-base, bge-reranker-large, jina-reranker-v1-turbo-en, jina-reranker-v2-base-multilingual | ONNX-based, CPU-friendly, no API costs |

### 📝 Model Details

**Voyage AI Models:**

| Model | Context Length | Description | Use Case |
|-------|----------------|-------------|----------|
| `rerank-2.5` | 32K tokens | Latest model, optimized for quality | Best quality, multilingual |
| `rerank-2.5-lite` | 32K tokens | Optimized for latency and quality | Balanced performance |
| `rerank-2` | 16K tokens | Second generation | General purpose |
| `rerank-2-lite` | 8K tokens | Optimized for speed | Fast reranking |
| `rerank-1` | 8K tokens | First generation | Legacy support |
| `rerank-lite-1` | 4K tokens | Fastest model | Low-latency needs |

**Cohere Models:**

| Model | Description | Use Case |
|-------|-------------|----------|
| `rerank-english-v3.0` | Latest English model | Best for English content |
| `rerank-multilingual-v3.0` | Latest multilingual model | 100+ languages |
| `rerank-english-v2.0` | Previous generation English | Legacy support |
| `rerank-multilingual-v2.0` | Previous generation multilingual | Legacy support |

**Jina AI Models:**

| Model | Context Length | Description | Use Case |
|-------|----------------|-------------|----------|
| `jina-reranker-v3` | 131K tokens | Latest model with long context | Long documents |
| `jina-reranker-v2-base-multilingual` | 1K tokens | Multilingual with auto-chunking | Multilingual content |
| `jina-reranker-v1-base-en` | 1K tokens | English-only | English content |
| `jina-colbert-v2` | 8K tokens | ColBERT architecture | Fast retrieval |

**FastEmbed Models (Local):**

| Model | Description | Use Case |
|-------|-------------|----------|
| `bge-reranker-base` | BAAI base model | Balanced quality/speed |
| `bge-reranker-large` | BAAI large model | Best quality |
| `jina-reranker-v1-turbo-en` | Jina turbo model | Fast inference |
| `jina-reranker-v2-base-multilingual` | Jina multilingual | Multilingual local |

**Recommendations:**
- **Best Quality**: Voyage `rerank-2.5` or Jina `jina-reranker-v3`
- **Balanced**: Cohere `rerank-english-v3.0` or Voyage `rerank-2.5-lite`
- **Fast/Local**: FastEmbed `bge-reranker-base` (no API costs)
- **Multilingual**: Cohere `rerank-multilingual-v3.0` or Jina `jina-reranker-v2-base-multilingual`
- **Long Documents**: Jina `jina-reranker-v3` (131K context)

## 🚀 Quick Start

### Environment Setup

**API-Based Providers:**
```bash
# Set API keys for providers you want to use
export VOYAGE_API_KEY="your_voyage_key"
export COHERE_API_KEY="your_cohere_key"
export JINA_API_KEY="your_jina_key"
```

**Local Providers:**
```bash
# Enable fastembed feature in Cargo.toml
cargo build --features fastembed
# No API keys needed!
```

### Basic Reranking

**Voyage AI:**
```rust
use octolib::reranker::rerank;

async fn voyage_example() -> anyhow::Result<()> {
    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.".to_string(),
        "Cooking recipes for beginners.".to_string(),
        "Deep learning uses neural networks.".to_string(),
    ];

    let response = rerank(
        query,
        documents,
        "voyage",
        "rerank-2.5",
        Some(2)  // Return top 2 results
    ).await?;

    for result in response.results {
        println!("Score: {:.4} - {}", result.relevance_score, result.document);
    }

    Ok(())
}
```

**Cohere:**
```rust
use octolib::reranker::rerank;

async fn cohere_example() -> anyhow::Result<()> {
    let response = rerank(
        "machine learning tutorial",
        vec!["AI guide".to_string(), "Cooking tips".to_string()],
        "cohere",
        "rerank-english-v3.0",
        Some(1)
    ).await?;

    println!("Top result: {}", response.results[0].document);
    Ok(())
}
```

**Jina AI:**
```rust
use octolib::reranker::rerank;

async fn jina_example() -> anyhow::Result<()> {
    let response = rerank(
        "artificial intelligence",
        vec!["ML basics".to_string(), "Cooking recipes".to_string()],
        "jina",
        "jina-reranker-v3",
        Some(1)
    ).await?;

    println!("Top result: {}", response.results[0].document);
    Ok(())
}
```

**FastEmbed (Local):**
```rust
use octolib::reranker::rerank;

async fn fastembed_example() -> anyhow::Result<()> {
    // No API key needed - runs locally!
    let response = rerank(
        "deep learning",
        vec!["Neural networks".to_string(), "Pasta recipes".to_string()],
        "fastembed",
        "bge-reranker-base",
        Some(1)
    ).await?;

    println!("Top result: {}", response.results[0].document);
    Ok(())
}
```

### Rerank All Documents

```rust
use octolib::reranker::rerank;

async fn rerank_all_example() -> anyhow::Result<()> {
    let query = "Apple conference call schedule";
    let documents = vec![
        "Mediterranean diet benefits".to_string(),
        "Photosynthesis in plants".to_string(),
        "Apple's Q4 conference call on Nov 2, 2023 at 2pm PT".to_string(),
        "Shakespeare's literary works".to_string(),
    ];

    // Rerank all documents (no top_k limit)
    let response = rerank(query, documents, "voyage", "rerank-2.5", None).await?;

    for result in response.results {
        println!("Score: {:.4} - {}", result.relevance_score, result.document);
    }

    Ok(())
}
```

### Custom Truncation Control

```rust
use octolib::reranker::rerank_with_truncation;

async fn truncation_example() -> anyhow::Result<()> {
    let query = "long query text...";
    let documents = vec!["document 1".to_string(), "document 2".to_string()];

    // Disable truncation - will error if content exceeds context length
    let response = rerank_with_truncation(
        query,
        documents,
        "voyage",
        "rerank-2.5",
        Some(5),
        false  // truncation disabled
    ).await?;

    Ok(())
}
```

## 🔧 Configuration

### Environment Variables

Set your Voyage API key:

```bash
export VOYAGE_API_KEY="your-voyage-api-key"
```

### API Limits

**Voyage AI Limits:**
- Maximum 1,000 documents per request
- Query token limits: 1K-8K depending on model
- Total tokens = (query_tokens × num_documents) + sum(document_tokens)
- Total token limits: 300K-600K depending on model

## 🎯 Response Structure

```rust
pub struct RerankResponse {
    pub results: Vec<RerankResult>,  // Sorted by relevance (descending)
    pub total_tokens: usize,         // Token usage for billing
}

pub struct RerankResult {
    pub index: usize,           // Original position in input
    pub document: String,       // Document text
    pub relevance_score: f64,   // Relevance score (higher = more relevant)
}
```

## 💡 Best Practices

### 1. Two-Stage Retrieval Pipeline

```rust
use octolib::embedding::generate_embeddings;
use octolib::reranker::rerank;

async fn rag_pipeline(query: &str, corpus: Vec<String>) -> anyhow::Result<Vec<String>> {
    // Stage 1: Fast embedding-based retrieval (top 100)
    let query_embedding = generate_embeddings(query, "voyage", "voyage-3.5").await?;
    let candidates = retrieve_top_k_by_similarity(&query_embedding, &corpus, 100);

    // Stage 2: Precise reranking (top 5)
    let response = rerank(query, candidates, "voyage", "rerank-2.5", Some(5)).await?;

    Ok(response.results.into_iter().map(|r| r.document).collect())
}
```

### 2. Instruction-Following (rerank-2.5 models)

```rust
// Add instructions to query for better relevance
let query = "Instruction: Focus on technical accuracy. Query: What is quantum computing?";
let response = rerank(query, documents, "voyage", "rerank-2.5", Some(3)).await?;
```

### 3. Batch Processing

Process large document sets in chunks:

```rust
async fn rerank_large_corpus(
    query: &str,
    documents: Vec<String>,
) -> anyhow::Result<Vec<RerankResult>> {
    let mut all_results = Vec::new();

    // Process in chunks of 1000 (API limit)
    for chunk in documents.chunks(1000) {
        let response = rerank(
            query,
            chunk.to_vec(),
            "voyage",
            "rerank-2.5",
            None
        ).await?;
        all_results.extend(response.results);
    }

    // Sort all results by score
    all_results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());

    Ok(all_results)
}
```

### 4. Error Handling

```rust
use octolib::reranker::rerank;

async fn robust_reranking(query: &str, docs: Vec<String>) -> anyhow::Result<()> {
    match rerank(query, docs, "voyage", "rerank-2.5", Some(5)).await {
        Ok(response) => {
            println!("Reranked {} documents", response.results.len());
        }
        Err(e) => {
            eprintln!("Reranking failed: {}", e);
            // Fallback to embedding-based ranking
        }
    }
    Ok(())
}
```

## 🔗 Integration Patterns

### RAG (Retrieval-Augmented Generation)

```rust
use octolib::reranker::rerank;
use octolib::llm::{ProviderFactory, ChatCompletionParams, Message};

async fn rag_with_reranking(
    user_query: &str,
    candidate_docs: Vec<String>,
) -> anyhow::Result<String> {
    // Rerank candidates
    let reranked = rerank(
        user_query,
        candidate_docs,
        "voyage",
        "rerank-2.5",
        Some(3)
    ).await?;

    // Build context from top results
    let context = reranked.results
        .iter()
        .map(|r| r.document.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    // Generate answer with LLM
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;
    let messages = vec![
        Message::system(&format!("Context:\n{}", context)),
        Message::user(user_query),
    ];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);
    let response = provider.chat_completion(params).await?;

    Ok(response.content)
}
```

### Semantic Search Enhancement

```rust
// Combine embedding similarity with reranking
async fn hybrid_search(
    query: &str,
    corpus: Vec<String>,
) -> anyhow::Result<Vec<String>> {
    // 1. Embedding-based retrieval (fast, broad)
    let embedding_candidates = embedding_search(query, &corpus, 50).await?;

    // 2. Reranking (precise, focused)
    let response = rerank(
        query,
        embedding_candidates,
        "voyage",
        "rerank-2.5",
        Some(10)
    ).await?;

    Ok(response.results.into_iter().map(|r| r.document).collect())
}
```

## 📊 Performance Considerations

### Model Selection

| Scenario | Recommended Model | Reason |
|----------|------------------|---------|
| Best quality | `rerank-2.5` | Highest accuracy |
| Balanced | `rerank-2.5-lite` | Good quality, faster |
| High throughput | `rerank-2-lite` | Optimized latency |
| Long documents | `rerank-2.5` | 32K context |

### Cost Optimization

1. **Use top_k**: Only rerank what you need
2. **Pre-filter**: Use cheap methods first (embeddings, BM25)
3. **Batch wisely**: Balance latency vs. throughput
4. **Cache results**: Reuse reranking for identical queries

## 🔗 Related Features

- [Embedding Generation](05-embedding.md) - For initial retrieval
- [Advanced Guide](04-advanced-guide.md) - Complete usage patterns
- [Tool Calling](07-tool-calling.md) - LLM integration

## 📚 Additional Resources

- [Voyage AI Reranker Documentation](https://docs.voyageai.com/docs/reranker)
- [Cross-Encoders Explained](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Two-Stage Retrieval Best Practices](https://www.pinecone.io/learn/series/rag/rerankers/)
