# Octolib: Self-Sufficient AI Provider Library

**© 2025 Muvon Un Limited (Hong Kong)** | [Website](https://muvon.io) | [Product Page](https://octolib.muvon.io)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## 🚀 Overview

Octolib is a comprehensive, self-sufficient AI provider library that provides a unified, type-safe interface for interacting with multiple AI services. It offers intelligent model selection, robust error handling, and advanced features like cross-provider tool calling and vision support.

## ✨ Key Features

- **🔌 Multi-Provider Support**: OpenAI, Anthropic, OpenRouter, Google, Amazon, Cloudflare, DeepSeek
- **🛡️ Unified Interface**: Consistent API across different providers
- **🔍 Intelligent Model Validation**: Strict `provider:model` format parsing
- **📋 Structured Output**: JSON and JSON Schema support for OpenAI, OpenRouter, and DeepSeek
- **💰 Cost Tracking**: Automatic token usage and cost calculation
- **🖼️ Vision Support**: Image attachment handling for compatible models
- **🧰 Tool Calling**: Cross-provider tool call standardization
- **⏱️ Retry Management**: Configurable exponential backoff
- **🔒 Secure Design**: Environment-based API key management
- **🎯 Embedding Support**: Multi-provider embedding generation with Jina, Voyage, Google, OpenAI, FastEmbed, and HuggingFace

## 📦 Quick Installation

```bash
# Add to Cargo.toml
octolib = { git = "https://github.com/muvon/octolib" }
```

## 🚀 Quick Start

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message};

async fn example() -> anyhow::Result<()> {
    // Parse model and get provider
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    // Create messages
    let messages = vec![
        Message::user("Hello, how are you?"),
    ];

    // Create completion parameters
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    // Get completion (requires OPENAI_API_KEY environment variable)
    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);

    Ok(())
}
```

### 📋 Structured Output

Get structured JSON responses with schema validation:

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message, StructuredOutputRequest};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct PersonInfo {
    name: String,
    age: u32,
    skills: Vec<String>,
}

async fn structured_example() -> anyhow::Result<()> {
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    // Check if provider supports structured output
    if !provider.supports_structured_output(&model) {
        return Err(anyhow::anyhow!("Provider does not support structured output"));
    }

    let messages = vec![
        Message::user("Tell me about a software engineer in JSON format"),
    ];

    // Request structured JSON output
    let structured_request = StructuredOutputRequest::json();
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_structured_output(structured_request);

    let response = provider.chat_completion(params).await?;

    if let Some(structured) = response.structured_output {
        let person: PersonInfo = serde_json::from_value(structured)?;
        println!("Person: {:?}", person);
    }

    Ok(())
}
```

### 🎯 Embedding Generation

Generate embeddings using multiple providers:

```rust
use octolib::embedding::{generate_embeddings, generate_embeddings_batch, InputType};

async fn embedding_example() -> anyhow::Result<()> {
    // Single embedding generation
    let embedding = generate_embeddings(
        "Hello, world!",
        "voyage",  // provider
        "voyage-3.5-lite"  // model
    ).await?;

    println!("Embedding dimension: {}", embedding.len());

    // Batch embedding generation
    let texts = vec![
        "First document".to_string(),
        "Second document".to_string(),
    ];

    let embeddings = generate_embeddings_batch(
        texts,
        "jina",  // provider
        "jina-embeddings-v4",  // model
        InputType::Document,  // input type for better embeddings
        16,  // batch size
        100_000,  // max tokens per batch
    ).await?;

    println!("Generated {} embeddings", embeddings.len());

    Ok(())
}

// Supported embedding providers:
// - Jina: jina-embeddings-v4, jina-clip-v2, etc.
// - Voyage: voyage-3.5, voyage-code-2, etc.
// - Google: gemini-embedding-001, text-embedding-005
// - OpenAI: text-embedding-3-small, text-embedding-3-large
// - FastEmbed: Local models (feature-gated)
// - HuggingFace: sentence-transformers models
```

## 🎯 Provider Support Matrix

| Provider | Structured Output | Vision | Tool Calls | Caching |
|----------|------------------|--------|------------|---------|
| **OpenAI** | ✅ JSON + Schema | ✅ Yes | ✅ Yes | ✅ Yes |
| **OpenRouter** | ✅ JSON + Schema | ✅ Yes | ✅ Yes | ✅ Yes |
| **DeepSeek** | ✅ JSON Mode | ❌ No | ❌ No | ✅ Yes |
| **Anthropic** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Google Vertex** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Amazon Bedrock** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Cloudflare** | ❌ No | ❌ No | ❌ No | ❌ No |

### Structured Output Details

- **JSON Mode**: Basic JSON object output
- **JSON Schema**: Full schema validation with strict mode
- **Provider Detection**: Use `provider.supports_structured_output(&model)` to check capability

## 📚 Complete Documentation

📖 **Quick Navigation**

- **[Overview](doc/01-overview.md)** - Library introduction and core concepts
- **[Installation Guide](doc/02-installation.md)** - Setup and configuration
- **[Advanced Usage](doc/03-advanced-usage.md)** - Advanced features and customization
- **[Advanced Guide](doc/04-advanced-guide.md)** - Comprehensive usage patterns
- **[Embedding Guide](doc/05-embedding.md)** - Embedding generation with multiple providers

## 🌐 Supported Providers

| Provider | Status | Capabilities |
|----------|--------|--------------|
| OpenAI | ✅ Full Support | Chat, Vision, Tools, Structured Output |
| Anthropic | ✅ Full Support | Claude Models, Vision, Tools, Caching |
| OpenRouter | ✅ Full Support | Multi-Provider Proxy, Vision, Caching, Structured Output |
| DeepSeek | ✅ Full Support | Open-Source AI Models, Structured Output |
| Google Vertex AI | ✅ Supported | Enterprise AI Integration |
| Amazon Bedrock | ✅ Supported | Cloud AI Services |
| Cloudflare Workers AI | ✅ Supported | Edge AI Compute |

## 🔒 Privacy & Security

- **🏠 Local-first design**
- **🔑 Secure API key management**
- **📁 Respects .gitignore**
- **🛡️ Comprehensive error handling**

## 🤝 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/Muvon/octolib/issues)
- **📧 Email**: [opensource@muvon.io](mailto:opensource@muvon.io)
- **🏢 Company**: Muvon Un Limited (Hong Kong)

## ⚖️ License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by the Muvon team in Hong Kong**
