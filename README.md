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
- **💰 Cost Tracking**: Automatic token usage and cost calculation
- **🖼️ Vision Support**: Image attachment handling for compatible models
- **🧰 Tool Calling**: Cross-provider tool call standardization
- **⏱️ Retry Management**: Configurable exponential backoff
- **🔒 Secure Design**: Environment-based API key management

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

## 📚 Complete Documentation

📖 **Quick Navigation**

- **[Overview](doc/01-overview.md)** - Library introduction and core concepts
- **[Installation Guide](doc/02-installation.md)** - Setup and configuration
- **[Advanced Usage](doc/03-advanced-usage.md)** - Advanced features and customization
- **[Advanced Guide](doc/04-advanced-guide.md)** - Comprehensive usage patterns

## 🌐 Supported Providers

| Provider | Status | Capabilities |
|----------|--------|--------------|
| OpenAI | ✅ Full Support | Chat, Vision, Tools |
| Anthropic | ✅ Full Support | Claude Models, Vision, Tools, Caching |
| OpenRouter | ✅ Full Support | Multi-Provider Proxy, Vision, Caching |
| Google Vertex AI | ✅ Supported | Enterprise AI Integration |
| Amazon Bedrock | ✅ Supported | Cloud AI Services |
| Cloudflare Workers AI | ✅ Supported | Edge AI Compute |
| DeepSeek | ✅ Supported | Open-Source AI Models |

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