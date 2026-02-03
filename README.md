# Octolib: Self-Sufficient AI Provider Library

**© 2025 Muvon Un Limited (Hong Kong)** | [Website](https://muvon.io) | [Product Page](https://octolib.muvon.io)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

## 🚀 Overview

Octolib is a comprehensive, self-sufficient AI provider library that provides a unified, type-safe interface for interacting with multiple AI services. It offers intelligent model selection, robust error handling, and advanced features like cross-provider tool calling and vision support.

## ✨ Key Features

- **🔌 Multi-Provider Support**: OpenAI, Anthropic, OpenRouter, Google, Amazon, Cloudflare, DeepSeek, MiniMax, Z.ai, CLI proxies
- **🛡️ Unified Interface**: Consistent API across different providers
- **🔍 Intelligent Model Validation**: Strict `provider:model` format parsing with case-insensitive model support
- **📋 Structured Output**: JSON and JSON Schema support for OpenAI, OpenRouter, and DeepSeek
- **💰 Cost Tracking**: Automatic token usage and cost calculation
- **🖼️ Vision Support**: Image attachment handling for compatible models
- **🧰 Tool Calling**: Cross-provider tool call standardization
- **🧩 CLI Provider**: Use `cli:<backend>/<model>` (e.g. `cli:codex/gpt-5.2-codex`). Proxy-only: tools/MCP are not used or controllable.
- **⏱️ Retry Management**: Configurable exponential backoff
- **🔒 Secure Design**: Environment-based API key management
- **🎯 Embedding Support**: Multi-provider embedding generation with Jina, Voyage, Google, OpenAI, FastEmbed, and HuggingFace
- **🔄 Reranking**: Document relevance scoring with cross-encoder models (Voyage AI)

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

### 🧩 CLI Provider (Proxy Mode)

Use local CLIs as a lightweight proxy. This mode is prompt-only; tool calling/MCP integration is not used or controllable.

```rust
let (provider, model) = ProviderFactory::get_provider_for_model(\"cli:codex/gpt-5.2-codex\")?;
// or: \"cli:claude/claude-sonnet-4-5\"
// or: \"cli:gemini/gemini-2.5-pro\"
// or: \"cli:cursor/auto\"
```

Set a backend-specific command if it is not on PATH:

```
CLI_CODEX_COMMAND=/path/to/codex
CLI_CLAUDE_COMMAND=/path/to/claude
CLI_GEMINI_COMMAND=/path/to/gemini
CLI_CURSOR_COMMAND=/path/to/cursor-agent
```

### 🧰 Tool Calling

Use AI models to call functions with automatic parameter extraction:

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message, FunctionDefinition, ToolCall};
use serde_json::json;

async fn tool_calling_example() -> anyhow::Result<()> {
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    // Define available tools/functions
    let tools = vec![
        FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a location".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }),
            cache_control: None,
        },
        FunctionDefinition {
            name: "calculate".to_string(),
            description: "Perform a mathematical calculation".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }),
            cache_control: None,
        },
    ];

    let mut messages = vec![
        Message::user("What's the weather in Tokyo and calculate 15 * 23?"),
    ];

    // Initial request with tools
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_tools(tools.clone());

    let response = provider.chat_completion(params).await?;

    // Check if model wants to call tools
    if let Some(tool_calls) = response.tool_calls {
        println!("Model requested {} tool calls", tool_calls.len());

        // Add assistant's response with tool calls to conversation
        let mut assistant_msg = Message::assistant(&response.content);
        assistant_msg.tool_calls = Some(serde_json::to_value(&tool_calls)?);
        messages.push(assistant_msg);

        // Execute each tool call and add results
        for tool_call in tool_calls {
            println!("Calling tool: {} with args: {}", tool_call.name, tool_call.arguments);

            // Execute the tool (your implementation)
            let result = match tool_call.name.as_str() {
                "get_weather" => {
                    let location = tool_call.arguments["location"].as_str().unwrap_or("Unknown");
                    json!({
                        "location": location,
                        "temperature": 22,
                        "unit": "celsius",
                        "condition": "sunny"
                    })
                }
                "calculate" => {
                    let expr = tool_call.arguments["expression"].as_str().unwrap_or("0");
                    // Simple calculation (in real app, use proper eval)
                    json!({
                        "expression": expr,
                        "result": 345  // 15 * 23
                    })
                }
                _ => json!({"error": "Unknown tool"}),
            };

            // Add tool result to conversation
            messages.push(Message::tool(
                &serde_json::to_string(&result)?,
                &tool_call.id,
                &tool_call.name,
            ));
        }

        // Get final response with tool results
        let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
            .with_tools(tools);

        let final_response = provider.chat_completion(params).await?;
        println!("Final response: {}", final_response.content);
    } else {
        println!("Direct response: {}", response.content);
    }

    Ok(())
}
```

**Tool Calling Features:**
- ✅ Cross-provider support (OpenAI, Anthropic, Google, Amazon, OpenRouter)
- ✅ Automatic parameter validation via JSON Schema
- ✅ Multi-turn conversations with tool results
- ✅ Parallel tool execution support
- ✅ Standardized `ToolCall` and `GenericToolCall` formats across all providers
- ✅ Provider-specific metadata preservation (e.g., Gemini thought signatures)
- ✅ Clean conversion API with `to_generic_tool_calls()` method

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

### 🎯 Document Reranking

Improve search results by scoring document relevance with cross-encoder models:

```rust
use octolib::reranker::rerank;

async fn reranking_example() -> anyhow::Result<()> {
    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of AI.".to_string(),
        "Cooking recipes for beginners.".to_string(),
        "Deep learning uses neural networks.".to_string(),
    ];

    // Rerank documents by relevance to query
    let response = rerank(
        query,
        documents,
        "voyage",           // provider: voyage, cohere, jina, fastembed
        "rerank-2.5",       // model
        Some(2)             // top_k: return top 2 results
    ).await?;

    for (rank, result) in response.results.iter().enumerate() {
        println!("Rank {}: Score {:.4}", rank + 1, result.relevance_score);
        println!("  Document: {}", result.document);
    }

    println!("Total tokens used: {}", response.total_tokens);

    Ok(())
}

// Supported Providers:
//
// API-Based (require API keys):
// - Voyage AI (VOYAGE_API_KEY): rerank-2.5, rerank-2.5-lite, rerank-2, rerank-2-lite
// - Cohere (COHERE_API_KEY): rerank-english-v3.0, rerank-multilingual-v3.0
// - Jina AI (JINA_API_KEY): jina-reranker-v3, jina-reranker-v2-base-multilingual
//
// Local (no API keys, requires features):
// - FastEmbed (fastembed feature): bge-reranker-base, bge-reranker-large, jina-reranker-v1-turbo-en
```

### 🔐 OAuth Authentication

Octolib supports OAuth authentication for ChatGPT subscriptions and Anthropic:

**OpenAI OAuth** (ChatGPT Plus/Pro/Team/Enterprise):
```bash
export OPENAI_OAUTH_ACCESS_TOKEN="your_oauth_token"
export OPENAI_OAUTH_ACCOUNT_ID="your_account_id"
```

**Anthropic OAuth**:
```bash
export ANTHROPIC_OAUTH_TOKEN="your_bearer_token"
```

The library automatically detects OAuth credentials and prefers them over API keys. See `examples/openai_oauth.rs` and `examples/anthropic_oauth.rs` for full usage examples.

## 🎯 Provider Support Matrix

| Provider | Structured Output | Vision | Tool Calls | Caching |
|----------|------------------|--------|------------|---------|
| **OpenAI** | ✅ JSON + Schema | ✅ Yes | ✅ Yes | ✅ Yes |
| **OpenRouter** | ✅ JSON + Schema | ✅ Yes | ✅ Yes | ✅ Yes |
| **DeepSeek** | ✅ JSON Mode | ❌ No | ❌ No | ✅ Yes |
| **Anthropic** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **MiniMax** | ✅ JSON Mode | ❌ No | ✅ Yes | ✅ Yes |
| **Z.ai** | ✅ JSON Mode | ❌ No | ✅ Yes | ✅ Yes |
| **Google Vertex** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Amazon Bedrock** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Cloudflare** | ❌ No | ❌ No | ❌ No | ❌ No |


### Structured Output Details

- **JSON Mode**: Basic JSON object output
- **JSON Schema**: Full schema validation with strict mode
- **Provider Detection**: Use `provider.supports_structured_output(&model)` to check capability

### 🧠 Thinking/Reasoning Support

Octolib provides first-class support for models that produce thinking/reasoning content. Thinking is stored **separately** from the main response content, similar to how `tool_calls` are separate from content.

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message, ThinkingBlock};

async fn thinking_example() -> anyhow::Result<()> {
    // MiniMax and OpenAI o-series models support thinking
    let (provider, model) = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2")?;

    let messages = vec![
        Message::user("Solve this complex math problem step by step"),
    ];

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);
    let response = provider.chat_completion(params).await?;

    // Access thinking content (separate from response.content)
    if let Some(ref thinking) = response.thinking {
        println!("=== MODEL THINKING ({}) ===", thinking.tokens);
        println!("{}", thinking.content);
        println!("==========================");
    }

    // Final response (clean, no thinking prefix)
    println!("Response: {}", response.content);

    // Token usage breakdown
    if let Some(usage) = &response.exchange.usage {
        println!("Prompt tokens: {}", usage.prompt_tokens);
        println!("Output tokens: {}", usage.output_tokens);
        println!("Reasoning tokens: {}", usage.reasoning_tokens);
    }

    Ok(())
}
```

#### Supported Providers

| Provider | Thinking Format | Notes |
|----------|----------------|-------|
| **MiniMax** | Content blocks (`{"type": "thinking"}`) | Full thinking block extraction |
| **OpenAI o-series** | `reasoning_content` field | o1, o3, o4 models |
| **OpenRouter** | `reasoning_details` | Gemini and other providers |

#### Token Tracking

Thinking tokens are tracked separately in `TokenUsage.reasoning_tokens`:

```rust
if let Some(usage) = &response.exchange.usage {
    println!("Total tokens: {}", usage.total_tokens);
    println!("  - Prompt: {}", usage.prompt_tokens);
    println!("  - Output: {}", usage.output_tokens);
    println!("  - Reasoning: {}", usage.reasoning_tokens);
}
```

## 📚 Complete Documentation

📖 **Quick Navigation**

- **[Overview](doc/01-overview.md)** - Library introduction and core concepts
- **[Installation Guide](doc/02-installation.md)** - Setup and configuration
- **[Advanced Usage](doc/03-advanced-usage.md)** - Advanced features and customization
- **[Advanced Guide](doc/04-advanced-guide.md)** - Comprehensive usage patterns
- **[Embedding Guide](doc/05-embedding.md)** - Embedding generation with multiple providers
- **[Reranking Guide](doc/06-reranking.md)** - Document relevance scoring
- **[Tool Calling](doc/07-tool-calling.md)** - Cross-provider tool calling
- **[Thinking/Reasoning](doc/08-thinking.md)** - Reasoning model support

## 🌐 Supported Providers

| Provider | Status | Capabilities |
|----------|--------|--------------|
| OpenAI | ✅ Full Support | Chat, Vision, Tools, Structured Output, Caching |
| Anthropic | ✅ Full Support | Claude Models, Vision, Tools, Caching |
| OpenRouter | ✅ Full Support | Multi-Provider Proxy, Vision, Caching, Structured Output |
| DeepSeek | ✅ Full Support | Open-Source AI Models, Structured Output, Caching |
| MiniMax | ✅ Full Support | Anthropic-Compatible API, Tools, Caching, Thinking, Structured Output |
| Z.ai | ✅ Full Support | GLM Models, Caching, Structured Output |
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
