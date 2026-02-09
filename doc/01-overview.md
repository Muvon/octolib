# Octolib: Overview

## What is Octolib?

Octolib is a comprehensive, self-sufficient Rust library designed to provide a unified, type-safe interface for interacting with multiple AI service providers. It abstracts away the complexities of working with different AI APIs, offering a consistent and developer-friendly experience.

## Core Design Principles

### 1. Provider Abstraction
Octolib provides a unified `AiProvider` trait that all providers must implement. This allows seamless switching between different AI services without changing your core application logic.

### 2. Type Safety
Every aspect of the library is designed with Rust's type system in mind:
- Strongly typed messages
- Validated model selection
- Comprehensive error handling
- Immutable configuration

### 3. Flexibility
- Support for multiple providers
- Configurable retry mechanisms
- Detailed token and cost tracking
- Cross-provider tool calling

## Key Components

### Provider Factory
The `ProviderFactory` handles:
- Dynamic provider creation
- Model validation
- Provider selection based on model string

### Message System
A robust `Message` type that supports:
- Multiple message roles (user, assistant, system, tool)
- Image attachments
- Caching markers
- Tool call tracking

### Embedding Generation
Multi-provider embedding support:
- Jina, Voyage, Google, OpenAI providers
- Batch processing with token limits
- Input type specification (query/document)
- Optional local models (FastEmbed, HuggingFace)

### Document Reranking
Cross-encoder models for relevance scoring:
- Voyage AI reranker models
- Query-document relevance scoring
- Configurable top-k results
- Token usage tracking

### Error Handling
Comprehensive error types for:
- Provider-specific errors
- Message validation
- Tool call processing
- Configuration issues

### Tool Calling
Cross-provider tool call standardization:
- Unified `ToolCall` structure
- Provider-specific extraction strategies
- Flexible tool definition

## Supported Providers

### LLM Providers
- OpenAI
- Anthropic
- OpenRouter
- DeepSeek
- Moonshot AI (Kimi)
- MiniMax
- Z.ai
- Google Vertex AI
- Amazon Bedrock
- Cloudflare Workers AI
- Local (Ollama, LM Studio, LocalAI, Jan, vLLM)
- CLI proxies (codex, claude, gemini, etc.)

### Embedding Providers
- Jina AI
- Voyage AI
- Google (Gemini)
- OpenAI
- FastEmbed (local, feature-gated)
- HuggingFace (local, feature-gated)

### Reranker Providers
- Voyage AI
- Cohere
- Jina AI
- FastEmbed (local, feature-gated)

## Use Cases

- AI-powered applications
- Chatbots and conversational AI
- Code generation tools
- Intelligent assistants
- Multi-provider AI platforms
- Semantic search with embeddings
- Document relevance ranking
- RAG (Retrieval-Augmented Generation) systems

## Performance Considerations

- Minimal overhead
- Async-first design
- Configurable retry and timeout mechanisms
- Lightweight error handling

## Security

- Environment-based API key management
- No external network calls for provider selection
- Comprehensive input validation
- Respects provider-specific security requirements

## Future Roadmap

- More provider integrations
- Enhanced caching mechanisms
- Improved vision support
- Advanced tool calling capabilities
- Additional reranker providers

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

Apache License 2.0 - Designed for maximum flexibility and open-source collaboration.
