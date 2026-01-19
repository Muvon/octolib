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

- OpenAI
- Anthropic
- OpenRouter
- DeepSeek
- MiniMax
- Z.ai
- Google Vertex AI
- Amazon Bedrock
- Cloudflare Workers AI

## Use Cases

- AI-powered applications
- Chatbots
- Code generation tools
- Intelligent assistants
- Multi-provider AI platforms

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

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

Apache License 2.0 - Designed for maximum flexibility and open-source collaboration.
