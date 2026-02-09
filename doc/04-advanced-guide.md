# Octolib: Advanced AI Provider Library

## 🌐 Supported Providers

| Provider | Status | Capabilities |
|----------|--------|--------------|
| OpenAI | ✅ Full Support | Chat, Vision, Tools, Structured Output, Caching |
| Anthropic | ✅ Full Support | Claude Models, Vision, Tools, Caching |
| OpenRouter | ✅ Full Support | Multi-Provider Proxy, Vision, Caching, Structured Output |
| DeepSeek | ✅ Full Support | Open-Source AI Models, Structured Output, Caching |
| Moonshot AI (Kimi) | ✅ Full Support | Kimi K2 Series, Vision (kimi-k2.5), Tools, Structured Output, Caching |
| MiniMax | ✅ Full Support | Anthropic-Compatible API, Tools, Caching, Thinking |
| Z.ai | ✅ Full Support | GLM Models, Caching, Structured Output |
| Google Vertex AI | ✅ Supported | Enterprise AI Integration |
| Amazon Bedrock | ✅ Supported | Cloud AI Services |
| Cloudflare Workers AI | ✅ Supported | Edge AI Compute |
| Local LLM | ✅ Supported | Ollama, LM Studio, LocalAI, Jan, vLLM |
| CLI Proxy | ✅ Supported | Codex, Claude, Gemini, Cursor |

## 🚀 Key Features

- **🔌 Multi-Provider Support**: Unified interface for multiple AI providers
- **🛡️ Robust Error Handling**: Comprehensive error types and context
- **🔍 Intelligent Model Selection**: Provider-aware model parsing
- **🧰 Tool Calling**: Cross-provider tool call standardization
- **🖼️ Vision Support**: Image attachment handling
- **⏱️ Retry & Timeout Management**: Configurable retry strategies
- **📊 Token & Cost Tracking**: Detailed usage and cost metrics
- **🔒 Secure Configuration**: Environment-based API key management

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
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

## 🔑 Environment Variables

### API Keys

Each provider requires its specific API key:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `GOOGLE_APPLICATION_CREDENTIALS`: Google Vertex AI credentials
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: Amazon Bedrock credentials

### Custom API Endpoints (Optional)

All providers support custom API URLs via environment variables. If not set, default provider URLs are used:

| Provider | API Key Env | API URL Env | Default URL |
|----------|-------------|-------------|-------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_API_URL` | `https://api.openai.com/v1/chat/completions` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_URL` | `https://api.anthropic.com/v1/messages` |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_API_URL` | `https://openrouter.ai/api/v1/chat/completions` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_API_URL` | `https://api.deepseek.com/chat/completions` |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_API_URL` | `https://api.minimax.io/anthropic/v1/messages` |
| Z.ai | `ZAI_API_KEY` | `ZAI_API_URL` | `https://api.z.ai/v1/llm/chat/completions` |

Example usage with custom endpoints:

```bash
# Use OpenAI with custom proxy
export OPENAI_API_URL="https://custom-proxy.example.com/v1/chat/completions"

# Use Anthropic with enterprise endpoint
export ANTHROPIC_API_URL="https://enterprise.anthropic.com/v1/messages"
```

### OAuth Authentication (Optional)

For ChatGPT subscriptions and Anthropic OAuth:

**OpenAI OAuth** (ChatGPT Plus/Pro/Team/Enterprise):
- `OPENAI_OAUTH_ACCESS_TOKEN`: OAuth access token
- `OPENAI_OAUTH_ACCOUNT_ID`: ChatGPT account ID
- Both required; library automatically uses OAuth if both are set, falls back to API key otherwise

**Anthropic OAuth**:
- `ANTHROPIC_OAUTH_TOKEN`: OAuth bearer token
- Library automatically uses OAuth if set, falls back to API key otherwise

**Note**: The library only handles authentication detection and request sending. Your application must implement:
- OAuth flow (authorization, token exchange)
- Token refresh (OpenAI tokens expire ~8 days)
- Token storage and security

See `examples/openai_oauth.rs` and `examples/anthropic_oauth.rs` for usage examples.

## 🛠️ Advanced Features

### Provider Selection

```rust
// Get provider dynamically
let (provider, model) = ProviderFactory::get_provider_for_model("anthropic:claude-3.5-sonnet")?;
```

### Tool Calling

```rust
let tools = vec![FunctionDefinition {
    name: "get_weather".to_string(),
    description: "Get current weather for a location".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        }
    }),
}];

let params = params.with_tools(tools);
```

### Error Handling

```rust
match provider.chat_completion(params).await {
    Ok(response) => println!("Response: {}", response.content),
    Err(e) => match e {
        ProviderError::RateLimitExceeded { provider } => {
            // Handle rate limit
        },
        ProviderError::ApiKeyNotFound { provider } => {
            // Handle missing API key
        },
        _ => println!("Unexpected error: {}", e),
    }
}
```

## 📋 Structured Output

Get structured JSON responses with schema validation:

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message, StructuredOutputRequest};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct PersonInfo {
    name: String,
    age: u32,
    occupation: String,
    skills: Vec<String>,
}

async fn structured_output_example() -> anyhow::Result<()> {
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    // Check if provider supports structured output
    if !provider.supports_structured_output(&model) {
        println!("Provider {} does not support structured output", provider.name());
        return Ok(());
    }

    let messages = vec![
        Message::system("You are a helpful assistant that responds with valid JSON."),
        Message::user("Tell me about a fictional software engineer. Respond with JSON containing name, age, occupation, and skills array."),
    ];

    // Basic JSON output
    let structured_request = StructuredOutputRequest::json();
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_structured_output(structured_request);

    let response = provider.chat_completion(params).await?;

    if let Some(structured) = response.structured_output {
        let person: PersonInfo = serde_json::from_value(structured)?;
        println!("Parsed person: {:?}", person);
    }

    // JSON Schema with strict validation (OpenAI/OpenRouter only)
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 18, "maximum": 100},
            "occupation": {"type": "string"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            }
        },
        "required": ["name", "age", "occupation", "skills"],
        "additionalProperties": false
    });

    let structured_request = StructuredOutputRequest::json_schema(schema)
        .with_strict_mode();

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_structured_output(structured_request);

    let response = provider.chat_completion(params).await?;

    if let Some(structured) = response.structured_output {
        match serde_json::from_value::<PersonInfo>(structured) {
            Ok(person) => println!("✅ Schema-validated person: {:?}", person),
            Err(e) => println!("❌ Schema validation failed: {}", e),
        }
    }

    Ok(())
}
```

### Provider Support for Structured Output

| Provider | JSON Mode | JSON Schema | Strict Mode |
|----------|-----------|-------------|-------------|
| OpenAI | ✅ Yes | ✅ Yes | ✅ Yes |
| OpenRouter | ✅ Yes | ✅ Yes | ✅ Yes |
| DeepSeek | ✅ Yes | ❌ No* | ❌ No |
| MiniMax | ✅ Yes | ❌ No | ❌ No |
| Z.ai | ✅ Yes | ❌ No | ❌ No |
| Others | ❌ No | ❌ No | ❌ No |

*DeepSeek falls back to JSON mode when JSON Schema is requested.

## 📊 Token Usage & Tracking

```rust
let response = provider.chat_completion(params).await?;
if let Some(usage) = response.exchange.usage {
    println!("Prompt Tokens: {}", usage.prompt_tokens);
    println!("Output Tokens: {}", usage.output_tokens);
    println!("Total Cost: ${}", usage.cost.unwrap_or(0.0));
}
```

## 🔒 Security & Privacy

- Local-first design
- No external network calls for search
- Respects `.gitignore`
- Secure API key management
- Comprehensive error handling

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

**Built with ❤️ by the Muvon team in Hong Kong**
