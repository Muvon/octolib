# Octolib: Advanced AI Provider Library

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

Each provider requires its specific API key:

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `GOOGLE_APPLICATION_CREDENTIALS`: Google Vertex AI credentials
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: Amazon Bedrock credentials

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