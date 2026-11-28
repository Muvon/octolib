# Octolib: Advanced AI Provider Library

## 🌐 Supported Providers

| Provider | Status | Capabilities |
|----------|--------|--------------|
| OpenAI | ✅ Full Support | Chat, Vision, Tools, Structured Output, Caching |
| Anthropic | ✅ Full Support | Claude Models, Vision, Tools, Caching |
| OpenRouter | ✅ Full Support | Multi-Provider Proxy, Vision, Caching, Structured Output |
| Groq | ✅ Full Support | Fast Inference, Structured Output, Caching |
| BytePlus | ✅ Full Support | Seed Models, Structured Output, Caching |
| Alibaba Model Studio | ✅ Full Support | Qwen Models + Resold DeepSeek/GLM, Vision, Video, Tools, Thinking, Caching |
| DeepSeek | ✅ Full Support | Open-Source AI Models, Structured Output, Caching |
| Moonshot AI (Kimi) | ✅ Full Support | Kimi K2 Series, Vision (kimi-k2.5), Tools, Structured Output, Caching, Thinking |
| MiniMax | ✅ Full Support | Anthropic-Compatible API, Tools, Caching, Thinking |
| Z.ai | ✅ Full Support | GLM Models, Caching, Structured Output |
| NVIDIA NIM | ✅ Full Support | 100+ Hosted Models, Tools, Structured Output, Reference Pricing |
| Together AI | ✅ Full Support | Multi-Provider Proxy, Vision, Tools, Structured Output |
| Cerebras | ✅ Full Support | Fast Inference, Structured Output |
| Isoquant | ✅ Supported | Models Read From Its Catalog, Anthropic Messages or OpenAI Chat Completions, Vision, Tools, Structured Output, Caching on Chat Completions, Zero Data Retention on Inference |
| Reflection AI | ✅ Supported | Beam Reasoning Model, Tools, Structured Output, Caching |
| OctoHub | ✅ Supported | Local AI Serving |
| Google Vertex AI | ✅ Supported | Enterprise AI Integration |
| Google AI Studio | ✅ Supported | Gemini API, API-Key Auth |
| Amazon Bedrock | ✅ Supported | Cloud AI Services |
| Cloudflare Workers AI | ✅ Supported | Edge AI Compute |
| Local LLM | ✅ Supported | Ollama, LM Studio, LocalAI, Jan, vLLM |
| WebWright | ✅ Supported | OpenAI-Compatible Chat, Caching |
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
    let params = ChatCompletionParams::new(&messages, &model, 1000);

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
- `GROQ_API_KEY`: Groq API key
- `BYTEPLUS_API_KEY`: BytePlus API key
- `ALIBABA_API_KEY`: Alibaba Cloud Model Studio (DashScope) API key
- `ALIBABA_WORKSPACE_ID` (optional): Model Studio workspace identity; with it and no `ALIBABA_API_URL`, the Singapore pay-as-you-go Messages endpoint becomes the default
- `NVIDIA_API_KEY`: NVIDIA NIM API key
- `CLOUDFLARE_API_KEY`: Cloudflare API key
- `TOGETHER_API_KEY`: Together AI API key
- `OCTOHUB_API_KEY`: OctoHub API key
- `GOOGLE_VERTEX_CREDENTIAL_FILE` (or `GOOGLE_APPLICATION_CREDENTIALS`): Google Vertex AI credentials
- `GOOGLE_VERTEX_PROJECT_ID`, `GOOGLE_VERTEX_LOCATION`: Optional Vertex project/location overrides (default location: `us-central1`)
- `GOOGLE_STUDIO_API_KEY`: Google AI Studio (Gemini API) key
- `WEBWRIGHT_API_KEY`: WebWright API key
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY`: Amazon Bedrock credentials

### Custom API Endpoints (Optional)

All providers support custom API URLs via environment variables. If not set, default provider URLs are used:

| Provider | API Key Env | API URL Env | Default URL |
|----------|-------------|-------------|-------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_API_URL` | `https://api.openai.com/v1/chat/completions` |
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_URL` | `https://api.anthropic.com/v1/messages` |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_API_URL` | `https://openrouter.ai/api/v1/chat/completions` |
| Groq | `GROQ_API_KEY` | `GROQ_API_URL` | `https://api.groq.com/openai/v1/chat/completions` |
| BytePlus | `BYTEPLUS_API_KEY` | `BYTEPLUS_API_URL` | `https://ark.ap-southeast.bytepluses.com/api/v3/chat/completions` |
| Alibaba Model Studio | `ALIBABA_API_KEY` | `ALIBABA_API_URL`, `ALIBABA_WORKSPACE_ID` | Chat Completions (`https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`) by default. With `ALIBABA_WORKSPACE_ID` set and no URL, Messages on the Singapore pay-as-you-go base (`https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages`); Token/Coding Plan accounts set their own full Messages base in `ALIBABA_API_URL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_API_URL` | `https://api.deepseek.com/anthropic/v1/messages` |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_API_URL` | `https://api.minimax.io/anthropic/v1/messages` |
| Moonshot | `MOONSHOT_API_KEY` | `MOONSHOT_API_URL` | Messages (`https://api.moonshot.ai/anthropic/v1/messages`); Chat Completions (`https://api.moonshot.ai/v1/chat/completions`) with `MOONSHOT_API_MODE=openai` |
| Z.ai | `ZAI_API_KEY` | `ZAI_API_URL` | Messages (`https://api.z.ai/api/anthropic/v1/messages`) for a model that takes no images; Chat Completions (`https://api.z.ai/api/paas/v4/chat/completions`) for a vision model, for the whole conversation |
| Isoquant | `ISOQUANT_API_KEY` | `ISOQUANT_API_URL` | Messages (`https://api.isoquant.ai/v1/messages`); Chat Completions (`https://api.isoquant.ai/v1/chat/completions`) with `ISOQUANT_API_MODE=openai` |
| Xiaomi MiMo | `XIAOMI_API_KEY` | `XIAOMI_API_URL`, `XIAOMI_API_MODE` | Chat Completions (`https://api.xiaomimimo.com/v1/chat/completions`); Messages (`https://api.xiaomimimo.com/anthropic/v1/messages`) with `XIAOMI_API_MODE=anthropic` |
| NVIDIA NIM | `NVIDIA_API_KEY` | `NVIDIA_API_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` |
| Together AI | `TOGETHER_API_KEY` | `TOGETHER_API_URL` | `https://api.together.xyz/v1/chat/completions` |
| Cerebras | `CEREBRAS_API_KEY` | `CEREBRAS_API_URL` | `https://api.cerebras.ai/v1/chat/completions` |
| OctoHub | `OCTOHUB_API_KEY` | `OCTOHUB_API_URL` | `http://127.0.0.1:8080` |
| Google Studio | `GOOGLE_STUDIO_API_KEY` | `GOOGLE_STUDIO_API_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions` |
| Google Vertex | `GOOGLE_VERTEX_CREDENTIAL_FILE` (service account) | `GOOGLE_VERTEX_API_URL` | Derived from `GOOGLE_VERTEX_PROJECT_ID`/`GOOGLE_VERTEX_LOCATION` |
| WebWright | `WEBWRIGHT_API_KEY` | `WEBWRIGHT_API_URL` | `https://webwright.ai/v1/chat/completions` |
| Tinker | `TINKER_API_KEY` | `TINKER_API_URL`, `TINKER_API_MODE` | Messages (`https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api/v1/messages`); Chat Completions (`https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions`) with `TINKER_API_MODE=openai` |
| Local | `LOCAL_API_KEY` (optional) | `LOCAL_API_URL` | `http://localhost:11434/v1/chat/completions` (Ollama) |

#### Messages or Chat Completions

DeepSeek, MiniMax, Moonshot, Alibaba, Z.ai, Isoquant, Xiaomi and Tinker also speak the Anthropic Messages protocol, as do some local servers. `DEEPSEEK_API_MODE`, `MINIMAX_API_MODE`, `MOONSHOT_API_MODE`, `ALIBABA_API_MODE`, `ZAI_API_MODE`, `ISOQUANT_API_MODE`, `XIAOMI_API_MODE`, `TINKER_API_MODE` and `LOCAL_API_MODE` take `anthropic` or `openai`; any other value is refused before a request is sent.

- A mode states the protocol. The URL is sent as configured and is never rewritten.
- Without a mode, the protocol comes from the URL's own path: one ending in `/messages` speaks Messages, one ending in `/chat/completions` speaks Chat Completions. A URL naming neither keeps the protocol it had before this option existed: Chat Completions for DeepSeek, Z.ai, Moonshot, Alibaba, Isoquant, Xiaomi, Tinker and local servers, Messages for MiniMax.
- With no URL, DeepSeek, MiniMax, Moonshot, Isoquant and Tinker default to their Messages endpoint (Isoquant to Chat Completions for a model its catalog serves there only), Z.ai to Messages for a model that takes no images and to Chat Completions for a vision model, and local servers to Ollama's Chat Completions endpoint. `LOCAL_API_MODE=anthropic` alone sends to Ollama's Messages endpoint, `http://localhost:11434/v1/messages`.
- Tinker takes its key in `x-api-key` on Messages. Images reach Inkling models there only, as inline base64 (also inside a tool result); other models and URL images are refused before a request is sent, and its Chat Completions endpoint documents no image input. Effort goes under `output_config.effort` for the families probed there: Inkling takes every level and a request for no reasoning switches its thinking off; gpt-oss takes `low`, `medium` and `high` only and has no off switch. Other models get no effort on Messages, because the endpoint answers 400 to models without effort support; `TINKER_API_MODE=openai` keeps the Chat Completions mapping for them. The endpoint reports an uncached prompt as `cache_creation_input_tokens` with `input_tokens` at 0, and Tinker bills both at the input rate. `cache_lifetime` was measured on Chat Completions and reports `None` on Messages.
- Z.ai chooses by the model, not by the turn: a vision model uses Chat Completions for every turn of a conversation — the text-only turn before its first image, and a compacted history that dropped the image, stay there — so the endpoint, the prefix cache and the signed thinking block never change mid-conversation. Its Messages endpoint takes no images on any model, its Chat Completions endpoint takes no video, and `supports_video()` reports `false` for every Z.ai model: a video request fails instead of being dropped.
- Contracts: `https://platform.kimi.ai/docs/api/messages` (Moonshot), `https://www.alibabacloud.com/help/en/model-studio/anthropic-api-messages` (Alibaba). On the Messages route, an image goes as inline base64: Moonshot's endpoint answered `400 unsupported image url` to a URL image on 2026-11-26, and Alibaba's Token Plan route is unproven for it, so a URL image is refused there instead of dropped. Alibaba rejects images whose height or width is 10px or less (`InvalidParameter`, measured with `qwen3.8-max`), and its Token Plan catalogue serves no `-vl` model, so an image on a plan host needs a model that plan actually carries. Neither protocol has a video block, so a video turn is refused there instead of being dropped. `MOONSHOT_API_MODE=openai` and `ALIBABA_API_MODE=openai` keep the Chat Completions route, which carries `video_url`. Capability methods follow the resolved route: `enforces_response_schema` and `supports_video` report true only while Chat Completions is in use, so a Messages turn takes the forced-tool schema path and never claims native schema enforcement.
- Isoquant takes its key in `x-api-key` on Messages and in `Authorization: Bearer` on Chat Completions, sends `Isoquant-ZDR: required` on both inference routes, and drops a caller's own copy of those headers; the public catalog (`GET /v1/models`, model metadata only) is read without the key or that header. The catalog is read beside the completion URL's path, so a trailing slash or a query on that URL (a credential included) never reaches discovery, and the completion URL itself is sent exactly as configured. On Messages an image goes as inline base64, and a URL image is refused because the endpoint does not document that shape. Reasoning effort goes under `output_config.effort` with the model's listed levels. `enforces_response_schema`, `supports_caching` and `cache_lifetime` were probed on Chat Completions and report nothing on Messages, `supports_required_tool_choice` is true on Chat Completions only because the Messages transport has no required tool choice, so a Messages turn with a `json_schema` takes the forced-tool path. A model outside the catalog claims no capability on Messages beyond its context window and price.
- Xiaomi keeps Chat Completions as its default because only that route decodes against a native `json_schema`: `enforces_response_schema` is true there alone, and a Messages turn with a schema takes the forced-tool path. `XIAOMI_API_MODE=anthropic` speaks Messages on `https://api.xiaomimimo.com/anthropic/v1/messages`. Xiaomi's guides address a host by a base, so a `XIAOMI_API_URL` ending in `/v1` or `/anthropic` names the host and the resolved protocol's endpoint is built on it: `https://token-plan-sgp.xiaomimimo.com/anthropic` sends to `/anthropic/v1/messages`, and the same host's `/v1` to `/v1/chat/completions`. A full endpoint or any other URL is sent as written. Messages reads the key as `Authorization: Bearer` and takes images (URL or base64) and video; it documents no response format and no cache lifetime, so `cache_read_input_tokens` is reported and `cache_lifetime` is `None` there, and a caller's cache breakpoints are not sent because the contract has no `cache_control`.
- Cache counters arrive in both routes' usage (`cache_creation_input_tokens`, `cache_read_input_tokens`, and the 5m/1h split on Moonshot's SSE), but a short prompt does not populate them: an identical 92-token `kimi-k3` prompt repeated immediately still read `cache_read_input_tokens: 0`, because it is below Moonshot's cacheable length. `TokenUsage` carries whatever the endpoint reports and never invents a hit.
- Alibaba resolves its endpoint in this order: `ALIBABA_API_URL` (used exactly as written, never rewritten), then `ALIBABA_API_MODE`, then `ALIBABA_WORKSPACE_ID`, then the Chat Completions default. A workspace identity selects the documented pay-as-you-go Messages base above: it is validated as a single DNS label (lowercase letters, digits, hyphens, 1-63 characters) before it enters the host, so no configured value can add a domain, path or port. Token Plan and Coding Plan bases are per plan and cannot be derived from a workspace, so those accounts set the full Messages URL. `ALIBABA_API_MODE=anthropic` with neither a URL nor a workspace is refused before a request is sent, and the legacy generic `dashscope-intl` `/api/v2/apps/claude-code-proxy` path is not a default.
- `LOCAL_ANTHROPIC_MODELS` (comma-separated model IDs) makes those local models speak Messages whatever the mode or the URL's path says.

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
let (provider, model) = ProviderFactory::get_provider_for_model("anthropic:claude-sonnet-5-5")?;
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
    let params = ChatCompletionParams::new(&messages, &model, 1000)
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

    let params = ChatCompletionParams::new(&messages, &model, 1000)
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
| NVIDIA NIM | ✅ Yes | ✅ Yes | ✅ Yes |
| Groq | ✅ Yes | ✅ Yes | ✅ Yes |
| BytePlus | ✅ Yes | ✅ Yes | ✅ Yes |
| Cerebras | ✅ Yes | ✅ Yes | ✅ Yes |
| Together AI | Per-model | Per-model | Per-model |
| DeepSeek | ✅ Yes | ❌ No* | ❌ No |
| Moonshot AI (Kimi) | ✅ Yes | ❌ No | ❌ No |
| MiniMax | ✅ Yes | ❌ No | ❌ No |
| Z.ai | ✅ Yes | ❌ No | ❌ No |
| Local/Ollama/OctoHub | Per-model | Per-model | Per-model |
| Alibaba Model Studio | ❌ No** | ❌ No | ❌ No |
| Others | ❌ No | ❌ No | ❌ No |

*DeepSeek falls back to JSON mode when JSON Schema is requested.

**Alibaba compatible mode maps JSON Schema onto JSON mode without enforcing the schema, and rejects the request unless the prompt itself contains the word "json" — so octolib never advertises structured output for it.

## 📊 Token Usage & Tracking

```rust
let response = provider.chat_completion(params).await?;
if let Some(usage) = response.exchange.usage {
    println!("Input Tokens: {}", usage.input_tokens);
    println!("Cache Read Tokens: {}", usage.cache_read_tokens);
    println!("Cache Write Tokens: {}", usage.cache_write_tokens);
    println!("Output Tokens: {}", usage.output_tokens);
    println!("Reasoning Tokens: {}", usage.reasoning_tokens);
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
