# Octolib

> One `provider:model` string — 30 AI providers, one trait, cost tracking built in.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Crates.io](https://img.shields.io/crates/v/octolib.svg)](https://crates.io/crates/octolib)
[![Documentation](https://docs.rs/octolib/badge.svg)](https://docs.rs/octolib)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmuvon%2Foctolib%2Fbadges%2Fcoverage.json&style=flat-square)](https://github.com/muvon/octolib/actions/workflows/ci.yml)
## Overview

Octolib is a self-sufficient Rust library for AI providers. One `provider:model` string — `openai:gpt-4o`, `anthropic:claude-opus-4`, `ollama:llama3.2` — resolves through `ProviderFactory` to a provider behind the `AiProvider` trait. Switching models is a string change, not a rewrite: the same `chat_completion` call drives OpenAI's Responses API, Anthropic's Messages API, an OpenAI-compatible proxy, or a local Ollama server.

Pricing and capability tables ship inside the crate, so every response reports token usage and USD cost — input, output, cache reads/writes, and reasoning tokens — with zero configuration. The API is `Result`-based end to end with no panics or printing in library code, and API keys are read from environment variables only.

**Contents:** [Features](#-key-features) · [Installation](#-quick-installation) · [Quick Start](#-quick-start) · [Media](#media-generation) · [Evaluation](#evaluation) · [Structured Output](#-structured-output) · [CLI Provider](#-cli-provider-proxy-mode) · [Tool Calling](#-tool-calling) · [Embeddings](#-embedding-generation) · [Reranking](#-document-reranking) · [OAuth](#-oauth-authentication) · [Provider Matrix](#-provider-support-matrix) · [Thinking](#-thinkingreasoning-support) · [Docs](#-complete-documentation) · [Security](#-privacy--security) · [Support](#-support--community) · [License](#-license)

## ✨ Key Features

- **🔌 30 providers, one interface** — OpenAI, Anthropic, xAI, OpenRouter, Google Vertex & Studio, Amazon Bedrock, DeepSeek, Moonshot (Kimi), MiniMax, Z.ai, BytePlus, Alibaba Model Studio, Groq, Cerebras, NVIDIA NIM, Together, Featherless, Fireworks, Hetzner, Inception Labs, Meta, Tinker, OpenCode Zen/Go, OctoHub, Cloudflare Workers AI, Ollama, Local, and CLI proxies (Codex, Claude, Gemini, Cursor)
- **💰 Cost tracking with zero config** — per-model pricing tables ship in the crate; every response carries input, output, cache, and reasoning tokens with USD cost
- **🧰 Tool calling** — one `ToolCall` format across providers, with JSON Schema parameter validation and multi-turn conversations
- **📋 Structured output** — JSON and JSON Schema modes, validated locally even when the upstream doesn't enforce them
- **🧠 Thinking/reasoning** — reasoning content and token counts surfaced separately from the answer; the `ReasoningEffort` hint maps to each provider's knob
- **🖼️ Vision & video** — image and video attachments on vision-capable models
- **🎯 Embeddings & reranking** — Jina, Voyage, Google, OpenAI, Together, OctoHub, plus local FastEmbed and HuggingFace backends
- **🎬 Media generation** — typed image, video, speech, and transcription APIs with durable jobs and dimensional cost reporting
- **🧩 CLI proxies** — drive `codex`, `claude`, `gemini`, or `cursor-agent` as a provider via `cli:<backend>/<model>` (prompt-only)
- **🛡️ Production posture** — `Result` everywhere, exponential-backoff retries, cancellation tokens, and API keys from the environment only

## 📦 Quick Installation

```toml
# From crates.io (recommended) — use the latest version from the badge above
octolib = "<latest>"

# Or latest from git
octolib = { git = "https://github.com/muvon/octolib" }
```

Every capability is on by default. Pick only what you need to cut compile time and dependencies:

| Feature | Module | Pulls in |
|---|---|---|
| `llm` | `octolib::llm` — chat completion, tool calling, structured output | `jsonschema`, `jsonwebtoken` |
| `embeddings` | `octolib::embedding` | `tiktoken-rs` |
| `reranker` | `octolib::reranker` | — |
| `media` | `octolib::media` — image, video, speech, transcription | `base64` |
| `evaluation` | structured evaluation (TypeSafe Jev, Cloudflare AI Gateway) | — |
| `fastembed` | local embedding backend (implies `embeddings`) | `fastembed` |
| `huggingface` | local embedding backend (implies `embeddings`) | `candle`, `tokenizers`, `hf-hub` |

```toml
# Chat only — no embedding, reranking, or media stack compiled.
octolib = { version = "<latest>", default-features = false, features = ["llm"] }
```

`octolib::errors`, `octolib::storage`, `octolib::utils` and `set_user_agent` are always available. Hardware acceleration is opt-in via features: `metal`, `cuda`, `cudnn`, `mkl`, `accelerate`.

## 🚀 Quick Start

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message};

async fn example() -> anyhow::Result<()> {
    // One string picks both the provider and the model.
    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    let messages = vec![Message::user("Hello, how are you?")];

    // Arguments: messages, model, temperature, top_p, top_k, max_tokens
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    // Requires OPENAI_API_KEY in the environment
    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);

    // Token usage and USD cost are attached to every response:
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Input: {}, Output: {}, Cost: ${:.6}",
            usage.input_tokens, usage.output_tokens,
            usage.cost.unwrap_or(0.0),
        );
    }

    Ok(())
}
```

Switching providers is the same call with a different string — `anthropic:claude-opus-4`, `ollama:llama3.2` — after exporting the matching `*_API_KEY` (see the [key table in the docs](doc/04-advanced-guide.md)).

### Media generation

Media delivery is separate from chat input attachments. The high-level helpers accept the same `provider:model` addressing used elsewhere and wait for asynchronous jobs when necessary:

```rust
use octolib::{generate_image, ImageGenerationRequest};

async fn image_example() -> octolib::MediaResult<()> {
    // Requires OPENROUTER_API_KEY.
    let request = ImageGenerationRequest::new("A red panda astronaut, studio lighting");
    let result = generate_image(
        "openrouter:openai/gpt-image-1",
        request,
    ).await?;

    // OpenRouter reports authoritative request cost in the response.
    let cost_usd = result
        .usage
        .as_ref()
        .and_then(|usage| usage.provider_reported_cost);
    println!("{} image(s), cost: {:?}", result.artifacts.len(), cost_usd);
    Ok(())
}
```

- **Coverage** — OpenRouter, Replicate, and fal serve all four task traits; Runway serves image and video; ElevenLabs serves speech and transcription; Cloudflare Workers AI serves image, speech, and transcription via `/ai/run`
- **Synchronous vs. durable** — ElevenLabs and Cloudflare answer synchronously, so results are complete on submission; the rest return a credential-free `JobHandle` you can persist and resume after a restart, and a local `wait_timeout` returns `MediaError::WaitTimeout { handle }` without cancelling the remote job
- **Honest costs** — `provider_reported_cost` only when the upstream returns a dollar amount; otherwise a caller-supplied `CostEstimate` or the crate's reference rates, always stored as `estimated_cost`, with the rate frozen into the `JobHandle` at submit
- **Safe downloads** — generated URLs are never fetched automatically; `download_artifact` takes an explicit byte limit, accepts HTTPS only, rejects embedded credentials and local/private addresses, and does not follow redirects
- **No duplicate paid work** — only idempotent schema and polling queries are retried; generation POSTs are never replayed after an ambiguous transport failure

Provider-specific fields live under the provider's own namespace (`provider_options["replicate"].input`), with optional `field_map` mappings for portable fields. The `media_openrouter`, `media_replicate`, and `media_fal` examples show the full contract.

### Evaluation

Evaluation models answer typed questions about one state with calibrated probabilities instead of generated text. Ask several independent questions in one call and branch on the numbers in code:

```rust
use octolib::{evaluate, Answer, EvaluationRequest, Question};

async fn triage() -> octolib::EvaluationResult<()> {
    // Requires TYPESAFE_API_KEY; use "cloudflare:typesafe/jev" to bill AI Gateway credits instead.
    let request = EvaluationRequest::new("Help! My payouts have been failing for 3 days.")
        .with_question("is_urgent", Question::noul("Does this convey urgency?"))
        .with_question(
            "department",
            Question::choice(
                "Which team should handle this?",
                [("billing", "Payments, refunds"), ("technical", "Bugs, outages")],
            ),
        )
        .with_question(
            "frustration",
            Question::score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
        );
    let response = evaluate("typesafe:jev-latest", request).await?;
    if let Answer::Noul { noul } = response.answers["is_urgent"] {
        println!("urgent with p={noul:.2}, cost {:?}", response.usage.cost);
    }
    Ok(())
}
```

Jev bills input tokens only ($0.042 per 1M, output free) and has a 32k context, so trim the state to what the questions need. The response's `model` field reports the versioned model that answered.

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
let (provider, model) = ProviderFactory::get_provider_for_model("cli:codex/gpt-5.2-codex")?;
// or: "cli:claude/claude-sonnet-4-5"
// or: "cli:gemini/gemini-2.5-pro"
// or: "cli:cursor/auto"
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
- ✅ Cross-provider support (OpenAI, Anthropic, xAI, Google Vertex, Google Studio, Amazon, OpenRouter, MiniMax, Moonshot, Z.ai, Alibaba, Hetzner, OpenCode, OctoHub, and other OpenAI-compatible providers)
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

### 🔎 Document Reranking

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

## 📊 Provider Support Matrix

Capabilities marked "Per-model" are resolved per model from the provider's capability tables.

| Provider | Highlights | Structured Output | Vision | Tool Calls | Caching |
|----------|------------|-------------------|--------|------------|---------|
| **OpenAI** | Responses API, OAuth support | ✅ JSON + Schema | ✅ | ✅ | ✅ |
| **xAI** | Grok 4.5/4.3/4.20/Build, encrypted reasoning | ✅ JSON + Schema | ✅ | ✅ | ✅ |
| **Anthropic** | Claude models, thinking blocks, OAuth support | ❌ | ✅ | ✅ | ✅ |
| **OpenRouter** | Multi-provider proxy | ✅ JSON + Schema | ✅ | ✅ | ✅ |
| **Google Vertex** | Enterprise, service-account auth | ❌ | ✅ | ✅ | ❌ |
| **Google Studio** | Gemini API, API-key auth | ✅ JSON + Schema | ✅ | ✅ | ✅ |
| **Amazon Bedrock** | Cloud AI services | ❌ | ✅ | ✅ | ❌ |
| **DeepSeek** | Open-source models | ✅ JSON Mode | ❌ | ❌ | ✅ |
| **Moonshot (Kimi)** | K2/K3 series | ✅ JSON Mode | ✅ kimi-k2.5 | ✅ | ✅ |
| **MiniMax** | Anthropic-compatible API | ✅ JSON Mode | ❌ | ✅ | ✅ |
| **Z.ai** | GLM models | ✅ JSON Mode | ❌ | ✅ | ✅ |
| **BytePlus** | Seed models | ✅ JSON + Schema | Per-model | ❌ | ✅ |
| **Alibaba Model Studio** | Qwen + resold DeepSeek/GLM | ❌ | Per-model | ✅ | ✅ |
| **Groq** | Fast inference | ✅ JSON + Schema | Per-model | ❌ | ✅ Select models |
| **Cerebras** | Fast inference | ✅ JSON + Schema | ❌ | ❌ | ❌ |
| **NVIDIA NIM** | 100+ hosted models | ✅ JSON + Schema | Per-model | ✅ | ❌ |
| **Together** | Multi-provider proxy | Per-model | Per-model | ✅ | ✅ (auto) |
| **Fireworks** | Auto prefix-cache | ✅ JSON + Schema | Per-model | ✅ | ✅ (auto) |
| **Featherless** | Open-weight models, subscription billing | ✅ JSON + Schema | ❌ | ❌ | ❌ |
| **Hetzner** | Open-weight models, free while experimental | ✅ JSON + Schema | Per-model | ✅ | ❌ |
| **Inception Labs** | Mercury diffusion LLMs | ✅ JSON + Schema | ❌ | ✅ | ✅ |
| **Meta** | Muse Spark models, 1M-token context | ✅ JSON + Schema | ✅ | ✅ | ✅ |
| **Tinker** | Inkling family, sampler checkpoints | ❌ | ❌ | ✅ | ❌ |
| **OpenCode Zen** | Multi-provider proxy, pay-as-you-go | Per-model | Per-model | ✅ | ❌ |
| **OpenCode Go** | Multi-provider proxy, subscription billing | ✅ JSON + Schema | Per-model | ✅ | ✅ |
| **OctoHub** | Local AI serving, evaluation proxy | Per-model | Per-model | ✅ | ✅ |
| **Cloudflare Workers AI** | Edge AI, media, evaluation | ❌ | ❌ | ❌ | ❌ |
| **Local** | Ollama, LM Studio, LocalAI, Jan, vLLM | Per-model | Per-model | Per-model | ❌ |
| **Ollama** | Local LLM runner | Per-model | Per-model | Per-model | ❌ |
| **CLI Proxy** | Codex, Claude, Gemini, Cursor — prompt-only | ❌ | ❌ | ❌ | ❌ |

### Structured Output Details

- **JSON Mode**: Basic JSON object output
- **JSON Schema**: Full schema validation with strict mode
- **Provider Detection**: Use `provider.supports_structured_output(&model)` to check capability

### 🧠 Thinking/Reasoning Support

Octolib provides first-class support for models that produce thinking/reasoning content. Thinking is stored **separately** from the main response content, similar to how `tool_calls` are separate from content.

```rust
use octolib::{ProviderFactory, ChatCompletionParams, Message, ThinkingBlock};

async fn thinking_example() -> anyhow::Result<()> {
    // Thinking-capable models: MiniMax, OpenAI o-series, Moonshot (kimi-k2-thinking*, K3), Z.ai (GLM hybrid thinking), xAI
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
        println!("Input tokens: {}", usage.input_tokens);
        println!("Cache read tokens: {}", usage.cache_read_tokens);
        println!("Cache write tokens: {}", usage.cache_write_tokens);
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
| **xAI** | Responses API reasoning items | Summary extraction plus encrypted reasoning preservation across tool rounds |
| **OpenAI o-series** | `reasoning_content` field | o1, o3, o4 models |
| **OpenRouter** | `reasoning_details` | Gemini and other providers |
| **Moonshot (Kimi)** | `reasoning_content` field | kimi-k2-thinking models; K3 always reasons |
| **Z.ai** | reasoning_content field with legacy think-tag fallback | GLM hybrid thinking models (4.5/4.6/4.7/5.x) |

#### Token Tracking

Thinking tokens are tracked separately in `TokenUsage.reasoning_tokens`:

```rust
if let Some(usage) = &response.exchange.usage {
    println!("Total tokens: {}", usage.total_tokens);
    println!("  - Input: {}", usage.input_tokens);
    println!("  - Cache Read: {}", usage.cache_read_tokens);
    println!("  - Cache Write: {}", usage.cache_write_tokens);
    println!("  - Output: {}", usage.output_tokens);
    println!("  - Reasoning: {}", usage.reasoning_tokens);
}
```

## 📚 Complete Documentation

- **[Overview](doc/01-overview.md)** — library introduction and core concepts
- **[Installation Guide](doc/02-installation.md)** — setup and API keys
- **[Advanced Usage](doc/03-advanced-usage.md)** and **[Advanced Guide](doc/04-advanced-guide.md)** — advanced features and the full environment-variable table
- **[Embedding Guide](doc/05-embedding.md)** — embedding generation with multiple providers
- **[Reranking Guide](doc/06-reranking.md)** — document relevance scoring
- **[Tool Calling](doc/07-tool-calling.md)** — cross-provider tool calling
- **[Thinking/Reasoning](doc/08-thinking.md)** — reasoning model support
- **[Configuration Migration](doc/09-configuration-migration.md)** — versioned TOML upgrades and safe file persistence

Also:

- **[Examples](examples/)** — one runnable file per feature; every snippet in this README has a fuller version there
- **[CHANGELOG](CHANGELOG.md)** — release history
- **[API reference](https://docs.rs/octolib)** — generated rustdoc
- **[AGENTS.md](AGENTS.md)** — repository guide: project layout, conventions, and how to add a provider

## 🔒 Privacy & Security

- **Keys from the environment only** — read at call time, never accepted as function parameters
- **No panics in library code** — every fallible path returns `Result`; no `unwrap()`, `expect()`, or `panic!()` outside tests
- **No hidden output** — the library never prints; diagnostics go through `tracing` when you enable it
- **Local-first option** — Ollama, Local, and OctoHub run against your own infrastructure with no external calls

## 🤝 Support & Community

- **🐛 Issues**: [GitHub Issues](https://github.com/Muvon/octolib/issues)
- **📧 Email**: [opensource@muvon.io](mailto:opensource@muvon.io)
- **🏢 Company**: [Muvon Un Limited](https://muvon.io) (Hong Kong)

## ⚖️ License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

**© 2026 Muvon Un Limited (Hong Kong)** · Built with ❤️ by the [Muvon team](https://muvon.io)
