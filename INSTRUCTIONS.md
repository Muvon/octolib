# OCTOLIB DEVELOPMENT GUIDE

> **Octolib** - Self-sufficient AI provider library used by Octomind and other projects. Multi-provider support, embeddings, structured output, cost tracking.

## 🚀 QUICK START

```bash
# Setup
git clone https://github.com/muvon/octolib.git && cd octolib
export OPENAI_API_KEY="your_key"  # or ANTHROPIC_API_KEY, OPENROUTER_API_KEY

# Development cycle (ALWAYS in this order)
cargo check --message-format=short                              # 1. Fast check
cargo clippy --all-features --all-targets -- -D warnings        # 2. Fix warnings
cargo test                                                      # 3. Test
cargo run --example basic_chat                                  # 4. Try it
```

## 🎯 ARCHITECTURE (5-MINUTE READ)

**This is a library, not an application** - No panics, no println, always return Result.

### Core Concepts
```
User Code → ProviderFactory → AiProvider trait → Specific Provider → API
                                    ↓
                            ProviderResponse (unified)
```

**Key Design:**
- **Trait-based**: All providers implement `AiProvider` trait
- **Factory pattern**: `ProviderFactory::get_provider_for_model("openai:gpt-4o")`
- **Self-sufficient**: No external app dependencies
- **Cost tracking**: Automatic token usage and pricing
- **Unified types**: Same `Message`, `ProviderResponse` for all providers

### What Lives Where
```
src/llm/
├── traits.rs          → AiProvider trait (THE interface)
├── factory.rs         → Model parsing, provider selection
├── types.rs           → Message, ProviderResponse, TokenUsage
├── providers/         → OpenAI, Anthropic, OpenRouter, etc.
├── strategies.rs      → Provider-specific tool call handling
└── retry.rs           → Exponential backoff logic

src/embedding/
├── mod.rs             → Main API: generate_embeddings()
├── types.rs           → EmbeddingProvider trait
└── constants.rs       → Provider configs (Jina, Voyage, etc.)

src/errors.rs          → ProviderError, ConfigError, etc.
```

## 📍 TASK-BASED GUIDE

### "Add support for new AI provider"
1. **Create**: `src/llm/providers/newprovider.rs`
2. **Implement**: `AiProvider` trait (copy from `openai.rs` as template)
3. **Add pricing**: Const table at top of file
4. **Register**: Add to `src/llm/providers/mod.rs`
5. **Factory**: Add case in `src/llm/factory.rs` → `get_provider_for_model()`
6. **Export**: Add to `src/lib.rs` re-exports
7. **Test**: Create example in `examples/`

**Files to touch**: `providers/newprovider.rs`, `providers/mod.rs`, `factory.rs`, `lib.rs`

### "Fix provider API issue"
1. **Find provider**: `src/llm/providers/<provider>.rs`
2. **Check**: `chat_completion()` method - request/response parsing
3. **Debug**: Add `tracing::debug!()` (NOT println!)
4. **Test**: `cargo test <provider>_provider`

**Common issues**: Wrong API endpoint, missing headers, incorrect response parsing

### "Add structured output support"
1. **Provider file**: `src/llm/providers/<provider>.rs`
2. **Update**: `supports_structured_output()` → return true
3. **Modify**: `chat_completion()` → add `response_format` to request
4. **Parse**: Extract `structured_output` from response
5. **Test**: Use `examples/structured_output.rs`

**Check**: OpenAI/OpenRouter/DeepSeek for working examples

### "Add new embedding provider"
1. **Constants**: Add to `src/embedding/constants.rs` (models, URL, API key env)
2. **Function**: Add `generate_embeddings_<provider>()` in `src/embedding/mod.rs`
3. **Route**: Add case in `generate_embeddings()` match statement
4. **Test**: `cargo test embedding_<provider>`

**Files to touch**: `embedding/constants.rs`, `embedding/mod.rs`

### "Fix cost calculation"
1. **Provider file**: `src/llm/providers/<provider>.rs`
2. **Update**: `PRICING` const table at top
3. **Check**: `calculate_cost()` function logic
4. **Verify**: Token usage parsing in `chat_completion()`

**Pricing sources**: Provider's official pricing page

### "Add caching support"
1. **Provider file**: `src/llm/providers/<provider>.rs`
2. **Update**: `supports_caching()` → return true
3. **Modify**: `chat_completion()` → add cache headers/params
4. **Parse**: Extract cache token usage from response
5. **Cost**: Update `calculate_cost()` for cache pricing

**Check**: Anthropic provider for cache implementation example

## 🚫 CRITICAL RULES

### Library Code - NEVER DO
```rust
// ❌ Panic in library
panic!("Unknown provider");
.unwrap()  // in public functions
.expect()  // in public functions

// ✅ Return Result
Err(ProviderError::UnsupportedProvider(name.to_string()))

// ❌ Print to console
println!("Debug info");

// ✅ Use tracing
tracing::debug!("Making API request");

// ❌ Accept API keys as parameters
pub fn new(api_key: String) -> Self

// ✅ Read from environment
fn get_api_key(&self) -> Result<String> {
    std::env::var("OPENAI_API_KEY")
        .map_err(|_| ProviderError::MissingApiKey("OPENAI_API_KEY".to_string()).into())
}
```

### Development Workflow
```bash
# ✅ ALWAYS this order
cargo check --message-format=short                       # Fast
cargo clippy --all-features --all-targets -- -D warnings # Fix ALL warnings
cargo test                                               # Verify

# ❌ NEVER during development
cargo build --release  # Too slow
```

## 🐛 QUICK DEBUG

**Problem: Provider not working**
→ Check `src/llm/providers/<provider>.rs` → `chat_completion()` method
→ Verify API key: `echo $OPENAI_API_KEY`
→ Enable logs: `RUST_LOG=debug cargo test <provider>`

**Problem: Model format error**
→ Must be `provider:model` (e.g., `openai:gpt-4o`)
→ Check `src/llm/factory.rs` → `get_provider_for_model()`

**Problem: Structured output not working**
→ Only OpenAI, OpenRouter, DeepSeek support it
→ Check `provider.supports_structured_output(&model)`
→ See `examples/structured_output.rs`

**Problem: Cost calculation wrong**
→ Update `PRICING` const in provider file
→ Check provider's official pricing page

**Problem: Compilation error**
→ `cargo clean && cargo check --message-format=short`
→ Missing feature? `cargo build --features fastembed,huggingface`

## 📚 EXAMPLES AS TEMPLATES

**Adding new provider?** → Copy `src/llm/providers/openai.rs` structure
**Adding embeddings?** → Check `src/embedding/mod.rs` → Jina/Voyage functions
**Structured output?** → See `examples/structured_output.rs`
**Tool calls?** → Check `src/llm/strategies.rs` → AnthropicStrategy

## 🎯 COMMON PATTERNS

### Provider Implementation Template
```rust
// src/llm/providers/newprovider.rs
use crate::llm::traits::AiProvider;
use async_trait::async_trait;

const PRICING: &[(&str, f64, f64)] = &[
    ("model-name", 1.0, 2.0), // input, output per 1M tokens
];

#[derive(Debug, Clone, Default)]
pub struct NewProvider;

impl NewProvider {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl AiProvider for NewProvider {
    fn name(&self) -> &str { "newprovider" }
    
    fn supports_model(&self, model: &str) -> bool {
        model.starts_with("prefix-")
    }
    
    fn get_api_key(&self) -> Result<String> {
        std::env::var("NEWPROVIDER_API_KEY")
            .map_err(|_| ProviderError::MissingApiKey("NEWPROVIDER_API_KEY".to_string()).into())
    }
    
    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        // 1. Get API key
        // 2. Build request
        // 3. Make HTTP call
        // 4. Parse response
        // 5. Calculate cost
        // 6. Return ProviderResponse
    }
}
```

### Error Handling Pattern
```rust
// ✅ Always return Result with context
pub fn parse_model(s: &str) -> Result<(String, String)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(ProviderError::InvalidModelFormat {
            provided: s.to_string(),
            expected: "provider:model".to_string(),
        }.into());
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}
```

---

**Need more details?** Check README.md and doc/ folder.
**This guide is for**: Getting started fast and knowing where to look when you get a task.
