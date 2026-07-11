#! Octolib — AI Provider Library Guide

Rust library providing a unified interface to multiple AI providers (OpenAI, Anthropic, OpenRouter, NVIDIA, Ollama, Google, Amazon, Cloudflare, DeepSeek, Moonshot, Z.ai, BytePlus, Groq, Cerebras, Together, Featherless, Fireworks, OctoHub, CLI proxies). Handles chat completions, embeddings, reranking, cost tracking, caching, structured output, vision, and tool calls. No panics, no `println!`, always `Result`. Copyright year is **2026**.

## Project Structure

```
src/
├── lib.rs                          → Public re-exports (all public types live here)
├── errors.rs                       → ProviderError, ConfigError, MessageError, ToolCallError + ErrorContext trait
├── storage.rs                      → Cache dirs for FastEmbed/HuggingFace models
├── llm/
│   ├── traits.rs                   → AiProvider trait — THE interface every provider implements
│   ├── types.rs                    → Message, ChatCompletionParams, ProviderResponse, TokenUsage, ModelPricing, ReasoningEffort, etc.
│   ├── factory.rs                  → ProviderFactory::get_provider_for_model("provider:model")
│   ├── config.rs                   → CacheConfig, CacheTTL, CacheType
│   ├── strategies.rs               → ProviderStrategy trait + AnthropicStrategy/OpenAIStrategy (tool call formatting)
│   ├── tool_calls.rs               → GenericToolCall unified format
│   ├── retry.rs                    → Exponential backoff logic
│   ├── utils.rs                    → normalize_model_name, sanitize_model_name, is_model_in_pricing_table, calculate_cost_from_pricing_table
│   ├── reference_pricing.rs        → Baseline pricing for open/proxy models (substring-matched, specific-first)
│   ├── reference_capabilities.rs   → Baseline capabilities (vision, video, structured_output, context window)
│   └── providers/
│       ├── openai_compat.rs        → Shared OpenAI-compatible request/response layer; auto-passes reasoning_effort; also OpenAiCompatConfig, get_optional_api_key, get_api_url (internal, not pub)
│       ├── shared.rs               → HTTP client (arc-swap pool), cache control helpers, tool call parsers
│       ├── openai.rs               → Native — PRICING table, Responses API, OAuth support
│       ├── anthropic.rs            → Native — PRICING table, caching, thinking blocks, OAuth support
│       ├── google.rs               → Native — Vertex AI
│       ├── amazon.rs               → Native — Bedrock
│       ├── deepseek.rs             → Native — PRICING table
│       ├── moonshot.rs             → Native — PRICING table
│       ├── minimax.rs              → Native — PRICING table
│       ├── zai.rs                  → Native — PRICING table
│       ├── byteplus.rs             → Native — PRICING table (ByteDance Seed models)
│       ├── groq.rs                 → Native — PRICING table
│       ├── nvidia.rs               → Proxy — delegates to openai_compat, reference pricing
│       ├── cerebras.rs             → Proxy — delegates to openai_compat, reference pricing
│       ├── together.rs             → Proxy — delegates to openai_compat, reference pricing
│       ├── ollama.rs               → Proxy — delegates to openai_compat, reference pricing
│       ├── local.rs                → Proxy — delegates to openai_compat, reference pricing
│       ├── cloudflare.rs           → Proxy — delegates to openai_compat
│       ├── openrouter.rs           → Proxy — delegates to openai_compat
│       ├── octohub.rs              → Proxy — delegates to openai_compat
│       ├── featherless.rs          → Proxy — delegates to openai_compat
│       ├── fireworks.rs            → Proxy — delegates to openai_compat (auto prefix-cache)
│       └── cli/                    → CLI proxy: claude, codex, cursor, gemini, generic backends
├── embedding/
│   ├── mod.rs                      → generate_embeddings(), generate_embeddings_batch(), count_tokens(), truncate_output()
│   ├── types.rs                    → EmbeddingProviderType, EmbeddingConfig, InputType, parse_provider_model()
│   ├── constants.rs                → Token limits, batch sizes
│   └── provider/                   → Jina, Voyage, Google, OpenAI, OpenRouter, Together, OctoHub, FastEmbed, HuggingFace
└── reranker/
    ├── mod.rs                      → rerank(), rerank_with_truncation()
    ├── types.rs                    → RerankProviderType, RerankResult, RerankResponse, parse_provider_model()
    └── provider/                   → Voyage, Cohere, Jina, Mixedbread, FastEmbed, HuggingFace

examples/                           → One file per feature — use as integration test templates
```

## Where to Look

| Task | Start here |
|------|------------|
| Add LLM provider (native API) | Copy `src/llm/providers/openai.rs` or `anthropic.rs` |
| Add LLM provider (OpenAI-compat proxy) | Copy `src/llm/providers/nvidia.rs` or `ollama.rs` |
| Add embedding provider | Copy `src/embedding/provider/jina.rs`, register in `provider/mod.rs` |
| Add reranker provider | Copy `src/reranker/provider/voyage.rs`, register in `provider/mod.rs` |
| Fix provider API / response parsing | `src/llm/providers/<provider>.rs` → `chat_completion()` |
| Fix cost calculation (native) | `PRICING` const in provider file |
| Fix cost calculation (proxy) | `src/llm/reference_pricing.rs` — check pattern exists and is specific-first |
| Fix capabilities (vision/context/structured) | `src/llm/reference_capabilities.rs` for proxies; override trait methods for native |
| Add structured output to a provider | `supports_structured_output()` + `response_format` in request + parse `structured_output` |
| Add caching to a provider | `supports_caching()` + cache headers in request + parse cache token fields |
| Configure reasoning / thinking effort | `src/llm/types.rs` → `ReasoningEffort` enum; per-provider mapping inside each `chat_completion()` |
| Tool call handling | `src/llm/strategies.rs` → `AnthropicStrategy` / `OpenAIStrategy` |
| Shared HTTP / cache helpers | `src/llm/providers/shared.rs` |
| Model format parsing | `src/llm/factory.rs` → `ProviderFactory::parse_model()` |
| Public API surface | `src/lib.rs` re-exports |
| Model name normalization | `src/llm/utils.rs` → `normalize_model_name`, `sanitize_model_name` |

## How Things Work

### Provider Shape: Native vs. Proxy

**Native** (OpenAI, Anthropic, Google, Amazon, DeepSeek, Moonshot, MiniMax, Z.ai, BytePlus, Groq):
- Own `PRICING` const table in the provider file — `(model, input, output, cache_write, cache_read)` per 1M tokens
- Own `chat_completion()` implementation with provider-specific request/response structs
- Override `supports_caching()`, `supports_vision()`, `get_max_input_tokens()`, `supports_structured_output()` directly
- `get_model_pricing()` reads from the local `PRICING` table via `get_model_pricing()` from `utils.rs`
- `supports_model()` uses `is_model_in_pricing_table()` — strict, unknown models rejected

**Proxy / OpenAI-compatible** (NVIDIA, Cerebras, Together, Ollama, Local, Cloudflare, OpenRouter, OctoHub, Featherless):
- Delegates to `openai_compat_chat_completion(OpenAiCompatConfig { provider_name, usage_fallback_cost, use_response_cost }, api_key, api_url, params)`
- `OpenAiCompatConfig.use_response_cost = true` → use cost from API response if present; `usage_fallback_cost` → fixed fallback cost (rarely used)
- `supports_model()` returns `!model.is_empty()` — accepts anything
- `get_model_pricing()` → `reference_pricing::get_reference_pricing(model)` (trait default already does this; explicit override is optional)
- After the call: if `usage.cost.is_none()`, fill it via `reference_pricing::calculate_reference_cost(&model, input_tokens, output_tokens)`
- Capabilities (vision, context, structured output) resolved via `reference_capabilities` trait defaults

### AiProvider Trait — Required vs. Defaulted

```rust
// MUST implement:
fn name(&self) -> &str
fn supports_model(&self, model: &str) -> bool
fn get_api_key(&self) -> Result<String>
async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse>

// Override for native providers (defaults use reference tables):
fn supports_caching(&self, model: &str) -> bool          // default: false
fn supports_vision(&self, model: &str) -> bool            // default: reference_capabilities lookup
fn supports_video(&self, model: &str) -> bool             // default: reference_capabilities lookup
fn supports_structured_output(&self, model: &str) -> bool // default: reference_capabilities lookup
fn get_max_input_tokens(&self, model: &str) -> usize      // default: reference_capabilities lookup (8192 fallback)
fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> // default: reference_pricing lookup
fn supported_sampling_params(&self, model: &str) -> SamplingSupport // default: ALL
```

### Reference Tables

`reference_pricing.rs` and `reference_capabilities.rs` — substring-matched against `sanitize_model_name(normalize_model_name(model))`:
- **More specific patterns must come before less specific ones** (e.g., `"gpt-5.1-codex-mini"` before `"gpt-5.1-codex"` before `"gpt-5"`)
- `sanitize_model_name` converts dots to dashes and strips colons/tags: `qwen2.5:7b` → `qwen-2-5-7b`
- Used by all proxy providers and as fallback for native providers that don't override trait methods

### Model Format

Always `provider:model` — e.g., `openai:gpt-4o`, `anthropic:claude-opus-4`, `ollama:llama3.2`.

CLI provider uses `cli:backend/model` — e.g., `cli:codex/gpt-5.2-codex`, `cli:claude/claude-opus-4`.

`ProviderFactory::get_provider_for_model("provider:model")` → `(Box<dyn AiProvider>, String)`.

### Pricing Table Format

```rust
// PricingTuple = (&str, f64, f64, f64, f64)
// (model_pattern, input_per_1m, output_per_1m, cache_write_per_1m, cache_read_per_1m)
const PRICING: &[PricingTuple] = &[
    ("claude-opus-4-7", 5.00, 25.00, 6.25, 0.50),
    // For models without cache support: set cache_write = input, cache_read = input
    ("some-model", 1.00, 2.00, 1.00, 1.00),
];
```

### TokenUsage Fields

`input_tokens` = clean input only (never includes cache tokens). Separate fields: `cache_write_tokens`, `cache_read_tokens`, `output_tokens`, `reasoning_tokens`, `cost: Option<f64>`.

### Reasoning Effort

`ReasoningEffort` enum (`Low`, `Medium`, `High`, `XHigh`, `Max`) — provider-agnostic. Set via `ChatCompletionParams::with_reasoning_effort(effort)`. `None` (default) = provider default behavior.

Per-provider mapping:
- **Anthropic** → `thinking` block with `budget_tokens` (Low: 2048, Medium: 8192, High: 16384, XHigh: 32768, Max: 65536); clamped below `max_tokens`
- **OpenAI** → `reasoning.effort` string (`"low"` / `"medium"` / `"high"` / `"xhigh"`; Max maps to `"xhigh"`)
- **openai_compat** (NVIDIA, Cerebras, Groq, etc.) → passthrough `reasoning_effort` string; unsupported models ignore it
- **OpenRouter / Together** → passthrough `reasoning_effort` string
- **OctoHub** → passthrough string (supports all five levels including `"max"`)
- **Z.ai** → binary `thinking: { "type": "enabled" }` for any non-None effort (no budget knob)
- **Codex CLI** → `model_reasoning_effort` config (accepts `low` / `medium` / `high` only)

### Sampling Parameters

Use `SamplingSupport` constants to declare what a model accepts:
- `SamplingSupport::ALL` — temperature + top_p + top_k
- `SamplingSupport::TEMPERATURE_AND_TOP_P` — no top_k (most OpenAI-compat APIs)
- `SamplingSupport::TEMPERATURE_ONLY`
- `SamplingSupport::NONE` — reasoning models (o1, o3, claude-opus-4-7, etc.)

### Error Handling

```rust
// ✅ Return typed errors
Err(ProviderError::UnsupportedProvider(name.to_string()).into())
Err(ProviderError::MissingApiKey("OPENAI_API_KEY".to_string()).into())

// ✅ Use ErrorContext trait for context
some_result.with_provider_context("openai")?
some_result.with_context("parsing response")?

// ✅ Use helpers
api_error("anthropic", 400, "Bad Request")
config_error("Invalid endpoint URL")

// ❌ Never
.unwrap()  .expect()  panic!()   // in any non-test code
```

### Logging

```rust
// ✅ Always tracing
tracing::debug!("Making API request to {}", url);
tracing::warn!("Rate limit hit, retrying");

// ❌ Never
println!()  eprintln!()
```

### API Keys

```rust
// ✅ Always from environment
fn get_api_key(&self) -> Result<String> {
    std::env::var("PROVIDER_API_KEY")
        .map_err(|_| ProviderError::MissingApiKey("PROVIDER_API_KEY".to_string()).into())
}

// For optional keys (Ollama, local):
fn get_api_key(&self) -> Result<String> {
    Ok(get_optional_api_key("OLLAMA_API_KEY"))  // returns "" if unset
}
```

### Copyright Header

Every `.rs` file must start with:
```rust
// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// ...
```
Files still showing `Copyright 2025` (`src/errors.rs`, `src/storage.rs`, `src/llm/strategies.rs`, `src/llm/config.rs`, `src/embedding/constants.rs`, `src/embedding/mod_test.rs`, `examples/structured_output.rs`, `examples/test_structured_output.rs`) must be updated when touched.

## Adding a New LLM Provider

### Native API Provider

1. Create `src/llm/providers/newprovider.rs` — copy `openai.rs` or `anthropic.rs` as template
2. Add `PRICING` const table (specific patterns first); verify prices from provider's official pricing page
3. Implement `AiProvider` — override all capability methods explicitly (don't rely on reference tables for native providers)
4. Register: `pub mod newprovider;` + `pub use newprovider::NewProvider;` in `providers/mod.rs`
5. Add match arm in `ProviderFactory::create_provider()` in `factory.rs`
6. Add to `ProviderFactory::supported_providers()` list in `factory.rs`
7. Re-export from `src/lib.rs`
8. Add example in `examples/`

### OpenAI-Compatible Proxy

1. Create `src/llm/providers/newprovider.rs` — copy `nvidia.rs` or `ollama.rs` as template
2. `supports_model()` → `!model.is_empty()`
3. `get_model_pricing()` → `reference_pricing::get_reference_pricing(model)` (or omit — trait default does this)
4. In `chat_completion()`: call `openai_compat_chat_completion(OpenAiCompatConfig { provider_name: "newprovider", usage_fallback_cost: None, use_response_cost: true }, api_key, api_url, params).await?`, then if `response.exchange.usage.cost.is_none()` fill it via `calculate_reference_cost(&params.model, input_tokens, output_tokens)`
5. Add missing model families to `reference_pricing.rs` and `reference_capabilities.rs` if needed
6. Register in `providers/mod.rs`, `factory.rs`, `lib.rs`; add example

### Embedding Provider

1. Create `src/embedding/provider/newprovider.rs` — implement `EmbeddingProvider` trait (`generate_embedding`, `generate_embeddings_batch`, `get_dimension`)
2. Add variant to `EmbeddingProviderType` enum in `embedding/types.rs`
3. Add `from_str` match arm in `EmbeddingProviderType` impl
4. Register module + re-export in `embedding/provider/mod.rs`
5. Add match arm in `create_embedding_provider_from_parts()` in `embedding/provider/mod.rs`
6. If feature-gated: wrap in `#[cfg(feature = "...")]`

### Reranker Provider

1. Create `src/reranker/provider/newprovider.rs` — implement `RerankProvider` trait
2. Add variant to `RerankProviderType` in `reranker/types.rs`
3. Register in `reranker/provider/mod.rs` + `create_rerank_provider_from_parts()`

## Gotchas

- **Model name matching uses `sanitize_model_name` + `normalize_model_name`** before substring matching in reference tables. `qwen2.5:7b` becomes `qwen-2-5-7b`. Patterns in reference tables must match the sanitized form.
- **Reference table order is critical** — first match wins. Put `"gpt-5.1-codex-mini"` before `"gpt-5.1-codex"` before `"gpt-5"`.
- **`input_tokens` in `TokenUsage` is always clean** — never add cache tokens to it. Cache tokens go in dedicated fields.
- **CLI provider model format** is `backend/model` (slash, not colon) — `cli:codex/gpt-5.2-codex`. No tool calls, no structured output, prompt-only.
- **OpenAI uses the Responses API** (`/v1/responses`), not `/v1/chat/completions`. The `openai_compat` layer uses `/v1/chat/completions`.
- **`fastembed` and `huggingface` are default features** — code gated on them must use `#[cfg(feature = "...")]`.
- **Shared HTTP client** in `providers/shared.rs` is process-wide and arc-swapped on connection errors — don't create per-request clients.
- **`supports_model()` for native providers** should use `is_model_in_pricing_table()` — this enforces that only known/priced models are accepted.
- **ReasoningEffort is a hint, not a guarantee** — models without thinking support silently ignore it. Anthropic clamps budget_tokens below max_tokens.

## Never

- `unwrap()` or `expect()` outside `#[cfg(test)]` or `LazyLock`/`static` initializers
- `println!()` / `eprintln!()` anywhere in library code — use `tracing`
- Accept API keys as function parameters — always read from environment variables
- Add a model to a native provider's `PRICING` table without verifying the price from the provider's official pricing page
- Put a less-specific pattern before a more-specific one in `REFERENCE_PRICING` or `REFERENCE_CAPABILITIES`
