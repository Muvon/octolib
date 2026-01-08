# Thinking/Reasoning Content Guide

## Overview

Octolib provides first-class support for models that produce thinking/reasoning content. This guide explains the architecture, usage patterns, and provider-specific details.

## Architecture

### ThinkingBlock Structure

Thinking content is stored in a separate `ThinkingBlock` structure, parallel to how `tool_calls` are separate from content:

```rust
use octolib::ThinkingBlock;

/// Thinking/reasoning block from models that support extended reasoning
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThinkingBlock {
    /// The thinking/reasoning text content
    pub content: String,
    /// Token count for cost tracking (may not be available from all providers)
    pub tokens: u64,
}

impl ThinkingBlock {
    /// Create a new thinking block with the given content
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
            tokens: 0,
        }
    }

    /// Create a thinking block with token count
    pub fn with_tokens(content: &str, tokens: u64) -> Self {
        Self {
            content: content.to_string(),
            tokens,
        }
    }
}
```

### Message with Thinking

Messages can now contain thinking alongside content and tool_calls:

```rust
use octolib::{Message, ThinkingBlock};

let msg = Message::assistant("The answer is 42.")
    .with_thinking(ThinkingBlock::with_tokens(
        "First, I solved for x by subtracting 7...",
        150  // thinking tokens
    ));
```

### ProviderResponse with Thinking

Provider responses include thinking as a separate field:

```rust
let response = provider.chat_completion(params).await?;

// Thinking is separate from content
if let Some(ref thinking) = response.thinking {
    println!("Thinking ({} tokens): {}", thinking.tokens, thinking.content);
}

println!("Response: {}", response.content);
```

## Provider Support

### MiniMax

MiniMax uses Anthropic-compatible API with content blocks:

**Request/Response Format:**
```json
{
  "content": [
    {"type": "thinking", "thinking": "Let me analyze this problem..."},
    {"type": "text", "text": "The answer is 42."}
  ]
}
```

**Usage:**
```rust
let (provider, model) = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2")?;
let response = provider.chat_completion(params).await?;
```

### OpenAI o-series

OpenAI o1, o3, and o4 models use `reasoning_content` field:

**Response Format:**
```json
{
  "choices": [{
    "message": {
      "content": "The answer is 42.",
      "reasoning_content": "First, I need to..."
    }
  }],
  "usage": {
    "completion_tokens_details": {
      "reasoning_tokens": 150
    }
  }
}
```

**Usage:**
```rust
let (provider, model) = ProviderFactory::get_provider_for_model("openai:o1")?;
let response = provider.chat_completion(params).await?;
```

### OpenRouter

OpenRouter passes through reasoning from underlying providers:

**Response Format (Gemini):**
```json
{
  "choices": [{
    "message": {
      "content": "The answer is 42.",
      "reasoning_details": [
        {"type": "reasoning.text", "text": "Let me think..."}
      ]
    }
  }]
}
```

**Usage:**
```rust
let (provider, model) = ProviderFactory::get_provider_for_model("openrouter:google/gemini-pro")?;
let response = provider.chat_completion(params).await?;
```

## Token Usage Tracking

Thinking tokens are tracked separately in `TokenUsage`:

```rust
if let Some(usage) = &response.exchange.usage {
    println!("Total tokens: {}", usage.total_tokens);
    println!("  - Prompt: {}", usage.prompt_tokens);
    println!("  - Output: {}", usage.output_tokens);
    println!("  - Reasoning: {}", usage.reasoning_tokens);

    // Cost calculation includes reasoning tokens
    if let Some(cost) = usage.cost {
        println!("Cost: ${:.4}", cost);
    }
}
```

## Complete Example

```rust
use octolib::{
    llm::{ProviderFactory, ChatCompletionParams, Message},
    ThinkingBlock,
};

async fn thinking_example() -> anyhow::Result<()> {
    // Use MiniMax for thinking capability
    let (provider, model) = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2")?;

    // Create conversation
    let messages = vec![
        Message::user("Solve the equation 2x + 7 = -23 and show your thinking"),
    ];

    // Create parameters
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000);

    // Get completion
    let response = provider.chat_completion(params).await?;

    // Display thinking (for debugging/educational purposes)
    if let Some(ref thinking) = response.thinking {
        println!("=" .repeat(60));
        println!("MODEL THINKING ({} tokens)", thinking.tokens);
        println!("=" .repeat(60));
        println!("{}", thinking.content);
        println!("=" .repeat(60));
        println!();
    }

    // Display final response (clean, no thinking prefix)
    println!("FINAL ANSWER:");
    println!("{}", response.content);
    println!();

    // Show token breakdown
    if let Some(usage) = &response.exchange.usage {
        println!("TOKEN USAGE:");
        println!("  Input: {} tokens", usage.prompt_tokens);
        println!("  Output: {} tokens", usage.output_tokens);
        println!("  Reasoning: {} tokens", usage.reasoning_tokens);
        println!("  Total: {} tokens", usage.total_tokens);

        if let Some(cost) = usage.cost {
            println!("  Cost: ${:.4}", cost);
        }
    }

    // Store thinking in conversation history for round-trip
    let mut history = messages;
    history.push(Message::assistant(&response.content)
        .with_thinking(response.thinking.clone()));

    Ok(())
}
```

## Message Builder with Thinking

```rust
use octolib::{Message, MessageBuilder, ThinkingBlock};

let msg = Message::builder()
    .role("assistant")
    .content("The answer is 42.")
    .thinking(ThinkingBlock::new("Step by step reasoning..."))
    .build()?;

assert!(msg.thinking.is_some());
```

## Provider Capabilities

Use `AiProvider` trait methods to check thinking support:

```rust
let (provider, model) = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2")?;

// Check if model supports thinking (this is provider-specific)
let supports_thinking = model.starts_with("MiniMax") ||
                        model.starts_with("o1") ||
                        model.starts_with("o3") ||
                        model.starts_with("o4");

// Thinking is automatically extracted when available
let response = provider.chat_completion(params).await?;
let has_thinking = response.thinking.is_some();
```

## Best Practices

### 1. Display Thinking Separately

```rust
// User-facing: show only content
fn display_response(response: &ProviderResponse) {
    println!("{}", response.content);
}

// Debug/educational: show thinking too
fn display_with_thinking(response: &ProviderResponse) {
    if let Some(ref thinking) = response.thinking {
        println!("[Thinking: {} tokens]", thinking.tokens);
        println!("{}\n", thinking.content);
    }
    println!("Answer: {}", response.content);
}
```

### 2. Preserve Thinking in Conversation History

```rust
// Store thinking for round-trip compatibility
let mut conversation = Vec::new();

for message in &messages {
    conversation.push(message.clone());
}

// Add assistant response with thinking
conversation.push(Message::assistant(&response.content)
    .with_thinking(response.thinking.clone()));

// Use conversation in next request
let params = ChatCompletionParams::new(&conversation, &model, 0.7, 1.0, 50, 1000);
```

### 3. Token Budgeting

```rust
let max_thinking_tokens = 500;
let max_output_tokens = 500;
let total_budget = max_thinking_tokens + max_output_tokens;

if let Some(ref thinking) = response.thinking {
    if thinking.tokens > max_thinking_tokens {
        println!("Warning: Thinking exceeded budget by {} tokens",
            thinking.tokens - max_thinking_tokens);
    }
}
```

## Cost Considerations

Thinking tokens are often charged differently than output tokens:

| Provider | Input Price | Output Price | Thinking Price |
|----------|-------------|--------------|----------------|
| MiniMax | $0.3/1M | $1.2/1M | Same as output |
| OpenAI o-series | Varies | Varies | Often discounted |

Check provider pricing documentation for current rates.

## Troubleshooting

### Thinking Not Appearing

**Problem:** `response.thinking` is always `None`

**Solutions:**
1. Verify you're using a thinking-capable model (MiniMax-M2, OpenAI o1/o3/o4)
2. Check that thinking wasn't stripped by provider
3. Verify API response contains thinking content

```rust
// Debug: check raw response
let exchange = &response.exchange;
println!("Raw response: {:?}", exchange.response);
```

### Token Count Missing

**Problem:** `thinking.tokens` is always 0

**Solutions:**
1. Not all providers return thinking token counts
2. MiniMax: token count may not be available
3. OpenAI: check `completion_tokens_details.reasoning_tokens`

```rust
// Fallback: estimate from content length
let estimated_tokens = thinking.content.split_whitespace().count() / 4;
```

---

**Need help?** [Open an issue](https://github.com/Muvon/octolib/issues) or contact [opensource@muvon.io](mailto:opensource@muvon.io)
