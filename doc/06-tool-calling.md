# Tool Calling Guide

## Overview

Octolib provides comprehensive tool calling support across multiple AI providers with automatic format conversion and metadata preservation.

## Architecture

### Two-Type System

Octolib uses two distinct types for tool calls, each serving a specific purpose:

```rust
// Runtime execution type - used during tool execution
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

// Storage/serialization type - used in conversation history
pub struct GenericToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    pub meta: Option<serde_json::Map<String, serde_json::Value>>,
}
```

**Why two types?**

1. **`ToolCall` (Runtime)**:
   - Clean, minimal structure
   - No provider-specific baggage
   - Fast and efficient for execution
   - Used when actually calling tools

2. **`GenericToolCall` (Storage)**:
   - Preserves provider-specific metadata
   - Used in conversation history
   - Serialized to JSON in messages
   - Supports round-trip through API calls

This separation follows the **Interface Segregation Principle** - keeping runtime code clean while storage code remains flexible.

## Conversion API

### Converting to Runtime Format

Use `to_tool_calls()` when you need to execute tools:

```rust
use octolib::ProviderToolCalls;

// Extract from provider response
let provider_calls = ProviderToolCalls::extract_from_exchange(&exchange)?;

// Convert to runtime format
let tool_calls = provider_calls.to_tool_calls()?;

// Execute tools
for tool_call in tool_calls {
    match tool_call.name.as_str() {
        "get_weather" => {
            let location = tool_call.arguments["location"].as_str().unwrap();
            let result = fetch_weather(location);
            // ...
        }
        _ => {}
    }
}
```

### Converting to Storage Format

Use `to_generic_tool_calls()` when storing in conversation history:

```rust
use octolib::{ProviderToolCalls, Message};

// Extract from provider response
let provider_calls = ProviderToolCalls::extract_from_exchange(&exchange)?;

// Convert to storage format
let generic_calls = provider_calls.to_generic_tool_calls();

// Store in message
let mut assistant_msg = Message::assistant(&response.content);
assistant_msg.tool_calls = Some(serde_json::to_value(&generic_calls)?);
messages.push(assistant_msg);
```

## Provider-Specific Metadata

### Gemini Thought Signatures

Gemini 3 models require thought signatures to be preserved during multi-turn function calling. Octolib handles this automatically:

**OpenRouter + Gemini:**
```rust
// Response from OpenRouter with Gemini
// reasoning_details are automatically extracted and stored in meta

let provider_calls = ProviderToolCalls::extract_from_exchange(&exchange)?;
let generic_calls = provider_calls.to_generic_tool_calls();

// generic_calls[0].meta contains:
// {
//   "reasoning_details": [
//     {"type": "reasoning.text", "text": "Let me think..."}
//   ]
// }

// When converting back to provider format, reasoning_details
// are automatically restored at the message level
```

**Direct Google API:**
```rust
// For direct Google Gemini API (future support)
// thought_signature would be stored in meta and restored
// to the function call part when sending back
```

### Metadata Structure

The `meta` field is a flexible JSON object that can store any provider-specific data:

```rust
pub struct GenericToolCall {
    pub meta: Option<serde_json::Map<String, serde_json::Value>>,
}

// Example metadata:
// {
//   "reasoning_details": [...],  // OpenRouter Gemini
//   "thought_signature": "...",  // Direct Google API
//   "custom_field": "..."        // Future providers
// }
```

## Complete Example

### Multi-Turn Conversation with Tool Calls

```rust
use octolib::{
    ProviderFactory, ChatCompletionParams, Message, 
    FunctionDefinition, ProviderToolCalls
};
use serde_json::json;

async fn tool_calling_example() -> anyhow::Result<()> {
    let (provider, model) = ProviderFactory::get_provider_for_model(
        "openrouter:google/gemini-3-pro-preview"
    )?;

    // Define tools
    let tools = vec![
        FunctionDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }),
            cache_control: None,
        },
    ];

    let mut messages = vec![
        Message::user("What's the weather in Tokyo?"),
    ];

    // First request
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_tools(tools.clone());

    let response = provider.chat_completion(params).await?;

    // Extract tool calls using new API
    if let Some(provider_calls) = ProviderToolCalls::extract_from_exchange(&response.exchange)? {
        // Convert to storage format (preserves metadata)
        let generic_calls = provider_calls.to_generic_tool_calls();
        
        // Add assistant message with tool calls
        let mut assistant_msg = Message::assistant(&response.content);
        assistant_msg.tool_calls = Some(serde_json::to_value(&generic_calls)?);
        messages.push(assistant_msg);

        // Convert to runtime format for execution
        let tool_calls = provider_calls.to_tool_calls()?;
        
        // Execute tools
        for tool_call in tool_calls {
            let result = match tool_call.name.as_str() {
                "get_weather" => {
                    json!({"temperature": 22, "condition": "sunny"})
                }
                _ => json!({"error": "Unknown tool"}),
            };

            // Add tool result
            messages.push(Message::tool(
                &serde_json::to_string(&result)?,
                &tool_call.id,
                &tool_call.name,
            ));
        }

        // Second request with tool results
        // Metadata (thought signatures) automatically preserved
        let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
            .with_tools(tools);

        let final_response = provider.chat_completion(params).await?;
        println!("Final response: {}", final_response.content);
    }

    Ok(())
}
```

## Provider Support

### Metadata Support Matrix

| Provider | Metadata Type | Automatic Handling |
|----------|--------------|-------------------|
| **OpenRouter + Gemini** | `reasoning_details` | ✅ Yes |
| **Direct Google API** | `thought_signature` | 🔜 Future |
| **OpenAI** | None | N/A |
| **Anthropic** | None | N/A |
| **DeepSeek** | None | N/A |

### Provider-Specific Notes

**OpenRouter with Gemini 3:**
- Automatically extracts `reasoning_details` from response
- Stores in `GenericToolCall.meta`
- Restores at message level when sending back
- Fixes 400 errors: "Function call is missing a thought_signature"

**Direct Google Gemini API:**
- Future support for `thought_signature` in function call parts
- Will use same `meta` field architecture
- Seamless integration with existing code

## Best Practices

### 1. Use Appropriate Type for Context

```rust
// ✅ GOOD: Use ToolCall for execution
let tool_calls = provider_calls.to_tool_calls()?;
for tool_call in tool_calls {
    execute_tool(&tool_call);
}

// ✅ GOOD: Use GenericToolCall for storage
let generic_calls = provider_calls.to_generic_tool_calls();
assistant_msg.tool_calls = Some(serde_json::to_value(&generic_calls)?);

// ❌ BAD: Don't mix types
let tool_calls = provider_calls.to_tool_calls()?;
assistant_msg.tool_calls = Some(serde_json::to_value(&tool_calls)?); // Loses metadata!
```

### 2. Always Preserve Metadata

```rust
// ✅ GOOD: Preserve metadata in conversation history
let generic_calls = provider_calls.to_generic_tool_calls();
// Metadata automatically preserved

// ❌ BAD: Manual conversion loses metadata
let manual_calls: Vec<GenericToolCall> = tool_calls
    .iter()
    .map(|tc| GenericToolCall {
        id: tc.id.clone(),
        name: tc.name.clone(),
        arguments: tc.arguments.clone(),
        meta: None, // Lost!
    })
    .collect();
```

### 3. Let Octolib Handle Provider Differences

```rust
// ✅ GOOD: Use octolib's conversion methods
let generic_calls = provider_calls.to_generic_tool_calls();
// Works for all providers, handles metadata automatically

// ❌ BAD: Manual provider-specific handling
match provider_name {
    "openrouter" => { /* extract reasoning_details */ }
    "google" => { /* extract thought_signature */ }
    _ => {}
}
// Duplicates logic, error-prone
```

## Troubleshooting

### Gemini 3 "Missing thought_signature" Error

**Error:**
```
400 Bad Request: Function call is missing a thought_signature in functionCall parts
```

**Solution:**
Use `to_generic_tool_calls()` to preserve metadata:

```rust
// ✅ Correct
let generic_calls = provider_calls.to_generic_tool_calls();
assistant_msg.tool_calls = Some(serde_json::to_value(&generic_calls)?);

// ❌ Wrong
let tool_calls = provider_calls.to_tool_calls()?;
assistant_msg.tool_calls = Some(serde_json::to_value(&tool_calls)?);
```

### Metadata Not Preserved

**Problem:** Metadata lost between turns

**Solution:** Always use `GenericToolCall` for storage:

```rust
// Extract from exchange
let provider_calls = ProviderToolCalls::extract_from_exchange(&exchange)?;

// Convert to storage format (preserves metadata)
let generic_calls = provider_calls.to_generic_tool_calls();

// Store in message
assistant_msg.tool_calls = Some(serde_json::to_value(&generic_calls)?);
```

## References

- [OpenRouter Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Google Gemini Thought Signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)
- [Octolib Advanced Usage](03-advanced-usage.md)

---

**Need help?** [Open an issue](https://github.com/Muvon/octolib/issues) or contact [opensource@muvon.io](mailto:opensource@muvon.io)
