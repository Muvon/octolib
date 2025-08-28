# Octolib: Advanced Usage

## Tool Calling

### Basic Tool Definition

```rust
let tools = vec![FunctionDefinition {
    name: "get_weather".to_string(),
    description: "Get current weather for a location".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "location": {
                "type": "string", 
                "description": "City name"
            }
        },
        "required": ["location"]
    }),
}];

let params = params.with_tools(tools);
```

### Handling Tool Calls

```rust
let response = provider.chat_completion(params).await?;

if let Some(tool_calls) = response.tool_calls {
    for tool_call in tool_calls {
        match tool_call.name.as_str() {
            "get_weather" => {
                let location = tool_call.arguments["location"].as_str().unwrap();
                let weather_result = fetch_weather(location);
                
                // Prepare tool result
                let tool_result = Message::tool(
                    &weather_result, 
                    &tool_call.id, 
                    "get_weather"
                );
            }
            _ => {}
        }
    }
}
```

## Vision Support

### Image Attachment

```rust
let image = ImageAttachment {
    data: ImageData::Base64(base64_image),
    media_type: "image/png".to_string(),
    source_type: SourceType::File(PathBuf::from("/path/to/image.png")),
    dimensions: Some((800, 600)),
    size_bytes: Some(1024),
};

let message = Message::user("Describe this image")
    .with_images(vec![image]);
```

## Error Handling

### Comprehensive Error Management

```rust
match provider.chat_completion(params).await {
    Ok(response) => {
        println!("Response: {}", response.content);
        
        // Token usage tracking
        if let Some(usage) = response.exchange.usage {
            println!("Prompt Tokens: {}", usage.prompt_tokens);
            println!("Output Tokens: {}", usage.output_tokens);
            println!("Total Cost: ${}", usage.cost.unwrap_or(0.0));
        }
    },
    Err(e) => match e {
        ProviderError::RateLimitExceeded { provider } => {
            // Implement backoff strategy
            tokio::time::sleep(Duration::from_secs(5)).await;
        },
        ProviderError::ApiKeyNotFound { provider } => {
            // Handle missing API key
            eprintln!("API key missing for {}", provider);
        },
        ProviderError::ModelNotSupported { provider, model } => {
            eprintln!("Model {} not supported by {}", model, provider);
        },
        _ => {
            eprintln!("Unexpected error: {}", e);
        }
    }
}
```

## Retry and Timeout Management

### Configurable Retry Strategy

```rust
let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
    .with_max_retries(5)  // Maximum retry attempts
    .with_retry_timeout(Duration::from_secs(2)); // Base timeout for exponential backoff

// Optional cancellation
let (tx, rx) = tokio::sync::watch::channel(false);
let params = params.with_cancellation_token(rx);

// Cancel operation if needed
tx.send(true)?;
```

## Caching

### Cache Configuration

```rust
let cache_config = CacheConfig::new(
    CacheTTL::Minutes(30),  // 30-minute cache
    CacheType::Ephemeral    // Temporary cache
);

// Provider-specific cache control
let function_def = FunctionDefinition {
    name: "cached_function".to_string(),
    description: "A function with caching".to_string(),
    parameters: serde_json::json!({}),
    cache_control: Some(cache_config.to_json()),
};
```

## Provider Selection

### Dynamic Provider Handling

```rust
// Get provider based on model
let (provider, model) = ProviderFactory::get_provider_for_model("anthropic:claude-3.5-sonnet")?;

// Check provider capabilities
if provider.supports_vision(&model) {
    // Use vision-enabled features
}

if provider.supports_caching(&model) {
    // Enable caching strategies
}
```

## Performance Considerations

- Use `clone()` sparingly
- Leverage async/await for non-blocking operations
- Configure appropriate timeouts
- Monitor token usage and costs

## Security Best Practices

- Use environment variables for API keys
- Implement proper error handling
- Validate and sanitize inputs
- Use the latest library version

## Debugging and Logging

```rust
// Capture full provider exchange for debugging
let response = provider.chat_completion(params).await?;
println!("Request: {}", response.exchange.request);
println!("Response: {}", response.exchange.response);
```

## Contribution

Discover an advanced use case? [Contribute to our documentation!](https://github.com/Muvon/octomind/issues)