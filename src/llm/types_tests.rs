// Copyright 2026 Muvon Un Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::*;

#[test]
fn test_message_constructors() {
    let user_msg = Message::user("Hello");
    assert_eq!(user_msg.role, "user");
    assert_eq!(user_msg.content, "Hello");
    assert!(!user_msg.cached);
    assert!(user_msg.tool_call_id.is_none());
    assert!(user_msg.images.is_none());

    let assistant_msg = Message::assistant("Hi there");
    assert_eq!(assistant_msg.role, "assistant");
    assert_eq!(assistant_msg.content, "Hi there");

    let system_msg = Message::system("You are helpful");
    assert_eq!(system_msg.role, "system");
    assert_eq!(system_msg.content, "You are helpful");

    let tool_msg = Message::tool("Result", "call_123", "test_tool");
    assert_eq!(tool_msg.role, "tool");
    assert_eq!(tool_msg.content, "Result");
    assert_eq!(tool_msg.tool_call_id, Some("call_123".to_string()));
    assert_eq!(tool_msg.name, Some("test_tool".to_string()));
}

#[test]
fn test_message_with_cache_marker() {
    let msg = Message::user("Test").with_cache_marker();
    assert!(msg.cached);
}

#[test]
fn test_chat_completion_params() {
    let messages = vec![Message::user("Hello")];
    let params = ChatCompletionParams::new(&messages, "openai:gpt-4o", 0.7, 1.0, 50, 1000);

    assert_eq!(params.model, "openai:gpt-4o");
    assert_eq!(params.temperature, 0.7);
    assert_eq!(params.top_p, 1.0);
    assert_eq!(params.top_k, 50);
    assert_eq!(params.max_tokens, 1000);
    assert_eq!(params.max_retries, 3); // Default
    assert!(params.cancellation_token.is_none());
    assert!(params.tools.is_none()); // Default

    let params_with_retries = params.with_max_retries(5);
    assert_eq!(params_with_retries.max_retries, 5);

    // Test with tools
    let tools = vec![FunctionDefinition {
        name: "test_function".to_string(),
        description: "A test function".to_string(),
        parameters: serde_json::json!({"type": "object", "properties": {}}),
        cache_control: None,
    }];
    let params_with_tools = params_with_retries.with_tools(tools.clone());
    assert!(params_with_tools.tools.is_some());
    assert_eq!(params_with_tools.tools.unwrap().len(), 1);
}

#[test]
fn test_token_usage() {
    let usage = TokenUsage {
        input_tokens: 100,
        cache_read_tokens: 50,
        cache_write_tokens: 25,
        output_tokens: 50,
        reasoning_tokens: 30,
        total_tokens: 255, // 100 + 50 + 25 + 50 + 30 (if provider includes reasoning)
        cost: Some(0.01),
        request_time_ms: Some(1500),
    };

    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.cache_read_tokens, 50);
    assert_eq!(usage.cache_write_tokens, 25);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.reasoning_tokens, 30);
    assert_eq!(usage.total_tokens, 255);
    assert_eq!(usage.cost, Some(0.01));
    assert_eq!(usage.request_time_ms, Some(1500));
}

#[test]
fn test_provider_exchange() {
    let request = serde_json::json!({"model": "test", "messages": []});
    let response = serde_json::json!({"choices": []});
    let usage = TokenUsage {
        input_tokens: 10,
        cache_read_tokens: 5,
        cache_write_tokens: 0,
        output_tokens: 5,
        reasoning_tokens: 3,
        total_tokens: 23,
        cost: None,
        request_time_ms: None,
    };

    let exchange = ProviderExchange::new(
        request.clone(),
        response.clone(),
        Some(usage.clone()),
        "test_provider",
    );

    assert_eq!(exchange.request, request);
    assert_eq!(exchange.response, response);
    assert_eq!(exchange.provider, "test_provider");
    assert!(exchange.usage.is_some());
    assert!(exchange.timestamp > 0);
}

#[test]
fn test_tool_call() {
    let tool_call = ToolCall {
        id: "call_123".to_string(),
        name: "test_function".to_string(),
        arguments: serde_json::json!({"param": "value"}),
    };

    assert_eq!(tool_call.id, "call_123");
    assert_eq!(tool_call.name, "test_function");
    assert_eq!(tool_call.arguments["param"], "value");
}

#[test]
fn test_function_definition() {
    let func_def = FunctionDefinition {
        name: "test_function".to_string(),
        description: "A test function for demonstration".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"}
            },
            "required": ["param1"]
        }),
        cache_control: None,
    };

    assert_eq!(func_def.name, "test_function");
    assert_eq!(func_def.description, "A test function for demonstration");
    assert_eq!(func_def.parameters["type"], "object");
    assert!(func_def.parameters["properties"]["param1"].is_object());
    assert!(func_def.cache_control.is_none());

    // Test with cache control
    let func_def_with_cache = FunctionDefinition {
        name: "cached_function".to_string(),
        description: "A cached function".to_string(),
        parameters: serde_json::json!({"type": "object"}),
        cache_control: Some(serde_json::json!({
            "type": "ephemeral",
            "ttl": "5m"
        })),
    };

    assert!(func_def_with_cache.cache_control.is_some());
    assert_eq!(
        func_def_with_cache.cache_control.unwrap()["type"],
        "ephemeral"
    );
}

#[test]
fn test_image_attachment() {
    let attachment = ImageAttachment {
        data: ImageData::Base64("base64data".to_string()),
        media_type: "image/png".to_string(),
        source_type: SourceType::File(std::path::PathBuf::from("/path/to/image.png")),
        dimensions: Some((800, 600)),
        size_bytes: Some(1024),
    };

    match &attachment.data {
        ImageData::Base64(data) => assert_eq!(data, "base64data"),
        _ => panic!("Expected Base64 data"),
    }

    assert_eq!(attachment.media_type, "image/png");
    assert_eq!(attachment.dimensions, Some((800, 600)));
    assert_eq!(attachment.size_bytes, Some(1024));

    match &attachment.source_type {
        SourceType::File(path) => {
            assert_eq!(path, &std::path::PathBuf::from("/path/to/image.png"))
        }
        _ => panic!("Expected File source type"),
    }
}

#[test]
fn test_thinking_block() {
    let thinking = ThinkingBlock::new("Let me solve this step by step...");
    assert_eq!(thinking.content, "Let me solve this step by step...");
    assert_eq!(thinking.tokens, 0);

    let thinking_with_tokens = ThinkingBlock::with_tokens("Reasoning...", 42);
    assert_eq!(thinking_with_tokens.content, "Reasoning...");
    assert_eq!(thinking_with_tokens.tokens, 42);
}

#[test]
fn test_message_with_thinking() {
    let thinking = ThinkingBlock::with_tokens("Let me solve this step by step...", 50);
    let msg = Message::assistant("The answer is 42.").with_thinking(thinking);

    assert!(msg.thinking.is_some());
    assert_eq!(
        msg.thinking.as_ref().unwrap().content,
        "Let me solve this step by step..."
    );
    assert_eq!(msg.thinking.as_ref().unwrap().tokens, 50);
    assert_eq!(msg.content, "The answer is 42.");
}

#[test]
fn test_message_builder_with_thinking() {
    let thinking = ThinkingBlock::new("First, I'll analyze the problem...");
    let msg = Message::builder()
        .role("assistant")
        .content("The answer is 42.")
        .thinking(thinking)
        .build()
        .unwrap();

    assert!(msg.thinking.is_some());
    assert_eq!(
        msg.thinking.unwrap().content,
        "First, I'll analyze the problem..."
    );
}

#[test]
fn test_sampling_support_all() {
    let sp = SamplingSupport::ALL;
    assert!(sp.temperature);
    assert!(sp.top_p);
    assert!(sp.top_k);
}

#[test]
fn test_sampling_support_none() {
    let sp = SamplingSupport::NONE;
    assert!(!sp.temperature);
    assert!(!sp.top_p);
    assert!(!sp.top_k);
}

#[test]
fn test_sampling_support_default() {
    let sp = SamplingSupport::default();
    assert!(sp.temperature);
    assert!(sp.top_p);
    assert!(sp.top_k);
}

#[test]
fn test_effective_sampling_params() {
    // All supported — user values pass through
    let sp = SamplingSupport::ALL.effective(0.3, 0.8, 10);
    assert_eq!(sp.temperature, Some(0.3));
    assert_eq!(sp.top_p, Some(0.8));
    assert_eq!(sp.top_k, Some(10));

    // None supported — user values are ignored
    let sp = SamplingSupport::NONE.effective(0.3, 0.8, 10);
    assert_eq!(sp.temperature, None);
    assert_eq!(sp.top_p, None);
    assert_eq!(sp.top_k, None);

    // Partial support — only supported params pass through
    let sp = SamplingSupport {
        temperature: true,
        top_p: false,
        top_k: true,
    }
    .effective(0.5, 0.9, 20);
    assert_eq!(sp.temperature, Some(0.5));
    assert_eq!(sp.top_p, None);
    assert_eq!(sp.top_k, Some(20));
}
