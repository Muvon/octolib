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
use crate::llm::types::TokenUsage;
use serde_json::json;

#[test]
fn test_anthropic_tool_call_extraction() {
    let exchange = ProviderExchange::new(
        json!({}),
        json!({
            "tool_calls": [
                {
                    "id": "toolu_123",
                    "name": "test_tool",
                    "arguments": {"param": "value"}
                }
            ]
        }),
        Some(TokenUsage {
            input_tokens: 100,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
            output_tokens: 50,
            reasoning_tokens: 0,
            total_tokens: 150,
            cost: Some(0.01),
            request_time_ms: Some(1000),
        }),
        "anthropic",
    );

    let result = ProviderToolCalls::extract_from_exchange(&exchange).unwrap();
    assert!(result.is_some());

    let tool_calls = result.unwrap();
    assert_eq!(tool_calls.provider(), "anthropic");
    assert_eq!(tool_calls.len(), 1);

    // Validate
    tool_calls.validate().unwrap();

    // Convert to generic format
    let generic_calls = tool_calls.to_tool_calls().unwrap();
    assert_eq!(generic_calls.len(), 1);
    assert_eq!(generic_calls[0].name, "test_tool");
    assert_eq!(generic_calls[0].id, "toolu_123");
}

#[test]
fn test_openai_tool_call_extraction() {
    let exchange = ProviderExchange::new(
        json!({}),
        json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": "{\"param\": \"value\"}"
                        }
                    }]
                }
            }],
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "test_tool",
                    "arguments": {"param": "value"}
                }
            ]
        }),
        None,
        "openai",
    );

    let result = ProviderToolCalls::extract_from_exchange(&exchange).unwrap();
    assert!(result.is_some());

    let tool_calls = result.unwrap();
    assert_eq!(tool_calls.provider(), "openai");
    assert_eq!(tool_calls.len(), 1);

    // Validate
    tool_calls.validate().unwrap();

    // Convert to generic format
    let generic_calls = tool_calls.to_tool_calls().unwrap();
    assert_eq!(generic_calls.len(), 1);
    assert_eq!(generic_calls[0].name, "test_tool");
    assert_eq!(generic_calls[0].id, "call_123");
}

#[test]
fn test_invalid_tool_call_format() {
    let exchange = ProviderExchange::new(
        json!({}),
        json!({
            "tool_calls": [
                {
                    // Missing required id field
                    "name": "test_tool",
                    "arguments": {}
                }
            ]
        }),
        None,
        "anthropic",
    );

    let result = ProviderToolCalls::extract_from_exchange(&exchange).unwrap();
    // Should return None because deserialization fails with missing required fields
    assert!(result.is_none());
}

#[test]
fn test_validation_errors() {
    let tool_calls = ProviderToolCalls::Anthropic {
        content: vec![AnthropicToolUse {
            id: "".to_string(), // Empty ID should fail validation
            name: "test".to_string(),
            input: json!({}),
        }],
    };

    let result = tool_calls.validate();
    assert!(result.is_err());

    if let Err(ToolCallError::MissingField { field }) = result {
        assert_eq!(field, "id");
    } else {
        panic!("Expected MissingField error");
    }
}

#[test]
fn test_invalid_json_arguments() {
    let tool_calls = ProviderToolCalls::OpenAI {
        tool_calls: vec![OpenAIToolCall {
            id: "call_123".to_string(),
            call_type: "function".to_string(),
            function: OpenAIFunction {
                name: "test".to_string(),
                arguments: "invalid json".to_string(),
            },
        }],
    };

    let result = tool_calls.to_tool_calls();
    assert!(result.is_err());
}
