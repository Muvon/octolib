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

fn exchange_with_tool_calls(provider: &str, tool_calls: serde_json::Value) -> ProviderExchange {
	ProviderExchange::new(json!({}), json!({ "tool_calls": tool_calls }), None, provider)
}

fn sample_calls() -> serde_json::Value {
	json!([
		{ "id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"} },
		{ "id": "call_2", "name": "get_time", "arguments": {} }
	])
}

#[test]
fn test_extract_openrouter_and_deepseek_variants() {
	for provider in ["openrouter", "deepseek"] {
		let exchange = exchange_with_tool_calls(provider, sample_calls());
		let calls = ProviderToolCalls::extract_from_exchange(&exchange).unwrap().unwrap();
		assert_eq!(calls.provider(), provider);
		assert_eq!(calls.len(), 2);
		calls.validate().unwrap();
		let tool_calls = calls.to_tool_calls().unwrap();
		assert_eq!(tool_calls[0].name, "get_weather");
		assert_eq!(tool_calls[0].arguments["city"], "Paris");
	}
}

#[test]
fn test_extract_fallback_for_unknown_provider() {
	let exchange = exchange_with_tool_calls("someproxy", sample_calls());
	let calls = ProviderToolCalls::extract_from_exchange(&exchange).unwrap().unwrap();
	assert_eq!(calls.provider(), "generic");
	assert_eq!(calls.len(), 2);
	assert!(matches!(calls, ProviderToolCalls::Generic { .. }));
}

#[test]
fn test_extract_returns_none_when_no_tool_calls() {
	for provider in ["anthropic", "openai", "openrouter", "deepseek", "anything"] {
		let exchange = ProviderExchange::new(json!({}), json!({ "content": "hi" }), None, provider);
		assert!(ProviderToolCalls::extract_from_exchange(&exchange).unwrap().is_none());
	}
}

#[test]
fn test_extract_returns_none_for_empty_tool_calls() {
	for provider in ["anthropic", "openai", "unknown"] {
		let exchange = exchange_with_tool_calls(provider, json!([]));
		assert!(ProviderToolCalls::extract_from_exchange(&exchange).unwrap().is_none());
	}
}

#[test]
fn test_to_tool_calls_empty_arguments_string_yields_empty_object() {
	let calls = ProviderToolCalls::OpenAI {
		tool_calls: vec![OpenAIToolCall {
			id: "call_1".to_string(),
			call_type: "function".to_string(),
			function: OpenAIFunction {
				name: "noop".to_string(),
				arguments: String::new(),
			},
		}],
	};
	let tool_calls = calls.to_tool_calls().unwrap();
	assert_eq!(tool_calls[0].arguments, json!({}));
}

#[test]
fn test_to_generic_tool_calls_openai_wraps_invalid_json() {
	let calls = ProviderToolCalls::OpenAI {
		tool_calls: vec![
			OpenAIToolCall {
				id: "call_1".to_string(),
				call_type: "function".to_string(),
				function: OpenAIFunction {
					name: "valid".to_string(),
					arguments: "{\"a\": 1}".to_string(),
				},
			},
			OpenAIToolCall {
				id: "call_2".to_string(),
				call_type: "function".to_string(),
				function: OpenAIFunction {
					name: "broken".to_string(),
					arguments: "not json".to_string(),
				},
			},
			OpenAIToolCall {
				id: "call_3".to_string(),
				call_type: "function".to_string(),
				function: OpenAIFunction {
					name: "empty".to_string(),
					arguments: "   ".to_string(),
				},
			},
		],
	};
	let generic = calls.to_generic_tool_calls();
	assert_eq!(generic.len(), 3);
	assert_eq!(generic[0].arguments, json!({"a": 1}));
	assert_eq!(generic[1].arguments, json!({"raw_arguments": "not json"}));
	assert_eq!(generic[2].arguments, json!({}));
	assert!(generic.iter().all(|c| c.meta.is_none()));
}

#[test]
fn test_to_generic_tool_calls_anthropic_and_generic_roundtrip() {
	let anthropic = ProviderToolCalls::Anthropic {
		content: vec![AnthropicToolUse {
			id: "toolu_1".to_string(),
			name: "tool".to_string(),
			input: json!({"x": 1}),
		}],
	};
	let generic = anthropic.to_generic_tool_calls();
	assert_eq!(generic[0].id, "toolu_1");
	assert_eq!(generic[0].arguments, json!({"x": 1}));

	let original = ProviderToolCalls::Generic {
		calls: vec![GenericToolCall {
			id: "g1".to_string(),
			name: "tool".to_string(),
			arguments: json!({"y": 2}),
			meta: Some(serde_json::Map::new()),
		}],
	};
	let cloned = original.to_generic_tool_calls();
	assert_eq!(cloned[0].id, "g1");
	assert_eq!(cloned[0].arguments, json!({"y": 2}));
}

#[test]
fn test_is_empty_and_len() {
	let empty = ProviderToolCalls::Generic { calls: vec![] };
	assert!(empty.is_empty());
	assert_eq!(empty.len(), 0);

	let two = ProviderToolCalls::Anthropic {
		content: vec![
			AnthropicToolUse {
				id: "a".to_string(),
				name: "n".to_string(),
				input: json!({}),
			},
			AnthropicToolUse {
				id: "b".to_string(),
				name: "n".to_string(),
				input: json!({}),
			},
		],
	};
	assert_eq!(two.len(), 2);
	assert!(!two.is_empty());
}

#[test]
fn test_validate_missing_fields_per_variant() {
	// OpenAI variant: empty function name
	let openai = ProviderToolCalls::OpenAI {
		tool_calls: vec![OpenAIToolCall {
			id: "call_1".to_string(),
			call_type: "function".to_string(),
			function: OpenAIFunction {
				name: String::new(),
				arguments: "{}".to_string(),
			},
		}],
	};
	match openai.validate() {
		Err(ToolCallError::MissingField { field }) => assert_eq!(field, "function.name"),
		other => panic!("expected MissingField, got {:?}", other),
	}

	// OpenAI variant: invalid JSON arguments
	let bad_json = ProviderToolCalls::OpenRouter {
		tool_calls: vec![OpenAIToolCall {
			id: "call_1".to_string(),
			call_type: "function".to_string(),
			function: OpenAIFunction {
				name: "n".to_string(),
				arguments: "{broken".to_string(),
			},
		}],
	};
	assert!(matches!(bad_json.validate(), Err(ToolCallError::InvalidArguments(_))));

	// Generic variant: empty name
	let generic = ProviderToolCalls::Generic {
		calls: vec![GenericToolCall {
			id: "g1".to_string(),
			name: String::new(),
			arguments: json!({}),
			meta: None,
		}],
	};
	match generic.validate() {
		Err(ToolCallError::MissingField { field }) => assert_eq!(field, "name"),
		other => panic!("expected MissingField, got {:?}", other),
	}
}

#[test]
fn test_validate_accepts_valid_calls_all_variants() {
	let valid = [
		ProviderToolCalls::Anthropic {
			content: vec![AnthropicToolUse {
				id: "a".to_string(),
				name: "n".to_string(),
				input: json!({}),
			}],
		},
		ProviderToolCalls::OpenAI {
			tool_calls: vec![OpenAIToolCall {
				id: "c".to_string(),
				call_type: "function".to_string(),
				function: OpenAIFunction {
					name: "n".to_string(),
					arguments: "{\"a\": 1}".to_string(),
				},
			}],
		},
		ProviderToolCalls::DeepSeek {
			tool_calls: vec![OpenAIToolCall {
				id: "c".to_string(),
				call_type: "function".to_string(),
				function: OpenAIFunction {
					name: "n".to_string(),
					arguments: String::new(),
				},
			}],
		},
	];
	for calls in &valid {
		calls.validate().unwrap();
	}
}

#[test]
fn test_serde_roundtrip_provider_tagged() {
	let calls = ProviderToolCalls::OpenRouter {
		tool_calls: vec![OpenAIToolCall {
			id: "call_1".to_string(),
			call_type: "function".to_string(),
			function: OpenAIFunction {
				name: "tool".to_string(),
				arguments: "{\"k\": \"v\"}".to_string(),
			},
		}],
	};
	let serialized = serde_json::to_string(&calls).unwrap();
	assert!(serialized.contains("\"provider\":\"openrouter\""));
	let deserialized: ProviderToolCalls = serde_json::from_str(&serialized).unwrap();
	assert_eq!(deserialized.provider(), "openrouter");
	assert_eq!(deserialized.len(), 1);
}

#[test]
fn test_generic_tool_call_meta_serialization_skip() {
	let call = GenericToolCall {
		id: "g1".to_string(),
		name: "tool".to_string(),
		arguments: json!({}),
		meta: None,
	};
	let serialized = serde_json::to_string(&call).unwrap();
	assert!(!serialized.contains("meta"));
}

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
