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

//! Unit coverage for the Responses API response parser: output-item
//! extraction, usage/cost normalization, rate-limit header plumbing, and
//! encrypted-reasoning metadata attachment.

use super::*;
use serde_json::json;
use std::collections::HashMap;

fn usage_json(input: u64, output: u64, total: u64) -> serde_json::Value {
	json!({"input_tokens": input, "output_tokens": output, "total_tokens": total})
}

#[test]
fn message_outputs_join_with_newlines_and_set_content() {
	let response = json!({
		"id": "resp_1",
		"output": [
			{"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
			{"type": "message", "content": [{"type": "output_text", "text": "world"}]}
		],
		"usage": usage_json(10, 5, 15)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert_eq!(parsed.content, "hello\nworld");
	assert!(parsed.tool_calls.is_none());
	assert!(parsed.thinking.is_none());
	assert_eq!(parsed.id, Some("resp_1".to_string()));
	assert!(parsed.finish_reason.is_none());
}

#[test]
fn non_text_content_items_are_ignored() {
	let response = json!({
		"output": [
			{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
		],
		"usage": usage_json(1, 1, 2)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert_eq!(parsed.content, "");
}

#[test]
fn unknown_output_types_are_skipped() {
	let response = json!({
		"output": [
			{"type": "web_search_call"},
			{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
		],
		"usage": usage_json(1, 1, 2)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert_eq!(parsed.content, "ok");
}

#[test]
fn function_call_arguments_parse_from_string_and_object_forms() {
	let response = json!({
		"output": [
			{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{\"path\":\"a.rs\"}"},
			{"type": "function_call", "call_id": "call_2", "name": "edit", "arguments": {"path": "b.rs"}}
		],
		"usage": usage_json(10, 5, 15)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let calls = parsed.tool_calls.expect("tool calls parsed");
	assert_eq!(calls.len(), 2, "parallel tool calls must all be kept");
	assert_eq!(calls[0].id, "call_1");
	assert_eq!(calls[0].name, "view");
	assert_eq!(calls[0].arguments, json!({"path": "a.rs"}));
	assert_eq!(calls[1].arguments, json!({"path": "b.rs"}));
}

#[test]
fn malformed_string_arguments_fall_back_to_empty_object() {
	let response = json!({
		"output": [
			{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "not-json{"}
		],
		"usage": usage_json(1, 1, 2)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let calls = parsed.tool_calls.expect("tool calls parsed");
	assert_eq!(calls[0].arguments, json!({}));
}

#[test]
fn function_call_missing_fields_is_dropped() {
	let response = json!({
		"output": [
			{"type": "function_call", "name": "view"}
		],
		"usage": usage_json(1, 1, 2)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert!(parsed.tool_calls.is_none());
}

#[test]
fn reasoning_output_builds_a_thinking_block_with_reasoning_tokens() {
	let response = json!({
		"output": [
			{"type": "reasoning", "content": [{"type": "output_text", "text": "thinking hard"}]},
			{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}
		],
		"usage": {
			"input_tokens": 10,
			"output_tokens": 20,
			"total_tokens": 30,
			"output_tokens_details": {"reasoning_tokens": 12}
		}
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let thinking = parsed.thinking.expect("thinking block present");
	assert_eq!(thinking.content, "thinking hard");
	assert_eq!(thinking.tokens, 12);

	let usage = parsed.exchange.usage.expect("usage present");
	// output_tokens excludes reasoning; reasoning rides in its own field.
	assert_eq!(usage.output_tokens, 8);
	assert_eq!(usage.reasoning_tokens, 12);
}

#[test]
fn usage_splits_output_and_reports_clean_input_without_cache_tokens() {
	let response = json!({
		"output": [],
		"usage": {
			"input_tokens": 100,
			"output_tokens": 40,
			"total_tokens": 140,
			"input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 20}
		}
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let usage = parsed.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 50, "clean input excludes cache read+write");
	assert_eq!(usage.cache_read_tokens, 30);
	assert_eq!(usage.cache_write_tokens, 20);
	assert_eq!(usage.total_tokens, 140);
	assert!(usage.cost.is_some(), "cache details must trigger cost calculation");
}

#[test]
fn plain_usage_costs_without_cache_and_keeps_full_input() {
	let response = json!({
		"output": [],
		"usage": usage_json(100, 40, 140)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let usage = parsed.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 100);
	assert_eq!(usage.cache_read_tokens, 0);
	assert_eq!(usage.cache_write_tokens, 0);
	assert!(usage.cost.is_some());
}

#[test]
fn missing_model_in_request_body_leaves_cost_unset() {
	let response = json!({
		"output": [],
		"usage": usage_json(10, 5, 15)
	});

	let parsed = parse_responses_api_response(json!({}), response, "openai", 0, HashMap::new()).unwrap();

	let usage = parsed.exchange.usage.expect("usage present");
	assert_eq!(usage.cost, None);
}

#[test]
fn rate_limit_headers_are_attached_to_the_exchange() {
	let response = json!({
		"output": [],
		"usage": usage_json(1, 1, 2)
	});
	let mut headers = HashMap::new();
	headers.insert("requests_limit".to_string(), "100".to_string());
	headers.insert("tokens_remaining".to_string(), "9999".to_string());

	let parsed = parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 42, headers).unwrap();

	assert_eq!(parsed.exchange.provider, "openai");
	let headers = parsed
		.exchange
		.rate_limit_headers
		.as_ref()
		.expect("rate limit headers present");
	assert_eq!(headers.get("requests_limit").map(String::as_str), Some("100"));
	assert_eq!(headers.get("tokens_remaining").map(String::as_str), Some("9999"));
	assert_eq!(parsed.exchange.usage.expect("usage").request_time_ms, Some(42));
}

#[test]
fn encrypted_reasoning_rides_on_tool_calls_in_the_exchange() {
	let response = json!({
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "enc-blob"},
			{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{}"}
		],
		"usage": usage_json(10, 5, 15)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert!(parsed.tool_calls.is_some());
	let stored = parsed.exchange.response["tool_calls"]
		.as_array()
		.expect("stored tool calls");
	assert!(
		stored[0]["meta"].get(REASONING_META_KEY).is_some(),
		"encrypted reasoning items must be attached for stateless replay"
	);
}

#[test]
fn reasoning_without_encrypted_content_attaches_no_meta() {
	let response = json!({
		"output": [
			{"type": "reasoning", "id": "rs_1", "summary": []},
			{"type": "function_call", "call_id": "call_1", "name": "view", "arguments": "{}"}
		],
		"usage": usage_json(10, 5, 15)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	let stored = parsed.exchange.response["tool_calls"]
		.as_array()
		.expect("stored tool calls");
	assert!(stored[0]["meta"].get(REASONING_META_KEY).is_none());
}

#[test]
fn structured_output_is_parsed_from_message_text() {
	let response = json!({
		"output": [
			{"type": "message", "content": [{"type": "output_text", "text": "{\"answer\": 42}"}]}
		],
		"usage": usage_json(1, 1, 2)
	});

	let parsed =
		parse_responses_api_response(json!({"model": "gpt-4o"}), response, "openai", 0, HashMap::new()).unwrap();

	assert_eq!(parsed.structured_output, Some(json!({"answer": 42})));
}
