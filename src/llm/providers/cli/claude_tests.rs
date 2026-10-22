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
use crate::llm::providers::cli::CliProvider;
use crate::llm::types::Message;
use std::path::PathBuf;

fn provider() -> CliProvider {
	CliProvider {
		backend: crate::llm::providers::cli::CliBackend::from_name("claude"),
		command: PathBuf::from("claude"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	}
}

#[test]
fn test_claude_messages_to_json_skips_system_and_empty() {
	let messages = vec![
		Message::system("sys"),
		Message::user("  hello  "),
		Message::assistant(""),
		Message::user("again"),
	];
	let value = messages_to_claude_json(&messages);
	let arr = value.as_array().unwrap();
	assert_eq!(arr.len(), 2);
	assert_eq!(arr[0]["role"], "user");
	assert_eq!(arr[0]["content"][0]["text"], "hello");
	assert_eq!(arr[1]["content"][0]["text"], "again");
}

#[test]
fn test_claude_build_args_contains_prompt_and_model() {
	let messages = vec![Message::user("hi")];
	let json = messages_to_claude_json(&messages);
	let args = build_args(&provider(), "claude-opus-4", "be brief", &json);
	assert_eq!(args[0], "-p");
	assert_eq!(args[2], "--system-prompt");
	assert_eq!(args[3], "be brief");
	assert!(args.contains(&"--model".to_string()));
	assert!(args.contains(&"claude-opus-4".to_string()));
	assert!(args.contains(&"json".to_string()));
	assert!(args.contains(&"--verbose".to_string()));
}

#[test]
fn test_claude_build_args_empty_model_omits_model_flag() {
	let args = build_args(&provider(), "", "sys", &serde_json::json!([]));
	assert!(!args.contains(&"--model".to_string()));
}

#[test]
fn test_claude_parse_response_assistant_and_result_usage() {
	let body = r#"[
		{"type":"assistant","message":{"content":[{"type":"text","text":"answer"}],"usage":{"input_tokens":11,"output_tokens":6,"total_tokens":17}}},
		{"type":"result","usage":{"input_tokens":999}}
	]"#;
	// the parser joins lines then parses one JSON document
	let (text, usage) = parse_response(&[body.to_string()]).unwrap();
	assert_eq!(text, "answer");
	let usage = usage.unwrap();
	assert_eq!(usage.input_tokens, 11);
	assert_eq!(usage.output_tokens, 6);
	assert_eq!(usage.total_tokens, 17);
}

#[test]
fn test_claude_parse_response_usage_total_defaults_to_sum() {
	let body = r#"[{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}},{"type":"result","usage":{"input_tokens":5,"output_tokens":2}}]"#;
	let (_, usage) = parse_response(&[body.to_string()]).unwrap();
	assert_eq!(usage.unwrap().total_tokens, 7);
}

#[test]
fn test_claude_parse_response_no_usage_yields_none() {
	let body = r#"[{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}]"#;
	let (text, usage) = parse_response(&[body.to_string()]).unwrap();
	assert_eq!(text, "hi");
	assert!(usage.is_none());
}

#[test]
fn test_claude_parse_response_invalid_json_is_err() {
	let err = parse_response(&["not json".to_string()]).unwrap_err();
	assert!(err.to_string().contains("Failed to parse"));
}

#[test]
fn test_claude_parse_response_empty_text_is_err() {
	let body = r#"[{"type":"assistant","message":{"content":[]}}]"#;
	let err = parse_response(&[body.to_string()]).unwrap_err();
	assert!(err.to_string().contains("No text content"));
}
