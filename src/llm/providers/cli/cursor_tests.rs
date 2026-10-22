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
use std::path::PathBuf;

fn provider() -> CliProvider {
	CliProvider {
		backend: crate::llm::providers::cli::CliBackend::from_name("cursor"),
		command: PathBuf::from("cursor-agent"),
		extra_args: vec!["--extra".to_string()],
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	}
}

fn lines(values: &[&str]) -> Vec<String> {
	values.iter().map(|s| s.to_string()).collect()
}

#[test]
fn test_cursor_build_args_ordering_and_flags() {
	let args = build_args(&provider(), "model-x", "do it");
	assert_eq!(
		args,
		vec![
			"--model".to_string(),
			"model-x".to_string(),
			"--extra".to_string(),
			"-p".to_string(),
			"do it".to_string(),
			"--output-format".to_string(),
			"json".to_string(),
			"--force".to_string(),
		]
	);
}

#[test]
fn test_cursor_build_args_empty_model_omits_flag() {
	let args = build_args(&provider(), "", "prompt");
	assert!(!args.contains(&"--model".to_string()));
}

#[test]
fn test_cursor_parse_response_result_line() {
	let events = lines(&[r#"{"type":"result","result":"the answer"}"#]);
	assert_eq!(parse_response(&events).unwrap(), "the answer");
}

#[test]
fn test_cursor_parse_response_error_result_empty_text() {
	let err = parse_response(&lines(&[r#"{"type":"result","result":"  ","is_error":true}"#])).unwrap_err();
	assert!(err.to_string().contains("error response"));
}

#[test]
fn test_cursor_parse_response_non_error_empty_result() {
	let err = parse_response(&lines(&[r#"{"type":"result","result":""}"#])).unwrap_err();
	assert!(err.to_string().contains("without content"));
}

#[test]
fn test_cursor_parse_response_falls_back_to_raw_lines() {
	let events = lines(&["line one", "line two"]);
	assert_eq!(parse_response(&events).unwrap(), "line one\nline two");
}

#[test]
fn test_cursor_parse_response_empty_is_err() {
	assert!(parse_response(&[]).is_err());
}
