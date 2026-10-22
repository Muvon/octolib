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

fn provider(effort: &str, skip_git_check: bool) -> CliProvider {
	CliProvider {
		backend: crate::llm::providers::cli::CliBackend::from_name("codex"),
		command: PathBuf::from("codex"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: effort.to_owned(),
		skip_git_check,
	}
}

fn lines(values: &[&str]) -> Vec<String> {
	values.iter().map(|s| s.to_string()).collect()
}

#[test]
fn test_codex_build_args_defaults_with_model_and_effort_from_provider() {
	let args = build_args(&provider("medium", false), "gpt-5.2-codex", None);
	assert_eq!(args[0], "exec");
	assert!(args.contains(&"-m".to_string()));
	assert!(args.contains(&"gpt-5.2-codex".to_string()));
	assert!(args.contains(&"model_reasoning_effort=\"medium\"".to_string()));
	assert!(!args.contains(&"--skip-git-repo-check".to_string()));
	assert_eq!(*args.last().unwrap(), "-".to_string());
}

#[test]
fn test_codex_build_args_empty_model_omits_model_flag() {
	let args = build_args(&provider("low", true), "", None);
	assert!(!args.contains(&"-m".to_string()));
	assert!(args.contains(&"--skip-git-repo-check".to_string()));
}

#[test]
fn test_codex_build_args_effort_override_beats_provider_default() {
	for (effort, expected) in [
		(ReasoningEffort::None, "low"),
		(ReasoningEffort::Low, "low"),
		(ReasoningEffort::Medium, "medium"),
		(ReasoningEffort::High, "high"),
		(ReasoningEffort::XHigh, "high"),
		(ReasoningEffort::Max, "high"),
	] {
		let args = build_args(&provider("should-be-ignored", false), "m", Some(effort));
		assert!(
			args.contains(&format!("model_reasoning_effort=\"{expected}\"")),
			"effort {effort:?} should map to {expected}"
		);
	}
}

#[test]
fn test_codex_parse_response_item_completed_and_usage() {
	let events = lines(&[
		r#"{"type":"item.completed","item":{"type":"agent_message","text":"  hello  "}}"#,
		r#"{"type":"item.completed","item":{"type":"reasoning","text":"thinking hard"}}"#,
		r#"{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":20,"output_tokens":50,"reasoning_tokens":5,"total_tokens":175}}"#,
	]);
	let (text, thinking, usage) = parse_response(&events).unwrap();
	assert_eq!(text, "hello");
	let thinking = thinking.expect("thinking block");
	assert_eq!(thinking.content, "thinking hard");
	let usage = usage.expect("usage");
	assert_eq!(usage.input_tokens, 80); // clean input excludes cache
	assert_eq!(usage.cache_read_tokens, 20);
	assert_eq!(usage.output_tokens, 50);
	assert_eq!(usage.reasoning_tokens, 5);
	assert_eq!(usage.total_tokens, 175);
}

#[test]
fn test_codex_parse_response_skips_empty_item_texts() {
	let events = lines(&[r#"{"type":"item.completed","item":{"type":"agent_message","text":"   "}}"#]);
	// every event is JSON with a type, so the fallback text is empty too
	let err = parse_response(&events).unwrap_err();
	assert!(err.to_string().contains("Empty response"));
}

#[test]
fn test_codex_parse_response_legacy_message_content() {
	let events = lines(&[
		r#"{"type":"message","content":[{"text":"part one"},{"text":"  "}]}"#,
		r#"{"type":"assistant","text":"plain text"}"#,
	]);
	let (text, _, usage) = parse_response(&events).unwrap();
	assert_eq!(text, "part one\n\nplain text");
	assert!(usage.is_none());
}

#[test]
fn test_codex_parse_response_error_without_text_is_err() {
	let events = lines(&[r#"{"type":"error","message":"boom"}"#]);
	let err = parse_response(&events).unwrap_err();
	assert!(err.to_string().contains("boom"));
}

#[test]
fn test_codex_parse_response_error_nested_message_is_err() {
	let events = lines(&[r#"{"type":"turn.failed","error":{"message":"deep failure"}}"#]);
	let err = parse_response(&events).unwrap_err();
	assert!(err.to_string().contains("deep failure"));
}

#[test]
fn test_codex_parse_response_error_with_text_still_returns_text() {
	let events = lines(&[
		r#"{"type":"item.completed","item":{"type":"agent_message","text":"partial answer"}}"#,
		r#"{"type":"error","message":"boom"}"#,
	]);
	let (text, _, _) = parse_response(&events).unwrap();
	assert_eq!(text, "partial answer");
}

#[test]
fn test_codex_parse_response_empty_input_is_err() {
	let err = parse_response(&[]).unwrap_err();
	assert!(err.to_string().contains("Empty response"));
}

#[test]
fn test_codex_parse_response_plain_text_fallback() {
	let events = lines(&["not json at all", "second line"]);
	let (text, _, usage) = parse_response(&events).unwrap();
	assert_eq!(text, "not json at all\nsecond line");
	assert!(usage.is_none());
}

#[test]
fn test_codex_parse_response_first_usage_wins() {
	let events = lines(&[
		r#"{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}"#,
		r#"{"type":"result","usage":{"input_tokens":10,"output_tokens":4}}"#,
	]);
	let (_, _, usage) = parse_response(&events).unwrap();
	let usage = usage.unwrap();
	assert_eq!(usage.input_tokens, 10);
	assert_eq!(usage.output_tokens, 4);
	assert_eq!(usage.total_tokens, 14);
}

#[test]
fn test_codex_parse_response_usage_total_defaults_to_sum() {
	let events = lines(&[
		r#"{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}"#,
		r#"{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":3,"reasoning_tokens":2}}"#,
	]);
	let (_, _, usage) = parse_response(&events).unwrap();
	let usage = usage.unwrap();
	assert_eq!(usage.total_tokens, 12);
}
