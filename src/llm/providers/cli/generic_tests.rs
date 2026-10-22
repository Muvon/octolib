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
use crate::llm::providers::cli::CliBackend;
use std::path::PathBuf;

fn provider() -> CliProvider {
	CliProvider {
		backend: CliBackend::from_name("mycli"),
		command: PathBuf::from("mycli"),
		extra_args: Vec::new(),
		model_flag: "-m".to_string(),
		prompt_flag: "-p".to_string(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	}
}

#[test]
fn test_build_args_includes_model_and_prompt() {
	let args = build_args(&provider(), "some-model", "hello world");
	assert_eq!(args, vec!["-m", "some-model", "-p", "hello world"]);
}

#[test]
fn test_build_args_skips_empty_model() {
	let args = build_args(&provider(), "", "prompt");
	assert_eq!(args, vec!["-p", "prompt"]);
}

#[test]
fn test_build_args_respects_custom_flags_and_extra_args() {
	let mut p = provider();
	p.model_flag = "--model".to_string();
	p.prompt_flag = "--prompt".to_string();
	p.extra_args = vec!["--verbose".to_string(), "--json".to_string()];
	let args = build_args(&p, "m", "q");
	assert_eq!(args, vec!["--model", "m", "--verbose", "--json", "--prompt", "q"]);
}

#[test]
fn test_parse_response_joins_lines() {
	let lines = vec!["first".to_string(), "second".to_string()];
	assert_eq!(parse_response(&lines).unwrap(), "first\nsecond");
}

#[test]
fn test_parse_response_single_line() {
	assert_eq!(parse_response(&["only".to_string()]).unwrap(), "only");
}

#[test]
fn test_parse_response_rejects_empty() {
	assert!(parse_response(&[]).is_err());
	assert!(parse_response(&["   ".to_string(), "\t".to_string()]).is_err());
}
