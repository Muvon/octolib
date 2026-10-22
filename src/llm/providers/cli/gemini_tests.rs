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
		backend: crate::llm::providers::cli::CliBackend::from_name("gemini"),
		command: PathBuf::from("gemini"),
		extra_args: vec!["--flag".to_string()],
		model_flag: "--model".to_owned(),
		prompt_flag: "--prompt".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	}
}

fn lines(values: &[&str]) -> Vec<String> {
	values.iter().map(|s| s.to_string()).collect()
}

#[test]
fn test_gemini_build_args_uses_provider_flags() {
	let args = build_args(&provider(), "gemini-3-pro", "question");
	assert_eq!(
		args,
		vec![
			"--model".to_string(),
			"gemini-3-pro".to_string(),
			"--flag".to_string(),
			"--prompt".to_string(),
			"question".to_string(),
			"--yolo".to_string(),
		]
	);
}

#[test]
fn test_gemini_build_args_empty_model_omits_flag() {
	let args = build_args(&provider(), "", "q");
	assert!(!args.contains(&"--model".to_string()));
}

#[test]
fn test_gemini_parse_response_filters_credential_noise() {
	let events = lines(&["Loaded cached credentials from somewhere", "actual output"]);
	assert_eq!(parse_response(&events).unwrap(), "actual output");
}

#[test]
fn test_gemini_parse_response_keeps_all_other_lines() {
	let events = lines(&["one", "two"]);
	assert_eq!(parse_response(&events).unwrap(), "one\ntwo");
}

#[test]
fn test_gemini_parse_response_only_noise_is_err() {
	let events = lines(&["Loaded cached credentials"]);
	assert!(parse_response(&events).is_err());
}
