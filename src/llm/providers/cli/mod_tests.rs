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
use crate::errors::ProviderError;

#[tokio::test]
async fn test_cli_rejects_per_call_api_key_before_starting_process() {
	let provider = CliProvider {
		backend: CliBackend::from_name("codex"),
		command: PathBuf::from("/nonexistent/byok-test-cli"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: "high".to_owned(),
		skip_git_check: false,
	};
	let params = ChatCompletionParams::new(&[], "codex/gpt-5.2-codex", 1.0, 1.0, 0, 10)
		.with_api_key("private-request-key".to_owned());
	let error = provider.chat_completion(params).await.unwrap_err();
	assert!(matches!(
		error.downcast_ref::<ProviderError>(),
		Some(ProviderError::UnsupportedOperation { provider, .. }) if provider == "cli"
	));
	assert!(!format!("{error:?}").contains("private-request-key"));
}

#[test]
fn test_split_cli_model() {
	let (backend, model) = split_cli_model("codex/gpt-5.2-codex").unwrap();
	assert_eq!(backend, "codex");
	assert_eq!(model, "gpt-5.2-codex");
}

#[test]
fn test_messages_to_prompt() {
	let provider = CliProvider::new_for_model("codex/gpt-5.2-codex").unwrap_or_else(|_| CliProvider {
		backend: CliBackend::from_name("codex"),
		command: PathBuf::from("codex"),
		extra_args: Vec::new(),
		model_flag: "-m".to_string(),
		prompt_flag: "-p".to_string(),
		reasoning_effort: "high".to_string(),
		skip_git_check: false,
	});

	let messages = vec![
		Message::system("You are helpful."),
		Message::user("Hello"),
		Message::assistant("Hi"),
	];

	let prompt = provider.messages_to_prompt(&messages);
	assert!(prompt.starts_with("You are helpful."));
	assert!(prompt.contains("Human: Hello"));
	assert!(prompt.contains("Assistant: Hi"));
	assert!(prompt.ends_with("Assistant: "));
}

#[test]
fn test_normalize_backend_name_aliases() {
	for (raw, expected) in [
		("claude-code", "claude"),
		("claude_code", "claude"),
		("Claude-Code", "claude"),
		("cursor-agent", "cursor"),
		("cursor_agent", "cursor"),
		("gemini-cli", "gemini"),
		("gemini_cli", "gemini"),
		("codex-cli", "codex"),
		("codex_cli", "codex"),
		("  CustomTool  ", "customtool"),
	] {
		assert_eq!(normalize_backend_name(raw), expected, "alias {raw}");
	}
}

#[test]
fn test_split_cli_model_rejects_missing_parts() {
	assert!(split_cli_model("").is_err());
	assert!(split_cli_model("codex").is_err());
	assert!(split_cli_model("/model").is_err());
	assert!(split_cli_model("codex/").is_err());
	assert!(split_cli_model("   /   ").is_err());
}

#[test]
fn test_split_cli_model_normalizes_backend() {
	let (backend, model) = split_cli_model("Claude-Code/opus").unwrap();
	assert_eq!(backend, "claude");
	assert_eq!(model, "opus");
}

#[test]
fn test_cli_backend_from_name_kinds() {
	assert_eq!(CliBackend::from_name("codex").kind, CliBackendKind::Codex);
	assert_eq!(CliBackend::from_name("claude").kind, CliBackendKind::Claude);
	assert_eq!(CliBackend::from_name("cursor").kind, CliBackendKind::Cursor);
	assert_eq!(CliBackend::from_name("gemini").kind, CliBackendKind::Gemini);
	assert_eq!(CliBackend::from_name("anything-else").kind, CliBackendKind::Generic);
}

#[test]
fn test_supports_model_delegates_to_split() {
	let provider = CliProvider {
		backend: CliBackend::from_name("codex"),
		command: PathBuf::from("codex"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	};
	assert!(provider.supports_model("codex/anything"));
	assert!(!provider.supports_model("no-slash"));
}

#[test]
fn test_parse_bool_env_values_and_default() {
	// Use a unique key so this never races with other env-reading tests.
	let key = "CLI_TEST_BOOL_ENV_KEY";
	for (value, expected) in [
		("1", true),
		("true", true),
		("YES", true),
		(" On ", true),
		("0", false),
		("no", false),
		("", false),
	] {
		env::set_var(key, value);
		assert_eq!(parse_bool_env(key, false), expected, "value {value:?}");
	}
	env::remove_var(key);
	assert!(!parse_bool_env(key, false));
	assert!(parse_bool_env(key, true));
}

#[test]
fn test_parse_reasoning_effort_accepts_known_levels_and_defaults() {
	env::set_var(CLI_CODEX_REASONING_EFFORT_ENV, " Low ");
	assert_eq!(parse_reasoning_effort(), "low");
	env::remove_var(CLI_CODEX_REASONING_EFFORT_ENV);

	env::set_var(CODEX_REASONING_EFFORT_ENV, "ultra");
	assert_eq!(parse_reasoning_effort(), "high");
	env::set_var(CODEX_REASONING_EFFORT_ENV, "medium");
	assert_eq!(parse_reasoning_effort(), "medium");
	env::remove_var(CODEX_REASONING_EFFORT_ENV);

	assert_eq!(parse_reasoning_effort(), "high");
}

#[test]
fn test_parse_extra_args_splits_on_whitespace() {
	let key = "CLI_TESTARGS_EXTRA_ARGS";
	env::set_var(key, "  --one   --two  ");
	assert_eq!(parse_extra_args("testargs"), vec!["--one", "--two"]);
	env::remove_var(key);
	assert!(parse_extra_args("testargs").is_empty());
}

#[test]
fn test_resolve_flag_uses_env_then_default() {
	let key = "CLI_TESTFLAGS_MODEL_FLAG";
	env::set_var(key, "--model-flag");
	assert_eq!(resolve_flag("testflags", CLI_MODEL_FLAG_SUFFIX, "-m"), "--model-flag");
	env::remove_var(key);
	assert_eq!(resolve_flag("testflags", CLI_MODEL_FLAG_SUFFIX, "-m"), "-m");
}

#[test]
fn test_resolve_cli_command_accepts_executable_path() {
	let key = "CLI_TESTCMD_COMMAND";
	let path = env::temp_dir().join("octolib-cli-testcmd");
	std::fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
	}
	env::set_var(key, path.to_str().unwrap());
	let resolved = resolve_cli_command("testcmd").unwrap();
	assert_eq!(resolved, path);
	env::set_var(key, "/nonexistent/octolib-testcmd-missing");
	let err = resolve_cli_command("testcmd").unwrap_err();
	assert!(format!("{err}").contains("not found"));
	env::remove_var(key);
	let _ = std::fs::remove_file(&path);
}

#[test]
fn test_resolve_cli_command_bare_name_searches_path() {
	// Create a temp executable and prepend its directory to PATH so the
	// bare-name PATH search is exercised deterministically on every platform.
	let dir = env::temp_dir().join("octolib-cli-path-search");
	std::fs::create_dir_all(&dir).unwrap();
	let exe_name = "octolib-pathsearch";
	let path = dir.join(exe_name);
	std::fs::write(&path, b"#!/bin/sh\nexit 0\n").unwrap();
	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
	}
	let old_path = env::var_os("PATH");
	let mut new_path = dir.clone().into_os_string();
	if let Some(prev) = &old_path {
		new_path.push(if cfg!(windows) { ";" } else { ":" });
		new_path.push(prev);
	}
	env::set_var("PATH", new_path);
	let found = resolve_cli_command("octolib-pathsearch").unwrap();
	assert_eq!(found, path);
	match old_path {
		Some(prev) => env::set_var("PATH", prev),
		None => env::remove_var("PATH"),
	}
	let _ = std::fs::remove_file(&path);
	let _ = std::fs::remove_dir(&dir);
}

#[test]
fn test_messages_to_prompt_skips_unknown_roles_and_empty_content() {
	let provider = CliProvider {
		backend: CliBackend::from_name("codex"),
		command: PathBuf::from("codex"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	};
	let mut weird = Message::user("");
	weird.role = "tool".to_string();
	let prompt = provider.messages_to_prompt(&[Message::system("  "), weird, Message::user("hi")]);
	assert_eq!(prompt, "Human: hi\n\nAssistant: ");
}

#[tokio::test]
async fn test_cli_chat_completion_rejects_model_for_other_backend() {
	use crate::llm::traits::AiProvider;
	let provider = CliProvider {
		backend: CliBackend::from_name("codex"),
		command: PathBuf::from("codex"),
		extra_args: Vec::new(),
		model_flag: "-m".to_owned(),
		prompt_flag: "-p".to_owned(),
		reasoning_effort: String::new(),
		skip_git_check: false,
	};
	let params = ChatCompletionParams::new(&[Message::user("hi")], "claude/some-model", 1.0, 1.0, 0, 10);
	let err = provider.chat_completion(params).await.unwrap_err();
	assert!(format!("{err}").contains("backend"));
}
