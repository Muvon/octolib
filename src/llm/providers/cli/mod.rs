// Copyright 2025 Muvon Un Limited
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

//! Generic CLI provider implementation

mod claude;
mod codex;
mod cursor;
mod gemini;
mod generic;

use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, Message, ProviderExchange, ProviderResponse};
use anyhow::Result;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const CODEX_COMMAND_ENV: &str = "CODEX_COMMAND";
const CODEX_REASONING_EFFORT_ENV: &str = "CODEX_REASONING_EFFORT";
const CODEX_SKIP_GIT_CHECK_ENV: &str = "CODEX_SKIP_GIT_CHECK";

const CLI_COMMAND_PREFIX: &str = "CLI_";
const CLI_COMMAND_SUFFIX: &str = "_COMMAND";
const CLI_EXTRA_ARGS_SUFFIX: &str = "_EXTRA_ARGS";
const CLI_MODEL_FLAG_SUFFIX: &str = "_MODEL_FLAG";
const CLI_PROMPT_FLAG_SUFFIX: &str = "_PROMPT_FLAG";
const CLI_CODEX_REASONING_EFFORT_ENV: &str = "CLI_CODEX_REASONING_EFFORT";
const CLI_CODEX_SKIP_GIT_CHECK_ENV: &str = "CLI_CODEX_SKIP_GIT_CHECK";

const CODEX_REASONING_LEVELS: &[&str] = &["low", "medium", "high"];

#[derive(Debug, Clone, PartialEq, Eq)]
enum CliBackendKind {
    Codex,
    Claude,
    Cursor,
    Gemini,
    Generic,
}

#[derive(Debug, Clone)]
struct CliBackend {
    name: String,
    kind: CliBackendKind,
}

#[derive(Debug, Clone)]
pub struct CliProvider {
    backend: CliBackend,
    command: PathBuf,
    pub(crate) extra_args: Vec<String>,
    pub(crate) model_flag: String,
    pub(crate) prompt_flag: String,
    pub(crate) reasoning_effort: String,
    pub(crate) skip_git_check: bool,
}

impl CliProvider {
    pub fn new_for_model(model: &str) -> Result<Self> {
        let (backend_name, _) = split_cli_model(model)?;
        let backend = CliBackend::from_name(&backend_name);
        let command = resolve_cli_command(&backend.name)?;
        let extra_args = parse_extra_args(&backend.name);
        let model_flag = resolve_flag(&backend.name, CLI_MODEL_FLAG_SUFFIX, "-m");
        let prompt_flag = resolve_flag(&backend.name, CLI_PROMPT_FLAG_SUFFIX, "-p");

        let reasoning_effort = parse_reasoning_effort();
        let skip_git_check = parse_bool_env(CLI_CODEX_SKIP_GIT_CHECK_ENV, false)
            || parse_bool_env(CODEX_SKIP_GIT_CHECK_ENV, false);

        Ok(Self {
            backend,
            command,
            extra_args,
            model_flag,
            prompt_flag,
            reasoning_effort,
            skip_git_check,
        })
    }

    fn messages_to_prompt(&self, messages: &[Message]) -> String {
        let mut full_prompt = String::new();

        let system_text = messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");

        if !system_text.is_empty() {
            full_prompt.push_str(&system_text);
            full_prompt.push_str("\n\n");
        }

        for message in messages.iter().filter(|m| m.role != "system") {
            let role_prefix = match message.role.as_str() {
                "user" => "Human: ",
                "assistant" => "Assistant: ",
                _ => continue,
            };

            let trimmed = message.content.trim();
            if trimmed.is_empty() {
                continue;
            }

            full_prompt.push_str(role_prefix);
            full_prompt.push_str(trimmed);
            full_prompt.push('\n');
            full_prompt.push('\n');
        }

        full_prompt.push_str("Assistant: ");
        full_prompt
    }

    // execute_command moved to module-level helper to avoid capturing &self in retry closures
}

#[async_trait::async_trait]
impl AiProvider for CliProvider {
    fn name(&self) -> &str {
        "cli"
    }

    fn supports_model(&self, model: &str) -> bool {
        split_cli_model(model).is_ok()
    }

    fn get_api_key(&self) -> Result<String> {
        Ok(String::new())
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        128_000
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        false
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let (backend_name, backend_model) = split_cli_model(&params.model)?;
        if backend_name != self.backend.name {
            return Err(anyhow::anyhow!(
                "CLI provider initialized for backend '{}' but got model for '{}'",
                self.backend.name,
                backend_name
            ));
        }

        let request_time_start = std::time::Instant::now();

        let (lines, stderr, content, thinking, usage, used_args) = match self.backend.kind {
            CliBackendKind::Codex => {
                let prompt = self.messages_to_prompt(&params.messages);
                let args = codex::build_args(self, &backend_model);
                let command = self.command.clone();

                let (lines, stderr) = retry::retry_with_exponential_backoff(
                    || {
                        let args = args.clone();
                        let prompt = prompt.clone();
                        let command = command.clone();
                        Box::pin(async move { execute_command(command, args, Some(prompt)).await })
                    },
                    params.max_retries,
                    params.retry_timeout,
                    params.cancellation_token.as_ref(),
                )
                .await?;

                let (content, thinking, usage) = codex::parse_response(&lines)?;
                (lines, stderr, content, thinking, usage, args)
            }
            CliBackendKind::Claude => {
                let system_text = params
                    .messages
                    .iter()
                    .filter(|m| m.role == "system")
                    .map(|m| m.content.trim())
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let messages_json = claude::messages_to_claude_json(&params.messages);
                let args = claude::build_args(self, &backend_model, &system_text, &messages_json);
                let command = self.command.clone();

                let (lines, stderr) = retry::retry_with_exponential_backoff(
                    || {
                        let args = args.clone();
                        let command = command.clone();
                        Box::pin(async move { execute_command(command, args, None).await })
                    },
                    params.max_retries,
                    params.retry_timeout,
                    params.cancellation_token.as_ref(),
                )
                .await?;

                let (content, usage) = claude::parse_response(&lines)?;
                (lines, stderr, content, None, usage, args)
            }
            CliBackendKind::Gemini => {
                let prompt = self.messages_to_prompt(&params.messages);
                let args = gemini::build_args(self, &backend_model, &prompt);
                let command = self.command.clone();

                let (lines, stderr) = retry::retry_with_exponential_backoff(
                    || {
                        let args = args.clone();
                        let command = command.clone();
                        Box::pin(async move { execute_command(command, args, None).await })
                    },
                    params.max_retries,
                    params.retry_timeout,
                    params.cancellation_token.as_ref(),
                )
                .await?;

                let content = gemini::parse_response(&lines)?;
                (lines, stderr, content, None, None, args)
            }
            CliBackendKind::Cursor => {
                let prompt = self.messages_to_prompt(&params.messages);
                let args = cursor::build_args(self, &backend_model, &prompt);
                let command = self.command.clone();

                let (lines, stderr) = retry::retry_with_exponential_backoff(
                    || {
                        let args = args.clone();
                        let command = command.clone();
                        Box::pin(async move { execute_command(command, args, None).await })
                    },
                    params.max_retries,
                    params.retry_timeout,
                    params.cancellation_token.as_ref(),
                )
                .await?;

                let content = cursor::parse_response(&lines)?;
                (lines, stderr, content, None, None, args)
            }
            CliBackendKind::Generic => {
                let prompt = self.messages_to_prompt(&params.messages);
                let args = generic::build_args(self, &backend_model, &prompt);
                let command = self.command.clone();

                let (lines, stderr) = retry::retry_with_exponential_backoff(
                    || {
                        let args = args.clone();
                        let command = command.clone();
                        Box::pin(async move { execute_command(command, args, None).await })
                    },
                    params.max_retries,
                    params.retry_timeout,
                    params.cancellation_token.as_ref(),
                )
                .await?;

                let content = generic::parse_response(&lines)?;
                (lines, stderr, content, None, None, args)
            }
        };

        let request_time_ms = request_time_start.elapsed().as_millis() as u64;
        let mut usage = usage;
        if let Some(ref mut usage) = usage {
            usage.request_time_ms = Some(request_time_ms);
        }

        let request_json = serde_json::json!({
            "backend": backend_name,
            "command": self.command,
            "model": backend_model,
            "args": used_args,
        });

        let response_json = serde_json::json!({
            "lines": lines,
            "stderr": stderr,
        });

        let exchange = ProviderExchange::new(request_json, response_json, usage.clone(), "cli");

        let structured_output =
            if content.trim().starts_with('{') || content.trim().starts_with('[') {
                serde_json::from_str(&content).ok()
            } else {
                None
            };

        Ok(ProviderResponse {
            content,
            thinking,
            exchange,
            tool_calls: None,
            finish_reason: None,
            structured_output,
            id: None,
        })
    }
}

impl CliBackend {
    fn from_name(name: &str) -> Self {
        let normalized = normalize_backend_name(name);
        let kind = match normalized.as_str() {
            "codex" => CliBackendKind::Codex,
            "claude" => CliBackendKind::Claude,
            "cursor" => CliBackendKind::Cursor,
            "gemini" => CliBackendKind::Gemini,
            _ => CliBackendKind::Generic,
        };

        Self {
            name: normalized,
            kind,
        }
    }
}

fn split_cli_model(model: &str) -> Result<(String, String)> {
    let mut parts = model.splitn(2, '/');
    let backend_raw = parts.next().unwrap_or_default().trim();
    let model_name = parts.next().unwrap_or_default().trim();

    if backend_raw.is_empty() || model_name.is_empty() {
        return Err(anyhow::anyhow!(
            "Invalid cli model format. Use 'cli:<backend>/<model>' (e.g., 'cli:codex/gpt-5.2-codex')"
        ));
    }

    let backend = normalize_backend_name(backend_raw);
    Ok((backend, model_name.to_string()))
}

fn normalize_backend_name(name: &str) -> String {
    let normalized = name.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "claude-code" | "claude_code" => "claude".to_string(),
        "cursor-agent" | "cursor_agent" => "cursor".to_string(),
        "gemini-cli" | "gemini_cli" => "gemini".to_string(),
        "codex-cli" | "codex_cli" => "codex".to_string(),
        _ => normalized,
    }
}

fn resolve_cli_command(backend: &str) -> Result<PathBuf> {
    let backend_upper = backend.to_ascii_uppercase();
    let env_key = format!(
        "{}{}{}",
        CLI_COMMAND_PREFIX, backend_upper, CLI_COMMAND_SUFFIX
    );
    let legacy_command = if backend == "codex" {
        env::var(CODEX_COMMAND_ENV).ok()
    } else {
        None
    };

    let command = env::var(&env_key)
        .ok()
        .or(legacy_command)
        .unwrap_or_else(|| {
            if backend == "cursor" {
                "cursor-agent".to_string()
            } else {
                backend.to_string()
            }
        });

    let command_path = PathBuf::from(&command);

    if command.contains(std::path::MAIN_SEPARATOR) || command.contains('/') {
        if is_executable(&command_path) {
            return Ok(command_path);
        }
        return Err(anyhow::anyhow!(
            "CLI command not found at '{}'. Set {} to a valid path.",
            command,
            env_key
        ));
    }

    if let Some(found) = find_in_path(&command) {
        return Ok(found);
    }

    Err(anyhow::anyhow!(
        "CLI command '{}' not found in PATH. Install it or set {}.",
        command,
        env_key
    ))
}

fn parse_extra_args(backend: &str) -> Vec<String> {
    let backend_upper = backend.to_ascii_uppercase();
    let env_key = format!(
        "{}{}{}",
        CLI_COMMAND_PREFIX, backend_upper, CLI_EXTRA_ARGS_SUFFIX
    );
    env::var(env_key)
        .map(|value| {
            value
                .split_whitespace()
                .filter(|arg| !arg.is_empty())
                .map(|arg| arg.to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn resolve_flag(backend: &str, suffix: &str, default_flag: &str) -> String {
    let backend_upper = backend.to_ascii_uppercase();
    let env_key = format!("{}{}{}", CLI_COMMAND_PREFIX, backend_upper, suffix);
    env::var(env_key).unwrap_or_else(|_| default_flag.to_string())
}

fn parse_reasoning_effort() -> String {
    let effort = env::var(CLI_CODEX_REASONING_EFFORT_ENV)
        .or_else(|_| env::var(CODEX_REASONING_EFFORT_ENV))
        .unwrap_or_else(|_| "high".to_string());
    let effort_trimmed = effort.trim().to_lowercase();

    if CODEX_REASONING_LEVELS
        .iter()
        .any(|level| *level == effort_trimmed)
    {
        effort_trimmed
    } else {
        tracing::warn!(
            "Invalid {} value '{}'; using 'high'",
            CLI_CODEX_REASONING_EFFORT_ENV,
            effort
        );
        "high".to_string()
    }
}

fn parse_bool_env(key: &str, default: bool) -> bool {
    match env::var(key) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn find_in_path(command: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(command);
        if is_executable(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    path.is_file()
}

async fn execute_command(
    command: PathBuf,
    args: Vec<String>,
    prompt: Option<String>,
) -> Result<(Vec<String>, String)> {
    let output = tokio::task::spawn_blocking(move || {
        let mut cmd = std::process::Command::new(&command);
        cmd.args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            anyhow::anyhow!(
                "Failed to spawn CLI command '{:?}': {}. Ensure the CLI is installed and in PATH, or set CLI_<BACKEND>_COMMAND.",
                command,
                e
            )
        })?;

        if let Some(prompt) = prompt {
            if let Some(mut stdin) = child.stdin.take() {
                use std::io::Write;
                let mut prompt_bytes = prompt.into_bytes();
                if !prompt_bytes.ends_with(b"\n") {
                    prompt_bytes.push(b'\n');
                }

                if let Err(e) = stdin.write_all(&prompt_bytes) {
                    if e.kind() != std::io::ErrorKind::BrokenPipe {
                        return Err(anyhow::anyhow!("Failed to write prompt to CLI stdin: {}", e));
                    }
                }
            }
        }

        let output = child.wait_with_output().map_err(|e| {
            anyhow::anyhow!("Failed to read CLI output: {}", e)
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            let status = output.status.code().unwrap_or(-1);
            let stderr_trimmed = stderr.trim();
            let message = if stderr_trimmed.is_empty() {
                format!("CLI command failed with exit code {}", status)
            } else {
                format!(
                    "CLI command failed with exit code {}: {}",
                    status, stderr_trimmed
                )
            };
            return Err(anyhow::anyhow!(message));
        }

        let lines = stdout
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .map(|line| line.to_string())
            .collect::<Vec<_>>();

        Ok((lines, stderr))
    })
    .await??;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_cli_model() {
        let (backend, model) = split_cli_model("codex/gpt-5.2-codex").unwrap();
        assert_eq!(backend, "codex");
        assert_eq!(model, "gpt-5.2-codex");
    }

    #[test]
    fn test_messages_to_prompt() {
        let provider =
            CliProvider::new_for_model("codex/gpt-5.2-codex").unwrap_or_else(|_| CliProvider {
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
}
