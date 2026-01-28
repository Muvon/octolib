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

//! Codex CLI provider implementation

use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, ProviderExchange, ProviderResponse, ThinkingBlock, TokenUsage,
};
use anyhow::Result;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Stdio;

const CODEX_COMMAND_ENV: &str = "CODEX_COMMAND";
const CODEX_REASONING_EFFORT_ENV: &str = "CODEX_REASONING_EFFORT";
const CODEX_SKIP_GIT_CHECK_ENV: &str = "CODEX_SKIP_GIT_CHECK";

const CODEX_REASONING_LEVELS: &[&str] = &["low", "medium", "high"];

#[derive(Debug, Clone)]
pub struct CodexProvider {
    command: PathBuf,
    reasoning_effort: String,
    skip_git_check: bool,
}

impl CodexProvider {
    pub fn new() -> Result<Self> {
        let command = resolve_codex_command()?;
        let reasoning_effort = parse_reasoning_effort();
        let skip_git_check = parse_bool_env(CODEX_SKIP_GIT_CHECK_ENV, false);

        Ok(Self {
            command,
            reasoning_effort,
            skip_git_check,
        })
    }

    #[cfg(test)]
    fn new_for_test(command: PathBuf) -> Self {
        Self {
            command,
            reasoning_effort: "high".to_string(),
            skip_git_check: false,
        }
    }

    fn messages_to_prompt(&self, messages: &[crate::llm::types::Message]) -> String {
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

    fn build_command_args(&self, model: &str) -> Vec<String> {
        let mut args = vec!["exec".to_string()];

        if !model.is_empty() {
            args.push("-m".to_string());
            args.push(model.to_string());
        }

        args.push("-c".to_string());
        args.push(format!(
            "model_reasoning_effort=\"{}\"",
            self.reasoning_effort
        ));

        if self.skip_git_check {
            args.push("--skip-git-repo-check".to_string());
        }

        args.push("--json".to_string());
        args.push("-".to_string());

        args
    }

    async fn execute_command(
        &self,
        prompt: String,
        model: String,
    ) -> Result<(Vec<String>, String)> {
        let command = self.command.clone();
        let args_primary = self.build_command_args(&model);

        let output = tokio::task::spawn_blocking(move || {
            fn run_once(
                command: &PathBuf,
                args: &[String],
                prompt: String,
            ) -> Result<(Vec<String>, String), anyhow::Error> {
                let mut cmd = std::process::Command::new(command);
                cmd.args(args)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped());

                let mut child = cmd.spawn().map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to spawn Codex CLI '{:?}': {}. Ensure Codex CLI is installed and in PATH (npm i -g @openai/codex), or set CODEX_COMMAND.",
                        command,
                        e
                    )
                })?;

                if let Some(mut stdin) = child.stdin.take() {
                    use std::io::Write;
                    let mut prompt_bytes = prompt.into_bytes();
                    if !prompt_bytes.ends_with(b"\n") {
                        prompt_bytes.push(b'\n');
                    }

                    if let Err(e) = stdin.write_all(&prompt_bytes) {
                        if e.kind() != std::io::ErrorKind::BrokenPipe {
                            return Err(anyhow::anyhow!(
                                "Failed to write prompt to Codex stdin: {}",
                                e
                            ));
                        }
                    }
                }

                let output = child.wait_with_output().map_err(|e| {
                    anyhow::anyhow!("Failed to read Codex CLI output: {}", e)
                })?;

                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                if !output.status.success() {
                    let status = output.status.code().unwrap_or(-1);
                    let stderr_trimmed = stderr.trim();
                    let message = if stderr_trimmed.is_empty() {
                        format!("Codex CLI failed with exit code {}", status)
                    } else {
                        format!(
                            "Codex CLI failed with exit code {}: {}",
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
            }

            run_once(&command, &args_primary, prompt)
        })
        .await??;

        Ok(output)
    }

    fn parse_response(
        &self,
        lines: &[String],
    ) -> Result<(String, Option<ThinkingBlock>, Option<TokenUsage>)> {
        let mut all_text_content: Vec<String> = Vec::new();
        let mut reasoning_chunks: Vec<String> = Vec::new();
        let mut error_message: Option<String> = None;
        let mut input_tokens: Option<u64> = None;
        let mut output_tokens: Option<u64> = None;
        let mut cached_tokens: Option<u64> = None;
        let mut reasoning_tokens: Option<u64> = None;
        let mut total_tokens: Option<u64> = None;

        for line in lines {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                if let Some(event_type) = parsed.get("type").and_then(|t| t.as_str()) {
                    match event_type {
                        "item.completed" => {
                            if let Some(item) = parsed.get("item") {
                                let item_type = item.get("type").and_then(|t| t.as_str());
                                if item_type == Some("agent_message") {
                                    if let Some(text) = item
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|t| t.trim())
                                        .filter(|t| !t.is_empty())
                                    {
                                        all_text_content.push(text.to_string());
                                    }
                                } else if item_type == Some("reasoning") {
                                    if let Some(text) = item
                                        .get("text")
                                        .and_then(|t| t.as_str())
                                        .map(|t| t.trim())
                                        .filter(|t| !t.is_empty())
                                    {
                                        reasoning_chunks.push(text.to_string());
                                    }
                                }
                            }
                        }
                        "turn.completed" | "result" | "done" => {
                            if let Some(usage_info) = parsed.get("usage") {
                                if input_tokens.is_none() {
                                    input_tokens =
                                        usage_info.get("input_tokens").and_then(|v| v.as_u64());
                                }
                                if output_tokens.is_none() {
                                    output_tokens =
                                        usage_info.get("output_tokens").and_then(|v| v.as_u64());
                                }
                                if cached_tokens.is_none() {
                                    cached_tokens = usage_info
                                        .get("cached_input_tokens")
                                        .and_then(|v| v.as_u64());
                                }
                                if reasoning_tokens.is_none() {
                                    reasoning_tokens =
                                        usage_info.get("reasoning_tokens").and_then(|v| v.as_u64());
                                }
                                if total_tokens.is_none() {
                                    total_tokens =
                                        usage_info.get("total_tokens").and_then(|v| v.as_u64());
                                }
                            }
                            all_text_content.extend(extract_legacy_text(&parsed));
                        }
                        "error" | "turn.failed" => {
                            error_message = extract_error(&parsed);
                        }
                        "message" | "assistant" => {
                            all_text_content.extend(extract_legacy_text(&parsed));
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(err) = error_message {
            if all_text_content.is_empty() {
                return Err(anyhow::anyhow!("Codex CLI error: {}", err));
            }
        }

        if all_text_content.is_empty() {
            if let Some(fallback) = build_fallback_text(lines) {
                all_text_content.push(fallback);
            }
        }

        let combined_text = all_text_content.join("\n\n");
        if combined_text.trim().is_empty() {
            return Err(anyhow::anyhow!("Empty response from Codex CLI"));
        }

        let thinking = if reasoning_chunks.is_empty() {
            None
        } else {
            let content = reasoning_chunks.join("\n\n");
            Some(ThinkingBlock {
                tokens: (content.len() / 4) as u64,
                content,
            })
        };

        let usage = if input_tokens.is_some()
            || output_tokens.is_some()
            || cached_tokens.is_some()
            || reasoning_tokens.is_some()
            || total_tokens.is_some()
        {
            let prompt_tokens = input_tokens.unwrap_or(0);
            let output_tokens_val = output_tokens.unwrap_or(0);
            let reasoning_tokens_val = reasoning_tokens.unwrap_or(0);
            let total =
                total_tokens.unwrap_or(prompt_tokens + output_tokens_val + reasoning_tokens_val);

            Some(TokenUsage {
                prompt_tokens,
                output_tokens: output_tokens_val,
                reasoning_tokens: reasoning_tokens_val,
                total_tokens: total,
                cached_tokens: cached_tokens.unwrap_or(0),
                cost: None,
                request_time_ms: None,
            })
        } else {
            None
        };

        Ok((combined_text, thinking, usage))
    }
}

#[async_trait::async_trait]
impl AiProvider for CodexProvider {
    fn name(&self) -> &str {
        "codex"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // Codex CLI manages auth internally (via its own login/config).
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
        let prompt = self.messages_to_prompt(&params.messages);
        let request_time_start = std::time::Instant::now();

        let (lines, stderr) = retry::retry_with_exponential_backoff(
            || {
                let prompt = prompt.clone();
                let model = params.model.clone();
                let self_clone = self.clone();
                Box::pin(async move { self_clone.execute_command(prompt, model).await })
            },
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
        )
        .await?;

        let request_time_ms = request_time_start.elapsed().as_millis() as u64;

        let (content, thinking, mut usage) = self.parse_response(&lines)?;

        if let Some(ref mut usage) = usage {
            usage.request_time_ms = Some(request_time_ms);
        }

        let request_json = serde_json::json!({
            "command": self.command,
            "args": self.build_command_args(&params.model),
            "model": params.model,
            "reasoning_effort": self.reasoning_effort,
            "skip_git_check": self.skip_git_check,
            "prompt": prompt,
        });

        let response_json = serde_json::json!({
            "lines": lines,
            "stderr": stderr,
        });

        let exchange = ProviderExchange::new(request_json, response_json, usage.clone(), "codex");

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

fn parse_reasoning_effort() -> String {
    let effort = env::var(CODEX_REASONING_EFFORT_ENV).unwrap_or_else(|_| "high".to_string());
    let effort_trimmed = effort.trim().to_lowercase();

    if CODEX_REASONING_LEVELS
        .iter()
        .any(|level| *level == effort_trimmed)
    {
        effort_trimmed
    } else {
        tracing::warn!(
            "Invalid {} value '{}'; using 'high'",
            CODEX_REASONING_EFFORT_ENV,
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

fn resolve_codex_command() -> Result<PathBuf> {
    let command = env::var(CODEX_COMMAND_ENV).unwrap_or_else(|_| "codex".to_string());
    let command_path = PathBuf::from(&command);

    if command.contains(std::path::MAIN_SEPARATOR) || command.contains('/') {
        if is_executable(&command_path) {
            return Ok(command_path);
        }
        return Err(anyhow::anyhow!(
            "Codex CLI not found at '{}'. Install it with npm i -g @openai/codex or set {} to a valid path.",
            command,
            CODEX_COMMAND_ENV
        ));
    }

    if let Some(found) = find_in_path(&command) {
        return Ok(found);
    }

    Err(anyhow::anyhow!(
        "Codex CLI '{}' not found in PATH. Install it with npm i -g @openai/codex or set {} to a valid path.",
        command,
        CODEX_COMMAND_ENV
    ))
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

fn extract_error(parsed: &serde_json::Value) -> Option<String> {
    parsed
        .get("message")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            parsed
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
        })
}

fn extract_legacy_text(parsed: &serde_json::Value) -> Vec<String> {
    let mut texts = Vec::new();
    if let Some(content) = parsed.get("content").and_then(|c| c.as_array()) {
        for item in content {
            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    texts.push(trimmed.to_string());
                }
            }
        }
    }
    if let Some(text) = parsed.get("text").and_then(|t| t.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            texts.push(trimmed.to_string());
        }
    }
    if let Some(text) = parsed.get("result").and_then(|r| r.as_str()) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            texts.push(trimmed.to_string());
        }
    }
    texts
}

fn build_fallback_text(lines: &[String]) -> Option<String> {
    let response_text = lines
        .iter()
        .filter(|line| {
            !line.starts_with('{')
                || serde_json::from_str::<serde_json::Value>(line)
                    .map(|v| v.get("type").is_none())
                    .unwrap_or(true)
        })
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    if response_text.trim().is_empty() {
        None
    } else {
        Some(response_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_messages_to_prompt_with_system_and_messages() {
        let provider = CodexProvider::new_for_test(PathBuf::from("codex"));
        let messages = vec![
            crate::llm::types::Message::system("You are a helpful assistant."),
            crate::llm::types::Message::user("Hello"),
            crate::llm::types::Message::assistant("Hi there!"),
        ];

        let prompt = provider.messages_to_prompt(&messages);
        assert!(prompt.starts_with("You are a helpful assistant."));
        assert!(prompt.contains("Human: Hello"));
        assert!(prompt.contains("Assistant: Hi there!"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_parse_response_json_events() {
        let provider = CodexProvider::new_for_test(PathBuf::from("codex"));
        let lines = vec![
            r#"{"type":"item.completed","item":{"id":"item_0","type":"reasoning","text":"Thinking..."}}"#.to_string(),
            r#"{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Hello there!"}}"#.to_string(),
            r#"{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50,"cached_input_tokens":30}}"#.to_string(),
        ];

        let result = provider.parse_response(&lines);
        assert!(result.is_ok());

        let (content, thinking, usage) = result.unwrap();
        assert!(content.contains("Hello there!"));
        assert!(thinking.is_some());

        let usage = usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cached_tokens, 30);
    }

    #[test]
    fn test_parse_response_plain_text_fallback() {
        let provider = CodexProvider::new_for_test(PathBuf::from("codex"));
        let lines = vec!["Hello, world!".to_string()];

        let result = provider.parse_response(&lines);
        assert!(result.is_ok());
        let (content, _thinking, _usage) = result.unwrap();
        assert!(content.contains("Hello, world!"));
    }
}
