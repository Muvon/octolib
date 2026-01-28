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

use crate::llm::providers::cli::CliProvider;
use crate::llm::types::{ThinkingBlock, TokenUsage};
use anyhow::Result;

pub(crate) fn build_args(provider: &CliProvider, model: &str) -> Vec<String> {
    let mut args = vec!["exec".to_string()];

    if !model.is_empty() {
        args.push("-m".to_string());
        args.push(model.to_string());
    }

    args.push("-c".to_string());
    args.push(format!(
        "model_reasoning_effort=\"{}\"",
        provider.reasoning_effort
    ));

    if provider.skip_git_check {
        args.push("--skip-git-repo-check".to_string());
    }

    args.push("--json".to_string());
    args.push("-".to_string());

    args
}

pub(crate) fn parse_response(
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
