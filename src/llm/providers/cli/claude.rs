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
use crate::llm::types::{Message, TokenUsage};
use anyhow::Result;

pub(crate) fn messages_to_claude_json(messages: &[Message]) -> serde_json::Value {
    let mut claude_messages = Vec::new();

    for message in messages.iter().filter(|m| m.role != "system") {
        let role = match message.role.as_str() {
            "user" => "user",
            "assistant" => "assistant",
            _ => continue,
        };

        let trimmed = message.content.trim();
        if trimmed.is_empty() {
            continue;
        }

        let content = vec![serde_json::json!({
            "type": "text",
            "text": trimmed,
        })];

        claude_messages.push(serde_json::json!({
            "role": role,
            "content": content,
        }));
    }

    serde_json::Value::Array(claude_messages)
}

pub(crate) fn build_args(
    provider: &CliProvider,
    model: &str,
    system: &str,
    messages_json: &serde_json::Value,
) -> Vec<String> {
    let mut args = vec![
        provider.prompt_flag.clone(),
        messages_json.to_string(),
        "--system-prompt".to_string(),
        system.to_string(),
    ];

    if !model.is_empty() {
        args.push("--model".to_string());
        args.push(model.to_string());
    }

    args.push("--output-format".to_string());
    args.push("json".to_string());
    args.push("--verbose".to_string());

    args
}

pub(crate) fn parse_response(lines: &[String]) -> Result<(String, Option<TokenUsage>)> {
    let mut all_text_content = Vec::new();
    let mut input_tokens: Option<u64> = None;
    let mut output_tokens: Option<u64> = None;
    let mut total_tokens: Option<u64> = None;

    let full_response = lines.join("");
    let json_array: Vec<serde_json::Value> = serde_json::from_str(&full_response)
        .map_err(|e| anyhow::anyhow!("Failed to parse Claude CLI JSON response: {}", e))?;

    for parsed in json_array {
        if let Some(msg_type) = parsed.get("type").and_then(|t| t.as_str()) {
            match msg_type {
                "assistant" => {
                    if let Some(message) = parsed.get("message") {
                        if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
                            for item in content {
                                if let Some(content_type) =
                                    item.get("type").and_then(|t| t.as_str())
                                {
                                    if content_type == "text" {
                                        if let Some(text) =
                                            item.get("text").and_then(|t| t.as_str())
                                        {
                                            all_text_content.push(text.to_string());
                                        }
                                    }
                                }
                            }
                        }

                        if let Some(usage_info) = message.get("usage") {
                            if input_tokens.is_none() {
                                input_tokens =
                                    usage_info.get("input_tokens").and_then(|v| v.as_u64());
                            }
                            if output_tokens.is_none() {
                                output_tokens =
                                    usage_info.get("output_tokens").and_then(|v| v.as_u64());
                            }
                            if total_tokens.is_none() {
                                total_tokens =
                                    usage_info.get("total_tokens").and_then(|v| v.as_u64());
                            }
                        }
                    }
                }
                "result" => {
                    if let Some(result_usage) = parsed.get("usage") {
                        if input_tokens.is_none() {
                            input_tokens =
                                result_usage.get("input_tokens").and_then(|v| v.as_u64());
                        }
                        if output_tokens.is_none() {
                            output_tokens =
                                result_usage.get("output_tokens").and_then(|v| v.as_u64());
                        }
                        if total_tokens.is_none() {
                            total_tokens =
                                result_usage.get("total_tokens").and_then(|v| v.as_u64());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let combined_text = all_text_content.join("\n\n");
    if combined_text.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "No text content found in Claude CLI response"
        ));
    }

    let usage = if input_tokens.is_some() || output_tokens.is_some() || total_tokens.is_some() {
        let prompt_tokens = input_tokens.unwrap_or(0);
        let output_tokens_val = output_tokens.unwrap_or(0);
        let total = total_tokens.unwrap_or(prompt_tokens + output_tokens_val);
        Some(TokenUsage {
            prompt_tokens,
            output_tokens: output_tokens_val,
            reasoning_tokens: 0,
            total_tokens: total,
            cached_tokens: 0,
            cost: None,
            request_time_ms: None,
        })
    } else {
        None
    };

    Ok((combined_text, usage))
}
