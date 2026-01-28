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
use anyhow::Result;

pub(crate) fn build_args(provider: &CliProvider, model: &str, prompt: &str) -> Vec<String> {
    let mut args = Vec::new();

    if !model.is_empty() {
        args.push("--model".to_string());
        args.push(model.to_string());
    }

    args.extend(provider.extra_args.clone());

    args.push("-p".to_string());
    args.push(prompt.to_string());
    args.push("--output-format".to_string());
    args.push("json".to_string());
    args.push("--force".to_string());

    args
}

pub(crate) fn parse_response(lines: &[String]) -> Result<String> {
    for line in lines {
        if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(type_val) = json_value.get("type") {
                if type_val == "result" {
                    let result = json_value.get("result").and_then(|v| v.as_str());
                    let is_error = json_value
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    if let Some(result) = result {
                        if result.trim().is_empty() {
                            if is_error {
                                return Err(anyhow::anyhow!(
                                    "Cursor CLI returned an error response"
                                ));
                            }
                            return Err(anyhow::anyhow!("Cursor CLI completed without content"));
                        }
                        return Ok(result.to_string());
                    }
                }
            }
        }
    }

    let response_text = lines.join("\n");
    if response_text.trim().is_empty() {
        return Err(anyhow::anyhow!("Empty response from Cursor CLI"));
    }

    Ok(response_text)
}
