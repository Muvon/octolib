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
        args.push(provider.model_flag.clone());
        args.push(model.to_string());
    }

    args.extend(provider.extra_args.clone());

    args.push(provider.prompt_flag.clone());
    args.push(prompt.to_string());
    args.push("--yolo".to_string());

    args
}

pub(crate) fn parse_response(lines: &[String]) -> Result<String> {
    let filtered = lines
        .iter()
        .filter(|line| !line.starts_with("Loaded cached credentials"))
        .cloned()
        .collect::<Vec<_>>();

    let response_text = filtered.join("\n");

    if response_text.trim().is_empty() {
        return Err(anyhow::anyhow!("Empty response from Gemini CLI"));
    }

    Ok(response_text)
}
