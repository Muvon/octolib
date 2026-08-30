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
