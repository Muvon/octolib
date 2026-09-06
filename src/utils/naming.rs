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

//! Model-name canonicalization shared by the LLM, embedding, and media
//! pricing tables. Lives outside `llm` so media can match reference rates
//! without depending on the chat providers.

/// Convert a model name to lowercase for case-insensitive matching.
///
/// Uses ASCII-lowercase since model names are typically ASCII-only,
/// which is faster than full Unicode to_lowercase().
#[inline]
pub fn normalize_model_name(model: &str) -> String {
    model.to_ascii_lowercase()
}

/// Sanitize provider-specific model name formats into a canonical form
/// for matching against reference patterns.
///
/// Handles:
/// - Ollama format: `llama3.3:70b` → `llama-3.3-70b`
/// - HuggingFace/Together: `meta-llama/llama-3.3-70b-instruct` → strips org prefix irrelevant to matching
/// - Version dots without dashes: `qwen2.5` → `qwen-2.5`
pub(crate) fn sanitize_model_name(name: &str) -> String {
    let mut s = name.to_string();
    // Replace colons with dashes (Ollama uses `model:size`)
    s = s.replace(':', "-");
    // Insert dashes between letters and digits where missing (e.g., `llama3` → `llama-3`)
    let mut result = String::with_capacity(s.len() + 4);
    let chars: Vec<char> = s.chars().collect();
    for i in 0..chars.len() {
        result.push(chars[i]);
        if i + 1 < chars.len() {
            let curr = chars[i];
            let next = chars[i + 1];
            // letter→digit or digit→letter boundary, but NOT around dots/dashes
            if (curr.is_ascii_alphabetic() && next.is_ascii_digit())
                || (curr.is_ascii_digit() && next.is_ascii_alphabetic())
            {
                // Only insert dash if there isn't already a separator
                if curr != '-' && curr != '.' && next != '-' && next != '.' {
                    result.push('-');
                }
            }
        }
    }
    result
}
