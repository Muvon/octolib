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

//! Utility functions for LLM providers
//!
//! This module provides case-insensitive string matching utilities optimized
//! for model name comparisons. All functions use ASCII-lowercase conversion
//! which is faster than full Unicode to_lowercase() since model names are
//! typically ASCII-only.

/// Convert a model name to lowercase for case-insensitive matching.
///
/// Uses ASCII-lowercase since model names are typically ASCII-only,
/// which is faster than full Unicode to_lowercase().
#[inline]
pub fn normalize_model_name(model: &str) -> String {
    model.to_ascii_lowercase()
}

/// Check if a model name starts with a given prefix (case-insensitive).
///
/// More efficient than creating a normalized string when checking a single prefix.
#[inline]
pub fn starts_with_ignore_ascii_case(model: &str, prefix: &str) -> bool {
    if model.len() < prefix.len() {
        return false;
    }
    model[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Check if a model name contains a given substring (case-insensitive).
///
/// Optimized to avoid double allocation by only converting the model name once.
#[inline]
pub fn contains_ignore_ascii_case(model: &str, substring: &str) -> bool {
    model
        .to_ascii_lowercase()
        .contains(&substring.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(normalize_model_name("GPT-4o"), "gpt-4o");
        assert_eq!(normalize_model_name("claude-3-haiku"), "claude-3-haiku");
        assert_eq!(normalize_model_name("MiniMax-M2.1"), "minimax-m2.1");
    }

    #[test]
    fn test_normalize_model_name_edge_cases() {
        // Empty string
        assert_eq!(normalize_model_name(""), "");
        // Numbers and special chars
        assert_eq!(normalize_model_name("GPT-4.5-TURBO"), "gpt-4.5-turbo");
        assert_eq!(normalize_model_name("o1-preview"), "o1-preview");
        // Colons (common in Bedrock model IDs)
        assert_eq!(
            normalize_model_name("ANTHROPIC.CLAUDE-3-HAIKU-V1:0"),
            "anthropic.claude-3-haiku-v1:0"
        );
    }

    #[test]
    fn test_starts_with_ignore_ascii_case() {
        assert!(starts_with_ignore_ascii_case("GPT-4o-mini", "gpt-4o"));
        assert!(starts_with_ignore_ascii_case("gpt-4o", "GPT-4O"));
        assert!(!starts_with_ignore_ascii_case("gpt-3.5", "gpt-4"));
        assert!(!starts_with_ignore_ascii_case("gpt", "gpt-4"));
    }

    #[test]
    fn test_starts_with_ignore_ascii_case_edge_cases() {
        // Empty prefix
        assert!(starts_with_ignore_ascii_case("gpt-4", ""));
        // Prefix longer than model
        assert!(!starts_with_ignore_ascii_case("gpt", "gpt-4o-mini"));
        // Exact match
        assert!(starts_with_ignore_ascii_case("GPT-4O", "GPT-4O"));
    }

    #[test]
    fn test_contains_ignore_ascii_case() {
        assert!(contains_ignore_ascii_case(
            "anthropic.claude-3-haiku-v1:0",
            "claude"
        ));
        assert!(contains_ignore_ascii_case("CLAUDE-3", "claude"));
        assert!(!contains_ignore_ascii_case("gpt-4o", "claude"));
    }

    #[test]
    fn test_contains_ignore_ascii_case_edge_cases() {
        // Empty substring
        assert!(contains_ignore_ascii_case("gpt-4o", ""));
        // Empty model
        assert!(!contains_ignore_ascii_case("", "gpt"));
        // Both empty
        assert!(contains_ignore_ascii_case("", ""));
    }
}
