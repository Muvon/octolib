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

//! Google Vertex AI provider implementation

use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{contains_ignore_ascii_case, normalize_model_name};
use anyhow::Result;
use std::env;

/// Google Vertex AI provider
#[derive(Debug, Clone)]
pub struct GoogleVertexProvider;

impl Default for GoogleVertexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleVertexProvider {
    pub fn new() -> Self {
        Self
    }
}

const GOOGLE_APPLICATION_CREDENTIALS_ENV: &str = "GOOGLE_APPLICATION_CREDENTIALS";
const GOOGLE_API_KEY_ENV: &str = "GOOGLE_API_KEY";

#[allow(dead_code)] // Pricing table ready for when implementation is completed
const PRICING: &[(&str, f64, f64)] = &[
    // Model, Input price per 1M tokens, Output price per 1M tokens
    // Source: https://ai.google.dev/gemini-api/docs/pricing (as of Jan 2026)
    // Gemini 3 (Released Nov 18, 2025)
    ("gemini-3-pro", 2.00, 12.00),
    ("gemini-3-pro-preview", 2.00, 12.00),
    // Gemini 2.5
    ("gemini-2.5-pro", 1.25, 5.00),
    ("gemini-2.5-flash", 0.075, 0.30),
    ("gemini-2.5-flash-lite", 0.10, 0.30),
    // Gemini 2.0
    ("gemini-2.0-flash", 0.10, 0.40), // Updated Jan 2026: simplified pricing
    ("gemini-2.0-flash-lite", 0.10, 0.30),
    ("gemini-2.0-flash-live", 0.35, 1.50), // Text pricing, audio/video higher
    // Gemini 1.5
    ("gemini-1.5-pro", 1.25, 5.00),
    ("gemini-1.5-flash", 0.075, 0.30),
    // Gemini 1.0
    ("gemini-1.0-pro", 0.50, 1.50),
];

#[async_trait::async_trait]
impl AiProvider for GoogleVertexProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn supports_model(&self, model: &str) -> bool {
        // Google Vertex AI models (case-insensitive)
        let normalized = normalize_model_name(model);
        normalized.starts_with("text-")
            || normalized.starts_with("chat-")
            || normalized.contains("gemini")
            || normalized.contains("palm")
            || normalized.contains("text-bison")
            || normalized.contains("chat-bison")
    }

    fn get_api_key(&self) -> Result<String> {
        // Google Vertex AI can use either API key or service account credentials
        if let Ok(key) = env::var(GOOGLE_API_KEY_ENV) {
            Ok(key)
        } else if let Ok(_credentials) = env::var(GOOGLE_APPLICATION_CREDENTIALS_ENV) {
            // Service account credentials file path is available
            // In a full implementation, we would use this to authenticate
            Ok("service_account_auth".to_string()) // Placeholder
        } else {
            Err(anyhow::anyhow!(
                "Google authentication not found. Set either {} or {}",
                GOOGLE_API_KEY_ENV,
                GOOGLE_APPLICATION_CREDENTIALS_ENV
            ))
        }
    }

    fn supports_caching(&self, model: &str) -> bool {
        // Google Vertex AI caching (case-insensitive)
        let normalized = normalize_model_name(model);
        normalized.contains("gemini-1.5")
            || normalized.contains("gemini-2")
            || normalized.contains("gemini-3")
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Google Vertex AI vision (case-insensitive)
        normalize_model_name(model).contains("gemini")
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        // Search through pricing table for matching model
        for (pricing_model, input_price, output_price) in PRICING {
            if contains_ignore_ascii_case(model, pricing_model) {
                return Some(crate::llm::types::ModelPricing::without_cache(
                    *input_price,
                    *output_price,
                ));
            }
        }
        None
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Google Vertex AI model context window limits (case-insensitive)
        let normalized = normalize_model_name(model);
        if normalized.contains("gemini-3") {
            1_048_576 // Gemini 3.0 has ~1M context
        } else if normalized.contains("gemini-2") {
            2_000_000 // Gemini 2.0 has 2M context
        } else if normalized.contains("gemini-1.5") {
            1_000_000 // Gemini 1.5 has 1M context
        } else if normalized.contains("gemini-1.0") || normalized.contains("bison-32k") {
            32_768 // Gemini 1.0 and 32K variants have 32K context
        } else if normalized.contains("bison") {
            8_192 // Standard Bison models
        } else {
            32_768 // Conservative default
        }
    }

    async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
        Err(anyhow::anyhow!(
            "Google Vertex AI provider not fully implemented in octolib"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = GoogleVertexProvider::new();

        // Google models should be supported
        assert!(provider.supports_model("gemini-1.5-pro"));
        assert!(provider.supports_model("gemini-2.0-flash"));
        assert!(provider.supports_model("gemini-1.0-pro"));
        assert!(provider.supports_model("text-bison"));

        // Unsupported models
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("claude-3"));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = GoogleVertexProvider::new();

        // Test uppercase
        assert!(provider.supports_model("GEMINI-1.5-PRO"));
        assert!(provider.supports_model("GEMINI-2.0-FLASH"));
        // Test mixed case
        assert!(provider.supports_model("Gemini-1.5-Pro"));
        assert!(provider.supports_model("GEMINI-1.0-pro"));
    }

    #[test]
    fn test_supports_caching_case_insensitive() {
        let provider = GoogleVertexProvider::new();

        // Test lowercase
        assert!(provider.supports_caching("gemini-1.5-pro"));
        assert!(provider.supports_caching("gemini-2.0-flash"));

        // Test uppercase
        assert!(provider.supports_caching("GEMINI-1.5-PRO"));
        assert!(provider.supports_caching("Gemini-2.0-Flash"));
    }
}
