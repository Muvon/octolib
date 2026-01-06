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

//! DeepSeek provider implementation
//!
//! PRICING UPDATE: January 2026
//! Model-specific pricing (per 1M tokens in USD):
//!
//! deepseek-chat (V3):
//! - Cache Hit: $0.07
//! - Cache Miss (Input): $0.27
//! - Output: $1.10
//!
//! deepseek-reasoner (R1):
//! - Cache Hit: $0.14
//! - Cache Miss (Input): $0.55
//! - Output: $2.19

use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderExchange, ProviderResponse, TokenUsage};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

// Model pricing (per 1M tokens in USD) - Updated Jan 2026
// Source: https://api-docs.deepseek.com/quick_start/pricing
const PRICING: &[(&str, f64, f64, f64)] = &[
    // Model, Cache Hit price, Cache Miss (Input) price, Output price per 1M tokens
    ("deepseek-chat", 0.07, 0.27, 1.10),     // V3 model
    ("deepseek-reasoner", 0.14, 0.55, 2.19), // R1 model
];

/// Get pricing for a specific model
fn get_model_pricing(model: &str) -> (f64, f64, f64) {
    for (pricing_model, cache_hit, cache_miss, output) in PRICING {
        if model.contains(pricing_model) {
            return (*cache_hit, *cache_miss, *output);
        }
    }
    // Default to deepseek-chat pricing if model not found
    (0.07, 0.27, 1.10)
}

/// Calculate cost for DeepSeek models with cache-aware pricing (Jan 2026)
fn calculate_cost_with_cache(
    model: &str,
    regular_input_tokens: u64,
    cache_hit_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    let (cache_hit_price, cache_miss_price, output_price) = get_model_pricing(model);

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * cache_miss_price;
    let cache_hit_cost = (cache_hit_tokens as f64 / 1_000_000.0) * cache_hit_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

    Some(regular_input_cost + cache_hit_cost + output_cost)
}

/// Calculate cost for DeepSeek models without cache
fn calculate_cost(model: &str, prompt_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_with_cache(model, prompt_tokens, 0, completion_tokens)
}

/// DeepSeek provider
#[derive(Debug, Clone)]
pub struct DeepSeekProvider {
    client: Client,
}

impl Default for DeepSeekProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepSeekProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

// DeepSeek API request/response structures
#[derive(Serialize, Debug)]
struct DeepSeekRequest {
    model: String,
    messages: Vec<DeepSeekMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct DeepSeekMessage {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekResponse {
    choices: Vec<DeepSeekChoice>,
    usage: Option<DeepSeekUsage>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekChoice {
    message: DeepSeekMessage,
    finish_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
    #[serde(default)]
    prompt_cache_miss_tokens: u64,
}

#[async_trait::async_trait]
impl AiProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn supports_model(&self, model: &str) -> bool {
        matches!(model, "deepseek-chat" | "deepseek-reasoner")
    }

    fn get_api_key(&self) -> Result<String> {
        match env::var(DEEPSEEK_API_KEY_ENV) {
            Ok(key) => Ok(key),
            Err(_) => Err(anyhow::anyhow!(
                "DeepSeek API key not found in environment variable: {}",
                DEEPSEEK_API_KEY_ENV
            )),
        }
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true // DeepSeek supports caching
    }

    fn supports_vision(&self, _model: &str) -> bool {
        false // DeepSeek doesn't support vision yet
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        // DeepSeek supports JSON mode as per their API documentation
        true
    }

    fn get_max_input_tokens(&self, _model: &str) -> usize {
        64_000 // DeepSeek context window
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;

        // Convert messages to DeepSeek format
        let messages: Vec<DeepSeekMessage> = params
            .messages
            .iter()
            .map(|msg| DeepSeekMessage {
                role: msg.role.clone(),
                content: msg.content.clone(),
            })
            .collect();

        let mut request = DeepSeekRequest {
            model: params.model.clone(),
            messages,
            temperature: Some(params.temperature),
            max_tokens: Some(params.max_tokens),
            stream: Some(false), // We don't support streaming in octolib yet
            response_format: None,
        };

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    // DeepSeek supports JSON mode but not full JSON schema validation
                    // Fall back to json_object mode
                    request.response_format = Some(serde_json::json!({
                        "type": "json_object"
                    }));
                }
            }
        }

        let response = self
            .client
            .post("https://api.deepseek.com/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "DeepSeek API error {}: {}",
                status,
                error_text
            ));
        }

        let deepseek_response: DeepSeekResponse = response.json().await?;

        // Clone the response for exchange logging before moving parts of it
        let response_for_exchange = serde_json::to_value(&deepseek_response)?;

        let choice = deepseek_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No choices in DeepSeek response"))?;

        // Create exchange record for logging
        let exchange = ProviderExchange {
            request: serde_json::to_value(&request)?,
            response: response_for_exchange,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            usage: None, // Will be set below
            provider: self.name().to_string(),
            rate_limit_headers: None, // DeepSeek doesn't provide rate limit headers in response
        };

        // Calculate cost with new unified pricing
        let token_usage = if let Some(usage) = deepseek_response.usage {
            let prompt_tokens = usage.prompt_tokens;
            let completion_tokens = usage.completion_tokens;
            let total_tokens = usage.total_tokens;
            let cache_hit_tokens = usage.prompt_cache_hit_tokens;

            // For DeepSeek: Cache hit tokens get special pricing ($0.07/1M)
            // Regular input tokens are charged at cache miss rate ($0.56/1M)
            let regular_input_tokens = prompt_tokens.saturating_sub(cache_hit_tokens);

            // Calculate cost with unified pricing (Sept 5, 2025+)
            let cost = if cache_hit_tokens > 0 {
                calculate_cost_with_cache(
                    &params.model,
                    regular_input_tokens,
                    cache_hit_tokens,
                    completion_tokens,
                )
            } else {
                calculate_cost(&params.model, prompt_tokens, completion_tokens)
            };

            Some(TokenUsage {
                prompt_tokens,
                output_tokens: completion_tokens,
                total_tokens,
                cached_tokens: cache_hit_tokens, // Simple: total tokens that came from cache
                cost,                  // Pre-calculated with unified pricing (Sept 5, 2025+)
                request_time_ms: None, // Not tracked in octolib
            })
        } else {
            None
        };

        // Update exchange with token usage
        let mut final_exchange = exchange;
        final_exchange.usage = token_usage.clone();

        let content = &choice.message.content;

        // Try to parse structured output if it was requested
        let structured_output =
            if content.trim().starts_with('{') || content.trim().starts_with('[') {
                serde_json::from_str(content).ok()
            } else {
                None
            };

        Ok(ProviderResponse {
            content: choice.message.content,
            exchange: final_exchange,
            tool_calls: None, // DeepSeek doesn't support tool calls in octolib yet
            finish_reason: choice.finish_reason,
            structured_output,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = DeepSeekProvider::new();
        assert!(provider.supports_model("deepseek-chat"));
        assert!(provider.supports_model("deepseek-reasoner"));
        assert!(!provider.supports_model("gpt-4"));
        assert!(!provider.supports_model("deepseek-coder")); // Not in current API
    }

    #[test]
    fn test_calculate_cost() {
        // Test basic cost calculation with Jan 2026 pricing
        // deepseek-chat: Input: $0.27/1M, Output: $1.10/1M
        let cost = calculate_cost("deepseek-chat", 1_000_000, 500_000);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();

        // Expected: (1M * $0.27) + (0.5M * $1.10) = $0.27 + $0.55 = $0.82
        let expected = 0.27 + (0.5 * 1.10);
        assert!((cost_value - expected).abs() < 0.01); // Allow small floating point differences

        // Test with reasoner model - different pricing
        // deepseek-reasoner: Input: $0.55/1M, Output: $2.19/1M
        let cost2 = calculate_cost("deepseek-reasoner", 1_000_000, 500_000);
        assert!(cost2.is_some());
        let expected2 = 0.55 + (0.5 * 2.19);
        assert!((cost2.unwrap() - expected2).abs() < 0.01);
    }

    #[test]
    fn test_calculate_cost_with_cache() {
        // Test cache-aware cost calculation with Jan 2026 pricing
        // deepseek-chat: Cache hit: $0.07/1M, Cache miss: $0.27/1M, Output: $1.10/1M
        let cost = calculate_cost_with_cache("deepseek-chat", 500_000, 500_000, 250_000);
        assert!(cost.is_some());
        let cost_value = cost.unwrap();

        // Expected: (0.5M * $0.27) + (0.5M * $0.07) + (0.25M * $1.10)
        //         = $0.135 + $0.035 + $0.275 = $0.445
        let expected = (0.5 * 0.27) + (0.5 * 0.07) + (0.25 * 1.10);
        assert!((cost_value - expected).abs() < 0.01);

        // Cost with cache should be less than without cache for same total input
        let cost_no_cache = calculate_cost("deepseek-chat", 1_000_000, 250_000);
        assert!(cost_no_cache.is_some());
        assert!(cost_value < cost_no_cache.unwrap());
    }
}
