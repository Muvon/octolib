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

use crate::errors::ProviderError;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderExchange, ProviderResponse, TokenUsage};
use crate::llm::utils::{is_model_in_pricing_table, PricingTuple};
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

// Model pricing (per 1M tokens in USD) - Updated Jan 2026
// Source: https://api-docs.deepseek.com/quick_start/pricing
/// Format: (model, input, output, cache_write, cache_read)
/// Note: DeepSeek uses cache_hit/cache_miss model - cache_write = cache_miss (input), cache_read = cache_hit
const PRICING: &[PricingTuple] = &[
    ("deepseek-chat", 0.27, 1.10, 0.27, 0.07),     // V3 model
    ("deepseek-reasoner", 0.55, 2.19, 0.55, 0.14), // R1 model
];

/// Get pricing tuple for a specific model (case-insensitive)
/// Returns None if the model is not in the pricing table (not supported)
fn get_pricing_tuple(model: &str) -> Option<(f64, f64, f64, f64)> {
    crate::llm::utils::get_model_pricing(model, PRICING)
}

/// Calculate cost for DeepSeek models with cache-aware pricing (Jan 2026)
fn calculate_cost_with_cache(
    model: &str,
    regular_input_tokens: u64,
    cache_hit_tokens: u64,
    completion_tokens: u64,
) -> Option<f64> {
    let (input_price, output_price, _cache_write_price, cache_read_price) =
        get_pricing_tuple(model)?;

    let regular_input_cost = (regular_input_tokens as f64 / 1_000_000.0) * input_price;
    let cache_hit_cost = (cache_hit_tokens as f64 / 1_000_000.0) * cache_read_price;
    let output_cost = (completion_tokens as f64 / 1_000_000.0) * output_price;

    Some(regular_input_cost + cache_hit_cost + output_cost)
}

/// Calculate cost for DeepSeek models without cache
fn calculate_cost(model: &str, input_tokens: u64, completion_tokens: u64) -> Option<f64> {
    calculate_cost_with_cache(model, input_tokens, 0, completion_tokens)
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
#[derive(Serialize, Debug, Clone)]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DeepSeekResponse {
    id: String,
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
    input_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    prompt_cache_hit_tokens: u64,
    #[serde(default)]
    prompt_cache_miss_tokens: u64,
    #[serde(default)]
    completion_tokens_details: Option<DeepSeekCompletionTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
struct DeepSeekCompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

#[async_trait::async_trait]
impl AiProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn supports_model(&self, model: &str) -> bool {
        // DeepSeek models - check against pricing table (strict)
        is_model_in_pricing_table(model, PRICING)
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

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            crate::llm::utils::get_model_pricing(model, PRICING)?;

        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
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
                reasoning_content: None,
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

        let client = self.client.clone();
        let response = retry::retry_with_exponential_backoff(
            || {
                let client = client.clone();
                let api_key = api_key.clone();
                let request = request.clone();
                Box::pin(async move {
                    let response = client
                        .post("https://api.deepseek.com/chat/completions")
                        .header("Authorization", format!("Bearer {}", api_key))
                        .header("Content-Type", "application/json")
                        .json(&request)
                        .send()
                        .await
                        .map_err(anyhow::Error::from)?;

                    // Return Err for retryable HTTP errors so the retry loop catches them
                    if retry::is_retryable_status(response.status().as_u16()) {
                        let status = response.status();
                        let error_text = response.text().await.unwrap_or_default();
                        return Err(anyhow::anyhow!(
                            "DeepSeek API error {}: {}",
                            status,
                            error_text
                        ));
                    }

                    Ok(response)
                })
            },
            params.max_retries,
            params.retry_timeout,
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
            |e| {
                matches!(
                    e.downcast_ref::<ProviderError>(),
                    Some(ProviderError::Cancelled)
                )
            },
        )
        .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = retry::cancellable(
                async { response.text().await.map_err(anyhow::Error::from) },
                params.cancellation_token.as_ref(),
                || ProviderError::Cancelled.into(),
            )
            .await?;
            return Err(anyhow::anyhow!(
                "DeepSeek API error {}: {}",
                status,
                error_text
            ));
        }

        let deepseek_response: DeepSeekResponse = retry::cancellable(
            async { response.json().await.map_err(anyhow::Error::from) },
            params.cancellation_token.as_ref(),
            || ProviderError::Cancelled.into(),
        )
        .await?;

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

        // Calculate cost with the provider pricing table
        let token_usage = if let Some(usage) = deepseek_response.usage {
            let prompt_tokens = usage.input_tokens;
            let completion_tokens = usage.completion_tokens;
            let total_tokens = usage.total_tokens;

            // DeepSeek reports:
            // - prompt_cache_hit_tokens: tokens read from cache (cache READ)
            // - prompt_cache_miss_tokens: fresh input tokens (includes regular + cache WRITE)
            // DeepSeek doesn't separate regular input from cache write in their API
            let cache_read_tokens = usage.prompt_cache_hit_tokens;
            let cache_miss_tokens = usage.prompt_cache_miss_tokens;

            // For CLEAN input_tokens, we use cache_miss_tokens
            // (DeepSeek charges these at the "cache miss" rate which includes write cost)
            let input_tokens_clean = cache_miss_tokens;

            // DeepSeek doesn't expose cache_write separately - it's included in cache_miss
            let cache_write_tokens = 0_u64;

            // Calculate cost with pricing table values (Jan 2026)
            let cost = if cache_read_tokens > 0 {
                calculate_cost_with_cache(
                    &params.model,
                    input_tokens_clean,
                    cache_read_tokens,
                    completion_tokens,
                )
            } else {
                calculate_cost(&params.model, prompt_tokens, completion_tokens)
            };

            let reasoning_tokens = usage
                .completion_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens)
                .unwrap_or(0);

            Some(TokenUsage {
                input_tokens: input_tokens_clean, // CLEAN input (cache miss tokens)
                cache_read_tokens,                // Tokens read from cache
                cache_write_tokens,               // DeepSeek doesn't expose this (0)
                output_tokens: completion_tokens,
                reasoning_tokens,
                total_tokens,
                cost,
                request_time_ms: None,
            })
        } else {
            None
        };

        // Update exchange with token usage
        let mut final_exchange = exchange;
        final_exchange.usage = token_usage.clone();

        let content = &choice.message.content;

        // Extract thinking block from reasoning_content if present
        let thinking = choice
            .message
            .reasoning_content
            .as_ref()
            .and_then(|reasoning| {
                if reasoning.trim().is_empty() {
                    None
                } else {
                    // Estimate tokens from content length (4 chars per token)
                    let tokens = (reasoning.len() / 4) as u64;
                    Some(crate::llm::types::ThinkingBlock {
                        content: reasoning.clone(),
                        tokens,
                    })
                }
            });

        // Try to parse structured output if it was requested
        let structured_output =
            if content.trim().starts_with('{') || content.trim().starts_with('[') {
                serde_json::from_str(content).ok()
            } else {
                None
            };

        Ok(ProviderResponse {
            content: choice.message.content,
            thinking,
            exchange: final_exchange,
            tool_calls: None, // DeepSeek doesn't support tool calls in octolib yet
            finish_reason: choice.finish_reason,
            structured_output,
            id: Some(deepseek_response.id),
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
    fn test_supports_model_case_insensitive() {
        let provider = DeepSeekProvider::new();
        // Test uppercase
        assert!(provider.supports_model("DEEPSEEK-CHAT"));
        assert!(provider.supports_model("DEEPSEEK-REASONER"));
        // Test mixed case
        assert!(provider.supports_model("DeepSeek-Chat"));
        assert!(provider.supports_model("DEEPSEEK-reasoner"));
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

    #[test]
    fn test_thinking_block_extraction() {
        // Test with reasoning_content present
        let message_with_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: "The answer is 9.11".to_string(),
            reasoning_content: Some("Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.".to_string()),
        };

        // Verify reasoning_content is properly stored
        assert!(message_with_thinking.reasoning_content.is_some());
        let reasoning = message_with_thinking.reasoning_content.as_ref().unwrap();
        assert_eq!(reasoning, "Let me compare 9.11 and 9.8. Converting to same decimal places: 9.11 vs 9.80. Clearly 9.80 > 9.11.");

        // Test token estimation (length / 4)
        let estimated_tokens = (reasoning.len() / 4) as u64;
        assert!(estimated_tokens > 0);

        // Test without reasoning_content
        let message_without_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: "Hello".to_string(),
            reasoning_content: None,
        };

        assert!(message_without_thinking.reasoning_content.is_none());

        // Test with empty reasoning_content
        let message_empty_thinking = DeepSeekMessage {
            role: "assistant".to_string(),
            content: "Hello".to_string(),
            reasoning_content: Some("".to_string()),
        };

        assert!(message_empty_thinking.reasoning_content.is_some());
        assert!(message_empty_thinking
            .reasoning_content
            .as_ref()
            .unwrap()
            .is_empty());
    }
}
