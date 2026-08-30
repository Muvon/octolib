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

//! Alibaba Cloud Model Studio (DashScope) provider implementation.
//!
//! Uses the OpenAI-compatible endpoint at:
//! `https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`
//!
//! Hosts the Qwen family plus third-party models (DeepSeek, GLM) at Alibaba's
//! own rates. Mainland China accounts, Token Plan subscriptions and dedicated
//! workspace deployments use a different host — override `ALIBABA_API_URL`
//! with the full endpoint including `/chat/completions`.
//!
//! PRICING UPDATE: August 2026
//! Source: <https://www.alibabacloud.com/help/en/model-studio/model-pricing>
//!
//! Configuration:
//! - `ALIBABA_API_KEY`: Required API key
//! - `ALIBABA_API_URL`: Optional endpoint override (Token Plan, China, workspace)

use crate::errors::StructuredOutputError;
use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::providers::shared::parse_structured_output_from_text;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, OutputFormat, ProviderResponse, StructuredOutputRequest,
};
use crate::llm::utils::{normalize_model_name, PricingTuple};
use anyhow::Result;
use std::env;

/// Alibaba Cloud Model Studio provider
#[derive(Debug, Clone)]
pub struct AlibabaProvider;

impl Default for AlibabaProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AlibabaProvider {
    pub fn new() -> Self {
        Self
    }

    async fn chat_completion_once(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let api_url = get_api_url(ALIBABA_API_URL_ENV, ALIBABA_API_URL);
        let model = params.model.clone();

        // Selected Qwen families get native strict JSON Schema. Alibaba-hosted
        // models without that capability use JSON Object mode with the schema
        // in the prompt. Both paths are validated locally before returning.
        let (params, requested_schema) = prepare_schema_request(params);
        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "alibaba",
                usage_fallback_cost: None,
                use_response_cost: false,
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                let input_tokens = usage.input_tokens;
                let cache_write_tokens = usage.cache_write_tokens;
                let cache_read_tokens = usage.cache_read_tokens;
                let output_tokens = usage.billable_output_tokens();
                usage.cost = calculate_local_usage_cost(
                    &model,
                    input_tokens,
                    cache_write_tokens,
                    cache_read_tokens,
                    output_tokens,
                )
                .or_else(|| {
                    self.get_model_pricing(&model).map(|pricing| {
                        pricing.calculate_cost(
                            input_tokens,
                            cache_write_tokens,
                            cache_read_tokens,
                            output_tokens,
                        )
                    })
                });
            }
        }

        if let Some(schema) = requested_schema {
            enforce_schema_on_response(&mut response, &schema)?;
        }

        Ok(response)
    }
}

struct AlibabaTransport<'a>(&'a AlibabaProvider);

#[async_trait::async_trait]
impl AiProvider for AlibabaTransport<'_> {
    fn name(&self) -> &str {
        "alibaba"
    }

    fn supports_model(&self, model: &str) -> bool {
        self.0.supports_model(model)
    }

    fn get_api_key(&self) -> Result<String> {
        self.0.get_api_key()
    }

    fn enforces_response_schema(&self, model: &str) -> bool {
        natively_enforces_response_schema(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        self.0.chat_completion_once(params).await
    }
}

const ALIBABA_API_KEY_ENV: &str = "ALIBABA_API_KEY";
const ALIBABA_API_URL_ENV: &str = "ALIBABA_API_URL";
const ALIBABA_API_URL: &str =
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions";

// Model Studio international pricing (per 1M tokens in USD) - Aug 2026
// Source: https://www.alibabacloud.com/help/en/model-studio/model-pricing
// Format: (model, input, output, cache_write, cache_read)
// Context caching is implicit: writes bill at the input rate, hits at cache_read.
// Tiered models are priced at the 0-256K tier; longer contexts bill up to 4x more.
// Except for qwen3.8-max's separately published $0.25 rate and DeepSeek V4 Pro,
// implicit cache hits cost 20% of uncached input.
const PRICING: &[PricingTuple] = &[
    ("qwen3.8-max", 2.00, 6.00, 2.00, 0.25),
    ("qwen3.8-flash", 0.113, 0.382, 0.113, 0.0226),
    // Dated Qwen 3.7 snapshots retain list price; moving aliases have current promos.
    ("qwen3.7-max-2026-06-08", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-2026-05-20", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-2026-05-17", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max-preview", 2.50, 7.50, 2.50, 0.50),
    ("qwen3.7-max", 1.25, 3.75, 1.25, 0.25),
    ("qwen3.7-plus-2026-05-26", 0.40, 1.60, 0.40, 0.08),
    ("qwen3.7-plus", 0.32, 1.28, 0.32, 0.064),
    ("qwen3.6-plus", 0.50, 3.00, 0.50, 0.10),
    ("qwen3.6-flash", 0.25, 1.50, 0.25, 0.05),
    ("qwen3.5-flash", 0.10, 0.40, 0.10, 0.02),
    ("qwen3-coder-plus", 1.00, 5.00, 1.00, 0.20),
    ("qwen3-coder-flash", 0.30, 1.50, 0.30, 0.06),
    ("qwen3-vl-plus", 0.20, 1.60, 0.20, 0.04),
    ("qwen-max", 1.60, 6.40, 1.60, 0.32),
    ("qwen-plus", 0.40, 1.20, 0.40, 0.08),
    ("qwen-turbo", 0.05, 0.20, 0.05, 0.01),
    // Third-party models resold by Model Studio at Alibaba's own rates
    ("deepseek-v4-pro", 2.40, 4.80, 2.40, 0.24),
    ("deepseek-v4-flash", 0.20, 0.40, 0.20, 0.04),
    ("glm-5.2", 1.40, 4.40, 1.40, 0.28),
];

const QWEN_PLUS_LONG_CONTEXT_THRESHOLD: u64 = 256_000;

fn calculate_local_usage_cost(
    model: &str,
    input_tokens: u64,
    cache_write_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    let (mut input, mut output, mut cache_write, mut cache_read) =
        crate::llm::utils::get_model_pricing(model, PRICING)?;
    let total_input_tokens = input_tokens
        .saturating_add(cache_write_tokens)
        .saturating_add(cache_read_tokens);

    if normalize_model_name(model).contains("qwen3.7-plus")
        && total_input_tokens > QWEN_PLUS_LONG_CONTEXT_THRESHOLD
    {
        if normalize_model_name(model).contains("qwen3.7-plus-2026-05-26") {
            // Dated snapshot list-price tier for prompts in (256K, 1M].
            input = 1.20;
            output = 4.80;
            cache_write = 1.20;
            cache_read = 0.24;
        } else {
            // Moving alias: current 20%-off tier for prompts in (256K, 1M].
            input = 0.96;
            output = 3.84;
            cache_write = 0.96;
            cache_read = 0.192;
        }
    }

    Some(
        (input_tokens as f64 / 1_000_000.0) * input
            + (cache_write_tokens as f64 / 1_000_000.0) * cache_write
            + (cache_read_tokens as f64 / 1_000_000.0) * cache_read
            + (output_tokens as f64 / 1_000_000.0) * output,
    )
}

#[async_trait::async_trait]
impl AiProvider for AlibabaProvider {
    fn name(&self) -> &str {
        "alibaba"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        env::var(ALIBABA_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Alibaba API key not found in environment variable: {}",
                ALIBABA_API_KEY_ENV
            )
        })
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true
    }

    // supports_vision, supports_video, get_max_input_tokens are resolved via
    // reference capabilities (trait defaults)

    /// Alibaba supports native JSON Schema for selected Qwen families. Other
    /// hosted models use JSON Object mode plus client-side validation.
    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn enforces_response_schema(&self, model: &str) -> bool {
        natively_enforces_response_schema(model)
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        if let Some((input, output, cache_write, cache_read)) =
            crate::llm::utils::get_model_pricing(model, PRICING)
        {
            return Some(crate::llm::types::ModelPricing::new(
                input,
                output,
                cache_write,
                cache_read,
            ));
        }
        crate::llm::reference_models::get_reference_pricing(model)
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        crate::llm::schema_enforcement::chat_completion_enforced(&AlibabaTransport(self), params)
            .await
    }
}

/// JSON Schema is native only for the Alibaba model families explicitly listed
/// by Model Studio. Snapshot suffixes inherit their family's capability.
fn natively_enforces_response_schema(model: &str) -> bool {
    let model = normalize_model_name(model);
    [
        "qwen3.8-max",
        "qwen3.8-flash",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.7-flash",
    ]
    .iter()
    .any(|prefix| model.starts_with(prefix))
}

/// Prepare a JSON Schema request for Alibaba and retain the schema for local
/// validation. Native Qwen routes keep `json_schema`; other models are guided
/// with `json_object` plus a trailing system message.
fn prepare_schema_request(
    mut params: ChatCompletionParams,
) -> (ChatCompletionParams, Option<serde_json::Value>) {
    let Some(request) = params.response_format.as_ref() else {
        return (params, None);
    };
    if !matches!(request.format, OutputFormat::JsonSchema) {
        return (params, None);
    }
    let Some(schema) = request.schema.clone() else {
        return (params, None);
    };

    if natively_enforces_response_schema(&params.model) {
        return (params, Some(schema));
    }

    params.response_format = Some(StructuredOutputRequest::json());
    params.messages.push(Message::system(&format!(
        "Your next reply must be a single JSON object that conforms exactly to this JSON schema — no prose, no markdown fences, no trailing text:\n{schema}"
    )));
    (params, Some(schema))
}

/// Validate the final response before exposing it as structured output. Schema
/// requests are fail-closed: invalid, truncated, or non-JSON responses never
/// escape as successful completions.
fn enforce_schema_on_response(
    response: &mut ProviderResponse,
    schema: &serde_json::Value,
) -> Result<()> {
    // Mid-agent-loop replies (tool calls) are not the final answer; the schema
    // constrains only the closing text response.
    if response
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Ok(());
    }
    let Some(value) = response
        .structured_output
        .clone()
        .or_else(|| parse_structured_output_from_text(&response.content))
    else {
        return Err(StructuredOutputError::ParsingFailed {
            reason: format!(
                "Alibaba returned no parseable JSON (finish_reason={:?})",
                response.finish_reason
            ),
        }
        .into());
    };
    let validator =
        jsonschema::validator_for(schema).map_err(|err| StructuredOutputError::InvalidSchema {
            reason: err.to_string(),
        })?;
    validator
        .validate(&value)
        .map_err(|err| StructuredOutputError::ValidationFailed {
            reason: format!("Alibaba response does not match requested schema: {err}"),
        })?;
    response.content = value.to_string();
    response.structured_output = Some(value);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = AlibabaProvider::new();
        assert!(provider.supports_model("qwen3.8-max"));
        assert!(provider.supports_model("qwen3-coder-plus"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_default_capabilities() {
        let provider = AlibabaProvider::new();
        assert_eq!(provider.name(), "alibaba");
        assert!(provider.supports_caching("qwen3.8-max"));
        // Structured output is native for supported Qwen families.
        assert!(provider.supports_structured_output("qwen3.8-max"));
        assert!(provider.enforces_response_schema("qwen3.8-max"));
        // qwen3.8-max carries a 1M context window via reference capabilities
        assert_eq!(provider.get_max_input_tokens("qwen3.8-max"), 1_000_000);
        // Verified live: 3.8-max/3.7-plus/3.6-flash take images and video,
        // 3.7-max rejects image parts outright
        assert!(provider.supports_vision("qwen3.8-max"));
        assert!(provider.supports_vision("qwen3.7-plus"));
        assert!(provider.supports_vision("qwen3.6-flash"));
        assert!(provider.supports_video("qwen3.8-max"));
        assert!(provider.supports_video("qwen3.6-flash"));
        assert!(!provider.supports_vision("qwen3.7-max"));
    }

    #[test]
    fn test_pricing_qwen() {
        let provider = AlibabaProvider::new();

        let p = provider.get_model_pricing("qwen3.8-max").unwrap();
        assert_eq!(p.input_price_per_1m, 2.00);
        assert_eq!(p.output_price_per_1m, 6.00);
        assert_eq!(p.cache_read_price_per_1m, 0.25);

        // Moving alias currently has a 50% promotion.
        let p = provider.get_model_pricing("qwen3.7-max").unwrap();
        assert_eq!(p.input_price_per_1m, 1.25);
        assert_eq!(p.output_price_per_1m, 3.75);

        // Dated snapshots retain list price.
        let p = provider
            .get_model_pricing("qwen3.7-max-2026-06-08")
            .unwrap();
        assert_eq!(p.input_price_per_1m, 2.50);
        assert_eq!(p.output_price_per_1m, 7.50);

        let p = provider.get_model_pricing("qwen3.6-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 0.25);
        assert_eq!(p.cache_read_price_per_1m, 0.05);

        // Dated aliases must resolve to their family
        let p = provider
            .get_model_pricing("deepseek-v4-flash-0731")
            .unwrap();
        assert_eq!(p.input_price_per_1m, 0.20);

        // Unversioned aliases must not shadow more specific entries
        let p = provider.get_model_pricing("qwen-plus-latest").unwrap();
        assert_eq!(p.input_price_per_1m, 0.40);
        assert_eq!(p.output_price_per_1m, 1.20);
    }

    #[test]
    fn test_pricing_falls_back_to_reference() {
        let provider = AlibabaProvider::new();
        let p = provider.get_model_pricing("qwen3-max-2026-01-25").unwrap();
        assert!(p.input_price_per_1m > 0.0);
    }

    #[test]
    fn test_qwen3_8_flash_pricing() {
        let provider = AlibabaProvider::new();
        let p = provider.get_model_pricing("qwen3.8-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 0.113);
        assert_eq!(p.output_price_per_1m, 0.382);
        assert_eq!(p.cache_write_price_per_1m, 0.113);
        assert_eq!(p.cache_read_price_per_1m, 0.0226);
    }

    #[test]
    fn test_cost_calculation() {
        let provider = AlibabaProvider::new();
        let pricing = provider.get_model_pricing("qwen3.8-max").unwrap();

        // 1M input + 500K output, no cache
        let cost = pricing.calculate_cost(1_000_000, 0, 0, 500_000);
        let expected = 2.00 + 0.5 * 6.00; // $5.00
        assert!((cost - expected).abs() < 0.001);

        // Cache hits must be cheaper than fresh input
        let cost_cached = pricing.calculate_cost(500_000, 0, 500_000, 500_000);
        assert!(cost_cached < cost);

        let long = calculate_local_usage_cost("qwen3.7-plus", 256_001, 0, 0, 100_000).unwrap();
        let expected_long = 256_001.0 / 1_000_000.0 * 0.96 + 0.1 * 3.84;
        assert!((long - expected_long).abs() < 0.001);

        let boundary = calculate_local_usage_cost("qwen3.7-plus", 256_000, 0, 0, 100_000).unwrap();
        let expected_boundary = 0.256 * 0.32 + 0.1 * 1.28;
        assert!((boundary - expected_boundary).abs() < 0.001);

        let snapshot_long =
            calculate_local_usage_cost("qwen3.7-plus-2026-05-26", 256_001, 0, 0, 100_000).unwrap();
        let expected_snapshot = 256_001.0 / 1_000_000.0 * 1.20 + 0.1 * 4.80;
        assert!((snapshot_long - expected_snapshot).abs() < 0.001);
    }

    use crate::llm::types::{ProviderExchange, ResponseMode, ToolCall};

    fn schema_request(model: &str, schema: serde_json::Value) -> ChatCompletionParams {
        let mut params =
            ChatCompletionParams::new(&[Message::user("analyze")], model, 0.3, 0.7, 20, 1024);
        params.response_format =
            Some(StructuredOutputRequest::json_schema(schema).with_strict_mode());
        params
    }

    fn test_response(content: &str, tool_calls: Option<Vec<ToolCall>>) -> ProviderResponse {
        ProviderResponse {
            content: content.to_string(),
            thinking: None,
            exchange: ProviderExchange::new(
                serde_json::json!({}),
                serde_json::json!({}),
                None,
                "alibaba",
            ),
            tool_calls,
            finish_reason: None,
            structured_output: None,
            id: None,
        }
    }

    #[test]
    fn test_prepare_schema_request_downgrades_deepseek() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"]
        });
        let (rewritten, requested) =
            prepare_schema_request(schema_request("deepseek-v4-flash-0731", schema.clone()));

        // Wire format is plain json_object — never json_schema.
        let wire = rewritten.response_format.expect("json mode on the wire");
        assert!(matches!(wire.format, OutputFormat::Json));
        assert!(wire.schema.is_none());

        // Schema rides in a trailing system message that mentions "JSON",
        // satisfying DashScope's json-in-prompt keyword check.
        let last = rewritten.messages.last().expect("schema system message");
        assert_eq!(last.role, "system");
        assert!(last.content.contains("JSON schema"));
        assert_eq!(requested.as_ref(), Some(&schema));
    }

    #[test]
    fn test_prepare_schema_request_keeps_native_qwen_schema() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": false
        });
        let (prepared, requested) =
            prepare_schema_request(schema_request("qwen3.8-flash", schema.clone()));

        let wire = prepared.response_format.expect("native schema on the wire");
        assert!(matches!(wire.format, OutputFormat::JsonSchema));
        assert_eq!(wire.schema.as_ref(), Some(&schema));
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(requested.as_ref(), Some(&schema));
    }

    #[test]
    fn test_downgrade_schema_request_passthrough() {
        // Plain JSON mode: untouched.
        let mut params =
            ChatCompletionParams::new(&[Message::user("hi")], "qwen3.8-max", 0.3, 0.7, 20, 1024);
        params.response_format = Some(StructuredOutputRequest::json());
        let (rewritten, requested) = prepare_schema_request(params);
        assert!(matches!(
            rewritten.response_format.unwrap().format,
            OutputFormat::Json
        ));
        assert_eq!(rewritten.messages.len(), 1);
        assert!(requested.is_none());

        // JsonSchema without a schema: untouched (nothing to inject).
        let mut params =
            ChatCompletionParams::new(&[Message::user("hi")], "qwen3.8-max", 0.3, 0.7, 20, 1024);
        params.response_format = Some(StructuredOutputRequest {
            format: OutputFormat::JsonSchema,
            mode: ResponseMode::Strict,
            schema: None,
        });
        let (rewritten, requested) = prepare_schema_request(params);
        assert!(matches!(
            rewritten.response_format.unwrap().format,
            OutputFormat::JsonSchema
        ));
        assert_eq!(rewritten.messages.len(), 1);
        assert!(requested.is_none());
    }

    #[test]
    fn test_enforce_schema_on_response() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
            "additionalProperties": false
        });

        // Valid JSON matching the schema → attached as structured_output.
        let mut response = test_response("{\"verdict\":\"dispatched\"}", None);
        enforce_schema_on_response(&mut response, &schema).unwrap();
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"verdict": "dispatched"}))
        );

        // Schema violation is fail-closed.
        let mut response = test_response("{\"wrong\": 1}", None);
        let err = enforce_schema_on_response(&mut response, &schema).unwrap_err();
        assert!(err.to_string().contains("does not match requested schema"));
        assert_eq!(response.structured_output, None);
        assert_eq!(response.content, "{\"wrong\": 1}");

        // Mid-loop tool-call replies are never rewritten.
        let mut response = test_response(
            "ignored",
            Some(vec![ToolCall {
                id: "c1".to_string(),
                name: "read".to_string(),
                arguments: serde_json::json!({}),
            }]),
        );
        enforce_schema_on_response(&mut response, &schema).unwrap();
        assert_eq!(response.structured_output, None);
        assert_eq!(response.content, "ignored");
    }

    #[test]
    fn test_native_schema_capability_is_model_specific() {
        let provider = AlibabaProvider::new();
        assert!(provider.enforces_response_schema("qwen3.8-flash"));
        assert!(provider.enforces_response_schema("qwen3.8-flash-2026-08-26"));
        assert!(provider.enforces_response_schema("qwen3.7-plus"));
        assert!(!provider.enforces_response_schema("deepseek-v4-flash-0731"));
        assert!(!provider.enforces_response_schema("qwen3.6-flash"));
    }
}
