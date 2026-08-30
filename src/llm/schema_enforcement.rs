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

//! Fail-closed JSON-schema enforcement for providers that don't natively
//! guarantee schema-conformant structured output.
//!
//! Mirrors the "forced tool call" technique used by Instructor/LangChain for
//! providers that expose tool calling but no native `json_schema` response
//! format: the schema becomes a single tool's parameters, the model is made
//! to call it, and the arguments are validated with a bounded retry-on-failure
//! loop. This is the only enforcement technique available to a stateless proxy
//! with no access to any provider's decode loop (unlike self-hosted
//! grammar-constrained decoding, which requires running the inference engine
//! itself).

use crate::errors::StructuredOutputError;
use crate::llm::providers::shared::parse_structured_output_from_text;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, FunctionDefinition, Message, OutputFormat, ProviderResponse, TokenUsage,
    ToolChoice,
};
use anyhow::Result;

const SYNTHETIC_TOOL_NAME: &str = "emit_structured_response";
// ponytail: fixed retry ceiling (1 initial attempt + 2 self-corrections). Bump
// if real-world schemas need more rounds — not worth a config knob nobody has
// asked for yet.
const MAX_ATTEMPTS: u32 = 3;

#[derive(Default)]
struct UsageAccumulator {
    usage: Option<TokenUsage>,
    complete: bool,
}

impl UsageAccumulator {
    fn new() -> Self {
        Self {
            usage: None,
            complete: true,
        }
    }

    fn add(&mut self, response: &ProviderResponse) {
        let Some(next) = response.exchange.usage.as_ref() else {
            self.complete = false;
            return;
        };
        let Some(total) = self.usage.as_mut() else {
            self.usage = Some(next.clone());
            return;
        };

        total.input_tokens = total.input_tokens.saturating_add(next.input_tokens);
        total.cache_read_tokens = total
            .cache_read_tokens
            .saturating_add(next.cache_read_tokens);
        total.cache_write_tokens = total
            .cache_write_tokens
            .saturating_add(next.cache_write_tokens);
        total.output_tokens = total.output_tokens.saturating_add(next.output_tokens);
        total.reasoning_tokens = total.reasoning_tokens.saturating_add(next.reasoning_tokens);
        total.total_tokens = total.total_tokens.saturating_add(next.total_tokens);
        total.cost = match (total.cost, next.cost) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        };
        total.request_time_ms = match (total.request_time_ms, next.request_time_ms) {
            (Some(left), Some(right)) => Some(left.saturating_add(right)),
            _ => None,
        };
    }

    fn apply(self, response: &mut ProviderResponse) {
        // TokenUsage has no partial/completeness marker. Returning None is more
        // honest than exposing totals that omit an upstream attempt.
        response.exchange.usage = self.complete.then_some(self.usage).flatten();
    }
}

/// Run a chat completion, forcing the response to conform to a requested JSON
/// schema even when `provider` doesn't natively guarantee it.
///
/// Transparent passthrough when no schema was requested. Client-supplied tool
/// calls remain untouched. Once that path produces a final tool-free response,
/// it is validated and, if necessary, retried with only the synthetic schema
/// tool. The provider's native-enforcement declaration is only an optimization
/// hint; actual output is always validated before it is trusted.
pub async fn chat_completion_enforced(
    provider: &dyn AiProvider,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let wants_schema = params
        .response_format
        .as_ref()
        .map(|f| matches!(f.format, OutputFormat::JsonSchema) && f.schema.is_some())
        .unwrap_or(false);

    if !wants_schema {
        return provider.chat_completion(params).await;
    }

    let schema = params
        .response_format
        .as_ref()
        .and_then(|f| f.schema.clone())
        .expect("checked by wants_schema above");

    let has_client_tools = params.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    if has_client_tools {
        let response = provider.chat_completion(params.clone()).await?;
        if response
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            return Ok(response);
        }

        let mut usage = UsageAccumulator::new();
        usage.add(&response);
        if let Some(value) = validate_candidate(&response, &schema)? {
            let mut response = finalize(response, value);
            usage.apply(&mut response);
            return Ok(response);
        }

        tracing::warn!(
            model = %params.model,
            provider = provider.name(),
            finish_reason = ?response.finish_reason,
            thinking_len = response.thinking.as_ref().map(|t| t.content.len()).unwrap_or(0),
            content_len = response.content.len(),
            content_head = %response.content.chars().take(400).collect::<String>(),
            "client-tool path returned invalid final structured output; falling back to forced schema path"
        );
        return force_schema(provider, params, schema, usage).await;
    }

    let mut usage = UsageAccumulator::new();
    if provider.enforces_response_schema(&params.model) {
        let response = provider.chat_completion(params.clone()).await?;
        usage.add(&response);
        if let Some(value) = validate_candidate(&response, &schema)? {
            let mut response = finalize(response, value);
            usage.apply(&mut response);
            return Ok(response);
        }
        tracing::warn!(
            model = %params.model,
            provider = provider.name(),
            "provider claimed native schema enforcement but returned invalid or unparseable output; falling back to forced schema path"
        );
    }

    force_schema(provider, params, schema, usage).await
}

pub(crate) fn validate_response(
    response: ProviderResponse,
    schema: &serde_json::Value,
    provider: &str,
) -> Result<ProviderResponse> {
    // A tool call is an intermediate agent turn, not the schema-constrained
    // final answer. Its text content is normally empty and must pass through.
    if response
        .tool_calls
        .as_ref()
        .is_some_and(|calls| !calls.is_empty())
    {
        return Ok(response);
    }
    match validate_candidate(&response, schema)? {
        Some(value) => Ok(finalize(response, value)),
        None => {
            tracing::warn!(
                provider = provider,
                finish_reason = ?response.finish_reason,
                thinking_len = response.thinking.as_ref().map(|t| t.content.len()).unwrap_or(0),
                content_len = response.content.len(),
                content_head = %response.content.chars().take(400).collect::<String>(),
                "RAW-FAIL: schema validation failed, final output captured"
            );
            Err(StructuredOutputError::ValidationFailed {
                reason: format!(
                    "provider '{provider}' returned invalid or unparseable final structured output"
                ),
            }
            .into())
        }
    }
}

async fn force_schema(
    provider: &dyn AiProvider,
    mut params: ChatCompletionParams,
    schema: serde_json::Value,
    mut usage: UsageAccumulator,
) -> Result<ProviderResponse> {
    params.tools = Some(vec![FunctionDefinition {
        name: SYNTHETIC_TOOL_NAME.to_string(),
        description: "Return the final answer as arguments to this function. Arguments MUST conform exactly to the provided JSON schema.".to_string(),
        parameters: schema.clone(),
        cache_control: None,
    }]);
    params.response_format = None;
    params.messages.push(Message::system(&format!(
        "Call the `{SYNTHETIC_TOOL_NAME}` function with your final answer — never respond in plain text. Its arguments must conform exactly to this JSON schema:\n{schema}"
    )));

    let validator = jsonschema::validator_for(&schema)
        .map_err(|e| anyhow::anyhow!("invalid JSON schema in response_format: {e}"))?;

    for attempt in 1..=MAX_ATTEMPTS {
        let response = if provider.supports_required_tool_choice(&params.model) {
            provider
                .chat_completion_with_tool_choice(params.clone(), ToolChoice::Required)
                .await?
        } else {
            provider.chat_completion(params.clone()).await?
        };
        usage.add(&response);
        let Some(value) = extract_candidate(&response) else {
            if attempt == MAX_ATTEMPTS {
                return Err(StructuredOutputError::ParsingFailed {
                    reason: format!(
                        "model '{}' exhausted {MAX_ATTEMPTS} structured-output attempts without parseable output (finish_reason={:?})",
                        params.model, response.finish_reason
                    ),
                }
                .into());
            }
            params.messages.push(Message::user(
                "You did not call the function. Call it now with the required JSON arguments.",
            ));
            continue;
        };

        match validator.validate(&value) {
            Ok(()) => {
                let mut response = finalize(response, value);
                usage.apply(&mut response);
                return Ok(response);
            }
            Err(err) if attempt < MAX_ATTEMPTS => {
                params.messages.push(Message::user(&format!(
                    "Your arguments `{value}` do not match the schema: {err}. Call the function again with corrected arguments."
                )));
            }
            Err(err) => {
                return Err(StructuredOutputError::ValidationFailed {
                    reason: format!(
                        "model '{}' exhausted {MAX_ATTEMPTS} structured-output attempts: {err}",
                        params.model
                    ),
                }
                .into());
            }
        }
    }
    unreachable!("loop always returns by the final attempt")
}

/// Pull the candidate structured-output value out of a response: prefer the
/// forced tool's arguments, then whatever the provider already parsed, then a
/// loose parse of the raw text (the model may have ignored the
/// tool and just answered in prose).
fn extract_candidate(response: &ProviderResponse) -> Option<serde_json::Value> {
    response
        .tool_calls
        .as_ref()
        .and_then(|calls| calls.iter().find(|c| c.name == SYNTHETIC_TOOL_NAME))
        .map(|c| c.arguments.clone())
        .or_else(|| response.structured_output.clone())
        .or_else(|| parse_structured_output_from_text(&response.content))
}

fn validate_candidate(
    response: &ProviderResponse,
    schema: &serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let Some(value) = extract_candidate(response) else {
        return Ok(None);
    };

    let validator = jsonschema::validator_for(schema)
        .map_err(|e| anyhow::anyhow!("invalid JSON schema in response_format: {e}"))?;
    Ok(match validator.validate(&value) {
        Ok(()) => Some(value),
        Err(err) => {
            tracing::warn!(
                error = %err,
                "native structured output did not match requested JSON schema"
            );
            None
        }
    })
}

/// Attach the validated value as `structured_output`, mirror it
/// into `content` as compact JSON text, and drop the synthetic tool call so it
/// never leaks to the client as if it were a real tool invocation.
fn finalize(mut response: ProviderResponse, value: serde_json::Value) -> ProviderResponse {
    response.content = value.to_string();
    response.tool_calls = None;
    response.structured_output = Some(value);
    response
}

#[cfg(test)]
#[path = "schema_enforcement_tests.rs"]
mod tests;
