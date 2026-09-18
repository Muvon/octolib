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
//! format: the schema becomes a single tool's parameters and the model is made
//! to call it. This is the only enforcement technique available to a stateless
//! proxy with no access to any provider's decode loop (unlike self-hosted
//! grammar-constrained decoding, which requires running the inference engine
//! itself).
//!
//! Every path makes exactly ONE upstream call and fails closed when the answer
//! does not match the schema, the way the OpenRouter provider already does. A
//! re-ask re-sends the whole prompt and re-runs the model's reasoning: on prod,
//! GLM-5.3 Flash spent thousands of reasoning tokens per supervisor call, so a
//! correction round doubled 40-60s calls past the hub's proxy timeout while
//! billing every discarded attempt.

use crate::errors::StructuredOutputError;
use crate::llm::providers::shared::parse_structured_output_from_text;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, FunctionDefinition, Message, OutputFormat, ProviderResponse, ToolChoice,
};
use anyhow::Result;

const SYNTHETIC_TOOL_NAME: &str = "emit_structured_response";

/// Run a chat completion, forcing the response to conform to a requested JSON
/// schema even when `provider` doesn't natively guarantee it.
///
/// Transparent passthrough when no schema was requested. Client-supplied tool
/// calls remain untouched; only a final tool-free answer is held to the schema.
/// The provider's native-enforcement declaration picks how the schema is sent,
/// but actual output is always validated before it is trusted.
pub async fn chat_completion_enforced(
    provider: &dyn AiProvider,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let Some(schema) = params
        .response_format
        .as_ref()
        .filter(|f| matches!(f.format, OutputFormat::JsonSchema))
        .and_then(|f| f.schema.clone())
    else {
        return provider.chat_completion(params).await;
    };

    let has_client_tools = params.tools.as_ref().is_some_and(|tools| !tools.is_empty());
    if has_client_tools || provider.enforces_response_schema(&params.model) {
        let response = provider.chat_completion(params).await?;
        return validate_response(response, &schema, provider.name());
    }

    force_schema(provider, params, &schema).await
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
    checked(response, schema, provider)
}

async fn force_schema(
    provider: &dyn AiProvider,
    mut params: ChatCompletionParams,
    schema: &serde_json::Value,
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

    let response = if provider.supports_required_tool_choice(&params.model) {
        provider
            .chat_completion_with_tool_choice(params, ToolChoice::Required)
            .await?
    } else {
        provider.chat_completion(params).await?
    };
    checked(response, schema, provider.name())
}

/// Hold one response to the schema: extract the answer, validate it, and fail
/// closed with the reason. Never asks the model again.
fn checked(
    response: ProviderResponse,
    schema: &serde_json::Value,
    provider: &str,
) -> Result<ProviderResponse> {
    let Some(value) = extract_candidate(&response) else {
        tracing::warn!(
            provider = provider,
            finish_reason = ?response.finish_reason,
            thinking_len = response.thinking.as_ref().map(|t| t.content.len()).unwrap_or(0),
            content_len = response.content.len(),
            content_head = %response.content.chars().take(400).collect::<String>(),
            "RAW-FAIL: no parseable structured output, final output captured"
        );
        return Err(StructuredOutputError::ParsingFailed {
            reason: format!(
                "provider '{provider}' returned no parseable structured output (finish_reason={:?})",
                response.finish_reason
            ),
        }
        .into());
    };

    let validator = jsonschema::validator_for(schema)
        .map_err(|e| anyhow::anyhow!("invalid JSON schema in response_format: {e}"))?;
    if let Err(err) = validator.validate(&value) {
        tracing::warn!(
            provider = provider,
            error = %err,
            "structured output did not match requested JSON schema"
        );
        return Err(StructuredOutputError::ValidationFailed {
            reason: format!("provider '{provider}' returned structured output that does not match the schema: {err}"),
        }
        .into());
    }
    Ok(finalize(response, value))
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
