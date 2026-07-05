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

//! Best-effort JSON-schema enforcement for providers that don't natively
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

use crate::llm::providers::shared::parse_structured_output_from_text;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, FunctionDefinition, Message, OutputFormat, ProviderResponse,
};
use anyhow::Result;

const SYNTHETIC_TOOL_NAME: &str = "emit_structured_response";
// ponytail: fixed retry ceiling (1 initial attempt + 2 self-corrections). Bump
// if real-world schemas need more rounds — not worth a config knob nobody has
// asked for yet.
const MAX_ATTEMPTS: u32 = 3;

/// Run a chat completion, forcing the response to conform to a requested JSON
/// schema even when `provider` doesn't natively guarantee it.
///
/// Transparent passthrough when no schema was requested or the client already
/// supplied its own tools (forcing a synthetic tool would shadow them, and a
/// model mid-agent-loop calling a real tool must not be mistaken for a failed
/// structured-output attempt). `AiProvider::enforces_response_schema` is treated
/// as an optimization hint: try the provider's native path first, but validate
/// the actual response before trusting it.
pub async fn chat_completion_enforced(
    provider: &dyn AiProvider,
    params: ChatCompletionParams,
) -> Result<ProviderResponse> {
    let wants_schema = params
        .response_format
        .as_ref()
        .map(|f| matches!(f.format, OutputFormat::JsonSchema) && f.schema.is_some())
        .unwrap_or(false);

    if !wants_schema || params.tools.is_some() {
        return provider.chat_completion(params).await;
    }

    let schema = params
        .response_format
        .as_ref()
        .and_then(|f| f.schema.clone())
        .expect("checked by wants_schema above");

    if provider.enforces_response_schema(&params.model) {
        let response = provider.chat_completion(params.clone()).await?;
        if let Some(value) = validate_candidate(&response, &schema)? {
            return Ok(finalize(response, value));
        }
        tracing::warn!(
            model = %params.model,
            provider = provider.name(),
            "provider claimed native schema enforcement but returned invalid or unparseable output; falling back to forced schema path"
        );
    }

    force_schema(provider, params, schema).await
}

async fn force_schema(
    provider: &dyn AiProvider,
    mut params: ChatCompletionParams,
    schema: serde_json::Value,
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
        let response = provider.chat_completion(params.clone()).await?;
        let Some(value) = extract_candidate(&response) else {
            if attempt == MAX_ATTEMPTS {
                tracing::warn!(
                    model = %params.model,
                    "structured-output fallback exhausted retries without a parseable response"
                );
                return Ok(response);
            }
            params.messages.push(Message::user(
                "You did not call the function. Call it now with the required JSON arguments.",
            ));
            continue;
        };

        match validator.validate(&value) {
            Ok(()) => return Ok(finalize(response, value)),
            Err(err) if attempt < MAX_ATTEMPTS => {
                params.messages.push(Message::user(&format!(
                    "Your arguments `{value}` do not match the schema: {err}. Call the function again with corrected arguments."
                )));
            }
            Err(err) => {
                tracing::warn!(
                    model = %params.model,
                    error = %err,
                    "structured-output fallback exhausted retries without schema-valid output"
                );
                return Ok(finalize(response, value));
            }
        }
    }
    unreachable!("loop always returns by the final attempt")
}

/// Pull the candidate structured-output value out of a response: prefer the
/// forced tool's arguments, then whatever the provider already parsed, then a
/// loose best-effort parse of the raw text (the model may have ignored the
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

/// Attach the (possibly best-effort) value as `structured_output`, mirror it
/// into `content` as compact JSON text, and drop the synthetic tool call so it
/// never leaks to the client as if it were a real tool invocation.
fn finalize(mut response: ProviderResponse, value: serde_json::Value) -> ProviderResponse {
    response.content = value.to_string();
    response.tool_calls = None;
    response.structured_output = Some(value);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{ProviderExchange, StructuredOutputRequest, ToolCall};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    struct ScriptedProvider {
        responses: Mutex<VecDeque<ProviderResponse>>,
        enforces: bool,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<ProviderResponse>, enforces: bool) -> Self {
            Self {
                responses: Mutex::new(responses.into()),
                enforces,
            }
        }
    }

    #[async_trait::async_trait]
    impl AiProvider for ScriptedProvider {
        fn name(&self) -> &str {
            "scripted"
        }

        fn supports_model(&self, _model: &str) -> bool {
            true
        }

        fn get_api_key(&self) -> Result<String> {
            Ok("test".to_string())
        }

        fn enforces_response_schema(&self, _model: &str) -> bool {
            self.enforces
        }

        async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("no scripted response left"))
        }
    }

    fn response_with_tool_call(name: &str, arguments: serde_json::Value) -> ProviderResponse {
        ProviderResponse {
            content: String::new(),
            thinking: None,
            exchange: ProviderExchange::new(
                serde_json::json!({}),
                serde_json::json!({}),
                None,
                "scripted",
            ),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                name: name.to_string(),
                arguments,
            }]),
            finish_reason: Some("tool_calls".to_string()),
            structured_output: None,
            id: None,
        }
    }

    fn response_with_content(content: &str) -> ProviderResponse {
        ProviderResponse {
            content: content.to_string(),
            thinking: None,
            exchange: ProviderExchange::new(
                serde_json::json!({}),
                serde_json::json!({}),
                None,
                "scripted",
            ),
            tool_calls: None,
            finish_reason: Some("stop".to_string()),
            structured_output: None,
            id: None,
        }
    }

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "integer" } },
            "required": ["answer"]
        })
    }

    fn params_with_schema(model: &str) -> ChatCompletionParams {
        ChatCompletionParams::new(&[Message::user("what is 2+2?")], model, 0.7, 1.0, 50, 100)
            .with_structured_output(StructuredOutputRequest::json_schema(schema()))
    }

    #[tokio::test]
    async fn accepts_valid_native_schema_response() {
        let provider = ScriptedProvider::new(vec![response_with_content(r#"{"answer":4}"#)], true);
        let params = params_with_schema("model");
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"answer": 4}))
        );
        assert_eq!(response.content, r#"{"answer":4}"#);
    }

    #[tokio::test]
    async fn falls_back_when_native_enforcer_returns_unparseable_output() {
        let provider = ScriptedProvider::new(
            vec![
                response_with_content("not json"),
                response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": 4})),
            ],
            true,
        );
        let params = params_with_schema("model");
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"answer": 4}))
        );
        assert_eq!(response.content, r#"{"answer":4}"#);
    }

    #[tokio::test]
    async fn extracts_and_validates_forced_tool_call_on_first_try() {
        let provider = ScriptedProvider::new(
            vec![response_with_tool_call(
                SYNTHETIC_TOOL_NAME,
                serde_json::json!({"answer": 4}),
            )],
            false,
        );
        let params = params_with_schema("model");
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"answer": 4}))
        );
        assert!(
            response.tool_calls.is_none(),
            "synthetic tool call must not leak to the caller"
        );
    }

    #[tokio::test]
    async fn retries_on_schema_mismatch_then_succeeds() {
        let provider = ScriptedProvider::new(
            vec![
                response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": "four"})),
                response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": 4})),
            ],
            false,
        );
        let params = params_with_schema("model");
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"answer": 4}))
        );
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts_but_returns_best_effort() {
        let bad =
            || response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": "nope"}));
        let provider = ScriptedProvider::new(vec![bad(), bad(), bad()], false);
        let params = params_with_schema("model");
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        // Best-effort: still surfaces the last (invalid) candidate rather than nothing.
        assert_eq!(
            response.structured_output,
            Some(serde_json::json!({"answer": "nope"}))
        );
    }

    #[tokio::test]
    async fn passthrough_when_client_already_supplies_tools() {
        let provider = ScriptedProvider::new(
            vec![response_with_tool_call(
                "client_tool",
                serde_json::json!({"x": 1}),
            )],
            false,
        );
        let mut params = params_with_schema("model");
        params.tools = Some(vec![FunctionDefinition {
            name: "client_tool".to_string(),
            description: String::new(),
            parameters: serde_json::json!({}),
            cache_control: None,
        }]);
        let response = chat_completion_enforced(&provider, params).await.unwrap();
        assert_eq!(response.tool_calls.unwrap()[0].name, "client_tool");
    }
}
