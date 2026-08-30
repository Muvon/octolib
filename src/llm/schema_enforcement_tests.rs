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
use crate::llm::types::{ProviderExchange, StructuredOutputRequest, ToolCall};
use std::collections::VecDeque;
use std::sync::Mutex;

struct ScriptedProvider {
    responses: Mutex<VecDeque<ProviderResponse>>,
    requests: Mutex<Vec<ChatCompletionParams>>,
    enforces: bool,
}

impl ScriptedProvider {
    fn new(responses: Vec<ProviderResponse>, enforces: bool) -> Self {
        Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
            enforces,
        }
    }

    fn requests(&self) -> Vec<ChatCompletionParams> {
        self.requests.lock().unwrap().clone()
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

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        self.requests.lock().unwrap().push(params);
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

fn with_usage(
    mut response: ProviderResponse,
    input_tokens: u64,
    output_tokens: u64,
    cost: f64,
    request_time_ms: u64,
) -> ProviderResponse {
    response.exchange.usage = Some(TokenUsage {
        input_tokens,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens,
        reasoning_tokens: 0,
        total_tokens: input_tokens + output_tokens,
        cost: Some(cost),
        request_time_ms: Some(request_time_ms),
    });
    response
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
async fn successful_fallback_aggregates_all_attempt_usage() {
    let provider = ScriptedProvider::new(
        vec![
            with_usage(response_with_content("not json"), 10, 5, 0.10, 100),
            with_usage(
                response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": 4})),
                20,
                7,
                0.20,
                200,
            ),
        ],
        true,
    );
    let response = chat_completion_enforced(&provider, params_with_schema("model"))
        .await
        .unwrap();
    let usage = response.exchange.usage.unwrap();
    assert_eq!(usage.input_tokens, 30);
    assert_eq!(usage.output_tokens, 12);
    assert_eq!(usage.total_tokens, 42);
    assert_eq!(usage.request_time_ms, Some(300));
    assert!((usage.cost.unwrap() - 0.30).abs() < f64::EPSILON);
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
async fn gives_up_after_max_attempts_with_validation_error() {
    let bad =
        || response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": "nope"}));
    let provider = ScriptedProvider::new(vec![bad(), bad(), bad()], false);
    let params = params_with_schema("model");
    let err = chat_completion_enforced(&provider, params)
        .await
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("exhausted 3 structured-output attempts"));
}

#[tokio::test]
async fn gives_up_after_empty_attempts_with_parsing_error() {
    let empty = || response_with_content("");
    let provider = ScriptedProvider::new(vec![empty(), empty(), empty()], false);
    let params = params_with_schema("model");
    let err = chat_completion_enforced(&provider, params)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("without parseable output"));
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

#[tokio::test]
async fn client_tools_invalid_final_falls_back_to_synthetic_schema_tool() {
    let provider = ScriptedProvider::new(
        vec![
            response_with_content("not json"),
            response_with_tool_call(SYNTHETIC_TOOL_NAME, serde_json::json!({"answer": 4})),
        ],
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
    assert_eq!(
        response.structured_output,
        Some(serde_json::json!({"answer": 4}))
    );

    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].tools.as_ref().unwrap()[0].name, "client_tool");
    assert_eq!(
        requests[1].tools.as_ref().unwrap()[0].name,
        SYNTHETIC_TOOL_NAME
    );
    assert!(requests[1].response_format.is_none());
}

#[test]
fn response_validation_defers_real_tool_call_turns() {
    let response = response_with_tool_call("client_tool", serde_json::json!({}));
    assert!(response.content.is_empty());
    let validated = validate_response(response, &schema(), "scripted").unwrap();
    let call = &validated.tool_calls.unwrap()[0];
    assert_eq!(call.name, "client_tool");
    assert_eq!(call.arguments, serde_json::json!({}));
}
