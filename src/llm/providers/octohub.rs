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

//! OctoHub provider — standalone Responses API client.
//!
//! Speaks the OctoHub `/v1/completions` endpoint which uses the same format
//! as the OpenAI Responses API: `input` array, `previous_response_id` for
//! multi-turn, `instructions` for system messages, and `output` array in
//! responses.
//!
use super::shared;
use crate::llm::retry;
use crate::llm::traits::AiProvider;
use crate::llm::types::{
    ChatCompletionParams, Message, ProviderExchange, ProviderResponse, SamplingSupport,
    ThinkingBlock, TokenUsage, ToolCall,
};
use anyhow::Result;
use serde::Deserialize;

const OCTOHUB_API_KEY_ENV: &str = "OCTOHUB_API_KEY";
const OCTOHUB_API_URL_ENV: &str = "OCTOHUB_API_URL";
const OCTOHUB_DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080";

/// OctoHub provider — routes through an OctoHub proxy server using the
/// Responses API format (`/v1/completions`).
#[derive(Debug, Clone)]
pub struct OctoHubProvider;

impl Default for OctoHubProvider {
    fn default() -> Self {
        Self::new()
    }
}
impl OctoHubProvider {
    pub fn new() -> Self {
        Self
    }

    fn base_url() -> String {
        std::env::var(OCTOHUB_API_URL_ENV).unwrap_or_else(|_| OCTOHUB_DEFAULT_BASE_URL.to_string())
    }

    fn api_key() -> Option<String> {
        std::env::var(OCTOHUB_API_KEY_ENV).ok()
    }
}

#[async_trait::async_trait]
impl AiProvider for OctoHubProvider {
    fn name(&self) -> &str {
        "octohub"
    }

    fn supported_sampling_params(&self, _model: &str) -> SamplingSupport {
        // OctoHub uses OpenAI-compatible API — supports temperature and top_p, not top_k.
        SamplingSupport::TEMPERATURE_AND_TOP_P
    }

    /// OctoHub accepts any model — it routes to the appropriate provider.
    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // OctoHub API key is optional (server may run without auth)
        Ok(Self::api_key().unwrap_or_default())
    }

    fn supports_caching(&self, _model: &str) -> bool {
        true // Depends on underlying provider
    }

    // supports_vision, supports_video, supports_structured_output, get_max_input_tokens
    // are resolved via reference capabilities (trait defaults)

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let base_url = Self::base_url();
        let api_url = format!("{}/v1/completions", base_url.trim_end_matches('/'));

        // Resolve previous_response_id: explicit param > last message with id
        let previous_id = params.previous_id.clone().or_else(|| {
            params
                .messages
                .iter()
                .rev()
                .find(|m| m.id.is_some())
                .and_then(|m| m.id.clone())
        });

        // Extract system instructions from messages
        let instructions = extract_instructions(&params.messages);

        // Convert messages to input array
        let input_array = messages_to_input(&params.messages, previous_id.is_some());

        // Build request body
        let mut request_body = serde_json::json!({
            "model": params.model,
            "input": input_array,
        });

        if let Some(instr) = instructions {
            request_body["instructions"] = instr;
        }

        if let Some(ref prev_id) = previous_id {
            request_body["previous_completion_id"] = serde_json::json!(prev_id);
        }

        // Apply sampling parameters based on model support
        let sampling = self.effective_sampling_params(&params);
        if let Some(temp) = sampling.temperature {
            request_body["temperature"] = serde_json::json!(temp);
        }
        if let Some(top_p) = sampling.top_p {
            request_body["top_p"] = serde_json::json!(top_p);
        }
        // Note: OctoHub doesn't support top_k

        if params.max_tokens > 0 {
            request_body["max_output_tokens"] = serde_json::json!(params.max_tokens);
        }

        // Add tools
        if let Some(tools) = &params.tools {
            if !tools.is_empty() {
                let mut sorted_tools = tools.clone();
                sorted_tools.sort_by(|a, b| a.name.cmp(&b.name));

                let tool_defs: Vec<serde_json::Value> = sorted_tools
                    .iter()
                    .map(|f| {
                        let mut tool = serde_json::json!({
                            "type": "function",
                            "name": f.name,
                            "description": f.description,
                            "parameters": f.parameters
                        });
                        if let Some(ref cc) = f.cache_control {
                            tool["cache_control"] = cc.clone();
                        }
                        tool
                    })
                    .collect();

                request_body["tools"] = serde_json::json!(tool_defs);
            }
        }

        // Add structured output format if specified
        if let Some(response_format) = &params.response_format {
            match &response_format.format {
                crate::llm::types::OutputFormat::Json => {
                    request_body["text"] = serde_json::json!({
                        "format": { "type": "json_object" }
                    });
                }
                crate::llm::types::OutputFormat::JsonSchema => {
                    if let Some(schema) = &response_format.schema {
                        let mut format_obj = serde_json::json!({
                            "type": "json_schema",
                            "name": "response_schema",
                            "schema": schema
                        });

                        if matches!(
                            response_format.mode,
                            crate::llm::types::ResponseMode::Strict
                        ) {
                            format_obj["strict"] = serde_json::json!(true);
                        }

                        request_body["text"] = serde_json::json!({
                            "format": format_obj
                        });
                    }
                }
            }
        }

        // Execute request with retry
        let api_key = Self::api_key();
        let start_time = std::time::Instant::now();
        let request_timeout = params.request_timeout;

        let response = retry::retry_with_exponential_backoff(
            || {
                let client = shared::http_client();
                let api_key = api_key.clone();
                let api_url = api_url.clone();
                let request_body = request_body.clone();
                Box::pin(async move {
                    let mut req = client
                        .post(&api_url)
                        .header("Content-Type", "application/json");

                    if let Some(ref key) = api_key {
                        if !key.is_empty() {
                            req = req.header("Authorization", format!("Bearer {}", key));
                        }
                    }

                    let response =
                        shared::apply_request_timeout(req.json(&request_body), request_timeout)
                            .send()
                            .await
                            .map_err(anyhow::Error::from)?;

                    if retry::is_retryable_status(response.status().as_u16()) {
                        let status = response.status();
                        let error_text = response.text().await.unwrap_or_default();
                        return Err(anyhow::anyhow!(
                            "OctoHub API error {}: {}",
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
            || crate::errors::ProviderError::Cancelled.into(),
            |e| {
                matches!(
                    e.downcast_ref::<crate::errors::ProviderError>(),
                    Some(crate::errors::ProviderError::Cancelled)
                )
            },
            |e: &anyhow::Error| shared::is_connection_error(e),
        )
        .await?;

        let request_time_ms = start_time.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = retry::cancellable(
                async { response.text().await.map_err(anyhow::Error::from) },
                params.cancellation_token.as_ref(),
                || crate::errors::ProviderError::Cancelled.into(),
            )
            .await?;
            return Err(anyhow::anyhow!(
                "OctoHub API error {}: {}",
                status,
                error_text
            ));
        }

        let response_text = retry::cancellable(
            async { response.text().await.map_err(anyhow::Error::from) },
            params.cancellation_token.as_ref(),
            || crate::errors::ProviderError::Cancelled.into(),
        )
        .await?;

        let api_response: OctoHubResponse = serde_json::from_str(&response_text)?;

        // Parse output items
        let mut content = String::new();
        let mut tool_calls: Option<Vec<ToolCall>> = None;
        let mut reasoning_content: Option<String> = None;

        for output in &api_response.output {
            match output.output_type.as_str() {
                "message" => {
                    if let Some(content_array) = &output.content {
                        for item in content_array {
                            if item.content_type == "output_text" {
                                if let Some(text) = &item.text {
                                    if !content.is_empty() {
                                        content.push('\n');
                                    }
                                    content.push_str(text);
                                }
                            }
                        }
                    }
                }
                "function_call" => {
                    if let (Some(name), Some(args), Some(call_id)) =
                        (&output.name, &output.arguments, &output.call_id)
                    {
                        let arguments: serde_json::Value = if args.is_string() {
                            serde_json::from_str(args.as_str().unwrap_or("{}"))
                                .unwrap_or(serde_json::json!({}))
                        } else {
                            args.clone()
                        };

                        let new_call = ToolCall {
                            id: call_id.clone(),
                            name: name.clone(),
                            arguments,
                        };

                        if let Some(ref mut calls) = tool_calls {
                            calls.push(new_call);
                        } else {
                            tool_calls = Some(vec![new_call]);
                        }
                    }
                }
                "reasoning" => {
                    if let Some(content_array) = &output.content {
                        for item in content_array {
                            if item.content_type == "output_text" {
                                if let Some(text) = &item.text {
                                    reasoning_content = Some(text.clone());
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Build usage from OctoHub's flat usage format
        let usage = &api_response.usage;
        let cache_read_tokens = usage.cache_read_tokens.unwrap_or(0);
        let cache_write_tokens = usage.cache_write_tokens.unwrap_or(0);
        let input_tokens_clean = usage.input_tokens.saturating_sub(cache_read_tokens);
        let reasoning_tokens = usage.reasoning_tokens.unwrap_or(0);

        let thinking = reasoning_content.map(|rc| ThinkingBlock {
            content: rc,
            tokens: reasoning_tokens,
        });

        let token_usage = TokenUsage {
            input_tokens: input_tokens_clean,
            cache_read_tokens,
            cache_write_tokens,
            output_tokens: usage.output_tokens,
            reasoning_tokens,
            total_tokens: usage.total_tokens + reasoning_tokens,
            cost: usage.cost,
            request_time_ms: Some(usage.request_time_ms.unwrap_or(request_time_ms)),
        };

        // Build response JSON and store tool_calls in unified format
        let mut response_json: serde_json::Value = serde_json::from_str(&response_text)?;
        if let Some(ref tc) = tool_calls {
            shared::set_response_tool_calls(&mut response_json, tc, None);
        }

        let exchange =
            ProviderExchange::new(request_body, response_json, Some(token_usage), "octohub");

        let structured_output = shared::parse_structured_output_from_text(&content);

        Ok(ProviderResponse {
            content,
            thinking,
            exchange,
            tool_calls,
            finish_reason: None, // Responses API doesn't have finish_reason
            structured_output,
            id: api_response.id,
        })
    }
}

// ---------------------------------------------------------------------------
// Message conversion
// ---------------------------------------------------------------------------

/// Extract system instructions from messages. Returns a JSON value that is
/// either a plain string or a structured array with `cache_control` when any
/// system message is marked as cached.
fn extract_instructions(messages: &[Message]) -> Option<serde_json::Value> {
    let system_msgs: Vec<&Message> = messages.iter().filter(|m| m.role == "system").collect();
    if system_msgs.is_empty() {
        return None;
    }

    let text = system_msgs
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");

    let any_cached = system_msgs.iter().any(|m| m.cached);
    if any_cached {
        let ttl = system_msgs.iter().find_map(|m| m.cache_ttl.as_deref());
        let mut block = serde_json::json!([{
            "type": "text",
            "text": text,
        }]);
        block[0]["cache_control"] = shared::ephemeral_cache_control_with_ttl(ttl);
        Some(block)
    } else {
        Some(serde_json::json!(text))
    }
}

/// Convert conversation messages to OctoHub `input` array.
///
/// When `has_previous_response` is true, only sends NEW messages after the
/// last assistant message (tool results or user follow-ups). The server
/// reconstructs full history from `previous_response_id`.
///
/// When false, sends all non-system messages as the initial input.
fn messages_to_input(messages: &[Message], has_previous_response: bool) -> Vec<serde_json::Value> {
    if has_previous_response {
        // Find the last assistant message with an ID
        let last_assistant_idx = messages
            .iter()
            .enumerate()
            .rev()
            .find(|(_, m)| m.role == "assistant" && m.id.is_some())
            .map(|(idx, _)| idx);

        if let Some(assistant_idx) = last_assistant_idx {
            // Collect new tool results after the last assistant message
            let new_tool_results: Vec<_> = messages
                .iter()
                .skip(assistant_idx + 1)
                .filter_map(|msg| {
                    if msg.role == "tool" {
                        let call_id = msg.tool_call_id.clone().unwrap_or_default();
                        Some(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": msg.content
                        }))
                    } else {
                        None
                    }
                })
                .collect();

            if !new_tool_results.is_empty() {
                return new_tool_results;
            }
        }

        // No tool results — send new user messages after the last assistant
        messages
            .iter()
            .skip(last_assistant_idx.map(|idx| idx + 1).unwrap_or(0))
            .filter_map(|msg| match msg.role.as_str() {
                "user" => Some(user_message_value(msg)),
                _ => None,
            })
            .collect()
    } else {
        // Initial request: send all non-system messages
        messages
            .iter()
            .filter_map(|msg| match msg.role.as_str() {
                "user" => Some(user_message_value(msg)),
                _ => None,
            })
            .collect()
    }
}

/// Build a single user message JSON value, attaching `cache_control` when the
/// message is marked as cached.
fn user_message_value(msg: &Message) -> serde_json::Value {
    let content: serde_json::Value = if msg.cached {
        let mut block = serde_json::json!([{
            "type": "input_text",
            "text": msg.content,
        }]);
        block[0]["cache_control"] =
            shared::ephemeral_cache_control_with_ttl(msg.cache_ttl.as_deref());
        block
    } else {
        serde_json::json!(msg.content)
    };

    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": content
    })
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct OctoHubResponse {
    #[serde(default)]
    id: Option<String>,
    output: Vec<OutputItem>,
    usage: OctoHubUsage,
}

#[derive(Deserialize, Debug)]
struct OutputItem {
    #[serde(rename = "type")]
    output_type: String,
    #[serde(default)]
    call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    content: Option<Vec<OutputContent>>,
}

#[derive(Deserialize, Debug)]
struct OutputContent {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OctoHubUsage {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    #[serde(default)]
    cache_read_tokens: Option<u64>,
    #[serde(default)]
    cache_write_tokens: Option<u64>,
    #[serde(default)]
    reasoning_tokens: Option<u64>,
    #[serde(default)]
    cost: Option<f64>,
    #[serde(default)]
    request_time_ms: Option<u64>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_name() {
        let provider = OctoHubProvider::new();
        assert_eq!(provider.name(), "octohub");
    }

    #[test]
    fn test_supports_any_model() {
        let provider = OctoHubProvider::new();
        assert!(provider.supports_model("gpt-4o"));
        assert!(provider.supports_model("claude-sonnet-4-20250514"));
        assert!(provider.supports_model("any-model-name"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_capabilities() {
        let provider = OctoHubProvider::new();
        assert!(provider.supports_caching("any"));
        // Vision/video/structured_output now resolved via reference capabilities
        assert!(provider.supports_vision("llava:latest"));
        assert!(!provider.supports_vision("llama-3.1-8b"));
        assert!(provider.supports_video("qwen-2.5-vl-72b"));
        assert!(!provider.supports_video("llama-3.1-8b"));
        assert!(provider.supports_structured_output("llama-3.1-8b"));
        assert_eq!(provider.get_max_input_tokens("llama-3.1-8b"), 131_072);
    }

    #[test]
    fn test_extract_instructions_single() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello")];
        let instr = extract_instructions(&messages).unwrap();
        assert_eq!(instr, serde_json::json!("You are helpful."));
    }

    #[test]
    fn test_extract_instructions_none() {
        let messages = vec![Message::user("Hello")];
        assert_eq!(extract_instructions(&messages), None);
    }

    #[test]
    fn test_extract_instructions_cached() {
        let messages = vec![
            Message::system("You are helpful.").with_cache_marker(),
            Message::user("Hello"),
        ];
        let instr = extract_instructions(&messages).unwrap();
        let arr = instr.as_array().expect("should be array when cached");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "You are helpful.");
        assert_eq!(arr[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_user_message_value_plain() {
        let msg = Message::user("Hello");
        let val = user_message_value(&msg);
        assert_eq!(val["content"], "Hello");
    }

    #[test]
    fn test_user_message_value_cached() {
        let msg = Message::user("Hello").with_cache_marker();
        let val = user_message_value(&msg);
        let content = val["content"].as_array().expect("should be array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "Hello");
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_messages_to_input_initial() {
        let messages = vec![Message::system("You are helpful."), Message::user("Hello!")];

        let input = messages_to_input(&messages, false);
        // System messages go to instructions, not input
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "Hello!");
    }

    #[test]
    fn test_messages_to_input_continuation_user() {
        let mut assistant = Message::assistant("Rust is a systems language.");
        assistant.id = Some("resp_abc".to_string());
        let messages = vec![
            Message::user("What is Rust?"),
            assistant,
            Message::user("Tell me more."),
        ];

        let input = messages_to_input(&messages, true);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "Tell me more.");
    }

    #[test]
    fn test_messages_to_input_tool_results() {
        let mut assistant_msg = Message::assistant("");
        assistant_msg.tool_calls = Some(serde_json::json!([{
            "id": "call_xyz",
            "name": "get_weather",
            "arguments": {"location": "NYC"}
        }]));
        assistant_msg.id = Some("resp_123".to_string());
        let messages = vec![
            Message::user("What is the weather?"),
            assistant_msg,
            Message::tool("72°F sunny", "call_xyz", "get_weather"),
        ];

        let input = messages_to_input(&messages, true);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_xyz");
        assert_eq!(input[0]["output"], "72°F sunny");
    }

    #[test]
    fn test_parse_response() {
        let json = r#"{
            "id": "resp_abc123",
            "object": "response",
            "model": "gpt-4o",
            "output": [
                {
                    "type": "message",
                    "id": "msg_001",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "Hello!"}
                    ]
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cost": 0.0001,
                "request_time_ms": 500
            },
            "created_at": 1700000000
        }"#;

        let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, Some("resp_abc123".to_string()));
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output[0].output_type, "message");
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert_eq!(resp.usage.cost, Some(0.0001));
        assert_eq!(resp.usage.request_time_ms, Some(500));
    }

    #[test]
    fn test_parse_function_call_response() {
        let json = r#"{
            "id": "resp_xyz",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "{\"location\":\"NYC\"}"
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30
            }
        }"#;

        let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output[0].output_type, "function_call");
        assert_eq!(resp.output[0].name, Some("get_weather".to_string()));
        assert_eq!(resp.output[0].call_id, Some("call_abc".to_string()));
    }

    #[test]
    fn test_parse_usage_with_cache() {
        let json = r#"{
            "id": "resp_cache",
            "output": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "cache_read_tokens": 80,
                "cache_write_tokens": 20,
                "cost": 0.005,
                "request_time_ms": 200
            }
        }"#;

        let resp: OctoHubResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.usage.cache_read_tokens, Some(80));
        assert_eq!(resp.usage.cache_write_tokens, Some(20));
        assert_eq!(resp.usage.cost, Some(0.005));
    }
}
