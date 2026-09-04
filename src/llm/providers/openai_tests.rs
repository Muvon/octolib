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
use serial_test::serial;

#[test]
fn test_supported_sampling_params() {
    let provider = OpenAiProvider::new();

    // Models that should support temperature/top_p (but never top_k)
    let sp = provider.supported_sampling_params("gpt-4");
    assert!(sp.temperature);
    assert!(sp.top_p);
    assert!(!sp.top_k); // OpenAI never supports top_k

    let sp = provider.supported_sampling_params("gpt-4o");
    assert!(sp.temperature);
    assert!(sp.top_p);

    let sp = provider.supported_sampling_params("gpt-4o-mini");
    assert!(sp.temperature);

    let sp = provider.supported_sampling_params("chatgpt-4o-latest");
    assert!(sp.temperature);

    // Reasoning models should NOT support temperature/top_p
    let sp = provider.supported_sampling_params("o1");
    assert!(!sp.temperature);
    assert!(!sp.top_p);
    assert!(!sp.top_k);

    let sp = provider.supported_sampling_params("o1-preview");
    assert!(!sp.temperature);

    let sp = provider.supported_sampling_params("o3");
    assert!(!sp.temperature);

    let sp = provider.supported_sampling_params("o4");
    assert!(!sp.temperature);

    let sp = provider.supported_sampling_params("gpt-5");
    assert!(!sp.temperature);
    assert!(!sp.top_p);

    let sp = provider.supported_sampling_params("gpt-5-mini");
    assert!(!sp.temperature);

    let sp = provider.supported_sampling_params("gpt-5-nano");
    assert!(!sp.temperature);
}

#[test]
fn test_supports_model_gpt5() {
    let provider = OpenAiProvider::new();

    // GPT-5 models should be supported
    assert!(provider.supports_model("gpt-5"));
    assert!(provider.supports_model("gpt-5-2025-08-07"));
    assert!(provider.supports_model("gpt-5-mini"));
    assert!(provider.supports_model("gpt-5-mini-2025-08-07"));
    assert!(provider.supports_model("gpt-5-nano"));
    assert!(provider.supports_model("gpt-5-nano-2025-08-07"));
    assert!(provider.supports_model("gpt-5.5"));
    assert!(provider.supports_model("gpt-5.5-pro"));
    assert!(provider.supports_model("gpt-5.6"));
    assert!(provider.supports_model("gpt-5.6-sol"));
    assert!(provider.supports_model("gpt-5.6-terra"));
    assert!(provider.supports_model("gpt-5.6-luna"));
    assert!(provider.supports_model("gpt-5.2-codex"));
    assert!(provider.supports_model("gpt-5.3-codex"));
    assert!(provider.supports_model("gpt-5.2-chat-latest"));
    assert!(provider.supports_model("codex-mini-latest"));

    // Other models should still be supported
    assert!(provider.supports_model("gpt-4o"));
    assert!(provider.supports_model("gpt-audio-mini"));
    assert!(provider.supports_model("gpt-4"));
    assert!(provider.supports_model("gpt-3.5-turbo"));
    assert!(provider.supports_model("o1"));

    // Unsupported models
    assert!(!provider.supports_model("claude-3"));
    assert!(!provider.supports_model("llama-2"));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = OpenAiProvider::new();

    // Test uppercase
    assert!(provider.supports_model("GPT-5"));
    assert!(provider.supports_model("GPT-4O"));
    assert!(provider.supports_model("GPT-4"));
    // Test mixed case
    assert!(provider.supports_model("Gpt-5"));
    assert!(provider.supports_model("gPT-4o"));
    assert!(provider.supports_model("O1"));
    assert!(provider.supports_model("o3-mini"));
}

#[test]
fn test_get_max_input_tokens_gpt5() {
    let provider = OpenAiProvider::new();

    // GPT-5.6 models have a 1.05M context window.
    assert_eq!(provider.get_max_input_tokens("gpt-5.6"), 1_050_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.6-sol"), 1_050_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.6-terra"), 1_050_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.6-luna"), 1_050_000);

    // GPT-5.5 models have a 1.05M context window.
    assert_eq!(provider.get_max_input_tokens("gpt-5.5"), 1_050_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.5-pro"), 1_050_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.6-cyber"), 400_000);

    // GPT-5 models should have 400K context window
    assert_eq!(provider.get_max_input_tokens("gpt-5"), 400_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5-2025-08-07"), 400_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5-mini"), 400_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5-nano"), 400_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.2-codex"), 400_000);
    assert_eq!(provider.get_max_input_tokens("gpt-5.3-codex"), 400_000);
    assert_eq!(provider.get_max_input_tokens("codex-mini-latest"), 200_000);

    // Other models should maintain their existing limits
    assert_eq!(provider.get_max_input_tokens("gpt-4o"), 128_000);
    assert_eq!(provider.get_max_input_tokens("gpt-4"), 8_192);
    assert_eq!(provider.get_max_input_tokens("gpt-3.5-turbo"), 16_384);
}

#[test]
fn test_supports_vision() {
    let provider = OpenAiProvider::new();

    // Models that should support vision
    assert!(provider.supports_vision("gpt-4o"));
    assert!(provider.supports_vision("gpt-4o-mini"));
    assert!(provider.supports_vision("gpt-4o-2024-05-13"));
    assert!(provider.supports_vision("gpt-4-turbo"));
    assert!(provider.supports_vision("gpt-4-vision-preview"));
    assert!(provider.supports_vision("gpt-4.1"));
    assert!(provider.supports_vision("gpt-5-mini"));
    assert!(provider.supports_vision("gpt-5.2-codex"));
    assert!(provider.supports_vision("gpt-5.3-codex"));
    assert!(provider.supports_vision("codex-mini-latest"));
    assert!(provider.supports_vision("gpt-realtime"));

    // Models that should NOT support vision
    assert!(!provider.supports_vision("gpt-3.5-turbo"));
    assert!(!provider.supports_vision("gpt-4"));
    assert!(!provider.supports_vision("o1-preview"));
    assert!(!provider.supports_vision("o1-mini"));
    assert!(!provider.supports_vision("text-davinci-003"));
}

#[test]
#[serial]
fn test_oauth_token_priority() {
    let provider = OpenAiProvider::new();

    // Set OAuth tokens
    env::set_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV, "test-oauth-token");
    env::set_var(OPENAI_OAUTH_ACCOUNT_ID_ENV, "test-account-id");

    // get_api_key should return error when OAuth is set
    let result = provider.get_api_key();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("OAuth authentication"));

    // Clean up
    env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
    env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);
}

#[test]
#[serial]
fn test_api_key_fallback() {
    let provider = OpenAiProvider::new();

    // Remove OAuth tokens if set
    env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
    env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);

    // Set API key
    env::set_var(OPENAI_API_KEY_ENV, "test-api-key");

    // get_api_key should return the API key
    let result = provider.get_api_key();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "test-api-key");

    // Clean up
    env::remove_var(OPENAI_API_KEY_ENV);
}

#[test]
#[serial]
fn test_no_auth_error() {
    let provider = OpenAiProvider::new();

    // Remove all auth env vars
    env::remove_var(OPENAI_OAUTH_ACCESS_TOKEN_ENV);
    env::remove_var(OPENAI_OAUTH_ACCOUNT_ID_ENV);
    env::remove_var(OPENAI_API_KEY_ENV);

    // get_api_key should return error
    let result = provider.get_api_key();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("OPENAI_API_KEY") || error_msg.contains("OPENAI_OAUTH"));
}

#[test]
fn test_messages_to_input() {
    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
        Message {
            role: "user".to_string(),
            content: "Hello!".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
    ];

    let input = messages_to_input(&messages, None, false);
    assert_eq!(input.len(), 2);

    // First message - content is plain string
    let first = &input[0];
    assert_eq!(first["role"], "system");
    assert_eq!(first["content"], "You are a helpful assistant.");

    // Second message - content is plain string
    let second = &input[1];
    assert_eq!(second["role"], "user");
    assert_eq!(second["content"], "Hello!");
}

#[test]
fn test_messages_to_input_with_images() {
    let image_attachment = crate::llm::types::ImageAttachment {
        data: ImageData::Base64("fakebase64data".to_string()),
        media_type: "image/png".to_string(),
        source_type: crate::llm::types::SourceType::File(std::path::PathBuf::from("test.png")),
        dimensions: None,
        size_bytes: None,
    };

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is in this image?").with_images(vec![image_attachment]),
    ];

    let input = messages_to_input(&messages, None, false);
    assert_eq!(input.len(), 2);

    // System message remains a plain string.
    assert_eq!(input[0]["role"], "system");
    assert_eq!(input[0]["content"], "You are a helpful assistant.");

    // User message with an image is an array of typed parts.
    assert_eq!(input[1]["role"], "user");
    let content = &input[1]["content"];
    assert!(content.is_array());
    assert_eq!(content.as_array().unwrap().len(), 2);
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["text"], "What is in this image?");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(
        content[1]["image_url"],
        "data:image/png;base64,fakebase64data"
    );
}

#[test]
fn test_messages_to_input_with_image_url_and_cache() {
    let image_attachment = crate::llm::types::ImageAttachment {
        data: ImageData::Url("https://example.com/image.png".to_string()),
        media_type: "image/png".to_string(),
        source_type: crate::llm::types::SourceType::Url,
        dimensions: None,
        size_bytes: None,
    };

    let messages = vec![Message::user("Describe this")
        .with_images(vec![image_attachment])
        .with_cache_marker()];

    let input = messages_to_input(&messages, None, true);
    assert_eq!(input.len(), 1);

    let content = &input[0]["content"];
    assert_eq!(content.as_array().unwrap().len(), 2);
    assert_eq!(content[0]["type"], "input_text");
    assert_eq!(content[0]["prompt_cache_breakpoint"]["mode"], "explicit");
    assert_eq!(content[1]["type"], "input_image");
    assert_eq!(content[1]["image_url"], "https://example.com/image.png");
}

#[test]
fn test_ollama_compaction_tail_rebases_and_keeps_summary() {
    let mut old_openai = Message::assistant("Older OpenAI answer");
    old_openai.id = Some("resp_old".to_string());

    let mut summary = Message::assistant("Compacted Ollama conversation");
    summary.name = Some("plan_compression".to_string());
    summary.id = Some("chatcmpl-18".to_string());

    let messages = vec![
        Message::system("System instructions"),
        old_openai,
        summary,
        Message::user("Please finalize the task"),
    ];

    let previous_id = resolve_previous_response_id(&messages, None);
    assert_eq!(
        previous_id, None,
        "Ollama ids must start a fresh OpenAI chain"
    );

    let input = messages_to_input(&messages, previous_id.as_deref(), false);
    assert_eq!(input.len(), 4);
    assert_eq!(input[2]["role"], "assistant");
    assert_eq!(input[2]["content"], "Compacted Ollama conversation");
    assert_eq!(input[3]["role"], "user");
    assert_eq!(input[3]["content"], "Please finalize the task");
}

#[test]
fn test_latest_openai_response_id_continues_exact_turn() {
    let mut assistant = Message::assistant("OpenAI answer");
    assistant.id = Some("resp_latest".to_string());
    let messages = vec![assistant, Message::user("Follow-up")];

    let previous_id = resolve_previous_response_id(&messages, None);
    assert_eq!(previous_id.as_deref(), Some("resp_latest"));

    let input = messages_to_input(&messages, previous_id.as_deref(), false);
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"], "Follow-up");
}

#[test]
fn test_invalid_explicit_previous_id_forces_rebase() {
    let messages = vec![Message::user("Fresh input")];
    let previous_id = resolve_previous_response_id(&messages, Some("chatcmpl-18".to_string()));

    assert_eq!(previous_id, None);
    let input = messages_to_input(&messages, previous_id.as_deref(), false);
    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["content"], "Fresh input");
}

#[test]
fn test_gpt_5_6_explicit_cache_breakpoint_wire_shape() {
    let messages = vec![
        Message::system("Stable instructions").with_cache_marker(),
        Message::user("Variable request"),
    ];

    let input = messages_to_input(&messages, None, true);
    assert_eq!(count_explicit_cache_breakpoints(&input), 1);
    assert_eq!(input[0]["content"][0]["type"], "input_text");
    assert_eq!(input[0]["content"][0]["text"], "Stable instructions");
    assert_eq!(
        input[0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
    assert_eq!(input[1]["content"], "Variable request");

    let mut request_body = serde_json::json!({"input": input});
    apply_explicit_cache_options(&mut request_body, 1).unwrap();
    assert_eq!(request_body["prompt_cache_options"]["mode"], "explicit");
    assert_eq!(request_body["prompt_cache_options"]["ttl"], "30m");
}

#[test]
fn test_gpt_5_6_cached_assistant_uses_output_text() {
    let mut summary = Message::assistant("Compressed task summary");
    summary.name = Some("plan_compression".to_string());
    summary.cached = true;

    let input = messages_to_input(&[summary], None, true);

    assert_eq!(input[0]["role"], "assistant");
    assert_eq!(input[0]["content"][0]["type"], "output_text");
    assert_eq!(
        input[0]["content"][0]["prompt_cache_breakpoint"]["mode"],
        "explicit"
    );
}

#[test]
fn test_explicit_cache_breakpoint_limit() {
    let mut request_body = serde_json::json!({});
    let error = apply_explicit_cache_options(&mut request_body, 5).unwrap_err();
    assert!(error.to_string().contains("at most 4"));
}

#[test]
fn test_pre_gpt_5_6_keeps_cached_messages_in_legacy_shape() {
    let messages = vec![Message::system("Stable instructions").with_cache_marker()];
    let input = messages_to_input(&messages, None, false);

    assert_eq!(input[0]["content"], "Stable instructions");
    assert_eq!(count_explicit_cache_breakpoints(&input), 0);
}

#[test]
fn test_messages_to_input_with_tool_response() {
    // Scenario: Assistant made a tool call, we're sending the tool result back
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "What is the weather?".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{}"
                }
            }])),
            tool_call_id: None,
            name: None,
            thinking: None,
            id: Some("resp_abc123".to_string()),
        },
        Message {
            role: "tool".to_string(),
            content: "{\"temperature\": \"22C\", \"condition\": \"sunny\"}".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: Some("call_12345".to_string()),
            name: Some("get_weather".to_string()),
            thinking: None,
            id: None,
        },
    ];

    // When there are NEW tool responses after assistant, send only those tool outputs
    let input = messages_to_input(&messages, Some("resp_abc123"), false);
    assert_eq!(input.len(), 1); // Only the NEW tool response

    // Tool response uses function_call_output format
    let tool_output = &input[0];
    assert_eq!(tool_output["type"], "function_call_output");
    assert_eq!(tool_output["call_id"], "call_12345");
    assert_eq!(
        tool_output["output"],
        "{\"temperature\": \"22C\", \"condition\": \"sunny\"}"
    );
}

#[test]
fn test_messages_to_input_continuation_without_tools() {
    // Scenario: Continuing conversation without tool calls (like "what else you can do?")
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "run date in shell".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_old",
                "type": "function",
                "function": {
                    "name": "shell",
                    "arguments": "{\"command\": \"date\"}"
                }
            }])),
            tool_call_id: None,
            name: None,
            thinking: None,
            id: Some("resp_first".to_string()),
        },
        Message {
            role: "tool".to_string(),
            content: "Mon Jan 19 22:12:18 +07 2026".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: Some("call_old".to_string()),
            name: Some("shell".to_string()),
            thinking: None,
            id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "The current date is Mon Jan 19 22:12:18 +07 2026".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: Some("resp_second".to_string()),
        },
        Message {
            role: "user".to_string(),
            content: "what else you can do?".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
    ];

    // Should send only the NEW user message, NOT the old tool result
    let input = messages_to_input(&messages, Some("resp_second"), false);
    assert_eq!(input.len(), 1);

    // Should be the new user message
    let user_msg = &input[0];
    assert_eq!(user_msg["role"], "user");
    assert_eq!(user_msg["content"], "what else you can do?");
}

/// Regression: after a multi-turn cancel that leaves a tool_result without
/// a follow-up assistant, the next user prompt must be sent alongside the
/// tool_result. Previously the function returned only the tool_result and
/// dropped the user message, causing the model to "drift" off-track.
#[test]
fn test_messages_to_input_tool_result_then_user_after_cancel() {
    let messages = vec![
        Message {
            role: "user".to_string(),
            content: "What is the weather?".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
        Message {
            role: "assistant".to_string(),
            content: "".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: Some(serde_json::json!([{
                "id": "call_w",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"}
            }])),
            tool_call_id: None,
            name: None,
            thinking: None,
            id: Some("resp_x".to_string()),
        },
        Message {
            role: "tool".to_string(),
            content: "72°F sunny".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: Some("call_w".to_string()),
            name: Some("get_weather".to_string()),
            thinking: None,
            id: None,
        },
        // User retried after cancelling the follow-up assistant.
        Message {
            role: "user".to_string(),
            content: "Now write me a poem about it.".to_string(),
            timestamp: 0,
            images: None,
            videos: None,
            cached: false,
            cache_ttl: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            thinking: None,
            id: None,
        },
    ];

    let input = messages_to_input(&messages, Some("resp_x"), false);
    assert_eq!(
        input.len(),
        2,
        "tool_result and follow-up user must both be sent"
    );
    assert_eq!(input[0]["type"], "function_call_output");
    assert_eq!(input[0]["call_id"], "call_w");
    assert_eq!(input[1]["role"], "user");
    assert_eq!(input[1]["content"], "Now write me a poem about it.");
}

#[test]
fn test_codex_pricing() {
    // Test that codex models have pricing defined
    let cost = calculate_cost("gpt-5-codex", 1000, 500);
    assert!(cost.is_some());
    let cost_value = cost.unwrap();
    // Expected: (1000/1M * 1.25) + (500/1M * 10.0) = 0.00125 + 0.005 = 0.00625
    assert!((cost_value - 0.00625).abs() < 0.0000001);

    // Verify gpt-5.2-codex pricing path exists
    let cost_52 = calculate_cost("gpt-5.2-codex", 1000, 500);
    assert!(cost_52.is_some());
    let cost_52_value = cost_52.unwrap();
    // Expected: (1000/1M * 1.75) + (500/1M * 14.0) = 0.00175 + 0.007 = 0.00875
    assert!((cost_52_value - 0.00875).abs() < 0.0000001);

    // Verify gpt-5.3-codex pricing path exists
    let cost_53 = calculate_cost("gpt-5.3-codex", 1000, 500);
    assert!(cost_53.is_some());
    let cost_53_value = cost_53.unwrap();
    assert!((cost_53_value - 0.00875).abs() < 0.0000001);
}

#[test]
fn test_cache_pricing_for_gpt_5_2_codex() {
    // (regular 1000 * 1.75 + cached 1000 * 0.175 + output 500 * 14) / 1M
    let cost = calculate_cost_with_cache("gpt-5.2-codex", 1000, 0, 1000, 500).unwrap();
    assert!((cost - 0.008925).abs() < 0.0000001);
}

#[test]
fn test_gpt_5_6_pricing_and_alias() {
    let provider = OpenAiProvider::new();
    let cases = [
        ("gpt-5.6", 4.00, 20.00, 5.00, 0.40),
        ("gpt-5.6-sol", 4.00, 20.00, 5.00, 0.40),
        ("gpt-5.6-terra", 2.00, 12.00, 2.50, 0.20),
        ("gpt-5.6-luna", 0.20, 1.20, 0.25, 0.02),
    ];

    for (model, input, output, cache_write, cache_read) in cases {
        let pricing = provider.get_model_pricing(model).unwrap();
        assert_eq!(pricing.input_price_per_1m, input);
        assert_eq!(pricing.output_price_per_1m, output);
        assert_eq!(pricing.cache_write_price_per_1m, cache_write);
        assert_eq!(pricing.cache_read_price_per_1m, cache_read);

        let reference = crate::llm::reference_models::get_reference_pricing(model).unwrap();
        assert_eq!(reference.input_price_per_1m, pricing.input_price_per_1m);
        assert_eq!(reference.output_price_per_1m, pricing.output_price_per_1m);
        assert_eq!(
            reference.cache_write_price_per_1m,
            pricing.cache_write_price_per_1m
        );
        assert_eq!(
            reference.cache_read_price_per_1m,
            pricing.cache_read_price_per_1m
        );
    }
}

#[test]
fn test_gpt_5_6_long_context_and_cache_write_pricing() {
    // Standard tier: regular input + cache write + cache read + output.
    let standard =
        calculate_cost_with_cache("gpt-5.6-terra", 100_000, 50_000, 50_000, 10_000).unwrap();
    assert!((standard - 0.455).abs() < 0.0000001);

    // Above 272K total input: 2x every input/cache rate and 1.5x output.
    let long = calculate_cost_with_cache("gpt-5.6-terra", 200_000, 50_000, 50_001, 10_000).unwrap();
    assert!((long - 1.2500004).abs() < 0.0000001);
}

#[test]
fn test_gpt_6_astra() {
    let provider = OpenAiProvider::new();

    assert!(provider.supports_model("gpt-6-astra"));
    assert_eq!(provider.get_max_input_tokens("gpt-6-astra"), 1_050_000);
    assert!(provider.supports_vision("gpt-6-astra"));
    assert!(provider.supports_caching("gpt-6-astra"));
    assert!(
        !provider
            .supported_sampling_params("gpt-6-astra")
            .temperature
    );

    let pricing = provider.get_model_pricing("gpt-6-astra").unwrap();
    assert_eq!(pricing.input_price_per_1m, 10.00);
    assert_eq!(pricing.output_price_per_1m, 50.00);
    assert_eq!(pricing.cache_write_price_per_1m, 12.50);
    assert_eq!(pricing.cache_read_price_per_1m, 1.00);

    let reference = crate::llm::reference_models::get_reference_pricing("gpt-6-astra").unwrap();
    assert_eq!(reference.input_price_per_1m, pricing.input_price_per_1m);
    assert_eq!(reference.output_price_per_1m, pricing.output_price_per_1m);

    // Standard tier: 100K input at $10 + 10K output at $50.
    let standard = calculate_cost_with_cache("gpt-6-astra", 100_000, 0, 0, 10_000).unwrap();
    assert!((standard - 1.5).abs() < 0.0000001);

    // Above 272K total input: 2x input/cache rates and 1.5x output.
    let long = calculate_cost_with_cache("gpt-6-astra", 300_000, 0, 0, 10_000).unwrap();
    assert!((long - 6.75).abs() < 0.0000001);
}

#[test]
fn test_gpt_5_6_usage_deserializes_cache_writes() {
    let usage: ResponseUsage = serde_json::from_value(serde_json::json!({
        "input_tokens": 3_000,
        "output_tokens": 500,
        "total_tokens": 3_500,
        "input_tokens_details": {
            "cached_tokens": 1_000,
            "cache_write_tokens": 500
        }
    }))
    .unwrap();

    let details = usage.input_tokens_details.unwrap();
    assert_eq!(details.cached_tokens, 1_000);
    assert_eq!(details.cache_write_tokens, 500);
}
