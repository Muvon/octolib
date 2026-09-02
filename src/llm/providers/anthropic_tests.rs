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
#[serial]
fn test_oauth_token_priority() {
    let provider = AnthropicProvider::new();

    // Set OAuth token
    env::set_var(ANTHROPIC_OAUTH_TOKEN_ENV, "test-oauth-token");

    // get_api_key should return error when OAuth is set
    let result = provider.get_api_key();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("OAuth authentication"));

    // Clean up
    env::remove_var(ANTHROPIC_OAUTH_TOKEN_ENV);
}

#[test]
#[serial]
fn test_api_key_fallback() {
    let provider = AnthropicProvider::new();

    // Remove OAuth token if set
    env::remove_var(ANTHROPIC_OAUTH_TOKEN_ENV);

    // Set API key
    env::set_var(ANTHROPIC_API_KEY_ENV, "test-api-key");

    // get_api_key should return the API key
    let result = provider.get_api_key();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "test-api-key");

    // Clean up
    env::remove_var(ANTHROPIC_API_KEY_ENV);
}

#[test]
#[serial]
fn test_no_auth_error() {
    let provider = AnthropicProvider::new();

    // Remove both OAuth and API key
    env::remove_var(ANTHROPIC_OAUTH_TOKEN_ENV);
    env::remove_var(ANTHROPIC_API_KEY_ENV);

    // get_api_key should return error
    let result = provider.get_api_key();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("ANTHROPIC_API_KEY") || error_msg.contains("ANTHROPIC_OAUTH_TOKEN"));
}

#[test]
fn test_supports_model_case_insensitive() {
    let provider = AnthropicProvider::new();

    // Test lowercase (already working)
    assert!(provider.supports_model("claude-3-haiku"));
    assert!(provider.supports_model("claude-3-5-sonnet"));
    assert!(provider.supports_model("claude-sonnet-4-6"));
    assert!(provider.supports_model("claude-opus-4-7"));

    // Test uppercase
    assert!(provider.supports_model("CLAUDE-3-HAIKU"));
    assert!(provider.supports_model("CLAUDE-3-5-SONNET"));
    assert!(provider.supports_model("CLAUDE-SONNET-4-6"));
    assert!(provider.supports_model("CLAUDE-OPUS-4-7"));
    // Test mixed case
    assert!(provider.supports_model("ClaUde-3-Haiku"));
    assert!(provider.supports_model("CLAUDE-3-7-sonnet"));
}

#[test]
fn test_supports_vision_case_insensitive() {
    let provider = AnthropicProvider::new();

    assert!(provider.supports_vision("claude-sonnet-5"));
    assert!(provider.supports_vision("claude-opus-4-6"));
    assert!(provider.supports_vision("claude-sonnet-4-6"));
    assert!(provider.supports_vision("claude-haiku-4-5"));

    // Test lowercase
    assert!(provider.supports_vision("claude-3-haiku"));
    assert!(provider.supports_vision("claude-3-5-sonnet"));

    // Test uppercase
    assert!(provider.supports_vision("CLAUDE-3-HAIKU"));
    assert!(provider.supports_vision("CLAUDE-3-5-SONNET"));
    // Test mixed case
    assert!(provider.supports_vision("ClaUde-3-7"));
}

#[test]
fn test_get_model_pricing() {
    let provider = AnthropicProvider::new();

    // Test Sonnet 4.6 pricing
    let pricing = provider.get_model_pricing("claude-sonnet-4-6").unwrap();
    assert_eq!(pricing.input_price_per_1m, 3.0);
    assert_eq!(pricing.output_price_per_1m, 15.0);
    assert_eq!(pricing.cache_write_price_per_1m, 3.75);
    assert_eq!(pricing.cache_read_price_per_1m, 0.30);

    // Test Opus 4.7 pricing
    let pricing = provider.get_model_pricing("claude-opus-4-7").unwrap();
    assert_eq!(pricing.input_price_per_1m, 5.0);
    assert_eq!(pricing.output_price_per_1m, 25.0);
    assert_eq!(pricing.cache_write_price_per_1m, 6.25);
    assert_eq!(pricing.cache_read_price_per_1m, 0.50);

    // Test Opus 4.7 context window
    assert_eq!(provider.get_max_input_tokens("claude-opus-4-7"), 1_000_000);

    // Test Opus 4.7 does not support any sampling parameters
    let sp = provider.supported_sampling_params("claude-opus-4-7");
    assert_eq!(sp, SamplingSupport::NONE);

    // Test Opus 4.1 supports temperature+top_k but not top_p
    let sp = provider.supported_sampling_params("claude-opus-4-1");
    assert!(sp.temperature);
    assert!(!sp.top_p);
    assert!(sp.top_k);

    // Test older Claude 3 supports all sampling params
    let sp = provider.supported_sampling_params("claude-3-haiku");
    assert!(sp.temperature);
    assert!(sp.top_p);
    assert!(sp.top_k);

    // Test Sonnet 4 pricing (from the pricing table)
    let pricing = provider.get_model_pricing("claude-sonnet-4").unwrap();
    assert_eq!(pricing.input_price_per_1m, 3.0);
    assert_eq!(pricing.output_price_per_1m, 15.0);
    assert_eq!(pricing.cache_write_price_per_1m, 3.75); // from pricing table
    assert_eq!(pricing.cache_read_price_per_1m, 0.30); // from pricing table

    // Test Haiku 3 pricing
    let pricing = provider.get_model_pricing("claude-3-haiku").unwrap();
    assert_eq!(pricing.input_price_per_1m, 0.25);
    assert_eq!(pricing.output_price_per_1m, 1.25);
    assert_eq!(pricing.cache_write_price_per_1m, 0.30); // from pricing table
    assert_eq!(pricing.cache_read_price_per_1m, 0.03); // from pricing table

    // Test case insensitive
    let pricing = provider.get_model_pricing("CLAUDE-SONNET-4").unwrap();
    assert_eq!(pricing.input_price_per_1m, 3.0);

    // Test unknown model
    assert!(provider.get_model_pricing("unknown-model").is_none());
}

#[test]
fn test_opus_5() {
    let provider = AnthropicProvider::new();
    let model = "claude-opus-5";

    assert!(provider.supports_model(model));

    let pricing = provider.get_model_pricing(model).unwrap();
    assert_eq!(pricing.input_price_per_1m, 5.0);
    assert_eq!(pricing.output_price_per_1m, 25.0);
    assert_eq!(pricing.cache_write_price_per_1m, 6.25);
    assert_eq!(pricing.cache_read_price_per_1m, 0.50);

    assert_eq!(provider.get_max_input_tokens(model), 1_000_000);
    assert!(provider.supports_vision(model));
    assert_eq!(
        provider.supported_sampling_params(model),
        SamplingSupport::NONE
    );

    assert!(THINKING_MODELS.contains(&"opus-5"));
    assert!(ADAPTIVE_THINKING_MODELS.contains(&"opus-5"));
    assert!(EFFORT_PARAM_MODELS.contains(&"opus-5"));
    assert_eq!(effort_value(model, ReasoningEffort::XHigh, true), "xhigh");
    assert_eq!(effort_value(model, ReasoningEffort::Max, true), "max");
}

#[test]
fn test_fable_5() {
    let provider = AnthropicProvider::new();

    let model = "claude-fable-5";
    assert!(provider.supports_model(model));

    // $10/$50, cache write 1.25x = 12.50, cache read 0.1x = 1.00
    let pricing = provider.get_model_pricing(model).unwrap();
    assert_eq!(pricing.input_price_per_1m, 10.0);
    assert_eq!(pricing.output_price_per_1m, 50.0);
    assert_eq!(pricing.cache_write_price_per_1m, 12.50);
    assert_eq!(pricing.cache_read_price_per_1m, 1.00);

    // 1M context window and vision (ID lacks claude-3/claude-4 substrings)
    assert_eq!(provider.get_max_input_tokens(model), 1_000_000);
    assert!(provider.supports_vision(model));

    // Mythos-class is adaptive-only: rejects ALL sampling parameters
    assert_eq!(
        provider.supported_sampling_params(model),
        SamplingSupport::NONE
    );
}

#[test]
fn test_fable_5_1() {
    let provider = AnthropicProvider::new();

    let model = "claude-fable-5-1";
    assert!(provider.supports_model(model));

    // Cache read is 0.025x input ($0.25), not the 0.1x every other Claude
    // uses — proves the 5.1 row is matched ahead of the "claude-fable-5" row.
    let pricing = provider.get_model_pricing(model).unwrap();
    assert_eq!(pricing.input_price_per_1m, 10.0);
    assert_eq!(pricing.output_price_per_1m, 50.0);
    assert_eq!(pricing.cache_write_price_per_1m, 12.50);
    assert_eq!(pricing.cache_read_price_per_1m, 0.25);

    assert_eq!(provider.get_max_input_tokens(model), 1_000_000);
    assert_eq!(
        provider.supported_sampling_params(model),
        SamplingSupport::NONE
    );
    assert_eq!(effort_value(model, ReasoningEffort::XHigh, true), "xhigh");
}

#[test]
fn test_sonnet_5() {
    let provider = AnthropicProvider::new();

    let model = "claude-sonnet-5";
    assert!(provider.supports_model(model));

    // Adaptive-only like the Opus 5 tier: manual thinking and every sampling
    // parameter return a 400.
    assert_eq!(
        provider.supported_sampling_params(model),
        SamplingSupport::NONE
    );
    assert!(THINKING_MODELS.contains(&"sonnet-5"));
    assert!(ADAPTIVE_THINKING_MODELS.contains(&"sonnet-5"));
    assert!(ADAPTIVE_ONLY_MODELS.contains(&"sonnet-5"));
    assert!(EFFORT_PARAM_MODELS.contains(&"sonnet-5"));
    assert_eq!(effort_value(model, ReasoningEffort::XHigh, true), "xhigh");
}

#[test]
fn test_mythos_5() {
    let provider = AnthropicProvider::new();

    // Project Glasswing only, but same pricing/capabilities as Fable 5
    let model = "claude-mythos-5";
    assert!(provider.supports_model(model));

    // $10/$50, cache write 1.25x = 12.50, cache read 0.1x = 1.00
    let pricing = provider.get_model_pricing(model).unwrap();
    assert_eq!(pricing.input_price_per_1m, 10.0);
    assert_eq!(pricing.output_price_per_1m, 50.0);
    assert_eq!(pricing.cache_write_price_per_1m, 12.50);
    assert_eq!(pricing.cache_read_price_per_1m, 1.00);

    // 1M context window and vision (ID lacks claude-3/claude-4 substrings)
    assert_eq!(provider.get_max_input_tokens(model), 1_000_000);
    assert!(provider.supports_vision(model));

    // Adaptive-only: rejects ALL sampling parameters
    assert_eq!(
        provider.supported_sampling_params(model),
        SamplingSupport::NONE
    );
}

#[test]
fn parallel_tool_results_share_one_user_message() {
    let mut assistant = Message::assistant("");
    assistant.tool_calls = Some(serde_json::json!([
        {"id": "toolu_a", "name": "first", "arguments": {}, "meta": null},
        {"id": "toolu_b", "name": "second", "arguments": {}, "meta": null}
    ]));

    let messages = vec![
        assistant,
        Message::tool("result A", "toolu_a", "first"),
        Message::tool("result B", "toolu_b", "second"),
    ];

    let converted = convert_messages(&messages);
    assert_eq!(converted.len(), 2);
    assert_eq!(converted[0].role, "assistant");
    assert_eq!(converted[1].role, "user");

    let user_content = serde_json::to_value(&converted[1].content).unwrap();
    let blocks = user_content.as_array().unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0]["type"], "tool_result");
    assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
    assert_eq!(blocks[1]["type"], "tool_result");
    assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
}

#[test]
fn tool_results_merge_following_user_hint() {
    let messages = vec![
        Message::tool("result A", "toolu_a", "first"),
        Message::tool("result B", "toolu_b", "second"),
        Message::user("Please use those results."),
    ];

    let converted = convert_messages(&messages);
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].role, "user");

    let content = serde_json::to_value(&converted[0].content).unwrap();
    let blocks = content.as_array().unwrap();
    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0]["tool_use_id"], "toolu_a");
    assert_eq!(blocks[1]["tool_use_id"], "toolu_b");
    assert_eq!(blocks[2]["type"], "text");
    assert_eq!(blocks[2]["text"], "Please use those results.");
}
