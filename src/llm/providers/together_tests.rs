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

#[test]
fn test_supports_model() {
    let provider = TogetherProvider::new();
    assert!(provider.supports_model("meta-llama/Llama-3.3-70B-Instruct-Turbo"));
    assert!(provider.supports_model("moonshotai/Kimi-K2.5"));
    assert!(provider.supports_model("any-model-name"));
    assert!(!provider.supports_model(""));
}

#[test]
fn current_serverless_pricing_uses_together_rates() {
    let provider = TogetherProvider::new();

    let qwen = provider
        .get_model_pricing("Qwen/Qwen3.8-2.4T-A95B")
        .unwrap();
    assert_eq!(qwen.input_price_per_1m, 2.00);
    assert_eq!(qwen.cache_read_price_per_1m, 0.25);
    assert_eq!(qwen.output_price_per_1m, 6.00);

    let qwen_max = provider.get_model_pricing("Qwen/Qwen3.7-Max").unwrap();
    assert_eq!(qwen_max.input_price_per_1m, 2.50);
    assert_eq!(qwen_max.cache_read_price_per_1m, 0.25);
    assert_eq!(qwen_max.output_price_per_1m, 7.50);

    let deepseek = provider
        .get_model_pricing("deepseek-ai/DeepSeek-V4-Flash-0731")
        .unwrap();
    assert_eq!(deepseek.input_price_per_1m, 0.14);
    assert_eq!(deepseek.cache_read_price_per_1m, 0.03);
    assert_eq!(deepseek.output_price_per_1m, 0.28);
    assert_eq!(
        provider.get_max_input_tokens("deepseek-ai/DeepSeek-V4-Flash-0731"),
        1_000_000
    );

    let gemma = provider.get_model_pricing("google/gemma-4-31B-it").unwrap();
    assert_eq!(gemma.input_price_per_1m, 0.39);
    assert_eq!(gemma.output_price_per_1m, 0.97);
    assert_eq!(
        provider.get_max_input_tokens("google/gemma-4-31B-it"),
        262_144
    );
}

#[test]
fn newly_listed_serverless_models_use_together_rates() {
    let provider = TogetherProvider::new();

    let glm = provider.get_model_pricing("zai-org/GLM-5.3").unwrap();
    assert_eq!(glm.input_price_per_1m, 1.40);
    assert_eq!(glm.cache_read_price_per_1m, 0.26);
    assert_eq!(glm.output_price_per_1m, 4.40);
    assert_eq!(provider.get_max_input_tokens("zai-org/GLM-5.3"), 1_048_575);

    let glm_flash = provider.get_model_pricing("zai-org/GLM-5.3-Flash").unwrap();
    assert_eq!(glm_flash.input_price_per_1m, 0.15);
    assert_eq!(glm_flash.cache_read_price_per_1m, 0.03);
    assert_eq!(glm_flash.output_price_per_1m, 0.50);
    assert_eq!(
        provider.get_max_input_tokens("zai-org/GLM-5.3-Flash"),
        1_048_575
    );

    let qwen_flash = provider.get_model_pricing("Qwen/Qwen3.8-Flash").unwrap();
    assert_eq!(qwen_flash.input_price_per_1m, 0.15);
    assert_eq!(qwen_flash.output_price_per_1m, 0.47);
    assert_eq!(
        provider.get_max_input_tokens("Qwen/Qwen3.8-Flash"),
        1_000_000
    );

    let deepseek = provider
        .get_model_pricing("deepseek-ai/DeepSeek-V4.1-Flash")
        .unwrap();
    assert_eq!(deepseek.input_price_per_1m, 0.30);
    assert_eq!(deepseek.cache_read_price_per_1m, 0.006);
    assert_eq!(deepseek.output_price_per_1m, 1.20);
    assert_eq!(
        provider.get_max_input_tokens("deepseek-ai/DeepSeek-V4.1-Flash"),
        1_000_000
    );

    let gpt_oss = provider.get_model_pricing("openai/gpt-oss-120b").unwrap();
    assert_eq!(gpt_oss.input_price_per_1m, 0.15);
    assert_eq!(gpt_oss.output_price_per_1m, 0.60);
    assert_eq!(
        provider.get_max_input_tokens("openai/gpt-oss-120b"),
        131_072
    );

    let qwen_9b = provider.get_model_pricing("Qwen/Qwen3.5-9B").unwrap();
    assert_eq!(qwen_9b.input_price_per_1m, 0.17);
    assert_eq!(qwen_9b.output_price_per_1m, 0.25);
    assert_eq!(provider.get_max_input_tokens("Qwen/Qwen3.5-9B"), 262_144);

    let llama = provider
        .get_model_pricing("meta-llama/Llama-3.3-70B-Instruct-Turbo")
        .unwrap();
    assert_eq!(llama.input_price_per_1m, 1.04);
    assert_eq!(llama.output_price_per_1m, 1.04);
    assert_eq!(
        provider.get_max_input_tokens("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        131_072
    );
}

#[test]
fn retired_serverless_models_leave_together_table() {
    assert!(together_model_pricing("moonshotai/Kimi-K2.7-Code").is_none());
    assert!(together_model_context("moonshotai/Kimi-K2.7-Code").is_none());
    assert!(together_model_pricing("deepseek-ai/DeepSeek-V4-Pro").is_none());
    assert!(together_model_context("deepseek-ai/DeepSeek-V4-Pro").is_none());
    assert!(together_model_pricing("deepseek-ai/DeepSeek-V4-Pro-0813").is_some());
}

#[test]
fn test_extract_thinking_reasoning_field() {
    let (thinking, content) =
        extract_thinking("the answer is 42", Some("let me think...".to_string()));
    assert_eq!(thinking.unwrap().content, "let me think...");
    assert_eq!(content, "the answer is 42"); // content untouched
}

#[test]
fn test_extract_thinking_think_tags() {
    let (thinking, content) = extract_thinking("<think>internal</think>visible answer", None);
    assert_eq!(thinking.unwrap().content, "internal");
    assert_eq!(content, "visible answer"); // tags stripped
}

#[test]
fn test_merge_sse_stream_content_and_usage() {
    let body = "\
data: {\"id\":\"abc\",\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\
data: {\"choices\":[{\"delta\":{\"reasoning\":\"think \"}}]}\n\
data: {\"choices\":[{\"delta\":{\"reasoning\":\"more\"}}]}\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\
data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15,\"cached_tokens\":4}}\n\
data: [DONE]\n";
    let m = merge_sse_stream(body);
    assert!(m.parsed_any);
    assert_eq!(m.id.as_deref(), Some("abc"));
    assert_eq!(m.content, "Hello world");
    assert_eq!(m.reasoning, "think more");
    assert_eq!(m.finish_reason.as_deref(), Some("stop"));
    let u = m.usage.unwrap();
    assert_eq!(u.prompt_tokens, Some(10));
    assert_eq!(u.cached_tokens, Some(4));
}

#[test]
fn test_merge_sse_stream_tool_call_fragments() {
    // id+name arrive once; arguments stream as fragments and must concatenate.
    let body = "\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\
data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ty\\\":\\\"NYC\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\
data: [DONE]\n";
    let m = merge_sse_stream(body);
    assert_eq!(m.tool_calls.len(), 1);
    assert_eq!(m.tool_calls[0].id, "call_1");
    assert_eq!(m.tool_calls[0].name, "get_weather");
    assert_eq!(m.tool_calls[0].arguments, "{\"city\":\"NYC\"}");
    assert_eq!(m.finish_reason.as_deref(), Some("tool_calls"));
}

#[test]
fn test_merge_sse_stream_ignores_garbage() {
    let m = merge_sse_stream(": keep-alive\n\ndata: not-json\n");
    assert!(!m.parsed_any);
    assert_eq!(m.content, "");
}

#[test]
fn test_extract_thinking_none() {
    let (thinking, content) = extract_thinking("plain answer", None);
    assert!(thinking.is_none());
    assert_eq!(content, "plain answer");
    // empty/whitespace reasoning is ignored
    let (thinking, _) = extract_thinking("x", Some("  ".to_string()));
    assert!(thinking.is_none());
}
