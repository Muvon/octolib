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
use crate::test_support::env::EnvGuard;

#[test]
fn test_supports_model() {
	let provider = TogetherProvider::new();
	assert!(provider.supports_model("meta-llama/Llama-3.3-70B-Instruct-Turbo"));
	assert!(provider.supports_model("moonshotai/Kimi-K2.5"));
	assert!(provider.supports_model("any-model-name"));
	assert!(!provider.supports_model(""));
}

#[test]
fn only_live_verified_models_enforce_a_response_schema() {
	let provider = TogetherProvider::new();
	for model in [
		"deepseek-ai/DeepSeek-V4.1-Flash",
		"thinkingmachines/Inkling",
		"MiniMaxAI/MiniMax-M3",
		"openai/gpt-oss-120b",
	] {
		assert!(provider.enforces_response_schema(model), "{model}");
	}
	for model in ["moonshotai/Kimi-K3", "Qwen/Qwen3.7-Plus", "unknown/model"] {
		assert!(!provider.enforces_response_schema(model), "{model}");
	}
}

#[test]
fn current_serverless_pricing_uses_together_rates() {
	let provider = TogetherProvider::new();

	let qwen = provider.get_model_pricing("Qwen/Qwen3.8-2.4T-A95B").unwrap();
	assert_eq!(qwen.input_price_per_1m, 2.00);
	assert_eq!(qwen.cache_read_price_per_1m, 0.25);
	assert_eq!(qwen.output_price_per_1m, 6.00);

	let qwen_max = provider.get_model_pricing("Qwen/Qwen3.7-Max").unwrap();
	assert_eq!(qwen_max.input_price_per_1m, 1.50);
	assert_eq!(qwen_max.cache_read_price_per_1m, 0.30);
	assert_eq!(qwen_max.output_price_per_1m, 4.50);

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
	assert_eq!(provider.get_max_input_tokens("google/gemma-4-31B-it"), 262_144);
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
	assert_eq!(provider.get_max_input_tokens("zai-org/GLM-5.3-Flash"), 1_048_575);

	let qwen_flash = provider.get_model_pricing("Qwen/Qwen3.8-Flash").unwrap();
	assert_eq!(qwen_flash.input_price_per_1m, 0.09);
	assert_eq!(qwen_flash.output_price_per_1m, 0.28);
	assert_eq!(provider.get_max_input_tokens("Qwen/Qwen3.8-Flash"), 1_000_000);

	let deepseek = provider.get_model_pricing("deepseek-ai/DeepSeek-V4.1-Flash").unwrap();
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
	assert_eq!(provider.get_max_input_tokens("openai/gpt-oss-120b"), 131_072);

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

	let muse = provider.get_model_pricing("meta-models/Muse-Glimmer-30B").unwrap();
	assert_eq!(muse.input_price_per_1m, 0.35);
	assert_eq!(muse.cache_read_price_per_1m, 0.04);
	assert_eq!(muse.output_price_per_1m, 1.50);
	assert_eq!(provider.get_max_input_tokens("meta-models/Muse-Glimmer-30B"), 131_072);

	let bonsai = provider.get_model_pricing("Prism-ML/Ternary-Bonsai-27B").unwrap();
	assert_eq!(bonsai.input_price_per_1m, 0.0);
	assert_eq!(bonsai.output_price_per_1m, 0.0);
	assert_eq!(provider.get_max_input_tokens("Prism-ML/Ternary-Bonsai-27B"), 262_144);
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
	let (thinking, content) = extract_thinking("the answer is 42", Some("let me think...".to_string()));
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

#[test]
fn tool_images_follow_tool_results_without_losing_call_association() {
	let messages = [shared::tests::tool_image_message("")];
	let value = serde_json::to_value(convert_messages(&messages).unwrap()).unwrap();
	assert_eq!(value[0]["role"], "tool");
	assert_eq!(value[0]["tool_call_id"], "call_image");
	assert_eq!(value[0]["content"], "");
	assert_eq!(value[1]["role"], "user");
	assert_eq!(
		value[1]["content"][1]["image_url"]["url"],
		"data:image/png;base64,aW1hZ2U="
	);
}

fn reasoning_request(model: &str, effort: Option<ReasoningEffort>) -> serde_json::Value {
	let mut body = serde_json::json!({ "model": model });
	apply_reasoning_effort(&mut body, model, effort);
	body
}

#[test]
fn unset_effort_sends_no_reasoning_fields() {
	let body = reasoning_request("moonshotai/Kimi-K3", None);
	assert!(body.get("reasoning_effort").is_none());
	assert!(body.get("reasoning").is_none());
}

#[test]
fn reasoning_off_disables_thinking_on_hybrid_models() {
	for model in [
		"moonshotai/Kimi-K3",
		"deepseek-ai/DeepSeek-V4-Pro-0813",
		"deepseek-ai/DeepSeek-V4.1-Flash",
		"zai-org/GLM-5.2",
		"MiniMaxAI/MiniMax-M3",
		"Qwen/Qwen3.6-Plus",
		"Qwen/Qwen3.5-9B",
	] {
		let body = reasoning_request(model, Some(ReasoningEffort::None));
		assert_eq!(body["reasoning"], serde_json::json!({ "enabled": false }), "{model}");
		assert!(body.get("reasoning_effort").is_none(), "{model}");
	}
}

#[test]
fn reasoning_off_runs_glm_5_3_at_low_and_leaves_gpt_oss_alone() {
	// GLM-5.3 cannot stop thinking; low is the documented closest equivalent.
	for model in ["zai-org/GLM-5.3", "zai-org/GLM-5.3-Flash"] {
		let body = reasoning_request(model, Some(ReasoningEffort::None));
		assert_eq!(body["reasoning_effort"], "low", "{model}");
		assert!(body.get("reasoning").is_none(), "{model}");
	}
	// GPT-OSS cannot be disabled and has no switch: nothing is sent.
	let body = reasoning_request("openai/gpt-oss-120b", Some(ReasoningEffort::None));
	assert!(body.get("reasoning_effort").is_none());
	assert!(body.get("reasoning").is_none());
}

#[test]
fn effort_levels_follow_each_model_ladder() {
	let effort =
		|model: &str, level: ReasoningEffort| reasoning_request(model, Some(level))["reasoning_effort"].clone();

	// Kimi K3 and GLM-5.3: low|medium|high|max; xhigh floors to high.
	for model in ["moonshotai/Kimi-K3", "zai-org/GLM-5.3"] {
		assert_eq!(effort(model, ReasoningEffort::Low), "low");
		assert_eq!(effort(model, ReasoningEffort::Medium), "medium");
		assert_eq!(effort(model, ReasoningEffort::High), "high");
		assert_eq!(effort(model, ReasoningEffort::XHigh), "high");
		assert_eq!(effort(model, ReasoningEffort::Max), "max");
	}

	// DeepSeek V4 Pro: high|max only.
	let deepseek = "deepseek-ai/DeepSeek-V4-Pro-0813";
	assert_eq!(effort(deepseek, ReasoningEffort::Low), "high");
	assert_eq!(effort(deepseek, ReasoningEffort::High), "high");
	assert_eq!(effort(deepseek, ReasoningEffort::XHigh), "max");
	assert_eq!(effort(deepseek, ReasoningEffort::Max), "max");

	// DeepSeek V4.1 Flash: low|high|xhigh|max; medium floors to low.
	let flash = "deepseek-ai/DeepSeek-V4.1-Flash";
	assert_eq!(effort(flash, ReasoningEffort::Low), "low");
	assert_eq!(effort(flash, ReasoningEffort::Medium), "low");
	assert_eq!(effort(flash, ReasoningEffort::High), "high");
	assert_eq!(effort(flash, ReasoningEffort::XHigh), "xhigh");
	assert_eq!(effort(flash, ReasoningEffort::Max), "max");

	// GPT-OSS tops out at high.
	let gpt_oss = "openai/gpt-oss-120b";
	assert_eq!(effort(gpt_oss, ReasoningEffort::Medium), "medium");
	assert_eq!(effort(gpt_oss, ReasoningEffort::Max), "high");

	// Undocumented models keep the OpenAI low|medium|high ladder.
	assert_eq!(effort("Qwen/Qwen3.8-Flash", ReasoningEffort::Max), "high");
	let body = reasoning_request("Qwen/Qwen3.8-Flash", Some(ReasoningEffort::None));
	assert!(body.get("reasoning_effort").is_none());
	assert!(body.get("reasoning").is_none());
}

#[test]
fn test_together_model_pricing_known_and_unknown() {
	let priced = together_model_pricing("Qwen/Qwen3.8-2.4T-A95B").unwrap();
	assert!((priced.input_price_per_1m - 2.00).abs() < 1e-9);
	assert!((priced.output_price_per_1m - 6.00).abs() < 1e-9);
	assert!((priced.cache_read_price_per_1m - 0.25).abs() < 1e-9);
	// Free model prices at zero.
	let free = together_model_pricing("Prism-ML/Ternary-Bonsai-27B").unwrap();
	assert_eq!(free.input_price_per_1m, 0.0);
	assert_eq!(free.output_price_per_1m, 0.0);
	assert!(together_model_pricing("unknown/model").is_none());
}

#[test]
fn test_together_model_context_families() {
	assert_eq!(together_model_context("ThinkingMachines/Inkling-Base"), Some(524_288));
	assert_eq!(together_model_context("MiniMaxAI/MiniMax-M3"), Some(524_288));
	assert_eq!(together_model_context("moonshotai/kimi-k3"), Some(1_048_576));
	assert_eq!(together_model_context("zai-org/glm-5.3"), Some(1_048_575));
	assert_eq!(together_model_context("google/gemma-4-31b-it"), Some(262_144));
	assert_eq!(together_model_context("qwen/qwen3.8-flash"), Some(1_000_000));
	assert_eq!(together_model_context("openai/gpt-oss-120b"), Some(131_072));
	assert_eq!(together_model_context("totally/unknown"), None);
}

#[test]
fn test_together_apply_reasoning_effort_families() {
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "moonshotai/kimi-k3", Some(ReasoningEffort::None));
	assert_eq!(body["reasoning"]["enabled"], false);

	// GLM-5.3 cannot stop thinking: None floors to low.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "zai-org/glm-5.3", Some(ReasoningEffort::None));
	assert_eq!(body["reasoning_effort"], "low");

	// GPT-OSS has no off switch: nothing is sent.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "openai/gpt-oss-120b", Some(ReasoningEffort::None));
	assert!(body.as_object().unwrap().is_empty());

	// Max passes through on Kimi K3 and GLM-5.3.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "moonshotai/kimi-k3", Some(ReasoningEffort::Max));
	assert_eq!(body["reasoning_effort"], "max");

	// DeepSeek V4 Pro: only high and max; lower levels floor to high.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "deepseek-ai/deepseek-v4-pro", Some(ReasoningEffort::Low));
	assert_eq!(body["reasoning_effort"], "high");
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "deepseek-ai/deepseek-v4-pro", Some(ReasoningEffort::XHigh));
	assert_eq!(body["reasoning_effort"], "max");

	// Hybrid models turn thinking off with the reasoning switch.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "minimaxai/minimax-m3", Some(ReasoningEffort::None));
	assert_eq!(body["reasoning"]["enabled"], false);

	// Other models use the generic ladder; XHigh collapses to high.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "some/other-model", Some(ReasoningEffort::XHigh));
	assert_eq!(body["reasoning_effort"], "high");

	// No effort requested: nothing is sent.
	let mut body = serde_json::json!({});
	apply_reasoning_effort(&mut body, "moonshotai/kimi-k3", None);
	assert!(body.as_object().unwrap().is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn chat_completion_via_mock_http_merges_sse_stream_and_prices_from_table() {
	// Together answers as an SSE stream; the provider folds `data:` chunks.
	let body = concat!(
		"data: {\"id\":\"t-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"together \"},\"finish_reason\":null}]}\n\n",
		"data: {\"id\":\"t-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"mocked answer\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n\n",
		"data: [DONE]\n\n"
	);
	let mock = crate::test_support::mock_http::MockHttp::start(vec![(200, body.to_string())]);
	let _env = EnvGuard::new().set("TOGETHER_API_URL", mock.url.clone());

	let provider = TogetherProvider::new();
	let params =
		ChatCompletionParams::new(&[], "Qwen/Qwen3.8-Flash", 1.0, 1.0, 0, 10).with_api_key("mock-key".to_owned());
	let response = provider.chat_completion(params).await.expect("mock completion");

	// Deltas from both chunks are concatenated.
	assert_eq!(response.content, "together mocked answer");
	let usage = response.exchange.usage.expect("usage present");
	assert_eq!(usage.input_tokens, 12);
	assert_eq!(usage.output_tokens, 7);
	assert!(usage.cost.unwrap_or(0.0) > 0.0, "cost computed from the pricing table");

	let requests = mock.requests();
	let req_body: serde_json::Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(req_body["model"], "Qwen/Qwen3.8-Flash");
}
