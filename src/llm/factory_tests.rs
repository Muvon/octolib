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
fn test_provider_setup_covers_supported_providers_once() {
	let setup = ProviderFactory::provider_setup_info();
	let expected: Vec<_> = ProviderFactory::supported_providers()
		.into_iter()
		.filter(|name| *name != "cli")
		.collect();
	let names: Vec<_> = setup.iter().map(|info| info.name).collect();
	assert_eq!(names, expected);
	assert_eq!(
		names.iter().collect::<std::collections::HashSet<_>>().len(),
		names.len()
	);
}

#[test]
fn test_provider_setup_credentials() {
	let setup = ProviderFactory::provider_setup_info();
	let info = |name| setup.iter().find(|info| info.name == name).unwrap();
	assert_eq!(info("openrouter").api_key_env, Some("OPENROUTER_API_KEY"));
	assert_eq!(info("cloudflare").api_key_env, Some("CLOUDFLARE_API_KEY"));
	assert_eq!(info("cloudflare").extra_env, &["CLOUDFLARE_ACCOUNT_ID"]);
	assert_eq!(info("amazon").api_key_env, Some("AWS_BEARER_TOKEN_BEDROCK"));
	assert!(info("amazon").extra_env.is_empty());
	assert_eq!(info("meta").api_key_env, Some("META_API_KEY"));
	assert!(info("meta").extra_env.is_empty());
	assert_eq!(info("google-vertex").api_key_env, None);
	assert!(info("google-vertex").extra_env.is_empty());
	assert_eq!(info("chatgpt").api_key_env, None);
	assert!(info("chatgpt").extra_env.is_empty());
	assert_eq!(info("local").api_key_env, Some("LOCAL_API_KEY"));
	assert_eq!(info("ollama").api_key_env, Some("OLLAMA_API_KEY"));
	assert_eq!(info("octohub").api_key_env, Some("OCTOHUB_API_KEY"));
	assert_eq!(info("opencode-zen").api_key_env, Some("OPENCODE_API_KEY"));
	assert_eq!(info("opencode-go").api_key_env, Some("OPENCODE_API_KEY"));
	assert!(info("openai").extra_env.is_empty());
	assert!(info("anthropic").extra_env.is_empty());
}

#[test]
fn test_provider_setup_models_are_unique_and_supported() {
	for info in ProviderFactory::provider_setup_info() {
		let mut seen = std::collections::HashSet::new();
		for model in &info.models {
			assert!(seen.insert(model), "duplicate model for {}: {model}", info.name);
			let full_model = format!("{}:{model}", info.name);
			assert!(
				ProviderFactory::get_provider_for_model(&full_model).is_ok(),
				"unsupported setup model: {full_model}"
			);
		}
	}
	let setup = ProviderFactory::provider_setup_info();
	let info = |name| setup.iter().find(|info| info.name == name).unwrap();
	assert!(!info("openai").models.is_empty());
	assert_eq!(info("cerebras").models, ["gpt-oss-120b", "qwen-3.8-27b"]);
	assert_eq!(info("hetzner").models, ["Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B-FP8"]);
	assert_eq!(info("reflection").models, ["Beam-1.0-501B-A23B"]);
	assert_eq!(
		info("isoquant").models,
		["deepseek-v4.1-flash-1017-fast", "glm-5.3-flash"]
	);
	assert_eq!(info("google-studio").models, info("google-vertex").models);
}

#[test]
fn test_provider_setup_omits_pricing_patterns_and_remote_catalogs() {
	let setup = ProviderFactory::provider_setup_info();
	let info = |name| setup.iter().find(|info| info.name == name).unwrap();
	for name in [
		"openrouter",
		"fireworks",
		"tinker",
		"chatgpt",
		"local",
		"nvidia",
		"octohub",
		"opencode-zen",
		"opencode-go",
	] {
		assert!(info(name).models.is_empty(), "unexpected local models for {name}");
	}
	assert!(!info("moonshot").models.iter().any(|model| model == "kimi-k2.7"));
	for pattern in [
		"gemini-omni",
		"gemini-robotics-er-2",
		"gemini-3.1-pro",
		"gemini-3-flash",
	] {
		assert!(!info("google-studio").models.iter().any(|model| model == pattern));
	}
	for pattern in ["google/gemma-4-31B", "google/gemma-4-26B"] {
		assert!(!info("featherless").models.iter().any(|model| model == pattern));
	}
	assert_eq!(
		info("amazon").models,
		[
			"global.moonshotai.kimi-k3",
			"moonshotai.kimi-k3",
			"global.xai.grok-4.7",
			"xai.grok-4.7",
			"global.xai.grok-4.6",
			"xai.grok-4.6",
			"qwen.qwen3-coder-next",
			"deepseek.v3.2",
			"deepseek.v3.1",
			"global.zai.glm-5.3",
			"zai.glm-5.3",
			"zai.glm-4.7-flash",
			"minimax.minimax-m2.1",
			"google.gemma-4-31b",
			"google.gemma-4-26b-a4b",
			"google.gemma-4-e2b",
		]
	);
	assert_eq!(
		info("byteplus").models,
		[
			"dola-seed-2.0-pro",
			"dola-seed-2.0-lite",
			"dola-seed-2.0-code",
			"bytedance-seed-code",
			"glm-5-3-flash-260828",
		]
	);
}

#[test]
fn test_provider_setup_deduplicates_models_in_table_order() {
	let info = ProviderSetupInfo::new("local", None, &[], ["second", "first", "second", "third", "first"]);
	assert_eq!(info.models, ["second", "first", "third"]);
}

#[test]
fn test_parse_model() {
	// Test with provider prefix
	let result = ProviderFactory::parse_model("openrouter:anthropic/claude-sonnet-4-6");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider, "openrouter");
	assert_eq!(model, "anthropic/claude-sonnet-4-6");

	// Test with different provider
	let result = ProviderFactory::parse_model("openai:gpt-4o");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider, "openai");
	assert_eq!(model, "gpt-4o");

	// Test DeepSeek provider
	let result = ProviderFactory::parse_model("deepseek:deepseek-chat");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider, "deepseek");
	assert_eq!(model, "deepseek-chat");

	// Test whitespace trimming
	let result = ProviderFactory::parse_model("  openai : gpt-4o  ");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider, "openai");
	assert_eq!(model, "gpt-4o");

	// Test invalid format (no colon)
	let result = ProviderFactory::parse_model("gpt-4o");
	assert!(result.is_err());

	// Test invalid format (empty provider)
	let result = ProviderFactory::parse_model(":gpt-4o");
	assert!(result.is_err());

	// Test invalid format (empty model)
	let result = ProviderFactory::parse_model("openai:");
	assert!(result.is_err());
}

#[test]
fn test_supported_providers() {
	let providers = ProviderFactory::supported_providers();
	assert!(providers.contains(&"openai"));
	assert!(providers.contains(&"anthropic"));
	assert!(providers.contains(&"openrouter"));
	assert!(providers.contains(&"cerebras"));
	assert!(providers.contains(&"ollama"));
	assert!(providers.contains(&"google-vertex"));
	assert!(providers.contains(&"google-studio"));
	assert!(providers.contains(&"amazon"));
	assert!(providers.contains(&"cloudflare"));
	assert!(providers.contains(&"deepseek"));
	assert!(providers.contains(&"minimax"));
	assert!(providers.contains(&"moonshot"));
	assert!(providers.contains(&"groq"));
	assert!(providers.contains(&"inception"));
	assert!(providers.contains(&"isoquant"));
	assert!(providers.contains(&"reflection"));
	assert!(providers.contains(&"meta"));
	assert!(providers.contains(&"xai"));
	assert!(providers.contains(&"xiaomi"));
	assert!(providers.contains(&"tinker"));
	assert!(providers.contains(&"webwright"));
	assert!(providers.contains(&"cli"));
}

#[test]
fn test_validate_model_format() {
	assert!(ProviderFactory::validate_model_format("openai:gpt-4o").is_ok());
	assert!(ProviderFactory::validate_model_format("anthropic:claude-sonnet-4-6").is_ok());
	assert!(ProviderFactory::validate_model_format("gpt-4o").is_err());
	assert!(ProviderFactory::validate_model_format(":model").is_err());
	assert!(ProviderFactory::validate_model_format("provider:").is_err());
}

#[test]
fn test_create_provider() {
	// Test creating valid providers
	assert!(ProviderFactory::create_provider("openai").is_ok());
	assert!(ProviderFactory::create_provider("anthropic").is_ok());
	assert!(ProviderFactory::create_provider("openrouter").is_ok());
	assert!(ProviderFactory::create_provider("cerebras").is_ok());
	assert!(ProviderFactory::create_provider("ollama").is_ok());
	assert!(ProviderFactory::create_provider("google-vertex").is_ok());
	assert!(ProviderFactory::create_provider("google-studio").is_ok());
	assert!(ProviderFactory::create_provider("google").is_err());
	assert!(ProviderFactory::create_provider("alibaba").is_ok());
	assert!(ProviderFactory::create_provider("amazon").is_ok());
	assert!(ProviderFactory::create_provider("cloudflare").is_ok());
	assert!(ProviderFactory::create_provider("deepseek").is_ok());
	assert!(ProviderFactory::create_provider("minimax").is_ok());
	assert!(ProviderFactory::create_provider("moonshot").is_ok());
	assert!(ProviderFactory::create_provider("nvidia").is_ok());
	assert!(ProviderFactory::create_provider("groq").is_ok());
	assert!(ProviderFactory::create_provider("inception").is_ok());
	assert!(ProviderFactory::create_provider("isoquant").is_ok());
	assert!(ProviderFactory::create_provider("reflection").is_ok());
	assert!(ProviderFactory::create_provider("xai").is_ok());
	assert!(ProviderFactory::create_provider("xiaomi").is_ok());
	assert!(ProviderFactory::create_provider("meta").is_ok());
	assert!(ProviderFactory::create_provider("tinker").is_ok());
	assert!(ProviderFactory::create_provider("webwright").is_ok());
	assert!(ProviderFactory::create_provider("cli").is_err());

	// Test case insensitive
	assert!(ProviderFactory::create_provider("OpenAI").is_ok());
	assert!(ProviderFactory::create_provider("ANTHROPIC").is_ok());
	assert!(ProviderFactory::create_provider("MiniMax").is_ok());
	assert!(ProviderFactory::create_provider("MOONSHOT").is_ok());
	assert!(ProviderFactory::create_provider("NVIDIA").is_ok());
	assert!(ProviderFactory::create_provider("OLLAMA").is_ok());
	assert!(ProviderFactory::create_provider("CEREBRAS").is_ok());
	assert!(ProviderFactory::create_provider("XAI").is_ok());

	// Test invalid provider
	assert!(ProviderFactory::create_provider("invalid").is_err());
}

#[test]
fn test_provider_capabilities() {
	let openai = ProviderFactory::create_provider("openai").unwrap();
	assert_eq!(openai.name(), "openai");
	assert!(openai.supports_model("gpt-4o"));
	assert!(openai.supports_vision("gpt-4o"));
	assert!(openai.supports_caching("gpt-4o"));

	let anthropic = ProviderFactory::create_provider("anthropic").unwrap();
	assert_eq!(anthropic.name(), "anthropic");
	assert!(anthropic.supports_model("claude-sonnet-4-6"));
	assert!(anthropic.supports_vision("claude-sonnet-4-6"));
	assert!(anthropic.supports_caching("claude-sonnet-4-6"));

	let openrouter = ProviderFactory::create_provider("openrouter").unwrap();
	assert_eq!(openrouter.name(), "openrouter");
	assert!(openrouter.supports_model("any-model"));
	assert!(openrouter.supports_vision("claude-sonnet-4-6"));
	assert!(openrouter.supports_caching("claude-sonnet-4-6"));

	let ollama = ProviderFactory::create_provider("ollama").unwrap();
	assert_eq!(ollama.name(), "ollama");
	assert!(ollama.supports_model("llama3.2"));
	assert!(ollama.supports_caching("llama3.2"));

	let cerebras = ProviderFactory::create_provider("cerebras").unwrap();
	assert_eq!(cerebras.name(), "cerebras");
	assert!(cerebras.supports_model("gpt-oss-120b"));
	assert!(!cerebras.supports_caching("gpt-oss-120b"));

	let nvidia = ProviderFactory::create_provider("nvidia").unwrap();
	assert_eq!(nvidia.name(), "nvidia");
	assert!(nvidia.supports_model("nvidia/llama-3.1-nemotron-ultra-253b-v1"));
	assert!(nvidia.supports_model("deepseek-ai/deepseek-v3.2"));
	assert!(!nvidia.supports_caching("any-model"));
}

#[test]
fn test_get_provider_for_model() {
	let result = ProviderFactory::get_provider_for_model("openai:gpt-4o");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "openai");
	assert_eq!(model, "gpt-4o");

	let result = ProviderFactory::get_provider_for_model("anthropic:claude-sonnet-4-6");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "anthropic");
	assert_eq!(model, "claude-sonnet-4-6");

	// Test MiniMax provider
	let result = ProviderFactory::get_provider_for_model("minimax:MiniMax-M2.1");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "minimax");
	assert_eq!(model, "MiniMax-M2.1");
	assert!(provider.supports_caching(&model));
	assert!(provider.supports_model(&model));

	// Test Moonshot provider
	let result = ProviderFactory::get_provider_for_model("moonshot:kimi-k3");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "moonshot");
	assert_eq!(model, "kimi-k3");

	// Generic multi-model providers should accept arbitrary non-empty model IDs
	let result = ProviderFactory::get_provider_for_model("google-vertex:any-gemini-variant");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "google-vertex");
	assert_eq!(model, "any-gemini-variant");

	let result = ProviderFactory::get_provider_for_model("google-studio:gemini-2.5-flash");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "google-studio");
	assert_eq!(model, "gemini-2.5-flash");

	let result = ProviderFactory::get_provider_for_model("cloudflare:@cf/custom/model");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "cloudflare");
	assert_eq!(model, "@cf/custom/model");

	let result = ProviderFactory::get_provider_for_model("amazon:custom.model-id-v1");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "amazon");
	assert_eq!(model, "custom.model-id-v1");

	let result = ProviderFactory::get_provider_for_model("ollama:llama3.2");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "ollama");
	assert_eq!(model, "llama3.2");

	let result = ProviderFactory::get_provider_for_model("cerebras:gpt-oss-120b");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "cerebras");
	assert_eq!(model, "gpt-oss-120b");

	// Test NVIDIA provider
	let result = ProviderFactory::get_provider_for_model("nvidia:nvidia/llama-3.1-nemotron-ultra-253b-v1");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "nvidia");
	assert_eq!(model, "nvidia/llama-3.1-nemotron-ultra-253b-v1");

	// Test Groq provider
	let result = ProviderFactory::get_provider_for_model("groq:llama-3.3-70b-versatile");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "groq");
	assert_eq!(model, "llama-3.3-70b-versatile");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_structured_output(&model));

	// Test Meta provider (closed catalog)
	let result = ProviderFactory::get_provider_for_model("meta:muse-spark-1.3");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "meta");
	assert_eq!(model, "muse-spark-1.3");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_structured_output(&model));
	assert!(ProviderFactory::get_provider_for_model("meta:muse-spark-9").is_err());

	// Test Inception provider (closed catalog)
	let result = ProviderFactory::get_provider_for_model("inception:mercury-2.5");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "inception");
	assert_eq!(model, "mercury-2.5");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_structured_output(&model));
	assert!(ProviderFactory::get_provider_for_model("inception:mercury-9").is_err());

	// Test Xiaomi MiMo provider (closed catalog)
	let result = ProviderFactory::get_provider_for_model("xiaomi:mimo-v2.6-pro");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "xiaomi");
	assert_eq!(model, "mimo-v2.6-pro");
	assert!(provider.supports_model(&model));
	assert!(ProviderFactory::get_provider_for_model("xiaomi:mimo-v2.5").is_err());

	// Test Featherless provider
	let result = ProviderFactory::get_provider_for_model("featherless:meta-llama/Meta-Llama-3.1-8B-Instruct");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "featherless");
	assert_eq!(model, "meta-llama/Meta-Llama-3.1-8B-Instruct");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_structured_output(&model));
	assert!(!provider.supports_caching(&model));

	// Test Fireworks provider
	let result = ProviderFactory::get_provider_for_model("fireworks:accounts/fireworks/models/kimi-k2-instruct-0905");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "fireworks");
	assert_eq!(model, "accounts/fireworks/models/kimi-k2-instruct-0905");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_structured_output(&model));
	assert!(provider.supports_caching(&model));
	assert!(provider.get_model_pricing(&model).is_some());

	// Test Tinker provider (model IDs contain colons; only the first splits)
	let result = ProviderFactory::get_provider_for_model("tinker:thinkingmachines/Inkling:peft:262144");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "tinker");
	assert_eq!(model, "thinkingmachines/Inkling:peft:262144");
	assert!(provider.supports_model(&model));

	// Test WebWright provider (OpenAI-compatible proxy)
	let result = ProviderFactory::get_provider_for_model("webwright:wren-r1");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "webwright");
	assert_eq!(model, "wren-r1");
	assert!(provider.supports_model(&model));
	assert!(provider.supports_caching(&model));

	// Test invalid format
	let result = ProviderFactory::get_provider_for_model("gpt-4o");
	assert!(result.is_err());

	// Test unsupported provider
	let result = ProviderFactory::get_provider_for_model("invalid:model");
	assert!(result.is_err());
}

#[test]
fn test_get_provider_for_model_case_insensitive() {
	let result = ProviderFactory::get_provider_for_model("OPENAI:gpt-4o");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "openai");
	assert_eq!(model, "gpt-4o");

	let result = ProviderFactory::get_provider_for_model("Anthropic:claude-sonnet-4-6");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "anthropic");
	assert_eq!(model, "claude-sonnet-4-6");

	let result = ProviderFactory::get_provider_for_model("openai:GPT-4O");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "openai");
	assert_eq!(model, "GPT-4O");
	assert!(provider.supports_model(&model));

	let result = ProviderFactory::get_provider_for_model("minimax:MINIMAX-M2.1");
	assert!(result.is_ok());
	let (provider, model) = result.unwrap();
	assert_eq!(provider.name(), "minimax");
	assert_eq!(model, "MINIMAX-M2.1");
	assert!(provider.supports_model(&model));
}
