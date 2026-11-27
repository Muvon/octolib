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

//! LLMTR provider example.
//!
//! Usage:
//! ```bash
//! export LLMTR_API_KEY="your_key"
//! cargo run --example llmtr_chat --features llm
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	let model_spec = "llmtr:qwen/qwen3.8-27b-free";
	let (provider, model) = ProviderFactory::get_provider_for_model(model_spec)?;

	println!("Provider: {}", provider.name());
	println!("Model: {}", model);

	let messages = vec![Message::user("Reply with a single word: ok")];
	let params = ChatCompletionParams::new(&messages, &model, 64);

	let response = provider.chat_completion(params).await?;
	println!("Response: {}", response.content);
	if let Some(usage) = &response.exchange.usage {
		println!(
			"Tokens: input={} output={} total={}",
			usage.input_tokens, usage.output_tokens, usage.total_tokens
		);
	}

	Ok(())
}
