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

//! Reflection AI chat completion example.
//!
//! Usage:
//! ```bash
//! export REFLECTION_API_KEY="your_key"
//! cargo run --example reflection_chat
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory, ReasoningEffort};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	let (provider, model) = ProviderFactory::get_provider_for_model("reflection:Beam-1.0-501B-A23B")?;

	println!("Provider: {}", provider.name());
	println!("Model: {}", model);

	let messages = vec![Message::user("In one short sentence, what is prompt caching?")];
	// Reasoning and the answer share this allowance
	let params =
		ChatCompletionParams::new(&messages, &model, 0.7, 0.9, 50, 2048).with_reasoning_effort(ReasoningEffort::Low);

	let response = provider.chat_completion(params).await?;
	println!("Response: {}", response.content);
	if let Some(usage) = &response.exchange.usage {
		println!(
			"Tokens: input={} output={} reasoning={} cached={} total={} cost={:?}",
			usage.input_tokens,
			usage.output_tokens,
			usage.reasoning_tokens,
			usage.cache_read_tokens,
			usage.total_tokens,
			usage.cost
		);
	}

	Ok(())
}
