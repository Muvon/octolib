//! Temporary live test for the Hetzner provider.
//!
//! Usage:
//! ```bash
//! export HETZNER_API_KEY="your_key"
//! cargo run --example hetzner_chat
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_spec = "hetzner:Qwen/Qwen3.6-35B-A3B-FP8";
    let (provider, model) = ProviderFactory::get_provider_for_model(model_spec)?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user(
        "In one short sentence, what is Hetzner known for?",
    )];
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 512);

    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);
    if let Some(thinking) = &response.thinking {
        println!("Thinking ({} tokens): captured", thinking.tokens);
    }
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Tokens: input={} output={} total={} cost={:?}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.cost
        );
    }

    Ok(())
}
