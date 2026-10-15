//! Minimal live test for the WebWright provider.
//!
//! Usage:
//! ```bash
//! export WEBWRIGHT_API_KEY="your_key"
//! cargo run --example webwright_chat
//! ```

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_spec = "webwright:wren-r1";
    let (provider, model) = ProviderFactory::get_provider_for_model(model_spec)?;

    println!("Provider: {}", provider.name());
    println!("Model: {}", model);

    let messages = vec![Message::user("In one short sentence, what is WebWright?")];
    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 512);

    let response = provider.chat_completion(params).await?;
    println!("Response: {}", response.content);
    if let Some(usage) = &response.exchange.usage {
        println!(
            "Tokens: input={} output={} total={} cost={:?}",
            usage.input_tokens, usage.output_tokens, usage.total_tokens, usage.cost
        );
    }

    Ok(())
}
