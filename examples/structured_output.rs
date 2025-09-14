// Copyright 2025 Muvon Un Limited
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

//! Example demonstrating structured output with OpenAI and OpenRouter providers

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory, StructuredOutputRequest};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct PersonInfo {
    name: String,
    age: u32,
    occupation: String,
    skills: Vec<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Example 1: Basic JSON output
    println!("=== Example 1: Basic JSON Output ===");

    let (provider, model) = ProviderFactory::get_provider_for_model("openai:gpt-4o")?;

    // Check if provider supports structured output
    if !provider.supports_structured_output(&model) {
        println!(
            "Provider {} does not support structured output for model {}",
            provider.name(),
            model
        );
        return Ok(());
    }

    let messages = vec![
        Message::system("You are a helpful assistant that responds with valid JSON."),
        Message::user("Tell me about a fictional software engineer. Respond with JSON containing name, age, occupation, and skills array."),
    ];

    let structured_request = StructuredOutputRequest::json();

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_structured_output(structured_request);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Content: {}", response.content);

            if let Some(structured) = response.structured_output {
                println!(
                    "Structured output: {}",
                    serde_json::to_string_pretty(&structured)?
                );

                // Try to parse as PersonInfo
                if let Ok(person) = serde_json::from_value::<PersonInfo>(structured) {
                    println!("Parsed person: {:?}", person);
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== Example 2: JSON Schema with Strict Mode ===");

    // Example 2: JSON Schema with strict validation
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Full name of the person"
            },
            "age": {
                "type": "integer",
                "minimum": 18,
                "maximum": 100,
                "description": "Age in years"
            },
            "occupation": {
                "type": "string",
                "description": "Job title or profession"
            },
            "skills": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "minItems": 1,
                "description": "List of professional skills"
            }
        },
        "required": ["name", "age", "occupation", "skills"],
        "additionalProperties": false
    });

    let structured_request = StructuredOutputRequest::json_schema(schema).with_strict_mode();

    let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, 1000)
        .with_structured_output(structured_request);

    match provider.chat_completion(params).await {
        Ok(response) => {
            println!("Content: {}", response.content);

            if let Some(structured) = response.structured_output {
                println!(
                    "Structured output: {}",
                    serde_json::to_string_pretty(&structured)?
                );

                // Parse as PersonInfo with validation
                match serde_json::from_value::<PersonInfo>(structured) {
                    Ok(person) => {
                        println!("✅ Successfully parsed person: {:?}", person);
                        println!("Name: {}", person.name);
                        println!("Age: {}", person.age);
                        println!("Occupation: {}", person.occupation);
                        println!("Skills: {:?}", person.skills);
                    }
                    Err(e) => println!("❌ Failed to parse person: {}", e),
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }

    println!("\n=== Example 3: DeepSeek Provider ===");

    // Example 3: Try with DeepSeek
    if let Ok((deepseek_provider, deepseek_model)) =
        ProviderFactory::get_provider_for_model("deepseek:deepseek-chat")
    {
        if deepseek_provider.supports_structured_output(&deepseek_model) {
            let structured_request = StructuredOutputRequest::json();

            let params = ChatCompletionParams::new(&messages, &deepseek_model, 0.7, 1.0, 50, 1000)
                .with_structured_output(structured_request);

            match deepseek_provider.chat_completion(params).await {
                Ok(response) => {
                    println!("DeepSeek Content: {}", response.content);

                    if let Some(structured) = response.structured_output {
                        println!(
                            "DeepSeek Structured output: {}",
                            serde_json::to_string_pretty(&structured)?
                        );
                    }
                }
                Err(e) => println!("DeepSeek Error: {}", e),
            }
        } else {
            println!("DeepSeek provider does not support structured output");
        }
    }

    println!("\n=== Example 4: OpenRouter Provider ===");

    // Example 4: Try with OpenRouter
    if let Ok((openrouter_provider, openrouter_model)) =
        ProviderFactory::get_provider_for_model("openrouter:openai/gpt-4o")
    {
        if openrouter_provider.supports_structured_output(&openrouter_model) {
            let structured_request = StructuredOutputRequest::json();

            let params =
                ChatCompletionParams::new(&messages, &openrouter_model, 0.7, 1.0, 50, 1000)
                    .with_structured_output(structured_request);

            match openrouter_provider.chat_completion(params).await {
                Ok(response) => {
                    println!("OpenRouter Content: {}", response.content);

                    if let Some(structured) = response.structured_output {
                        println!(
                            "OpenRouter Structured output: {}",
                            serde_json::to_string_pretty(&structured)?
                        );
                    }
                }
                Err(e) => println!("OpenRouter Error: {}", e),
            }
        } else {
            println!("OpenRouter provider does not support structured output");
        }
    }

    println!("\n=== Example 5: Unsupported Provider ===");

    // Example 5: Try with unsupported provider (should show clear error)
    if let Ok((anthropic_provider, anthropic_model)) =
        ProviderFactory::get_provider_for_model("anthropic:claude-3-5-sonnet")
    {
        if !anthropic_provider.supports_structured_output(&anthropic_model) {
            println!(
                "✅ Correctly detected: Provider {} does not support structured output",
                anthropic_provider.name()
            );
        }

        // Try anyway to demonstrate error handling
        let structured_request = StructuredOutputRequest::json();
        let params = ChatCompletionParams::new(&messages, &anthropic_model, 0.7, 1.0, 50, 1000)
            .with_structured_output(structured_request);

        match anthropic_provider.chat_completion(params).await {
            Ok(response) => {
                println!(
                    "Unexpected success with unsupported provider: {}",
                    response.content
                );
            }
            Err(e) => {
                println!("✅ Expected error with unsupported provider: {}", e);
            }
        }
    }

    Ok(())
}
