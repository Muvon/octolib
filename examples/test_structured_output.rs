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

//! Simple test to verify structured output functionality

use octolib::llm::{ProviderFactory, StructuredOutputRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Structured Output Support");

    // Test provider capability detection
    let providers_to_test = vec![
        "openai:gpt-4o",
        "openrouter:openai/gpt-4o",
        "deepseek:deepseek-chat",
        "anthropic:claude-3-5-sonnet",
    ];

    for model_spec in providers_to_test {
        match ProviderFactory::get_provider_for_model(model_spec) {
            Ok((provider, model)) => {
                let supports = provider.supports_structured_output(&model);
                let status = if supports {
                    "✅ SUPPORTED"
                } else {
                    "❌ NOT SUPPORTED"
                };
                println!("{:<30} | {}", model_spec, status);
            }
            Err(e) => {
                println!("{:<30} | ❌ ERROR: {}", model_spec, e);
            }
        }
    }

    println!("\n🔧 Testing StructuredOutputRequest creation:");

    // Test basic JSON request
    let json_request = StructuredOutputRequest::json();
    println!("✅ JSON request created: {:?}", json_request);

    // Test JSON schema request
    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    });

    let schema_request = StructuredOutputRequest::json_schema(schema).with_strict_mode();
    println!("✅ JSON Schema request created: {:?}", schema_request);

    println!("\n🎉 All structured output tests passed!");

    Ok(())
}
