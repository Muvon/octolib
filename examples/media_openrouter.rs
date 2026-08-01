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

//! Run with `OPENROUTER_API_KEY=... cargo run --example media_openrouter`.

use octolib::{generate_image, ArtifactSource, ImageGenerationRequest};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let request =
        ImageGenerationRequest::new("A small brass octopus library assistant, product photography");
    let result = generate_image("openrouter:openai/gpt-image-1", request).await?;

    for artifact in &result.artifacts {
        match &artifact.source {
            ArtifactSource::Inline(bytes) => {
                println!("{} bytes of {}", bytes.len(), artifact.media_type);
            }
            ArtifactSource::Url(url) => println!("remote artifact: {url}"),
            _ => println!("provider-managed artifact"),
        }
    }
    if let Some(usage) = result.usage {
        println!(
            "provider-reported cost: {:?} USD",
            usage.provider_reported_cost
        );
    }
    Ok(())
}
