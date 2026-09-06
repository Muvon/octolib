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

//! fal queue lifecycle against a cheap endpoint.
//! Run with `FAL_API_KEY=... cargo run --example media_fal`.

use octolib::{
    generate_image, ArtifactSource, ImageGenerationRequest, MediaProviderFactory, OutputGeometry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = "fal:fal-ai/flux/schnell";

    // Low-level path first: submit returns immediately with a resumable handle.
    let (provider, native_model) = MediaProviderFactory::get_image_provider_for_model(model)?;
    let mut submitted = ImageGenerationRequest::new("A brass octopus reading a book")
        .with_model(native_model.clone());
    submitted.geometry = Some(OutputGeometry::Dimensions {
        width: 1024,
        height: 1024,
    });
    let operation = provider.submit_image(submitted).await?;
    let handle = operation
        .handle
        .clone()
        .ok_or_else(|| anyhow::anyhow!("queued operation carried no handle"))?;
    println!(
        "queued: status {:?}, request {}",
        operation.status, handle.remote_id
    );

    // The handle is credential-free JSON, so it survives a process restart.
    let persisted: octolib::JobHandle = serde_json::from_str(&serde_json::to_string(&handle)?)?;
    let polled = provider.poll_image(&persisted).await?;
    println!("polled: status {:?}", polled.status);
    provider.cancel_image(&persisted).await.ok();

    // High-level path: submit and wait.
    let mut request = ImageGenerationRequest::new("A brass octopus reading a book");
    request.geometry = Some(OutputGeometry::Dimensions {
        width: 1024,
        height: 1024,
    });
    let result = generate_image(model, request).await?;
    for artifact in &result.artifacts {
        match &artifact.source {
            ArtifactSource::Url(url) => println!("image: {url}"),
            ArtifactSource::Inline(bytes) => {
                println!("image: {} bytes of {}", bytes.len(), artifact.media_type)
            }
            _ => println!("image: provider-managed artifact"),
        }
    }
    println!("warnings: {}", result.warnings.len());
    if let Some(usage) = result.usage.as_ref() {
        println!("usage: {}", serde_json::to_string(usage)?);
    }
    for warning in &result.warnings {
        println!("warning [{:?}]: {}", warning.code, warning.message);
    }
    Ok(())
}
