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

//! Higgsfield request lifecycle against the SOUL image endpoint.
//! Run with `HIGGSFIELD_API_KEY=<key_id>:<key_secret> cargo run --example media_higgsfield`.

use octolib::{
	generate_image, ArtifactSource, HiggsfieldMediaOptions, ImageGenerationRequest, MediaProviderFactory,
	OutputGeometry,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	let model = "higgsfield:higgsfield-ai/soul/standard";

	// Low-level path first: submit returns immediately with a resumable handle.
	let (provider, native_model) = MediaProviderFactory::get_image_provider_for_model(model)?;
	let submitted = ImageGenerationRequest::new("A brass octopus reading a book").with_model(native_model);
	let operation = provider.submit_image(submitted).await?;
	let handle = operation
		.handle
		.clone()
		.ok_or_else(|| anyhow::anyhow!("queued operation carried no handle"))?;
	println!("queued: status {:?}, request {}", operation.status, handle.remote_id);

	// The handle is credential-free JSON, so it survives a process restart.
	let persisted: octolib::JobHandle = serde_json::from_str(&serde_json::to_string(&handle)?)?;
	let polled = provider.poll_image(&persisted).await?;
	println!("polled: status {:?}", polled.status);
	// Cancellation only succeeds while the request is still queued.
	if let Err(error) = provider.cancel_image(&persisted).await {
		println!("cancel: {error}");
	}

	// High-level path: submit and wait. Endpoint-specific fields go in `input`.
	let mut request = ImageGenerationRequest::new("A brass octopus reading a book");
	request.geometry = Some(OutputGeometry::AspectRatio { width: 1, height: 1 });
	request.request_options.idempotency_key = Some(uuid::Uuid::new_v4().to_string());
	request.provider_options = HiggsfieldMediaOptions {
		input: serde_json::from_value(serde_json::json!({"resolution": "720p"}))?,
		..HiggsfieldMediaOptions::default()
	}
	.into_provider_options()?;
	let result = generate_image(model, request).await?;
	for artifact in &result.artifacts {
		match &artifact.source {
			ArtifactSource::Url(url) => println!("image: {url}"),
			_ => println!("image: provider-managed artifact"),
		}
	}
	if let Some(usage) = result.usage.as_ref() {
		println!("usage: {}", serde_json::to_string(usage)?);
	}
	for warning in &result.warnings {
		println!("warning [{:?}]: {}", warning.code, warning.message);
	}
	Ok(())
}
