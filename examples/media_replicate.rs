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

//! Run with `REPLICATE_API_TOKEN=... cargo run --example media_replicate`.

use octolib::{ImageGenerationRequest, MediaProviderFactory, OperationStatus};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (provider, model) = MediaProviderFactory::get_image_provider_for_model(
        "replicate:black-forest-labs/flux-1.1-pro",
    )?;
    let mut request =
        ImageGenerationRequest::new("A quiet library under the ocean").with_model(model);
    request.request_options.polling_interval = Duration::from_secs(2);

    // submit_image returns immediately. Persist this handle if your process may restart.
    let mut operation = provider.submit_image(request).await?;
    while !operation.status.is_terminal() {
        let handle = operation
            .handle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Replicate returned no job handle"))?;
        operation = provider.poll_image(handle).await?;
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    if operation.status != OperationStatus::Succeeded {
        return Err(anyhow::anyhow!("prediction failed: {:?}", operation.error));
    }
    let result = operation
        .result
        .ok_or_else(|| anyhow::anyhow!("completed prediction returned no result"))?;
    println!("generated {} artifact(s)", result.artifacts.len());
    Ok(())
}
