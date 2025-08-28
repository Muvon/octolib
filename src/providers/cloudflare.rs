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

//! Cloudflare Workers AI provider implementation

use crate::traits::AiProvider;
use crate::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// Cloudflare Workers AI provider
#[derive(Debug, Clone)]
pub struct CloudflareWorkersAiProvider;

impl Default for CloudflareWorkersAiProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl CloudflareWorkersAiProvider {
	pub fn new() -> Self {
		Self
	}
}

const CLOUDFLARE_API_TOKEN_ENV: &str = "CLOUDFLARE_API_TOKEN";

#[async_trait::async_trait]
impl AiProvider for CloudflareWorkersAiProvider {
	fn name(&self) -> &str {
		"cloudflare"
	}

	fn supports_model(&self, model: &str) -> bool {
		model.contains("llama") || model.contains("mistral") || model.contains("qwen")
	}

	fn get_api_key(&self) -> Result<String> {
		match env::var(CLOUDFLARE_API_TOKEN_ENV) {
			Ok(key) => Ok(key),
			Err(_) => Err(anyhow::anyhow!(
				"Cloudflare API token not found in environment variable: {}",
				CLOUDFLARE_API_TOKEN_ENV
			)),
		}
	}

	fn supports_caching(&self, _model: &str) -> bool {
		false
	}

	fn supports_vision(&self, _model: &str) -> bool {
		false
	}

	fn get_max_input_tokens(&self, _model: &str) -> usize {
		32_768
	}

	async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
		Err(anyhow::anyhow!(
			"Cloudflare Workers AI provider not fully implemented in octolib"
		))
	}
}
