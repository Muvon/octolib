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

//! DeepSeek provider implementation

use crate::traits::AiProvider;
use crate::types::{ChatCompletionParams, ProviderResponse};
use anyhow::Result;
use std::env;

/// DeepSeek provider
#[derive(Debug, Clone)]
pub struct DeepSeekProvider;

impl Default for DeepSeekProvider {
	fn default() -> Self {
		Self::new()
	}
}

impl DeepSeekProvider {
	pub fn new() -> Self {
		Self
	}
}

const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";

#[async_trait::async_trait]
impl AiProvider for DeepSeekProvider {
	fn name(&self) -> &str {
		"deepseek"
	}

	fn supports_model(&self, model: &str) -> bool {
		model.contains("deepseek")
	}

	fn get_api_key(&self) -> Result<String> {
		match env::var(DEEPSEEK_API_KEY_ENV) {
			Ok(key) => Ok(key),
			Err(_) => Err(anyhow::anyhow!(
				"DeepSeek API key not found in environment variable: {}",
				DEEPSEEK_API_KEY_ENV
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
		64_000
	}

	async fn chat_completion(&self, _params: ChatCompletionParams) -> Result<ProviderResponse> {
		Err(anyhow::anyhow!(
			"DeepSeek provider not fully implemented in octolib"
		))
	}
}
