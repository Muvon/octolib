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

//! Ollama decision endpoint (`POST https://ollama.com/v1/systemone`).
//!
//! The body is `{model, state, questions}` and the reply is the same
//! `{model, answers, usage}` payload as the Jev hosts. Any model id is
//! accepted here; Ollama decides what it serves.
//!
//! The key is optional, as on the Ollama chat provider. Ollama Cloud needs
//! one. A self-hosted Ollama server (0.35.0 or later), reached through the
//! URL override, takes none, and then no `Authorization` header is sent.
//!
//! Images and box questions are refused before sending.
//!
//! Pricing (<https://ollama.com/pricing>, 2026-11-14): the page lists one
//! decision model, `clef`, at $0.24 per 1M input tokens with no output rate.
//! The cost is reported on the default endpoint only, because a self-hosted
//! server is not billed.
//!
//! Configuration:
//! - `OLLAMA_API_KEY`: optional; sent as a bearer token when set
//! - `OLLAMA_EVALUATION_API_URL`: optional full endpoint URL, default
//!   `https://ollama.com/v1/systemone`

use super::shared;
use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use serde_json::{json, Value};

const PROVIDER: &str = "ollama";
const API_KEY_ENV: &str = "OLLAMA_API_KEY";
const API_URL_ENV: &str = "OLLAMA_EVALUATION_API_URL";
const DEFAULT_API_URL: &str = "https://ollama.com/v1/systemone";
const CLEF_MODEL: &str = "clef";
const CLEF_PRICING: EvaluationPricing = EvaluationPricing {
	input_price_per_1m: 0.24,
	output_price_per_1m: 0.0,
};

#[derive(Debug, Clone, Default)]
pub struct OllamaEvaluationProvider;

impl OllamaEvaluationProvider {
	pub fn new() -> Self {
		Self
	}

	fn endpoint(&self) -> String {
		std::env::var(API_URL_ENV).unwrap_or_else(|_| DEFAULT_API_URL.to_string())
	}
}

pub(crate) fn request_body(request: &EvaluationRequest) -> Value {
	json!({
		"model": request.model.trim(),
		"state": request.state,
		"questions": request.questions,
	})
}

#[async_trait::async_trait]
impl EvaluationProvider for OllamaEvaluationProvider {
	fn name(&self) -> &str {
		PROVIDER
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.trim().is_empty()
	}

	/// Ollama Cloud's listed rate; `None` for a model the pricing page does not list.
	fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
		(model.trim() == CLEF_MODEL).then_some(CLEF_PRICING)
	}

	async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
		if request.uses_images() {
			return Err(EvaluationError::InvalidRequest(
				"images and box questions are only supported on the typesafe and openrouter providers".to_string(),
			));
		}
		request.validate()?;
		// Optional: a self-hosted server takes no key.
		let key = shared::api_key(API_KEY_ENV).ok();
		let url = self.endpoint();
		let body = request_body(&request);
		let response = shared::send(PROVIDER, &request, || {
			let builder = crate::http::http_client().post(&url).json(&body);
			Ok(match &key {
				Some(key) => builder.bearer_auth(key),
				None => builder,
			})
		})
		.await?;
		let payload = shared::parse_json(PROVIDER, &response)?;
		// A self-hosted server is not billed, so the cloud rate applies on the cloud endpoint only.
		let pricing = if url == DEFAULT_API_URL {
			self.get_model_pricing(&request.model)
		} else {
			None
		};
		shared::parse_payload(PROVIDER, &payload, pricing)
	}
}

#[cfg(test)]
#[path = "ollama_tests.rs"]
mod tests;
