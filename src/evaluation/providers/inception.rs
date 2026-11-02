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

//! Inception Mercury Decide (`POST https://api.inceptionlabs.ai/v1/decisions`).
//!
//! The body is `{model, state, questions}` and the reply is the same
//! `{model, answers, usage}` payload as the Jev hosts. Only `mercury-decide`
//! is served: the dated `mercury-decide-20260930` answers 404 "Decision model
//! not found." on Inception's API.
//!
//! Images and box questions are refused before sending. Inception rejects box
//! questions (422 `union_tag_invalid`) and accepts image assets but ignores
//! them (an all-red 320x320 PNG scored `noul` 0.38 with 67 input tokens, live
//! probe 2026-11-02), so a request carrying one would silently get a
//! text-only answer.
//!
//! Pricing (Inception docs, Models page, 2026-11-02): $0.02 per 1M input
//! tokens (a 50% launch discount from $0.04); output tokens are reported but
//! free.
//!
//! Configuration:
//! - `INCEPTION_API_KEY`: required
//! - `INCEPTION_EVALUATION_API_URL`: optional full endpoint URL, default
//!   `https://api.inceptionlabs.ai/v1/decisions`

use super::shared;
use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use serde_json::{json, Value};

const PROVIDER: &str = "inception";
const API_KEY_ENV: &str = "INCEPTION_API_KEY";
const API_URL_ENV: &str = "INCEPTION_EVALUATION_API_URL";
const DEFAULT_API_URL: &str = "https://api.inceptionlabs.ai/v1/decisions";
const MODEL: &str = "mercury-decide";
const MERCURY_DECIDE_PRICING: EvaluationPricing = EvaluationPricing {
	input_price_per_1m: 0.02,
	output_price_per_1m: 0.0,
};

#[derive(Debug, Clone, Default)]
pub struct InceptionEvaluationProvider;

impl InceptionEvaluationProvider {
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
impl EvaluationProvider for InceptionEvaluationProvider {
	fn name(&self) -> &str {
		PROVIDER
	}

	fn supports_model(&self, model: &str) -> bool {
		model.trim() == MODEL
	}

	fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
		self.supports_model(model).then_some(MERCURY_DECIDE_PRICING)
	}

	async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
		if request.uses_images() {
			return Err(EvaluationError::InvalidRequest(
				"images and box questions are only supported on the typesafe and openrouter providers".to_string(),
			));
		}
		request.validate()?;
		if !self.supports_model(&request.model) {
			return Err(EvaluationError::UnsupportedModel {
				provider: PROVIDER.to_string(),
				model: request.model,
			});
		}
		let key = shared::api_key(API_KEY_ENV)?;
		let url = self.endpoint();
		let body = request_body(&request);
		let response = shared::send(PROVIDER, &request, || {
			Ok(crate::http::http_client().post(&url).bearer_auth(&key).json(&body))
		})
		.await?;
		let payload = shared::parse_json(PROVIDER, &response)?;
		shared::parse_payload(PROVIDER, &payload, self.get_model_pricing(&request.model))
	}
}

#[cfg(test)]
#[path = "inception_tests.rs"]
mod tests;
