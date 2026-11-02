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

//! Evaluation through OpenRouter's Decisions router
//! (`POST https://openrouter.ai/api/alpha/decisions`).
//!
//! The request is the Jev shape: `{model, state, questions}` plus, when images
//! are used, `assets` as a map of reference to
//! `{type: "image", format, data_base64}`. The answer is the usual
//! `{model, answers, usage{input_tokens, output_tokens, cost}}` payload, with
//! `usage.cost` as OpenRouter priced the call. Any model id is accepted here;
//! OpenRouter decides what it serves. `MODELS` lists the published input rates
//! of the decision models we route here (output is free on every one), so
//! callers can show a price before the first call. Features a model does not
//! support are rejected upstream with HTTP 400 (verified live, Nov 2026): for
//! example box questions on OpenAI models, and, on OpenAI, images only as
//! top-level state items.
//!
//! Configuration:
//! - `OPENROUTER_API_KEY`: required
//! - `OPENROUTER_EVALUATION_API_URL`: optional full URL override, default
//!   `https://openrouter.ai/api/alpha/decisions`

use super::shared;
use crate::evaluation::errors::EvaluationResult;
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Map, Value};

const PROVIDER: &str = "openrouter";
const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const API_URL_ENV: &str = "OPENROUTER_EVALUATION_API_URL";
const DEFAULT_URL: &str = "https://openrouter.ai/api/alpha/decisions";

/// Input USD per 1M tokens as OpenRouter lists them
/// (`/api/v1/models?output_modalities=decisions`, checked 2026-11-02).
const MODELS: &[(&str, f64)] = &[
	("openai/gpt-6-luna-decisions", 0.10),
	("perplexity/pplx-decider-v1.2", 0.02),
	("perplexity/pplx-decider-flash-v1.2", 0.01),
	("liquid/d1.1", 0.02),
	("liquid/d1-ultra", 0.042),
	("upstage/solar-decide", 0.05),
	("nace-ai/drex-max", 0.06),
	("microsoft/microsoft-decision-1", 0.042),
];

#[derive(Debug, Clone, Default)]
pub struct OpenRouterEvaluationProvider;

impl OpenRouterEvaluationProvider {
	pub fn new() -> Self {
		Self
	}

	fn endpoint(&self) -> String {
		std::env::var(API_URL_ENV).unwrap_or_else(|_| DEFAULT_URL.to_string())
	}
}

pub(crate) fn request_body(request: &EvaluationRequest) -> Value {
	let mut body = json!({
		"model": request.model.trim(),
		"state": request.state,
		"questions": request.questions,
	});
	if !request.assets.is_empty() {
		let assets: Map<String, Value> = request
			.assets
			.iter()
			.map(|(reference, asset)| {
				(
					reference.clone(),
					json!({
						"type": "image",
						"format": asset.format,
						"data_base64": STANDARD.encode(&asset.data),
					}),
				)
			})
			.collect();
		body["assets"] = Value::Object(assets);
	}
	body
}

#[async_trait::async_trait]
impl EvaluationProvider for OpenRouterEvaluationProvider {
	fn name(&self) -> &str {
		PROVIDER
	}

	fn supports_model(&self, model: &str) -> bool {
		!model.trim().is_empty()
	}

	/// The listed rate of a known model; OpenRouter's `usage.cost` stays the charge.
	fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
		let model = model.trim();
		MODELS
			.iter()
			.find(|(id, _)| id.eq_ignore_ascii_case(model))
			.map(|&(_, input_price_per_1m)| EvaluationPricing {
				input_price_per_1m,
				output_price_per_1m: 0.0,
			})
	}

	async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
		request.validate()?;
		let key = shared::api_key(API_KEY_ENV)?;
		let url = self.endpoint();
		let body = request_body(&request);
		let response = shared::send(PROVIDER, &request, || {
			Ok(crate::http::http_client().post(&url).bearer_auth(&key).json(&body))
		})
		.await?;
		let payload = shared::parse_json(PROVIDER, &response)?;
		let mut parsed = shared::parse_payload(PROVIDER, &payload, None)?;
		parsed.usage.cost = payload.pointer("/usage/cost").and_then(Value::as_f64);
		Ok(parsed)
	}
}

#[cfg(test)]
#[path = "openrouter_tests.rs"]
mod tests;
