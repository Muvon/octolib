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

//! Cloudflare evaluation models, over two routes.
//!
//! Jev (`typesafe/jev`) goes through AI Gateway unified billing:
//! `POST /accounts/{account_id}/ai/run` with `{model: "typesafe/jev",
//! input: {state, questions}}`. The answer is double-wrapped (verified live,
//! Sep 2026): `{"result": {"state": "Completed", "result": {model, answers,
//! usage}, "gatewayMetadata": {...}}, "success": true}`. Billing comes from
//! the account's prepaid AI Gateway credits; no gateway header is needed.
//! The model is not listed by the Workers AI catalog, so support is static.
//!
//! Clef (`@cf/cloudflare/clef`, `clef-flash`, `clef-omni`) runs on Workers AI:
//! `POST /accounts/{account_id}/ai/run/@cf/cloudflare/{name}` with a flat
//! body `{model: "{name}", state, questions}` (verified live, Nov 2026). The
//! answer has a single envelope with the payload directly under `result`:
//! `{"result": {model, answers, usage}, "success": true}`. Invalid requests
//! are HTTP 422 with `{"errors": [{"message": ...}], "success": false}`.
//!
//! Pricing per 1M input tokens, output free. Jev: $0.042 (dashboard, Sep
//! 2026). Clef: $0.240, clef-flash: $0.038, clef-omni: $0.150 (Workers AI
//! pricing page, Nov 2026).
//!
//! Images and box questions are rejected before sending for every model here.
//! Clef rejects `assets` and `box` upstream (422); its images are an
//! unreferenced `images` list of data URLs, which the reference-based assets
//! cannot map onto. Clef-omni audio and video are not representable either.
//!
//! Configuration:
//! - `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`: required
//! - `CLOUDFLARE_EVALUATION_API_URL`: optional replacement for the run URL
//!   (`.../ai/run`). Jev posts to it as is; Clef appends `/{model id}`.

use super::shared;
use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use serde_json::{json, Value};

const PROVIDER: &str = "cloudflare";
const API_KEY_ENV: &str = "CLOUDFLARE_API_KEY";
const ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
const API_URL_ENV: &str = "CLOUDFLARE_EVALUATION_API_URL";
const API_BASE: &str = "https://api.cloudflare.com/client/v4/accounts";
const JEV_MODEL: &str = "typesafe/jev";
const WORKERS_AI_PREFIX: &str = "@cf/cloudflare/";

/// Canonical model id and its price; the only models this provider serves.
const MODELS: &[(&str, EvaluationPricing)] = &[
	(
		JEV_MODEL,
		EvaluationPricing {
			input_price_per_1m: 0.042,
			output_price_per_1m: 0.0,
		},
	),
	(
		"@cf/cloudflare/clef",
		EvaluationPricing {
			input_price_per_1m: 0.240,
			output_price_per_1m: 0.0,
		},
	),
	(
		"@cf/cloudflare/clef-omni",
		EvaluationPricing {
			input_price_per_1m: 0.150,
			output_price_per_1m: 0.0,
		},
	),
	(
		"@cf/cloudflare/clef-flash",
		EvaluationPricing {
			input_price_per_1m: 0.038,
			output_price_per_1m: 0.0,
		},
	),
];

#[derive(Debug, Clone, Default)]
pub struct CloudflareEvaluationProvider;

impl CloudflareEvaluationProvider {
	pub fn new() -> Self {
		Self
	}

	fn run_url(&self) -> EvaluationResult<String> {
		if let Ok(url) = std::env::var(API_URL_ENV) {
			return Ok(url);
		}
		let account = shared::api_key(ACCOUNT_ID_ENV)?;
		if !account.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'-') {
			return Err(EvaluationError::InvalidRequest(
				"CLOUDFLARE_ACCOUNT_ID must be an account identifier".to_string(),
			));
		}
		Ok(format!("{API_BASE}/{account}/ai/run"))
	}

	/// Jev posts to the run URL itself; Workers AI models append their id.
	fn endpoint(&self, model_id: &str) -> EvaluationResult<String> {
		let run_url = self.run_url()?;
		if model_id == JEV_MODEL {
			return Ok(run_url);
		}
		Ok(format!("{run_url}/{model_id}"))
	}
}

fn find_model(model: &str) -> Option<&'static (&'static str, EvaluationPricing)> {
	let model = model.trim();
	MODELS.iter().find(|(id, _)| id.eq_ignore_ascii_case(model))
}

pub(crate) fn gateway_request_body(request: &EvaluationRequest) -> Value {
	json!({
		"model": JEV_MODEL,
		"input": {
			"state": request.state,
			"questions": request.questions,
		},
	})
}

/// `model_id` is the canonical id; the body names the model without its prefix.
pub(crate) fn workers_ai_request_body(model_id: &str, request: &EvaluationRequest) -> Value {
	json!({
		"model": model_id.trim_start_matches(WORKERS_AI_PREFIX),
		"state": request.state,
		"questions": request.questions,
	})
}

fn invalid_response(message: String) -> EvaluationError {
	EvaluationError::InvalidResponse {
		provider: PROVIDER.to_string(),
		message,
	}
}

/// Cloudflare reports some failures as HTTP 200 with `success: false`.
fn reject_unsuccessful(value: &Value) -> EvaluationResult<()> {
	if value.get("success").and_then(Value::as_bool) == Some(false) {
		return Err(EvaluationError::Api {
			provider: PROVIDER.to_string(),
			status: 200,
			message: shared::error_message(value.to_string().as_bytes()),
		});
	}
	Ok(())
}

/// Peel the gateway envelope down to the Jev payload.
pub(crate) fn unwrap_gateway_envelope(value: &Value) -> EvaluationResult<&Value> {
	reject_unsuccessful(value)?;
	let outer = value
		.get("result")
		.ok_or_else(|| invalid_response("response has no result field".to_string()))?;
	match outer.get("state").and_then(Value::as_str) {
		Some("Completed") => {}
		Some(state) => {
			return Err(invalid_response(format!(
				"gateway reported state '{state}' instead of Completed"
			)))
		}
		None => return Err(invalid_response("gateway result carries no state".to_string())),
	}
	outer
		.get("result")
		.ok_or_else(|| invalid_response("gateway result has no inner result".to_string()))
}

/// The Workers AI envelope carries the payload directly under `result`.
pub(crate) fn unwrap_workers_ai_envelope(value: &Value) -> EvaluationResult<&Value> {
	reject_unsuccessful(value)?;
	value
		.get("result")
		.ok_or_else(|| invalid_response("response has no result field".to_string()))
}

#[async_trait::async_trait]
impl EvaluationProvider for CloudflareEvaluationProvider {
	fn name(&self) -> &str {
		PROVIDER
	}

	fn supports_model(&self, model: &str) -> bool {
		find_model(model).is_some()
	}

	fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
		find_model(model).map(|(_, pricing)| *pricing)
	}

	async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
		if request.uses_images() {
			return Err(EvaluationError::InvalidRequest(
				"images and box questions are only supported on the typesafe and openrouter providers".to_string(),
			));
		}
		request.validate()?;
		let &(model_id, pricing) = find_model(&request.model).ok_or_else(|| EvaluationError::UnsupportedModel {
			provider: PROVIDER.to_string(),
			model: request.model.clone(),
		})?;
		let key = shared::api_key(API_KEY_ENV)?;
		let url = self.endpoint(model_id)?;
		let is_gateway = model_id == JEV_MODEL;
		let body = if is_gateway {
			gateway_request_body(&request)
		} else {
			workers_ai_request_body(model_id, &request)
		};
		let response = shared::send(PROVIDER, &request, || {
			Ok(crate::http::http_client().post(&url).bearer_auth(&key).json(&body))
		})
		.await?;
		let envelope = shared::parse_json(PROVIDER, &response)?;
		let payload = if is_gateway {
			unwrap_gateway_envelope(&envelope)?
		} else {
			unwrap_workers_ai_envelope(&envelope)?
		};
		shared::parse_payload(PROVIDER, payload, Some(pricing))
	}
}

#[cfg(test)]
#[path = "cloudflare_tests.rs"]
mod tests;
