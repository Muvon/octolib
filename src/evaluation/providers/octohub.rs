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

//! Evaluation through an OctoHub proxy (`POST /v1/evaluations`).
//!
//! The hub resolves the model alias against its own `[evaluation_models]`
//! map and returns the same `{model, answers, usage}` payload as the
//! upstream adapters, plus `usage.cost` as it priced the call. Any model id
//! is accepted here; the hub decides what it serves.
//!
//! Configuration:
//! - `OCTOHUB_API_KEY`: required
//! - `OCTOHUB_API_URL`: optional base, default `https://hub.octomind.run`

use super::shared;
use crate::evaluation::errors::EvaluationResult;
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use serde_json::{json, Value};

const PROVIDER: &str = "octohub";
const API_KEY_ENV: &str = "OCTOHUB_API_KEY";
const API_URL_ENV: &str = "OCTOHUB_API_URL";
const DEFAULT_BASE_URL: &str = "https://hub.octomind.run";

#[derive(Debug, Clone, Default)]
pub struct OctoHubEvaluationProvider;

impl OctoHubEvaluationProvider {
    pub fn new() -> Self {
        Self
    }

    fn endpoint(&self) -> String {
        let base = std::env::var(API_URL_ENV).unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        format!("{}/v1/evaluations", base.trim_end_matches('/'))
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
impl EvaluationProvider for OctoHubEvaluationProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.trim().is_empty()
    }

    /// The hub prices the call itself and reports it in `usage.cost`.
    fn get_model_pricing(&self, _model: &str) -> Option<EvaluationPricing> {
        None
    }

    async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
        request.validate()?;
        let key = shared::api_key(API_KEY_ENV)?;
        let url = self.endpoint();
        let body = request_body(&request);
        let response = shared::send(PROVIDER, &request, || {
            crate::http::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let payload = shared::parse_json(PROVIDER, &response)?;
        let mut parsed = shared::parse_payload(PROVIDER, &payload, None)?;
        parsed.usage.cost = payload.pointer("/usage/cost").and_then(Value::as_f64);
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::types::Question;

    #[test]
    fn body_sends_the_alias_state_and_questions_flat() {
        let request = EvaluationRequest::new("Help!")
            .with_model("jev")
            .with_question("is_urgent", Question::noul("Does this convey urgency?"));
        assert_eq!(
            request_body(&request),
            json!({
                "model": "jev",
                "state": "Help!",
                "questions": {
                    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
                }
            })
        );
    }

    #[test]
    fn any_alias_is_accepted_and_pricing_is_left_to_the_hub() {
        let provider = OctoHubEvaluationProvider::new();
        assert!(provider.supports_model("jev"));
        assert!(provider.supports_model("typesafe:jev-latest"));
        assert!(!provider.supports_model("  "));
        assert!(provider.get_model_pricing("jev").is_none());
    }

    #[test]
    fn endpoint_honours_the_base_url_override() {
        let provider = OctoHubEvaluationProvider::new();
        std::env::set_var(API_URL_ENV, "http://127.0.0.1:8080/");
        assert_eq!(provider.endpoint(), "http://127.0.0.1:8080/v1/evaluations");
        std::env::remove_var(API_URL_ENV);
        assert_eq!(
            provider.endpoint(),
            "https://hub.octomind.run/v1/evaluations"
        );
    }
}
