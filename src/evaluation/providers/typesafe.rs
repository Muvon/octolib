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

//! TypeSafe System One API (<https://docs.typesafe.ai/api>).
//!
//! `POST {base}/v1/systemone` with `{state, model, questions}` returns
//! `{model, answers, usage}`. Models are the `jev-*` family: `jev-latest`
//! and `jev-preview` are moving aliases, `jev-1.13.0` is the current
//! versioned id; the response's `model` reports which version answered.
//!
//! Pricing (models page, Sep 2026): $0.042 per 1M input tokens, output free.
//!
//! Configuration:
//! - `TYPESAFE_API_KEY`: required
//! - `TYPESAFE_BASE_URL`: optional API root, default `https://api.typesafe.ai`

use super::shared;
use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};
use serde_json::{json, Value};

const PROVIDER: &str = "typesafe";
const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
const API_BASE_ENV: &str = "TYPESAFE_BASE_URL";
const API_BASE: &str = "https://api.typesafe.ai";
const JEV_PRICING: EvaluationPricing = EvaluationPricing {
    input_price_per_1m: 0.042,
    output_price_per_1m: 0.0,
};

#[derive(Debug, Clone, Default)]
pub struct TypeSafeEvaluationProvider;

impl TypeSafeEvaluationProvider {
    pub fn new() -> Self {
        Self
    }

    fn endpoint(&self) -> String {
        let base = std::env::var(API_BASE_ENV).unwrap_or_else(|_| API_BASE.to_string());
        format!("{}/v1/systemone", base.trim_end_matches('/'))
    }
}

fn is_jev(model: &str) -> bool {
    let model = model.trim();
    model.starts_with("jev-")
        && model
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.'))
}

pub(crate) fn request_body(request: &EvaluationRequest) -> Value {
    json!({
        "state": request.state,
        "model": request.model.trim(),
        "questions": request.questions,
    })
}

#[async_trait::async_trait]
impl EvaluationProvider for TypeSafeEvaluationProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        is_jev(model)
    }

    fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
        is_jev(model).then_some(JEV_PRICING)
    }

    async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
        request.validate()?;
        if !is_jev(&request.model) {
            return Err(EvaluationError::UnsupportedModel {
                provider: PROVIDER.to_string(),
                model: request.model,
            });
        }
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
        shared::parse_payload(PROVIDER, &payload, self.get_model_pricing(&request.model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::types::Question;

    #[test]
    fn body_matches_the_documented_request() {
        let request = EvaluationRequest::new("Help! My payouts have been failing for 3 days.")
            .with_model("jev-latest")
            .with_question("is_urgent", Question::noul("Does this convey urgency?"));
        assert_eq!(
            request_body(&request),
            json!({
                "state": "Help! My payouts have been failing for 3 days.",
                "model": "jev-latest",
                "questions": {
                    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
                }
            })
        );
    }

    #[test]
    fn only_the_jev_family_is_served_and_priced() {
        let provider = TypeSafeEvaluationProvider::new();
        for model in ["jev-latest", "jev-preview", "jev-1.13.0"] {
            assert!(provider.supports_model(model));
            assert_eq!(provider.get_model_pricing(model), Some(JEV_PRICING));
        }
        assert!(!provider.supports_model("typesafe/jev"));
        assert!(!provider.supports_model("jev-latest/../models"));
        assert!(provider.get_model_pricing("gpt-5").is_none());
    }

    #[test]
    fn endpoint_honours_the_base_url_override() {
        let provider = TypeSafeEvaluationProvider::new();
        std::env::set_var(API_BASE_ENV, "https://proxy.example/typesafe/");
        assert_eq!(
            provider.endpoint(),
            "https://proxy.example/typesafe/v1/systemone"
        );
        std::env::remove_var(API_BASE_ENV);
        assert_eq!(provider.endpoint(), "https://api.typesafe.ai/v1/systemone");
    }
}
