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

//! Jev through Cloudflare AI Gateway unified billing.
//!
//! `POST /accounts/{account_id}/ai/run` with `{model: "typesafe/jev",
//! input: {state, questions}}`. The answer is double-wrapped (verified live,
//! Sep 2026): `{"result": {"state": "Completed", "result": {model, answers,
//! usage}, "gatewayMetadata": {...}}, "success": true}`. Billing comes from
//! the account's prepaid AI Gateway credits; no gateway header is needed.
//! The model is not listed by the Workers AI catalog, so support is static.
//!
//! Pricing (dashboard, Sep 2026): $0.042 per 1M input tokens, output free.
//!
//! Configuration:
//! - `CLOUDFLARE_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`: required
//! - `CLOUDFLARE_EVALUATION_API_URL`: optional full override of the run URL

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
const JEV_PRICING: EvaluationPricing = EvaluationPricing {
    input_price_per_1m: 0.042,
    output_price_per_1m: 0.0,
};

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
        if !account
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(EvaluationError::InvalidRequest(
                "CLOUDFLARE_ACCOUNT_ID must be an account identifier".to_string(),
            ));
        }
        Ok(format!("{API_BASE}/{account}/ai/run"))
    }
}

fn is_jev(model: &str) -> bool {
    model.trim().eq_ignore_ascii_case(JEV_MODEL)
}

pub(crate) fn request_body(request: &EvaluationRequest) -> Value {
    json!({
        "model": JEV_MODEL,
        "input": {
            "state": request.state,
            "questions": request.questions,
        },
    })
}

/// Peel the gateway envelope down to the Jev payload.
pub(crate) fn unwrap_envelope(value: &Value) -> EvaluationResult<&Value> {
    let invalid = |message: String| EvaluationError::InvalidResponse {
        provider: PROVIDER.to_string(),
        message,
    };
    if value.get("success").and_then(Value::as_bool) == Some(false) {
        return Err(EvaluationError::Api {
            provider: PROVIDER.to_string(),
            status: 200,
            message: shared::error_message(value.to_string().as_bytes()),
        });
    }
    let outer = value
        .get("result")
        .ok_or_else(|| invalid("response has no result field".to_string()))?;
    match outer.get("state").and_then(Value::as_str) {
        Some("Completed") => {}
        Some(state) => {
            return Err(invalid(format!(
                "gateway reported state '{state}' instead of Completed"
            )))
        }
        None => return Err(invalid("gateway result carries no state".to_string())),
    }
    outer
        .get("result")
        .ok_or_else(|| invalid("gateway result has no inner result".to_string()))
}

#[async_trait::async_trait]
impl EvaluationProvider for CloudflareEvaluationProvider {
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
        let url = self.run_url()?;
        let body = request_body(&request);
        let response = shared::send(PROVIDER, &request, || {
            crate::http::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let envelope = shared::parse_json(PROVIDER, &response)?;
        let payload = unwrap_envelope(&envelope)?;
        shared::parse_payload(PROVIDER, payload, Some(JEV_PRICING))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::types::{Answer, Question};

    #[test]
    fn body_nests_state_and_questions_under_input() {
        let request = EvaluationRequest::new("Help!")
            .with_model("typesafe/jev")
            .with_question("is_urgent", Question::noul("Does this convey urgency?"));
        assert_eq!(
            request_body(&request),
            json!({
                "model": "typesafe/jev",
                "input": {
                    "state": "Help!",
                    "questions": {
                        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
                    }
                }
            })
        );
    }

    #[test]
    fn envelope_is_unwrapped_only_when_completed() {
        // Trimmed from the live response.
        let live = json!({
            "result": {
                "state": "Completed",
                "result": {
                    "model": "jev-1.13.0",
                    "answers": {"is_urgent": {"type": "noul", "noul": 0.96}},
                    "usage": {"input_tokens": 429, "output_tokens": 73}
                },
                "gatewayMetadata": {"keySource": "Unified"}
            },
            "success": true,
            "errors": [],
            "messages": []
        });
        let payload = unwrap_envelope(&live).unwrap();
        let response = shared::parse_payload(PROVIDER, payload, Some(JEV_PRICING)).unwrap();
        assert_eq!(response.model, "jev-1.13.0");
        assert_eq!(response.answers["is_urgent"], Answer::Noul { noul: 0.96 });
        assert!((response.usage.cost.unwrap() - 429.0 * 0.042 / 1_000_000.0).abs() < 1e-15);

        let queued = json!({"result": {"state": "Queued", "id": "abc"}, "success": true});
        assert!(matches!(
            unwrap_envelope(&queued),
            Err(EvaluationError::InvalidResponse { message, .. }) if message.contains("Queued")
        ));

        let failed = json!({
            "result": {},
            "success": false,
            "errors": [{"code": 5035, "message": "not available on the Workers Free plan"}]
        });
        assert!(matches!(
            unwrap_envelope(&failed),
            Err(EvaluationError::Api { message, .. }) if message.contains("Free plan")
        ));
    }

    #[test]
    fn only_the_gateway_jev_id_is_served() {
        let provider = CloudflareEvaluationProvider::new();
        assert!(provider.supports_model("typesafe/jev"));
        assert!(provider.supports_model(" TypeSafe/Jev "));
        assert!(!provider.supports_model("jev-latest"));
        assert!(!provider.supports_model("@cf/zai-org/glm-5.3"));
        assert_eq!(
            provider.get_model_pricing("typesafe/jev"),
            Some(JEV_PRICING)
        );
    }

    #[test]
    fn run_url_honours_override_and_validates_account() {
        let provider = CloudflareEvaluationProvider::new();
        std::env::set_var(API_URL_ENV, "https://gateway.example/ai/run");
        assert_eq!(
            provider.run_url().unwrap(),
            "https://gateway.example/ai/run"
        );
        std::env::remove_var(API_URL_ENV);
        std::env::set_var(ACCOUNT_ID_ENV, "abc123");
        assert_eq!(
            provider.run_url().unwrap(),
            "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run"
        );
        std::env::set_var(ACCOUNT_ID_ENV, "abc/../123");
        assert!(provider.run_url().is_err());
        std::env::remove_var(ACCOUNT_ID_ENV);
    }
}
