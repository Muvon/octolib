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
//! and `jev-preview` are moving aliases. `jev-1.14.0` is the preview version;
//! `jev-latest` still points to `jev-1.13.0`. The response's `model` reports
//! which version answered.
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
use reqwest::multipart::{Form, Part};
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
    let mut body = json!({
        "state": request.state,
        "model": request.model.trim(),
        "questions": request.questions,
    });
    if !request.assets.is_empty() {
        body["assets"] = request
            .assets
            .iter()
            .enumerate()
            .map(|(index, (reference, asset))| {
                (
                    reference.clone(),
                    json!({
                        "type": "image",
                        "part": format!("image-{index}"),
                        "format": asset.format,
                    }),
                )
            })
            .collect();
    }
    body
}

fn file_parts(request: &EvaluationRequest) -> Vec<(String, &'static str, Vec<u8>)> {
    request
        .assets
        .values()
        .enumerate()
        .map(|(index, asset)| {
            (
                format!("image-{index}"),
                asset.format.mime(),
                asset.data.clone(),
            )
        })
        .collect()
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
        let files = file_parts(&request);
        let response = shared::send(PROVIDER, &request, || {
            let mut builder = crate::http::http_client().post(&url).bearer_auth(&key);
            if request.uses_images() {
                builder = builder.header("TypeSafe-Danger-Zone-Images", "true");
            }
            if files.is_empty() {
                return Ok(builder.json(&body));
            }
            // Forms are consumed by reqwest, so each retry needs a fresh one.
            let mut form = Form::new().part(
                "request",
                Part::text(body.to_string()).mime_str("application/json")?,
            );
            for (name, mime, bytes) in &files {
                let part = Part::bytes(bytes.clone())
                    .file_name(name.clone())
                    .mime_str(mime)?;
                form = form.part(name.clone(), part);
            }
            Ok(builder.multipart(form))
        })
        .await?;
        let payload = shared::parse_json(PROVIDER, &response)?;
        shared::parse_payload(PROVIDER, &payload, self.get_model_pricing(&request.model))
    }
}

#[cfg(test)]
#[path = "typesafe_tests.rs"]
mod tests;
