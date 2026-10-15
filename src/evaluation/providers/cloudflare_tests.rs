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

#[tokio::test]
async fn images_and_box_questions_are_rejected_before_sending() {
    use crate::evaluation::image::tests::png;
    use crate::evaluation::types::ImageAsset;

    let provider = CloudflareEvaluationProvider::new();
    let requests = [
        EvaluationRequest::new("photo-ref")
            .with_question("q", Question::noul("Is there a cat?"))
            .with_image("photo-ref", ImageAsset::png(png(512, 512))),
        EvaluationRequest::new("state")
            .with_question("q", Question::bounding_box("Locate the cat")),
    ];
    for request in requests {
        let request = request.with_model("typesafe/jev");
        assert!(request.validate().is_ok());
        assert!(matches!(
            provider.evaluate(request).await,
            Err(EvaluationError::InvalidRequest(message))
                if message == "images and box questions are only supported on the typesafe provider"
        ));
    }
}
