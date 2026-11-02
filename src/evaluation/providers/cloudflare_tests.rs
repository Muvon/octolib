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
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

const JEV_PRICING: EvaluationPricing = MODELS[0].1;

// Trimmed from the live response, Nov 2026.
const CLEF_BODY: &str = r#"{"result":{"model":"clef-flash","answers":{"urgent":{"type":"noul","noul":0.9884},"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.6277,"infra":0.3723},"confidence":0.0652}},"usage":{"input_tokens":212,"output_tokens":0}},"success":true,"errors":[],"messages":[]}"#;

const CLEF_ERROR_BODY: &str = r#"{"errors":[{"message":"AiError: AiError: Request body failed validation: questions.where: Input tag 'box' found","code":3030}],"success":false,"result":{},"messages":[]}"#;

#[test]
fn gateway_body_nests_state_and_questions_under_input() {
	let request = EvaluationRequest::new("Help!")
		.with_model("typesafe/jev")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));
	assert_eq!(
		gateway_request_body(&request),
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
fn workers_ai_body_is_flat_with_the_short_model_name() {
	let request = EvaluationRequest::new("Help!")
		.with_model("@cf/cloudflare/clef-flash")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));
	assert_eq!(
		workers_ai_request_body("@cf/cloudflare/clef-flash", &request),
		json!({
			"model": "clef-flash",
			"state": "Help!",
			"questions": {
				"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
			}
		})
	);
}

#[test]
fn gateway_envelope_is_unwrapped_only_when_completed() {
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
	let payload = unwrap_gateway_envelope(&live).unwrap();
	let response = shared::parse_payload(PROVIDER, payload, Some(JEV_PRICING)).unwrap();
	assert_eq!(response.model, "jev-1.13.0");
	assert_eq!(response.answers["is_urgent"], Answer::Noul { noul: 0.96 });
	assert!((response.usage.cost.unwrap() - 429.0 * 0.042 / 1_000_000.0).abs() < 1e-15);

	let queued = json!({"result": {"state": "Queued", "id": "abc"}, "success": true});
	assert!(matches!(
		unwrap_gateway_envelope(&queued),
		Err(EvaluationError::InvalidResponse { message, .. }) if message.contains("Queued")
	));

	let failed = json!({
		"result": {},
		"success": false,
		"errors": [{"code": 5035, "message": "not available on the Workers Free plan"}]
	});
	assert!(matches!(
		unwrap_gateway_envelope(&failed),
		Err(EvaluationError::Api { message, .. }) if message.contains("Free plan")
	));
}

#[test]
fn workers_ai_envelope_holds_the_payload_directly() {
	let live: Value = serde_json::from_str(CLEF_BODY).unwrap();
	let payload = unwrap_workers_ai_envelope(&live).unwrap();
	let pricing = CloudflareEvaluationProvider::new()
		.get_model_pricing("@cf/cloudflare/clef")
		.unwrap();
	let response = shared::parse_payload(PROVIDER, payload, Some(pricing)).unwrap();
	assert_eq!(response.model, "clef-flash");
	assert_eq!(response.answers["urgent"], Answer::Noul { noul: 0.9884 });
	assert_eq!(response.usage.input_tokens, 212);
	assert!((response.usage.cost.unwrap() - 212.0 * 0.24 / 1_000_000.0).abs() < 1e-15);

	let failed: Value = serde_json::from_str(CLEF_ERROR_BODY).unwrap();
	assert!(matches!(
		unwrap_workers_ai_envelope(&failed),
		Err(EvaluationError::Api { message, .. }) if message.contains("Input tag 'box'")
	));

	let empty = json!({"success": true});
	assert!(matches!(
		unwrap_workers_ai_envelope(&empty),
		Err(EvaluationError::InvalidResponse { .. })
	));
}

#[test]
fn only_the_gateway_jev_id_is_served_on_the_gateway() {
	let provider = CloudflareEvaluationProvider::new();
	assert!(provider.supports_model("typesafe/jev"));
	assert!(provider.supports_model(" TypeSafe/Jev "));
	assert!(!provider.supports_model("jev-latest"));
	assert!(!provider.supports_model("@cf/zai-org/glm-5.3"));
	assert_eq!(provider.get_model_pricing("typesafe/jev"), Some(JEV_PRICING));
}

#[test]
fn clef_models_are_served_with_their_input_prices() {
	let provider = CloudflareEvaluationProvider::new();
	for (model, input_price_per_1m) in [
		("@cf/cloudflare/clef", 0.240),
		("@cf/cloudflare/clef-omni", 0.150),
		("@cf/cloudflare/clef-flash", 0.038),
	] {
		assert!(provider.supports_model(model));
		assert_eq!(
			provider.get_model_pricing(model),
			Some(EvaluationPricing {
				input_price_per_1m,
				output_price_per_1m: 0.0,
			})
		);
	}
	assert!(provider.supports_model(" @CF/Cloudflare/Clef-Flash "));
	assert!(provider.get_model_pricing(" @CF/Cloudflare/Clef-Flash ").is_some());
	assert!(!provider.supports_model("@cf/cloudflare/clef-pro"));
	assert!(!provider.supports_model("clef"));
	assert!(provider.get_model_pricing("clef").is_none());
}

#[test]
#[serial_test::serial]
fn run_url_honours_override_and_validates_account() {
	let provider = CloudflareEvaluationProvider::new();
	std::env::set_var(API_URL_ENV, "https://gateway.example/ai/run");
	assert_eq!(provider.run_url().unwrap(), "https://gateway.example/ai/run");
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

#[test]
#[serial_test::serial]
fn endpoint_appends_the_model_path_only_for_workers_ai() {
	let provider = CloudflareEvaluationProvider::new();
	{
		let _env = EnvGuard::new().set(API_URL_ENV, "https://gateway.example/ai/run");
		assert_eq!(
			provider.endpoint("@cf/cloudflare/clef").unwrap(),
			"https://gateway.example/ai/run/@cf/cloudflare/clef"
		);
		assert_eq!(provider.endpoint(JEV_MODEL).unwrap(), "https://gateway.example/ai/run");
	}
	let _env = EnvGuard::new().remove(API_URL_ENV).set(ACCOUNT_ID_ENV, "abc123");
	assert_eq!(
		provider.endpoint("@cf/cloudflare/clef-omni").unwrap(),
		"https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/cloudflare/clef-omni"
	);
}

#[tokio::test]
#[serial_test::serial]
async fn clef_evaluate_posts_to_the_model_path_and_prices_the_input() {
	let mock = MockHttp::start(vec![(200, CLEF_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("Payouts failing for 3 days.")
		.with_model(" @CF/Cloudflare/Clef-Flash ")
		.with_question("urgent", Question::noul("Does this convey urgency?"))
		.with_question(
			"team",
			Question::choice("Which team?", [("billing", "Money"), ("infra", "Servers")]),
		);
	let response = CloudflareEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect("mock evaluation");

	assert_eq!(response.answers["urgent"], Answer::Noul { noul: 0.9884 });
	assert!(matches!(
		&response.answers["team"],
		Answer::Choice { choice, .. } if choice == "billing"
	));
	assert_eq!(response.usage.input_tokens, 212);
	assert!((response.usage.cost.unwrap() - 212.0 * 0.038 / 1_000_000.0).abs() < 1e-15);

	let requests = mock.requests();
	assert!(requests[0].path.ends_with("/@cf/cloudflare/clef-flash"));
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let sent: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "clef-flash");
	assert_eq!(sent["state"], "Payouts failing for 3 days.");
	assert!(sent.get("input").is_none());
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
		EvaluationRequest::new("state").with_question("q", Question::bounding_box("Locate the cat")),
	];
	for request in requests {
		let request = request.with_model("typesafe/jev");
		assert!(request.validate().is_ok());
		assert!(matches!(
			provider.evaluate(request).await,
			Err(EvaluationError::InvalidRequest(message))
				if message == "images and box questions are only supported on the typesafe and openrouter providers"
		));
	}
}
