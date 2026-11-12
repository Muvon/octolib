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

// Inception is deprecated (removal 2026-12-28); these tests exercise it until then.
#![allow(deprecated)]
use super::*;
use crate::evaluation::types::{Answer, Question};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

const CHOICE_BODY: &str = r#"{"model":"mercury-decide","answers":{"urgent":{"type":"noul","noul":0.9989012689316751},"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.9669140228030646,"infra":0.03308597719693549},"confidence":0.9338280456061292}},"usage":{"input_tokens":122,"output_tokens":2}}"#;

const SCORE_BODY: &str = r#"{"model":"mercury-decide","answers":{"how":{"type":"score","score":1.8792180608870601,"legend":{"0":"low","1":"medium","2":"high"},"probabilities":{"0":0.018634330928595448,"1":0.08351327725574878,"2":0.8978523918156557},"confidence":0.8188270913305904}},"usage":{"input_tokens":77,"output_tokens":1}}"#;

const NOT_FOUND_BODY: &str =
	r#"{"error":{"code":"model_not_found","type":"invalid_request_error","message":"Decision model not found."}}"#;

#[test]
fn body_matches_the_documented_request() {
	let request = EvaluationRequest::new("Help! My payouts have been failing for 3 days.")
		.with_model(" mercury-decide ")
		.with_question("urgent", Question::noul("Does this convey urgency?"));
	assert_eq!(
		request_body(&request),
		json!({
			"model": "mercury-decide",
			"state": "Help! My payouts have been failing for 3 days.",
			"questions": {
				"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
			}
		})
	);
}

#[test]
fn only_mercury_decide_is_served() {
	let provider = InceptionEvaluationProvider::new();
	assert!(provider.supports_model("mercury-decide"));
	assert!(provider.supports_model(" mercury-decide "));
	assert!(!provider.supports_model("mercury-decide-20260930"));
	assert!(!provider.supports_model("jev-latest"));
	assert!(!provider.supports_model("mercury-2"));
	assert!(!provider.supports_model(""));
}

#[test]
fn input_tokens_are_priced_and_output_is_free() {
	let provider = InceptionEvaluationProvider::new();
	assert_eq!(
		provider.get_model_pricing("mercury-decide"),
		Some(MERCURY_DECIDE_PRICING)
	);
	assert_eq!(MERCURY_DECIDE_PRICING.input_price_per_1m, 0.02);
	assert_eq!(MERCURY_DECIDE_PRICING.output_price_per_1m, 0.0);
	assert!(provider.get_model_pricing("mercury-decide-20260930").is_none());
	assert!(provider.get_model_pricing("jev-latest").is_none());
}

#[test]
#[serial_test::serial]
fn endpoint_honours_the_full_url_override() {
	let provider = InceptionEvaluationProvider::new();
	let _env = EnvGuard::new().set(API_URL_ENV, "http://127.0.0.1:8080/decisions");
	assert_eq!(provider.endpoint(), "http://127.0.0.1:8080/decisions");
	let _env = EnvGuard::new().remove(API_URL_ENV);
	assert_eq!(provider.endpoint(), "https://api.inceptionlabs.ai/v1/decisions");
}

#[tokio::test]
async fn images_and_box_questions_are_rejected_before_sending() {
	use crate::evaluation::image::tests::jpeg;
	use crate::evaluation::types::ImageAsset;

	let provider = InceptionEvaluationProvider::new();
	let requests = [
		EvaluationRequest::new("photo-ref")
			.with_question("q", Question::noul("Is there a cat?"))
			.with_image("photo-ref", ImageAsset::jpeg(jpeg(512, 512))),
		EvaluationRequest::new("state").with_question("q", Question::bounding_box("Locate the cat")),
	];
	for request in requests {
		let request = request.with_model("mercury-decide");
		assert!(request.validate().is_ok());
		assert!(matches!(
			provider.evaluate(request).await,
			Err(EvaluationError::InvalidRequest(message))
				if message == "images and box questions are only supported on the typesafe and openrouter providers"
		));
	}
}

#[tokio::test]
async fn unsupported_model_is_rejected_before_sending() {
	let request = EvaluationRequest::new("state")
		.with_model("mercury-decide-20260930")
		.with_question("q", Question::noul("Is it urgent?"));
	assert!(matches!(
		InceptionEvaluationProvider::new().evaluate(request).await,
		Err(EvaluationError::UnsupportedModel { provider, model })
			if provider == "inception" && model == "mercury-decide-20260930"
	));
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_parses_answers_and_prices_input_tokens_only() {
	let mock = MockHttp::start(vec![(200, CHOICE_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("Payouts failing for 3 days.")
		.with_model(" mercury-decide ")
		.with_question("urgent", Question::noul("Does this convey urgency?"))
		.with_question(
			"team",
			Question::choice("Which team?", [("billing", "Money"), ("infra", "Servers")]),
		);
	let response = InceptionEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect("mock evaluation");

	assert_eq!(response.model, "mercury-decide");
	// serde_json parses long decimals to within one ULP, so compare with a tolerance.
	assert!(matches!(
		response.answers["urgent"],
		Answer::Noul { noul } if (noul - 0.9989012689316751).abs() < 1e-12
	));
	assert!(matches!(
		&response.answers["team"],
		Answer::Choice { choice, .. } if choice == "billing"
	));
	assert_eq!(response.usage.input_tokens, 122);
	assert_eq!(response.usage.output_tokens, 2);
	let cost = response.usage.cost.expect("local pricing");
	assert!((cost - 122.0 * 0.02 / 1e6).abs() < 1e-12);

	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let sent: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "mercury-decide");
	assert_eq!(sent["state"], "Payouts failing for 3 days.");
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_parses_a_score_answer() {
	let mock = MockHttp::start(vec![(200, SCORE_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("Server is down.")
		.with_model("mercury-decide")
		.with_question("how", Question::score("How severe?", ["low", "medium", "high"]));
	let response = InceptionEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect("mock evaluation");

	assert!(matches!(
		&response.answers["how"],
		Answer::Score { score, .. } if (*score - 1.8792180608870601).abs() < 1e-12
	));
	assert_eq!(response.usage.input_tokens, 77);
	assert_eq!(response.usage.output_tokens, 1);
}

#[tokio::test]
#[serial_test::serial]
async fn upstream_not_found_surfaces_the_error_message() {
	let mock = MockHttp::start(vec![(404, NOT_FOUND_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("state")
		.with_model("mercury-decide")
		.with_question("q", Question::noul("Is it urgent?"));
	let error = InceptionEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect_err("upstream 404");

	assert!(matches!(
		error,
		EvaluationError::Api { status: 404, message, .. } if message == "Decision model not found."
	));
}
