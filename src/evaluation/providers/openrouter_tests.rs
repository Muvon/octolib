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
use crate::evaluation::errors::EvaluationError;
use crate::evaluation::types::{Answer, ImageAsset, Question};
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

const TEXT_BODY: &str = r#"{"model":"openai/gpt-6-luna-decisions-20261006","answers":{"is_urgent":{"type":"noul","noul":0.98},"team":{"type":"choice","choice":"billing","probabilities":{"billing":1,"support":0},"confidence":1},"severity":{"type":"score","score":1.98,"legend":{"0":"low","1":"medium","2":"high"},"probabilities":{"0":0,"1":0.02,"2":0.98},"confidence":0.97}},"usage":{"input_tokens":388,"output_tokens":0,"cost":0.0000388},"id":"gen-dec-1793589593-iHcwCt8QTgUhNuaiPzqF","provider":"OpenAI"}"#;

const ERROR_BODY: &str =
	r#"{"error":{"message":"OpenAI does not accept box questions; question \"where\" is a box","code":400}}"#;

#[test]
fn body_sends_the_model_state_and_questions_without_assets() {
	let request = EvaluationRequest::new("Help!")
		.with_model(" openai/gpt-6-luna-decisions ")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));
	assert_eq!(
		request_body(&request),
		json!({
			"model": "openai/gpt-6-luna-decisions",
			"state": "Help!",
			"questions": {
				"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
			}
		})
	);
}

#[test]
fn body_encodes_image_assets_as_base64() {
	use crate::evaluation::image::tests::png;

	let bytes = png(512, 512);
	let request = EvaluationRequest::new("photo-ref")
		.with_model("typesafe/jev")
		.with_question("red", Question::noul("Is it red?"))
		.with_image("photo-ref", ImageAsset::png(bytes.clone()));
	assert!(request.validate().is_ok());
	assert_eq!(
		request_body(&request)["assets"],
		json!({
			"photo-ref": {
				"type": "image",
				"format": "png",
				"data_base64": STANDARD.encode(&bytes),
			}
		})
	);
}

#[test]
fn any_model_is_accepted_and_pricing_is_left_to_openrouter() {
	let provider = OpenRouterEvaluationProvider::new();
	assert!(provider.supports_model("openai/gpt-6-luna-decisions"));
	assert!(!provider.supports_model("  "));
	assert!(provider.get_model_pricing("openai/gpt-6-luna-decisions").is_none());
}

#[test]
#[serial_test::serial]
fn endpoint_honours_the_full_url_override() {
	let provider = OpenRouterEvaluationProvider::new();
	let _env = EnvGuard::new().set(API_URL_ENV, "http://127.0.0.1:8080/decisions");
	assert_eq!(provider.endpoint(), "http://127.0.0.1:8080/decisions");
	let _env = EnvGuard::new().remove(API_URL_ENV);
	assert_eq!(provider.endpoint(), "https://openrouter.ai/api/alpha/decisions");
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_parses_answers_and_reports_the_upstream_cost() {
	let mock = MockHttp::start(vec![(200, TEXT_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("Payouts failing for 3 days.")
		.with_model(" openai/gpt-6-luna-decisions ")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"))
		.with_question(
			"team",
			Question::choice("Which team?", [("billing", "Money"), ("support", "Help")]),
		)
		.with_question("severity", Question::score("How severe?", ["low", "medium", "high"]));
	let response = OpenRouterEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect("mock evaluation");

	assert_eq!(response.model, "openai/gpt-6-luna-decisions-20261006");
	assert_eq!(response.answers["is_urgent"], Answer::Noul { noul: 0.98 });
	assert!(matches!(
		&response.answers["team"],
		Answer::Choice { choice, .. } if choice == "billing"
	));
	assert!(matches!(
		&response.answers["severity"],
		Answer::Score { score, .. } if (*score - 1.98).abs() < 1e-12
	));
	assert_eq!(response.usage.input_tokens, 388);
	assert_eq!(response.usage.output_tokens, 0);
	assert_eq!(response.usage.cost, Some(0.0000388));

	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let sent: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "openai/gpt-6-luna-decisions");
	assert!(sent.get("assets").is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn upstream_rejection_surfaces_the_error_message() {
	let mock = MockHttp::start(vec![(400, ERROR_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	let request = EvaluationRequest::new("state")
		.with_model("openai/gpt-6-luna-decisions")
		.with_question("where", Question::bounding_box("Locate the cat"));
	let error = OpenRouterEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect_err("upstream 400");

	// shared::require_success maps 400 to Api; only 422 becomes InvalidRequest.
	assert!(matches!(
		error,
		EvaluationError::Api { status: 400, message, .. } if message.contains("box questions")
	));
}
