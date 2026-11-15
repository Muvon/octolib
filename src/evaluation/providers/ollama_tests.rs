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

// Follows the example in Ollama's decision docs
// (https://docs.ollama.com/capabilities/decision), read 2026-11-14.
const CHOICE_BODY: &str = r#"{"model":"nimble","answers":{"label":{"type":"choice","choice":"bug","probabilities":{"billing":0.0125,"bug":0.9781,"account":0.0093},"confidence":0.8906}},"usage":{"input_tokens":174,"output_tokens":1}}"#;

#[test]
fn body_sends_the_trimmed_model_state_and_questions() {
	let request = EvaluationRequest::new("Help!")
		.with_model(" tev1:0.8b ")
		.with_question("is_urgent", Question::noul("Does this convey urgency?"));
	assert_eq!(
		request_body(&request),
		json!({
			"model": "tev1:0.8b",
			"state": "Help!",
			"questions": {
				"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
			}
		})
	);
}

#[test]
fn any_non_empty_model_is_accepted_and_only_clef_has_a_listed_rate() {
	let provider = OllamaEvaluationProvider::new();
	assert!(provider.supports_model("clef-flash"));
	assert!(provider.supports_model("tev1:0.8b"));
	assert!(!provider.supports_model("  "));
	assert!(!provider.supports_model(""));
	assert_eq!(provider.get_model_pricing(" clef "), Some(CLEF_PRICING));
	assert!(provider.get_model_pricing("clef-flash").is_none());
}

#[test]
#[serial_test::serial]
fn endpoint_honours_the_full_url_override() {
	let provider = OllamaEvaluationProvider::new();
	let _env = EnvGuard::new().set(API_URL_ENV, "http://localhost:11434/v1/systemone");
	assert_eq!(provider.endpoint(), "http://localhost:11434/v1/systemone");
	let _env = EnvGuard::new().remove(API_URL_ENV);
	assert_eq!(provider.endpoint(), "https://ollama.com/v1/systemone");
}

#[tokio::test]
async fn images_and_box_questions_are_rejected_before_sending() {
	use crate::evaluation::image::tests::jpeg;
	use crate::evaluation::types::ImageAsset;

	let provider = OllamaEvaluationProvider::new();
	let requests = [
		EvaluationRequest::new("photo-ref")
			.with_question("q", Question::noul("Is there a cat?"))
			.with_image("photo-ref", ImageAsset::jpeg(jpeg(512, 512))),
		EvaluationRequest::new("state").with_question("q", Question::bounding_box("Locate the cat")),
	];
	for request in requests {
		let request = request.with_model("clef");
		assert!(request.validate().is_ok());
		assert!(matches!(
			provider.evaluate(request).await,
			Err(EvaluationError::InvalidRequest(message))
				if message == "images and box questions are only supported on the typesafe and openrouter providers"
		));
	}
}

fn label_request(model: &str) -> EvaluationRequest {
	EvaluationRequest::new("The export button crashes the app.")
		.with_model(model)
		.with_question(
			"label",
			Question::choice(
				"Which label fits?",
				[("billing", "Money"), ("bug", "Defect"), ("account", "Login")],
			),
		)
}

#[tokio::test]
#[serial_test::serial]
async fn a_self_hosted_server_gets_no_authorization_header_and_reports_no_cost() {
	let mock = MockHttp::start(vec![(200, CHOICE_BODY.to_string())]);
	let _env = EnvGuard::new().set(API_URL_ENV, mock.base_url()).remove(API_KEY_ENV);

	// `clef` has a cloud rate, and it must not be charged to a server behind the override.
	let response = OllamaEvaluationProvider::new()
		.evaluate(label_request(" clef "))
		.await
		.expect("mock evaluation");

	assert_eq!(response.model, "nimble");
	assert!(matches!(
		&response.answers["label"],
		Answer::Choice { choice, .. } if choice == "bug"
	));
	assert_eq!(response.usage.input_tokens, 174);
	assert_eq!(response.usage.output_tokens, 1);
	assert!(response.usage.cost.is_none());

	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), None);
	let sent: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "clef");
	assert_eq!(sent["state"], "The export button crashes the app.");
}

#[tokio::test]
#[serial_test::serial]
async fn a_set_key_is_sent_as_a_bearer_token() {
	let mock = MockHttp::start(vec![(200, CHOICE_BODY.to_string())]);
	let _env = EnvGuard::new()
		.set(API_URL_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");

	OllamaEvaluationProvider::new()
		.evaluate(label_request("nimble"))
		.await
		.expect("mock evaluation");

	assert_eq!(mock.requests()[0].header("authorization"), Some("Bearer mock-key"));
}
