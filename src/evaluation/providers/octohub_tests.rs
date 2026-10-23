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
#[serial_test::serial]
fn endpoint_honours_the_base_url_override() {
	let provider = OctoHubEvaluationProvider::new();
	std::env::set_var(API_URL_ENV, "http://127.0.0.1:8080/");
	assert_eq!(provider.endpoint(), "http://127.0.0.1:8080/v1/evaluations");
	std::env::remove_var(API_URL_ENV);
	assert_eq!(provider.endpoint(), "https://hub.octomind.run/v1/evaluations");
}

#[tokio::test]
async fn images_and_box_questions_are_rejected_before_sending() {
	use crate::evaluation::image::tests::jpeg;
	use crate::evaluation::types::ImageAsset;

	let provider = OctoHubEvaluationProvider::new();
	let requests = [
		EvaluationRequest::new("photo-ref")
			.with_question("q", Question::noul("Is there a cat?"))
			.with_image("photo-ref", ImageAsset::jpeg(jpeg(512, 512))),
		EvaluationRequest::new("state").with_question("q", Question::bounding_box("Locate the cat")),
	];
	for request in requests {
		let request = request.with_model("jev");
		assert!(request.validate().is_ok());
		assert!(matches!(
			provider.evaluate(request).await,
			Err(EvaluationError::InvalidRequest(message))
				if message == "images and box questions are only supported on the typesafe provider"
		));
	}
}
