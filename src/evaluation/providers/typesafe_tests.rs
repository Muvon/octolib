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
#[serial_test::serial]
fn endpoint_honours_the_base_url_override() {
	let provider = TypeSafeEvaluationProvider::new();
	std::env::set_var(API_BASE_ENV, "https://proxy.example/typesafe/");
	assert_eq!(provider.endpoint(), "https://proxy.example/typesafe/v1/systemone");
	std::env::remove_var(API_BASE_ENV);
	assert_eq!(provider.endpoint(), "https://api.typesafe.ai/v1/systemone");
}

#[test]
fn assets_and_file_parts_follow_reference_order() {
	use crate::evaluation::image::tests::{jpeg, png};
	use crate::evaluation::types::ImageAsset;

	let request = EvaluationRequest::new("a-photo and z-photo")
		.with_model("jev-preview")
		.with_question("q", Question::bounding_box("Locate the cat"))
		.with_image("z-photo", ImageAsset::jpeg(jpeg(512, 512)))
		.with_image("a-photo", ImageAsset::png(png(512, 512)));
	assert_eq!(
		request_body(&request),
		json!({
			"model": "jev-preview",
			"state": "a-photo and z-photo",
			"questions": {"q": {"type": "box", "instructions": "Locate the cat"}},
			"assets": {
				"a-photo": {"type": "image", "part": "image-0", "format": "png"},
				"z-photo": {"type": "image", "part": "image-1", "format": "jpg"}
			}
		})
	);
	assert_eq!(
		file_parts(&request),
		vec![
			("image-0".to_string(), "image/png", png(512, 512)),
			("image-1".to_string(), "image/jpeg", jpeg(512, 512)),
		]
	);
}

#[test]
fn requests_without_assets_have_no_file_parts_or_assets_field() {
	let request = EvaluationRequest::new("state")
		.with_model("jev-preview")
		.with_question("q", Question::bounding_box("Locate the cat"));
	assert!(file_parts(&request).is_empty());
	assert!(request_body(&request).get("assets").is_none());
}
