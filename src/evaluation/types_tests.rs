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
use crate::evaluation::image::tests::{apng, jpeg, png};
use serde_json::json;

#[test]
fn questions_serialize_to_the_documented_wire_shape() {
	let request = EvaluationRequest::new("Help! My payouts have been failing for 3 days.")
		.with_question(
			"is_urgent",
			Question::noul_with_criteria(
				"Does this convey urgency?",
				"Explicitly time-sensitive",
				"No urgency expressed",
			),
		)
		.with_question(
			"department",
			Question::choice(
				"Which team should handle this?",
				[
					("billing", "Payments, invoicing, refunds"),
					("technical", "Bugs, outages, integrations"),
				],
			),
		)
		.with_question(
			"frustration",
			Question::score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
		);
	assert_eq!(
		serde_json::to_value(&request.questions).unwrap(),
		json!({
			"is_urgent": {
				"type": "noul",
				"instructions": "Does this convey urgency?",
				"criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
			},
			"department": {
				"type": "choice",
				"instructions": "Which team should handle this?",
				"criteria": {
					"billing": "Payments, invoicing, refunds",
					"technical": "Bugs, outages, integrations"
				}
			},
			"frustration": {
				"type": "score",
				"instructions": "How frustrated is the customer?",
				"criteria": ["Calm", "Frustrated", "Very angry"]
			}
		})
	);
	// A bare noul carries no criteria key at all.
	assert_eq!(
		serde_json::to_value(Question::noul("Is it urgent?")).unwrap(),
		json!({"type": "noul", "instructions": "Is it urgent?"})
	);
}

#[test]
fn answers_parse_from_the_live_response_shape() {
	let answers: BTreeMap<String, Answer> = serde_json::from_value(json!({
		"is_urgent": {"type": "noul", "noul": 0.96},
		"department": {
			"type": "choice",
			"choice": "billing",
			"probabilities": {"sales": 0, "billing": 0.93, "technical": 0.07},
			"confidence": 0.89
		},
		"frustration": {
			"type": "score",
			"score": 1.39,
			"legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
			"probabilities": {"0": 0, "1": 0.61, "2": 0.39},
			"confidence": 0.41
		}
	}))
	.unwrap();
	assert_eq!(answers["is_urgent"], Answer::Noul { noul: 0.96 });
	match &answers["department"] {
		Answer::Choice {
			choice,
			probabilities,
			confidence,
		} => {
			assert_eq!(choice, "billing");
			assert_eq!(probabilities["billing"], 0.93);
			assert_eq!(*confidence, 0.89);
		}
		other => panic!("unexpected {other:?}"),
	}
	match &answers["frustration"] {
		Answer::Score { score, legend, .. } => {
			assert_eq!(*score, 1.39);
			assert_eq!(legend["2"], "Very angry");
		}
		other => panic!("unexpected {other:?}"),
	}
}

#[test]
fn requests_are_validated_before_they_leave() {
	let empty = EvaluationRequest::new("state").with_model("jev-latest");
	assert!(empty.validate().is_err());

	let no_model = EvaluationRequest::new("state").with_question("q", Question::noul("Yes?"));
	assert!(no_model.validate().is_err());

	let one_level = EvaluationRequest::new("state")
		.with_model("jev-latest")
		.with_question("q", Question::score("Rate it", ["only"]));
	assert!(one_level.validate().is_err());

	let one_option = EvaluationRequest::new("state")
		.with_model("jev-latest")
		.with_question("q", Question::choice("Pick", [("a", "A")]));
	assert!(one_option.validate().is_err());

	let blank = EvaluationRequest::new("state")
		.with_model("jev-latest")
		.with_question("q", Question::noul("   "));
	assert!(blank.validate().is_err());

	let ok = EvaluationRequest::new(json!({"ticket": {"subject": "Duplicate charge"}}))
		.with_model("jev-latest")
		.with_question("q", Question::noul("Refund requested?"));
	assert!(ok.validate().is_ok());
}

#[test]
fn pricing_bills_input_only_when_output_is_free() {
	let pricing = EvaluationPricing {
		input_price_per_1m: 0.042,
		output_price_per_1m: 0.0,
	};
	assert!((pricing.cost(429, 73) - 0.000_018_018).abs() < 1e-12);
}

#[test]
fn image_formats_and_constructors_match_the_wire_contract() {
	let png_data = png(512, 512);
	let png = ImageAsset::png(png_data.clone());
	assert_eq!(png.data, png_data);
	assert_eq!(png.format, EvaluationImageFormat::Png);
	assert_eq!(serde_json::to_value(png.format).unwrap(), "png");
	assert_eq!(png.format.mime(), "image/png");

	let jpeg_data = jpeg(512, 512);
	let jpeg = ImageAsset::jpeg(jpeg_data.clone());
	assert_eq!(jpeg.data, jpeg_data);
	assert_eq!(jpeg.format, EvaluationImageFormat::Jpeg);
	assert_eq!(serde_json::to_value(jpeg.format).unwrap(), "jpg");
	assert_eq!(jpeg.format.mime(), "image/jpeg");
}

#[test]
fn box_question_serializes_without_criteria() {
	for instructions in [json!("Locate the cat"), json!({"target": ["cat"]})] {
		assert_eq!(
			serde_json::to_value(Question::bounding_box(instructions.clone())).unwrap(),
			json!({"type": "box", "instructions": instructions})
		);
	}
}

#[test]
fn box_answer_parses_the_documented_coordinates() {
	let wire = json!({
		"type": "box",
		"box": {"top": 0.1, "bottom": 0.8, "left": 0.2, "right": 0.9},
		"confidence": 0.95
	});
	let answer: Answer = serde_json::from_value(wire.clone()).unwrap();
	assert_eq!(
		answer,
		Answer::Box {
			bounds: BoundingBox {
				top: 0.1,
				bottom: 0.8,
				left: 0.2,
				right: 0.9,
			},
			confidence: 0.95,
		}
	);
	assert_eq!(serde_json::to_value(answer).unwrap(), wire);
}

#[test]
fn box_questions_reject_empty_instructions() {
	for instructions in [Value::Null, json!("  ")] {
		let request = EvaluationRequest::new("state")
			.with_model("jev-preview")
			.with_question("region", Question::bounding_box(instructions));
		assert!(matches!(
			request.validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message == "question 'region' has empty instructions"
		));
	}
}

#[test]
fn assets_reject_blank_references_empty_data_and_unused_references() {
	let request = EvaluationRequest::new("photo-ref")
		.with_model("jev-preview")
		.with_question("q", Question::noul("Is there a cat?"));
	for reference in ["", " \t\n"] {
		assert!(matches!(
			request.clone().with_image(reference, ImageAsset::png(png(512, 512))).validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message == "image asset reference must not be empty"
		));
	}
	assert!(matches!(
		request.clone().with_image("photo-ref", ImageAsset::jpeg(vec![])).validate(),
		Err(EvaluationError::InvalidRequest(message))
			if message == "image asset 'photo-ref' has empty data"
	));
	assert!(matches!(
		request.with_image("unused-ref", ImageAsset::png(png(512, 512))).validate(),
		Err(EvaluationError::InvalidRequest(message))
			if message.contains("unused-ref") && message.contains("not referenced")
	));
}

#[test]
fn object_keys_and_question_ids_never_reference_assets() {
	let requests = [
		EvaluationRequest::new(json!({"nested": [{"photo-ref": "text"}]}))
			.with_question("q", Question::noul("Is there a cat?")),
		EvaluationRequest::new("state").with_question("photo-ref", Question::noul("Is there a cat?")),
		EvaluationRequest::new("state")
			.with_question("q", Question::choice("Pick", [("photo-ref", "Cat"), ("other", "Dog")])),
		EvaluationRequest::new("state").with_question("q", Question::bounding_box(json!({"photo-ref": "Cat"}))),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::Score {
				instructions: json!("Rate"),
				criteria: vec![json!({"photo-ref": "Cat"}), json!("Dog")],
			},
		),
	];
	for request in requests {
		assert!(matches!(
			request.with_model("jev-preview")
				.with_image("photo-ref", ImageAsset::png(png(512, 512)))
				.validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message.contains("photo-ref") && message.contains("not referenced")
		));
	}
}

#[test]
fn references_match_nested_string_values_and_substrings_in_criteria() {
	let requests = [
		EvaluationRequest::new(json!({"nested": [{"image": "Look at photo-ref now"}]}))
			.with_question("q", Question::noul("Is there a cat?")),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::bounding_box(json!({"nested": ["Look at photo-ref now"]})),
		),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::noul_with_criteria("Cat?", "A cat in photo-ref", "No cat"),
		),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::Noul {
				instructions: json!("Cat?"),
				criteria: Some(NoulCriteria {
					yes: None,
					no: Some(json!({"nested": ["No cat in photo-ref"]})),
				}),
			},
		),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::Choice {
				instructions: json!("Pick"),
				criteria: BTreeMap::from([
					("cat".to_string(), Some(json!({"nested": ["Cat in photo-ref"]}))),
					("other".to_string(), None),
				]),
			},
		),
		EvaluationRequest::new("state").with_question(
			"q",
			Question::Score {
				instructions: json!("Rate"),
				criteria: vec![json!("None"), json!({"nested": ["Cat in photo-ref"]})],
			},
		),
	];
	for request in requests {
		assert!(request
			.with_model("jev-preview")
			.with_image("photo-ref", ImageAsset::png(png(512, 512)))
			.validate()
			.is_ok());
	}
}

#[test]
fn uses_images_for_assets_or_box_questions() {
	let text = EvaluationRequest::new("photo-ref").with_question("q", Question::noul("Cat?"));
	assert!(text.assets.is_empty());
	assert!(!text.uses_images());
	assert!(text
		.with_image("photo-ref", ImageAsset::png(png(512, 512)))
		.uses_images());

	let boxes = EvaluationRequest::new("state").with_question("q", Question::bounding_box("Locate the cat"));
	assert!(boxes.assets.is_empty());
	assert!(boxes.uses_images());
}

fn image_request(asset: ImageAsset) -> EvaluationRequest {
	EvaluationRequest::new("photo-ref")
		.with_model("jev-preview")
		.with_question("q", Question::noul("Is there a cat?"))
		.with_image("photo-ref", asset)
}

#[test]
fn declared_and_detected_image_formats_must_match() {
	for (asset, declared, detected) in [
		(ImageAsset::png(jpeg(512, 512)), "Png", "Jpeg"),
		(ImageAsset::jpeg(png(1, 1)), "Jpeg", "Png"),
	] {
		assert!(matches!(
			image_request(asset).validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message.contains("photo-ref")
					&& message.contains(&format!("declares {declared}"))
					&& message.contains(&format!("detected {detected}"))
		));
	}
}

#[test]
fn image_sides_must_be_within_inclusive_limits() {
	for (width, height, side, value) in [
		(255, 512, "width", 255),
		(1569, 512, "width", 1569),
		(512, 255, "height", 255),
		(512, 1569, "height", 1569),
		(u32::MAX, u32::MAX, "width", u32::MAX),
	] {
		assert!(matches!(
			image_request(ImageAsset::png(png(width, height))).validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message.contains("photo-ref")
					&& message.contains(side)
					&& message.contains("256..=1568 px")
					&& message.contains(&format!(": {value} px"))
		));
	}
}

#[test]
fn images_with_valid_sides_must_also_meet_the_pixel_limit() {
	for asset in [ImageAsset::png(png(256, 256)), ImageAsset::jpeg(jpeg(256, 256))] {
		assert!(matches!(
			image_request(asset).validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message.contains("photo-ref")
					&& message.contains("total pixels")
					&& message.contains("78400..=2458624")
					&& message.contains("65536 (256x256)")
		));
	}
}

#[test]
fn inclusive_image_boundaries_are_accepted_without_changing_bytes() {
	for (width, height) in [(256, 1568), (1568, 256), (280, 280), (1568, 1568)] {
		for asset in [
			ImageAsset::png(png(u32::from(width), u32::from(height))),
			ImageAsset::jpeg(jpeg(width, height)),
		] {
			let original = asset.clone();
			let request = image_request(asset);
			assert!(request.validate().is_ok(), "{width}x{height}");
			assert_eq!(request.assets["photo-ref"], original);
		}
	}
}

#[test]
fn image_file_size_is_checked_before_the_header_and_is_inclusive() {
	let oversized = ImageAsset::png(vec![0; image::MAX_FILE_BYTES + 1]);
	assert!(matches!(
		image_request(oversized).validate(),
		Err(EvaluationError::InvalidRequest(message))
			if message.contains("photo-ref")
				&& message.contains("file size limit of 16777216 bytes (16 MiB)")
				&& message.contains(": 16777217 bytes")
	));

	let mut data = png(512, 512);
	data.resize(image::MAX_FILE_BYTES, 0);
	assert!(image_request(ImageAsset::png(data)).validate().is_ok());
}

#[test]
fn invalid_image_headers_name_the_asset_and_the_failure() {
	let mut truncated_png = png(512, 512);
	truncated_png.truncate(20);
	let mut truncated_jpeg = jpeg(512, 512);
	truncated_jpeg.truncate(15);
	for (asset, failure) in [
		(ImageAsset::png(apng(512, 512)), "animated PNG (acTL)"),
		(ImageAsset::png(truncated_png), "truncated or corrupt PNG"),
		(ImageAsset::jpeg(truncated_jpeg), "truncated or corrupt JPEG"),
		(ImageAsset::png(b"GIF89a".to_vec()), "only PNG or JPEG still images"),
	] {
		assert!(matches!(
			image_request(asset).validate(),
			Err(EvaluationError::InvalidRequest(message))
				if message.contains("photo-ref") && message.contains(failure)
		));
	}
}

#[test]
fn invalid_image_error_is_prefixed_once() {
	let error = image_request(ImageAsset::png(b"GIF89a".to_vec()))
		.validate()
		.unwrap_err();
	assert_eq!(
        error.to_string(),
        "invalid evaluation request: image asset 'photo-ref': unsupported image format; only PNG or JPEG still images are allowed"
    );
}

#[test]
fn every_asset_is_validated() {
	let request = image_request(ImageAsset::png(png(512, 512)))
		.with_question("other", Question::noul("Is there a cat in z-photo?"))
		.with_image("z-photo", ImageAsset::jpeg(jpeg(256, 256)));
	assert!(matches!(
		request.validate(),
		Err(EvaluationError::InvalidRequest(message))
			if message.contains("z-photo") && message.contains("total pixels")
	));
}
