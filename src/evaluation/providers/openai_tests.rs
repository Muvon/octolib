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
use crate::evaluation::image::tests::png;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

const IMAGE_REFERENCE: &str = "img-5c1f0e";

fn request(state: impl Into<Value>) -> EvaluationRequest {
	EvaluationRequest::new(state)
		.with_model(MODEL)
		.with_question("ok", Question::noul("Is this about payments?"))
}

fn rejection(request: &EvaluationRequest) -> String {
	match request_body(request) {
		Err(EvaluationError::InvalidRequest(message)) => message,
		other => panic!("expected InvalidRequest, got {other:?}"),
	}
}

fn assert_close(actual: f64, expected: f64) {
	assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn usage_payload(input: u64, cached: u64, written: u64) -> Value {
	json!({
		"model": MODEL,
		"answers": [{"type": "predicate", "name": "ok", "probability": 0.5}],
		"usage": {
			"input_tokens": input,
			"input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
			"output_tokens": 0,
		}
	})
}

const LIVE_BODY: &str = r#"{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"is_urgent","probability":0.98},{"type":"choice","name":"team","choice":"billing","probabilities":[{"value":"billing","probability":1.0},{"value":"support","probability":0.0}],"confidence":1.0},{"type":"score","name":"severity","score":1.98,"probabilities":[{"value":0,"label":"low","probability":0.0},{"value":1,"label":"medium","probability":0.02},{"value":2,"label":"high","probability":0.98}],"confidence":0.97}],"usage":{"input_tokens":388,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":388}}"#;

const REFUSAL_BODY: &str = r#"{"model":"gpt-6-luna","answers":[{"type":"predicate","name":"ok","probability":0.99},{"type":"refusal","name":"weapon"}],"usage":{"input_tokens":165,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},"output_tokens":0,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":165}}"#;

#[test]
fn string_state_and_all_question_types_match_the_wire_format() {
	let mut request = EvaluationRequest::new("Payouts failing for 3 days.")
		.with_model(" gpt-6-luna ")
		.with_question("is_urgent", Question::noul("Is this urgent?"))
		.with_question(
			"severity",
			Question::score("How severe is it?", ["low", "medium", "high"]),
		);
	request.questions.insert(
		"team".to_string(),
		Question::Choice {
			instructions: json!("Which team owns it?"),
			criteria: BTreeMap::from([
				("billing".to_string(), Some(json!("Money questions"))),
				("support".to_string(), None),
			]),
		},
	);
	assert_eq!(
		request_body(&request).expect("body"),
		json!({
			"model": "gpt-6-luna",
			"input": "Payouts failing for 3 days.",
			"questions": [
				{"type": "predicate", "name": "is_urgent", "instructions": "Is this urgent?"},
				{
					"type": "score",
					"name": "severity",
					"instructions": "How severe is it?",
					"levels": [{"label": "low"}, {"label": "medium"}, {"label": "high"}]
				},
				{
					"type": "choice",
					"name": "team",
					"instructions": "Which team owns it?",
					"choices": [
						{"value": "billing", "description": "Money questions"},
						{"value": "support"}
					]
				}
			]
		})
	);
}

#[test]
fn object_state_becomes_an_input_json_message() {
	let request = request(json!({"ticket": "payout failing"}));
	assert_eq!(
		request_body(&request).expect("body"),
		json!({
			"model": "gpt-6-luna",
			"input": [{
				"type": "message",
				"role": "user",
				"content": [{"type": "input_json", "data": {"ticket": "payout failing"}}]
			}],
			"questions": [{"type": "predicate", "name": "ok", "instructions": "Is this about payments?"}]
		})
	);
}

#[test]
fn array_state_maps_each_item_to_a_part() {
	let image = png(512, 512);
	let data = base64::engine::general_purpose::STANDARD.encode(&image);
	let request =
		request(json!(["Photo:", IMAGE_REFERENCE, {"k": 1}])).with_image(IMAGE_REFERENCE, ImageAsset::png(image));
	assert_eq!(
		request_body(&request).expect("body")["input"],
		json!([{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "Photo:"},
				{"type": "input_image", "image_url": format!("data:image/png;base64,{data}")},
				{"type": "input_json", "data": {"k": 1}}
			]
		}])
	);
}

#[test]
fn string_state_equal_to_a_reference_is_one_image_part() {
	let image = png(512, 512);
	let data = base64::engine::general_purpose::STANDARD.encode(&image);
	let request = request(IMAGE_REFERENCE).with_image(IMAGE_REFERENCE, ImageAsset::png(image));
	assert_eq!(
		request_body(&request).expect("body")["input"],
		json!([{
			"type": "message",
			"role": "user",
			"content": [{"type": "input_image", "image_url": format!("data:image/png;base64,{data}")}]
		}])
	);
}

#[test]
fn noul_criteria_are_rejected() {
	let request = EvaluationRequest::new("state")
		.with_model(MODEL)
		.with_question("q", Question::noul_with_criteria("Urgent?", "yes", "no"));
	assert!(rejection(&request).contains("no true/false criteria"));
}

#[test]
fn box_questions_are_rejected() {
	let request = EvaluationRequest::new("state")
		.with_model(MODEL)
		.with_question("q", Question::bounding_box("Locate the cat"));
	assert!(rejection(&request).contains("does not support box questions"));
}

#[test]
fn non_string_instructions_are_rejected() {
	let request = EvaluationRequest::new("state").with_model(MODEL).with_question(
		"q",
		Question::Noul {
			instructions: json!({"rule": "urgent"}),
			criteria: None,
		},
	);
	assert!(rejection(&request).contains("string instructions"));
}

#[test]
fn non_string_choice_descriptions_are_rejected() {
	let request = EvaluationRequest::new("state").with_model(MODEL).with_question(
		"q",
		Question::Choice {
			instructions: json!("Pick"),
			criteria: BTreeMap::from([("a".to_string(), Some(json!(5))), ("b".to_string(), None)]),
		},
	);
	assert!(rejection(&request).contains("string description"));
}

#[test]
fn non_string_score_levels_are_rejected() {
	let request = EvaluationRequest::new("state").with_model(MODEL).with_question(
		"q",
		Question::Score {
			instructions: json!("Rate"),
			criteria: vec![json!("low"), json!(2)],
		},
	);
	assert!(rejection(&request).contains("string levels"));
}

#[test]
fn counts_above_the_native_limits_are_rejected() {
	let levels = EvaluationRequest::new("state").with_model(MODEL).with_question(
		"q",
		Question::score("Rate", (0..=MAX_SCORE_LEVELS).map(|i| format!("l{i}"))),
	);
	assert!(rejection(&levels).contains("at most 10"));

	let choices = EvaluationRequest::new("state").with_model(MODEL).with_question(
		"q",
		Question::choice("Pick", (0..=MAX_CHOICES).map(|i| (format!("c{i:03}"), "d"))),
	);
	assert!(rejection(&choices).contains("at most 255"));

	let mut questions = EvaluationRequest::new("state").with_model(MODEL);
	for index in 0..=MAX_QUESTIONS {
		questions = questions.with_question(format!("q{index:03}"), Question::noul("Ok?"));
	}
	assert!(rejection(&questions).contains("at most 400"));
}

#[test]
fn image_references_outside_top_level_state_items_are_rejected() {
	let asset = || ImageAsset::png(png(512, 512));
	let nested = request(json!({"photo": IMAGE_REFERENCE})).with_image(IMAGE_REFERENCE, asset());
	assert!(rejection(&nested).contains("top-level state items"));

	let embedded = request(format!("see {IMAGE_REFERENCE}")).with_image(IMAGE_REFERENCE, asset());
	assert!(rejection(&embedded).contains("top-level state items"));

	let in_instructions = EvaluationRequest::new("state")
		.with_model(MODEL)
		.with_question("q", Question::noul(format!("Is {IMAGE_REFERENCE} clean?")))
		.with_image(IMAGE_REFERENCE, asset());
	assert!(rejection(&in_instructions).contains("top-level state items"));
}

#[test]
fn only_gpt_6_luna_is_served_and_priced() {
	let provider = OpenAiEvaluationProvider::new();
	assert!(provider.supports_model("gpt-6-luna"));
	assert!(provider.supports_model(" gpt-6-luna "));
	assert!(!provider.supports_model("gpt-6"));
	assert!(!provider.supports_model("gpt-6-luna-2"));
	assert!(!provider.supports_model("openai/gpt-6-luna"));
	assert_eq!(provider.get_model_pricing("gpt-6-luna"), Some(MODEL_PRICING));
	assert_eq!(MODEL_PRICING.input_price_per_1m, 0.10);
	assert_eq!(MODEL_PRICING.output_price_per_1m, 0.0);
	assert!(provider.get_model_pricing("gpt-6").is_none());
}

#[tokio::test]
async fn evaluate_rejects_an_unsupported_model() {
	let request = request("state").with_model("gpt-6");
	let error = OpenAiEvaluationProvider::new()
		.evaluate(request)
		.await
		.expect_err("unsupported model");
	assert!(matches!(error, EvaluationError::UnsupportedModel { .. }), "{error}");
}

#[tokio::test]
#[serial_test::serial]
async fn evaluate_sends_the_request_and_parses_the_live_response() {
	let mock = MockHttp::start(vec![(200, LIVE_BODY.to_string())]);
	let _env = EnvGuard::new().set(API_URL_ENV, &mock.url).set(API_KEY_ENV, "mock-key");

	let response = OpenAiEvaluationProvider::new()
		.evaluate(request("Payouts failing for 3 days."))
		.await
		.expect("mock evaluation");

	assert_eq!(response.model, "gpt-6-luna");
	assert_eq!(response.answers.get("is_urgent"), Some(&Answer::Noul { noul: 0.98 }));
	assert_eq!(
		response.answers.get("team"),
		Some(&Answer::Choice {
			choice: "billing".to_string(),
			probabilities: BTreeMap::from([("billing".to_string(), 1.0), ("support".to_string(), 0.0)]),
			confidence: 1.0,
		})
	);
	assert_eq!(
		response.answers.get("severity"),
		Some(&Answer::Score {
			score: 1.98,
			legend: BTreeMap::from([
				("0".to_string(), json!("low")),
				("1".to_string(), json!("medium")),
				("2".to_string(), json!("high")),
			]),
			probabilities: BTreeMap::from([("0".to_string(), 0.0), ("1".to_string(), 0.02), ("2".to_string(), 0.98),]),
			confidence: 0.97,
		})
	);
	assert_eq!(response.usage.input_tokens, 388);
	assert_eq!(response.usage.output_tokens, 0);
	assert_close(response.usage.cost.expect("cost"), 388.0 * 0.10 / 1_000_000.0);

	let requests = mock.requests();
	assert_eq!(requests[0].header("authorization"), Some("Bearer mock-key"));
	let sent: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(sent["model"], "gpt-6-luna");
	assert_eq!(sent["input"], "Payouts failing for 3 days.");
}

#[tokio::test]
#[serial_test::serial]
async fn refusal_answers_sit_next_to_real_ones() {
	let mock = MockHttp::start(vec![(200, REFUSAL_BODY.to_string())]);
	let _env = EnvGuard::new().set(API_URL_ENV, &mock.url).set(API_KEY_ENV, "mock-key");

	let response = OpenAiEvaluationProvider::new()
		.evaluate(request("state"))
		.await
		.expect("mock evaluation");

	assert_eq!(response.answers.get("ok"), Some(&Answer::Noul { noul: 0.99 }));
	assert_eq!(response.answers.get("weapon"), Some(&Answer::Refusal));
}

#[test]
fn cost_splits_input_by_cache_state() {
	let read = parse_response(usage_payload(2000, 1024, 0)).expect("response");
	assert_close(
		read.usage.cost.expect("cost"),
		(976.0 * 0.10 + 1024.0 * 0.01) / 1_000_000.0,
	);
	assert_eq!(read.usage.input_tokens, 2000);

	let write = parse_response(usage_payload(2000, 0, 500)).expect("response");
	assert_close(
		write.usage.cost.expect("cost"),
		(1500.0 * 0.10 + 500.0 * 0.125) / 1_000_000.0,
	);
}

#[test]
fn missing_cache_details_count_as_zero() {
	let mut payload = usage_payload(1000, 0, 0);
	payload["usage"]
		.as_object_mut()
		.expect("usage object")
		.remove("input_tokens_details");
	let response = parse_response(payload).expect("response");
	assert_close(response.usage.cost.expect("cost"), 1000.0 * 0.10 / 1_000_000.0);
}

#[test]
fn malformed_responses_are_invalid_responses() {
	let mut missing_tokens = usage_payload(10, 0, 0);
	missing_tokens["usage"]
		.as_object_mut()
		.expect("usage object")
		.remove("output_tokens");
	let unknown_type = json!({
		"model": MODEL,
		"answers": [{"type": "mystery", "name": "q"}],
		"usage": {"input_tokens": 1, "output_tokens": 0}
	});
	let missing_name = json!({
		"model": MODEL,
		"answers": [{"type": "refusal"}],
		"usage": {"input_tokens": 1, "output_tokens": 0}
	});
	let duplicate = json!({
		"model": MODEL,
		"answers": [
			{"type": "refusal", "name": "q"},
			{"type": "predicate", "name": "q", "probability": 0.1}
		],
		"usage": {"input_tokens": 1, "output_tokens": 0}
	});
	for payload in [missing_tokens, unknown_type, missing_name, duplicate] {
		assert!(
			matches!(parse_response(payload), Err(EvaluationError::InvalidResponse { .. })),
			"payload should be rejected"
		);
	}
}

#[test]
fn boolean_choices_become_strings() {
	let payload = json!({
		"model": MODEL,
		"answers": [{
			"type": "choice",
			"name": "flag",
			"choice": true,
			"probabilities": [
				{"value": true, "probability": 0.75},
				{"value": false, "probability": 0.25}
			],
			"confidence": 0.5
		}],
		"usage": {"input_tokens": 1, "output_tokens": 0}
	});
	let response = parse_response(payload).expect("response");
	assert_eq!(
		response.answers.get("flag"),
		Some(&Answer::Choice {
			choice: "true".to_string(),
			probabilities: BTreeMap::from([("false".to_string(), 0.25), ("true".to_string(), 0.75)]),
			confidence: 0.5,
		})
	);
}

#[test]
#[serial_test::serial]
fn endpoint_honours_the_url_override() {
	let provider = OpenAiEvaluationProvider::new();
	let guard = EnvGuard::new().set(API_URL_ENV, "https://proxy.example/decisions");
	assert_eq!(provider.endpoint(), "https://proxy.example/decisions");
	drop(guard);
	let _env = EnvGuard::new().remove(API_URL_ENV);
	assert_eq!(provider.endpoint(), "https://api.openai.com/v1/decisions");
}
