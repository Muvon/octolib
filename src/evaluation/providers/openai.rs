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

//! OpenAI Decisions API (`POST https://api.openai.com/v1/decisions`).
//!
//! Only `gpt-6-luna` is served. The request is `{model, input, questions}`
//! and the response is `{model, answers, usage}`; both differ from the Jev
//! payload, so this provider maps them instead of reusing `shared::parse_payload`.
//!
//! Mapping:
//! - `state`: a plain string stays a string. A JSON object (or any other
//!   non-string value) becomes one `input_json` part of a user message. A JSON
//!   array becomes one part per top-level item: a string equal to an asset
//!   reference is an `input_image` data URL, any other string an `input_text`,
//!   anything else `input_json`.
//! - questions: `Noul` is a `predicate`, `Choice` a `choice` with `choices`,
//!   `Score` a `score` with `levels`; the question id is its `name`.
//! - answers: `predicate`, `choice`, `score` and `refusal` map to
//!   [`Answer::Noul`], [`Answer::Choice`], [`Answer::Score`] and
//!   [`Answer::Refusal`].
//!
//! Rejected with `InvalidRequest`, never emulated, because the native wire
//! format has no equivalent:
//! - `Box` questions (no localisation question type);
//! - `Noul` criteria (predicates take no true/false descriptions);
//! - non-string instructions, choice descriptions or score levels;
//! - an image reference anywhere but a top-level state item (images are
//!   input parts, not text substitutions);
//! - more than 400 questions, 255 choices or 10 score levels per question.
//!
//! Pricing (OpenAI pricing page, `gpt-6-luna`, verified 2026-11-02): uncached
//! input $0.10 per 1M tokens, cache read $0.01, cache write (30 minutes, the
//! only Decisions TTL) $0.125, output free. The reported cost splits the input
//! by `input_tokens_details`.
//!
//! Configuration:
//! - `OPENAI_API_KEY`: required
//! - `OPENAI_EVALUATION_API_URL`: optional full endpoint URL, default
//!   `https://api.openai.com/v1/decisions`

use super::shared;
use crate::evaluation::errors::{EvaluationError, EvaluationResult};
use crate::evaluation::traits::EvaluationProvider;
use crate::evaluation::types::{
	Answer, EvaluationPricing, EvaluationRequest, EvaluationResponse, EvaluationUsage, ImageAsset, Question,
};
use base64::Engine;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const PROVIDER: &str = "openai";
const API_KEY_ENV: &str = "OPENAI_API_KEY";
const API_URL_ENV: &str = "OPENAI_EVALUATION_API_URL";
const API_URL: &str = "https://api.openai.com/v1/decisions";
const MODEL: &str = "gpt-6-luna";

const MAX_QUESTIONS: usize = 400;
const MAX_CHOICES: usize = 255;
const MAX_SCORE_LEVELS: usize = 10;

const TOKENS_PER_PRICE_UNIT: f64 = 1_000_000.0;
const INPUT_PRICE_PER_1M: f64 = 0.10;
const CACHE_READ_PRICE_PER_1M: f64 = 0.01;
const CACHE_WRITE_PRICE_PER_1M: f64 = 0.125;
const MODEL_PRICING: EvaluationPricing = EvaluationPricing {
	input_price_per_1m: INPUT_PRICE_PER_1M,
	output_price_per_1m: 0.0,
};

#[derive(Debug, Clone, Default)]
pub struct OpenAiEvaluationProvider;

impl OpenAiEvaluationProvider {
	pub fn new() -> Self {
		Self
	}

	fn endpoint(&self) -> String {
		std::env::var(API_URL_ENV).unwrap_or_else(|_| API_URL.to_string())
	}
}

fn is_supported(model: &str) -> bool {
	model.trim() == MODEL
}

fn invalid_response(message: String) -> EvaluationError {
	EvaluationError::InvalidResponse {
		provider: PROVIDER.to_string(),
		message,
	}
}

pub(crate) fn request_body(request: &EvaluationRequest) -> EvaluationResult<Value> {
	if request.questions.len() > MAX_QUESTIONS {
		return Err(EvaluationError::InvalidRequest(format!(
			"openai accepts at most {MAX_QUESTIONS} questions: {}",
			request.questions.len()
		)));
	}
	let questions = request
		.questions
		.iter()
		.map(|(id, question)| question_body(id, question))
		.collect::<EvaluationResult<Vec<_>>>()?;
	Ok(json!({
		"model": request.model.trim(),
		"input": input_value(&request.state, &request.assets)?,
		"questions": questions,
	}))
}

fn input_value(state: &Value, assets: &BTreeMap<String, ImageAsset>) -> EvaluationResult<Value> {
	let mut used = BTreeSet::new();
	let parts = match state {
		Value::Array(items) => items
			.iter()
			.map(|item| state_part(item, assets, &mut used))
			.collect::<Vec<_>>(),
		other => vec![state_part(other, assets, &mut used)],
	};
	if let Some(reference) = assets.keys().find(|reference| !used.contains(reference.as_str())) {
		return Err(EvaluationError::InvalidRequest(format!(
			"openai reads images only as top-level state items; asset '{reference}' is referenced elsewhere"
		)));
	}
	if let (Value::String(text), true) = (state, used.is_empty()) {
		return Ok(Value::String(text.clone()));
	}
	Ok(json!([{"type": "message", "role": "user", "content": parts}]))
}

fn state_part<'a>(item: &Value, assets: &'a BTreeMap<String, ImageAsset>, used: &mut BTreeSet<&'a str>) -> Value {
	match item {
		Value::String(text) => match assets.get_key_value(text) {
			Some((reference, asset)) => {
				used.insert(reference.as_str());
				let data = base64::engine::general_purpose::STANDARD.encode(&asset.data);
				json!({"type": "input_image", "image_url": format!("data:{};base64,{data}", asset.format.mime())})
			}
			None => json!({"type": "input_text", "text": text}),
		},
		other => json!({"type": "input_json", "data": other}),
	}
}

fn question_body(id: &str, question: &Question) -> EvaluationResult<Value> {
	match question {
		Question::Noul { instructions, criteria } => {
			if criteria.is_some() {
				return Err(EvaluationError::InvalidRequest(format!(
					"openai predicates take no true/false criteria; question '{id}' sets them"
				)));
			}
			Ok(json!({
				"type": "predicate",
				"name": id,
				"instructions": instructions_text(id, instructions)?,
			}))
		}
		Question::Choice { instructions, criteria } => {
			if criteria.len() > MAX_CHOICES {
				return Err(EvaluationError::InvalidRequest(format!(
					"choice question '{id}' has {} options; openai accepts at most {MAX_CHOICES}",
					criteria.len()
				)));
			}
			let choices = criteria
				.iter()
				.map(|(value, description)| {
					let mut choice = json!({"value": value});
					match description {
						None => {}
						Some(Value::String(text)) => choice["description"] = json!(text),
						Some(_) => {
							return Err(EvaluationError::InvalidRequest(format!(
								"choice question '{id}' option '{value}' needs a string description"
							)));
						}
					}
					Ok(choice)
				})
				.collect::<EvaluationResult<Vec<_>>>()?;
			Ok(json!({
				"type": "choice",
				"name": id,
				"instructions": instructions_text(id, instructions)?,
				"choices": choices,
			}))
		}
		Question::Score { instructions, criteria } => {
			if criteria.len() > MAX_SCORE_LEVELS {
				return Err(EvaluationError::InvalidRequest(format!(
					"score question '{id}' has {} levels; openai accepts at most {MAX_SCORE_LEVELS}",
					criteria.len()
				)));
			}
			let levels = criteria
				.iter()
				.map(|level| match level {
					Value::String(label) => Ok(json!({"label": label})),
					_ => Err(EvaluationError::InvalidRequest(format!(
						"score question '{id}' needs string levels"
					))),
				})
				.collect::<EvaluationResult<Vec<_>>>()?;
			Ok(json!({
				"type": "score",
				"name": id,
				"instructions": instructions_text(id, instructions)?,
				"levels": levels,
			}))
		}
		Question::Box { .. } => Err(EvaluationError::InvalidRequest(
			"openai does not support box questions".to_string(),
		)),
	}
}

fn instructions_text<'a>(id: &str, instructions: &'a Value) -> EvaluationResult<&'a str> {
	instructions
		.as_str()
		.ok_or_else(|| EvaluationError::InvalidRequest(format!("question '{id}' needs string instructions")))
}

#[derive(Deserialize)]
struct WirePayload {
	model: String,
	answers: Vec<WireNamedAnswer>,
	usage: WireUsage,
}

#[derive(Deserialize)]
struct WireNamedAnswer {
	name: String,
	#[serde(flatten)]
	answer: WireAnswer,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WireAnswer {
	Predicate {
		probability: f64,
	},
	Choice {
		choice: WireChoiceValue,
		probabilities: Vec<WireChoiceProbability>,
		confidence: f64,
	},
	Score {
		score: f64,
		probabilities: Vec<WireScoreProbability>,
		confidence: f64,
	},
	Refusal,
}

/// The API reports yes/no choices as JSON booleans.
#[derive(Deserialize)]
#[serde(untagged)]
enum WireChoiceValue {
	Text(String),
	Flag(bool),
}

impl WireChoiceValue {
	fn label(&self) -> String {
		match self {
			Self::Text(text) => text.clone(),
			Self::Flag(flag) => flag.to_string(),
		}
	}
}

#[derive(Deserialize)]
struct WireChoiceProbability {
	value: WireChoiceValue,
	probability: f64,
}

#[derive(Deserialize)]
struct WireScoreProbability {
	value: u32,
	label: String,
	probability: f64,
}

#[derive(Deserialize)]
struct WireUsage {
	input_tokens: u64,
	output_tokens: u64,
	input_tokens_details: Option<WireInputDetails>,
}

#[derive(Deserialize)]
struct WireInputDetails {
	#[serde(default)]
	cached_tokens: u64,
	#[serde(default)]
	cache_write_tokens: u64,
}

impl From<WireAnswer> for Answer {
	fn from(answer: WireAnswer) -> Self {
		match answer {
			WireAnswer::Predicate { probability } => Answer::Noul { noul: probability },
			WireAnswer::Choice {
				choice,
				probabilities,
				confidence,
			} => Answer::Choice {
				choice: choice.label(),
				probabilities: probabilities
					.iter()
					.map(|entry| (entry.value.label(), entry.probability))
					.collect(),
				confidence,
			},
			WireAnswer::Score {
				score,
				probabilities,
				confidence,
			} => Answer::Score {
				score,
				legend: probabilities
					.iter()
					.map(|entry| (entry.value.to_string(), Value::String(entry.label.clone())))
					.collect(),
				probabilities: probabilities
					.iter()
					.map(|entry| (entry.value.to_string(), entry.probability))
					.collect(),
				confidence,
			},
			WireAnswer::Refusal => Answer::Refusal,
		}
	}
}

fn input_cost(usage: &WireUsage) -> f64 {
	let (cached, written) = usage
		.input_tokens_details
		.as_ref()
		.map_or((0, 0), |details| (details.cached_tokens, details.cache_write_tokens));
	let uncached = usage.input_tokens.saturating_sub(cached).saturating_sub(written);
	(uncached as f64 * INPUT_PRICE_PER_1M
		+ cached as f64 * CACHE_READ_PRICE_PER_1M
		+ written as f64 * CACHE_WRITE_PRICE_PER_1M)
		/ TOKENS_PER_PRICE_UNIT
}

pub(crate) fn parse_response(payload: Value) -> EvaluationResult<EvaluationResponse> {
	let wire: WirePayload = serde_json::from_value(payload)
		.map_err(|error| invalid_response(format!("response did not match the Decisions shape: {error}")))?;
	let mut answers = BTreeMap::new();
	for WireNamedAnswer { name, answer } in wire.answers {
		if answers.insert(name.clone(), Answer::from(answer)).is_some() {
			return Err(invalid_response(format!("duplicate answer '{name}'")));
		}
	}
	Ok(EvaluationResponse {
		model: wire.model,
		answers,
		usage: EvaluationUsage {
			input_tokens: wire.usage.input_tokens,
			output_tokens: wire.usage.output_tokens,
			cost: Some(input_cost(&wire.usage)),
		},
	})
}

#[async_trait::async_trait]
impl EvaluationProvider for OpenAiEvaluationProvider {
	fn name(&self) -> &str {
		PROVIDER
	}

	fn supports_model(&self, model: &str) -> bool {
		is_supported(model)
	}

	fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing> {
		is_supported(model).then_some(MODEL_PRICING)
	}

	async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse> {
		request.validate()?;
		if !is_supported(&request.model) {
			return Err(EvaluationError::UnsupportedModel {
				provider: PROVIDER.to_string(),
				model: request.model,
			});
		}
		let body = request_body(&request)?;
		let key = shared::api_key(API_KEY_ENV)?;
		let url = self.endpoint();
		tracing::debug!(provider = PROVIDER, %url, "sending evaluation request");
		let response = shared::send(PROVIDER, &request, || {
			Ok(crate::http::http_client().post(&url).bearer_auth(&key).json(&body))
		})
		.await?;
		parse_response(shared::parse_json(PROVIDER, &response)?)
	}
}

#[cfg(test)]
#[path = "openai_tests.rs"]
mod tests;
