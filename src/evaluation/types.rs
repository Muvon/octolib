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

use super::errors::{EvaluationError, EvaluationResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

/// Optional meaning of the two ends of a yes/no question.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<Value>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<Value>,
}

/// One typed question. Instructions and criteria are plain strings in the
/// common case; the API also accepts JSON structure in every one of them, so
/// the fields are `Value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Yes/no. The answer is the probability of yes.
    Noul {
        instructions: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One option from a set. Each option maps to a description, or null.
    Choice {
        instructions: Value,
        criteria: BTreeMap<String, Option<Value>>,
    },
    /// Ordered levels. The answer is probability-weighted across them.
    Score {
        instructions: Value,
        criteria: Vec<Value>,
    },
}

impl Question {
    pub fn noul(instructions: impl Into<String>) -> Self {
        Self::Noul {
            instructions: Value::String(instructions.into()),
            criteria: None,
        }
    }

    pub fn noul_with_criteria(
        instructions: impl Into<String>,
        yes: impl Into<String>,
        no: impl Into<String>,
    ) -> Self {
        Self::Noul {
            instructions: Value::String(instructions.into()),
            criteria: Some(NoulCriteria {
                yes: Some(Value::String(yes.into())),
                no: Some(Value::String(no.into())),
            }),
        }
    }

    pub fn choice<K: Into<String>, D: Into<String>>(
        instructions: impl Into<String>,
        options: impl IntoIterator<Item = (K, D)>,
    ) -> Self {
        Self::Choice {
            instructions: Value::String(instructions.into()),
            criteria: options
                .into_iter()
                .map(|(option, description)| {
                    (option.into(), Some(Value::String(description.into())))
                })
                .collect(),
        }
    }

    pub fn score<L: Into<String>>(
        instructions: impl Into<String>,
        levels: impl IntoIterator<Item = L>,
    ) -> Self {
        Self::Score {
            instructions: Value::String(instructions.into()),
            criteria: levels
                .into_iter()
                .map(|level| Value::String(level.into()))
                .collect(),
        }
    }

    fn validate(&self, id: &str) -> EvaluationResult<()> {
        let instructions = match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        };
        if instructions.is_null()
            || instructions
                .as_str()
                .is_some_and(|text| text.trim().is_empty())
        {
            return Err(EvaluationError::InvalidRequest(format!(
                "question '{id}' has empty instructions"
            )));
        }
        match self {
            Self::Choice { criteria, .. } if criteria.len() < 2 => {
                Err(EvaluationError::InvalidRequest(format!(
                    "choice question '{id}' needs at least two options"
                )))
            }
            Self::Score { criteria, .. } if criteria.len() < 2 => {
                Err(EvaluationError::InvalidRequest(format!(
                    "score question '{id}' needs at least two levels"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// One answer, under the same id as its question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        /// Probability that the answer is yes, 0 to 1.
        noul: f64,
    },
    Choice {
        /// The highest-probability option.
        choice: String,
        /// Every option mapped to its probability; they sum to 1.
        probabilities: BTreeMap<String, f64>,
        /// Certainty derived from the distribution, 0 to 1.
        confidence: f64,
    },
    Score {
        /// Probability-weighted level; may land between levels.
        score: f64,
        /// Level index (as a string key) back to its description.
        legend: BTreeMap<String, Value>,
        /// Level index (as a string key) to its probability; they sum to 1.
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

#[derive(Debug, Clone)]
pub struct EvaluationRequest {
    /// Provider-native model id, set by the high-level helper or `with_model`.
    pub model: String,
    /// Text, or a JSON object/array, evaluated by every question.
    pub state: Value,
    /// Questions keyed by the ids their answers come back under.
    pub questions: BTreeMap<String, Question>,
    /// Retries on 429, 529 and 5xx. Evaluations have no side effects, so a
    /// replay cannot duplicate work.
    pub max_retries: u32,
    /// Per-attempt timeout.
    pub timeout: Duration,
}

impl EvaluationRequest {
    pub fn new(state: impl Into<Value>) -> Self {
        Self {
            model: String::new(),
            state: state.into(),
            questions: BTreeMap::new(),
            max_retries: 2,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_question(mut self, id: impl Into<String>, question: Question) -> Self {
        self.questions.insert(id.into(), question);
        self
    }

    pub(crate) fn validate(&self) -> EvaluationResult<()> {
        if self.model.trim().is_empty() {
            return Err(EvaluationError::InvalidRequest(
                "model must not be empty".to_string(),
            ));
        }
        if self.questions.is_empty() {
            return Err(EvaluationError::InvalidRequest(
                "at least one question is required".to_string(),
            ));
        }
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                return Err(EvaluationError::InvalidRequest(
                    "question ids must not be empty".to_string(),
                ));
            }
            question.validate(id)?;
        }
        Ok(())
    }
}

/// Price per 1M tokens in USD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationPricing {
    pub input_price_per_1m: f64,
    pub output_price_per_1m: f64,
}

impl EvaluationPricing {
    pub fn cost(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 / 1_000_000.0) * self.input_price_per_1m
            + (output_tokens as f64 / 1_000_000.0) * self.output_price_per_1m
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// USD, computed from the provider's published rate when it is known.
    pub cost: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationResponse {
    /// The versioned model that answered, as reported by the provider.
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: EvaluationUsage,
}

#[cfg(test)]
mod tests {
    use super::*;
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
                Question::score(
                    "How frustrated is the customer?",
                    ["Calm", "Frustrated", "Very angry"],
                ),
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
}
