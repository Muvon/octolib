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
use super::image;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

/// Supported still-image formats for evaluation assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaluationImageFormat {
    #[serde(rename = "png")]
    Png,
    #[serde(rename = "jpg")]
    Jpeg,
}

impl EvaluationImageFormat {
    pub fn mime(&self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
        }
    }
}

/// Bytes are sent as-is and must already satisfy Jev's image limits; EXIF orientation is ignored upstream, so rotate pixels upright first.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAsset {
    pub data: Vec<u8>,
    pub format: EvaluationImageFormat,
}

impl ImageAsset {
    pub fn png(data: Vec<u8>) -> Self {
        Self {
            data,
            format: EvaluationImageFormat::Png,
        }
    }

    pub fn jpeg(data: Vec<u8>) -> Self {
        Self {
            data,
            format: EvaluationImageFormat::Jpeg,
        }
    }
}

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
    /// Locate a region in an image.
    Box { instructions: Value },
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

    pub fn bounding_box(instructions: impl Into<Value>) -> Self {
        Self::Box {
            instructions: instructions.into(),
        }
    }

    fn references_image(&self, reference: &str) -> bool {
        match self {
            Self::Noul {
                instructions,
                criteria,
            } => {
                contains_reference(instructions, reference)
                    || criteria.as_ref().is_some_and(|criteria| {
                        criteria
                            .yes
                            .iter()
                            .chain(&criteria.no)
                            .any(|value| contains_reference(value, reference))
                    })
            }
            Self::Choice {
                instructions,
                criteria,
            } => {
                contains_reference(instructions, reference)
                    || criteria
                        .values()
                        .flatten()
                        .any(|value| contains_reference(value, reference))
            }
            Self::Score {
                instructions,
                criteria,
            } => {
                contains_reference(instructions, reference)
                    || criteria
                        .iter()
                        .any(|value| contains_reference(value, reference))
            }
            Self::Box { instructions } => contains_reference(instructions, reference),
        }
    }

    fn validate(&self, id: &str) -> EvaluationResult<()> {
        let instructions = match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Box { instructions } => instructions,
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
    Box {
        #[serde(rename = "box")]
        bounds: BoundingBox,
        confidence: f64,
    },
}

/// Edges as fractions from 0 to 1, with the origin at the image's top-left.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// Fraction of image height from the top edge.
    pub top: f64,
    /// Fraction of image height from the top edge.
    pub bottom: f64,
    /// Fraction of image width from the left edge.
    pub left: f64,
    /// Fraction of image width from the left edge.
    pub right: f64,
}

#[derive(Debug, Clone)]
pub struct EvaluationRequest {
    /// Provider-native model id, set by the high-level helper or `with_model`.
    pub model: String,
    /// Text, or a JSON object/array, evaluated by every question.
    pub state: Value,
    /// Questions keyed by the ids their answers come back under.
    pub questions: BTreeMap<String, Question>,
    /// References are plain text substituted wherever they appear in string values.
    /// Use a unique token such as a UUID: `cat` also matches inside `category`.
    pub assets: BTreeMap<String, ImageAsset>,
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
            assets: BTreeMap::new(),
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

    /// The reference is plain text substituted wherever it appears in string values.
    /// Use a unique token such as a UUID: `cat` also matches inside `category`.
    pub fn with_image(mut self, reference: impl Into<String>, asset: ImageAsset) -> Self {
        self.assets.insert(reference.into(), asset);
        self
    }

    pub(crate) fn uses_images(&self) -> bool {
        !self.assets.is_empty()
            || self
                .questions
                .values()
                .any(|question| matches!(question, Question::Box { .. }))
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
        for (reference, asset) in &self.assets {
            if reference.trim().is_empty() {
                return Err(EvaluationError::InvalidRequest(
                    "image asset reference must not be empty".to_string(),
                ));
            }
            if asset.data.is_empty() {
                return Err(EvaluationError::InvalidRequest(format!(
                    "image asset '{reference}' has empty data"
                )));
            }
            if !contains_reference(&self.state, reference)
                && !self
                    .questions
                    .values()
                    .any(|question| question.references_image(reference))
            {
                return Err(EvaluationError::InvalidRequest(format!(
                    "image asset '{reference}' is not referenced in state, instructions or criteria"
                )));
            }
            if asset.data.len() > image::MAX_FILE_BYTES {
                return Err(EvaluationError::InvalidRequest(format!(
                    "image asset '{reference}' exceeds the file size limit of {} bytes (16 MiB): {} bytes",
                    image::MAX_FILE_BYTES,
                    asset.data.len()
                )));
            }
            let (format, (width, height)) = image::detect(&asset.data).map_err(|error| {
                EvaluationError::InvalidRequest(format!("image asset '{reference}': {error}"))
            })?;
            if format != asset.format {
                return Err(EvaluationError::InvalidRequest(format!(
                    "image asset '{reference}' declares {:?} but detected {format:?}; formats must match",
                    asset.format
                )));
            }
            for (side, value) in [("width", width), ("height", height)] {
                if !(image::MIN_SIDE..=image::MAX_SIDE).contains(&value) {
                    return Err(EvaluationError::InvalidRequest(format!(
                        "image asset '{reference}' {side} must be {}..={} px: {value} px",
                        image::MIN_SIDE,
                        image::MAX_SIDE
                    )));
                }
            }
            let pixels = u64::from(width) * u64::from(height);
            if !(image::MIN_PIXELS..=image::MAX_PIXELS).contains(&pixels) {
                return Err(EvaluationError::InvalidRequest(format!(
                    "image asset '{reference}' total pixels must be {}..={}: {pixels} ({width}x{height})",
                    image::MIN_PIXELS,
                    image::MAX_PIXELS
                )));
            }
        }
        Ok(())
    }
}

fn contains_reference(value: &Value, reference: &str) -> bool {
    match value {
        Value::String(text) => text.contains(reference),
        Value::Array(values) => values
            .iter()
            .any(|value| contains_reference(value, reference)),
        Value::Object(values) => values
            .values()
            .any(|value| contains_reference(value, reference)),
        _ => false,
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
#[path = "types_tests.rs"]
mod tests;
