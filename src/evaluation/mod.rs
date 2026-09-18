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

//! Structured evaluation: calibrated answers to typed questions.
//!
//! An evaluation model is not a chat model. It takes one `state` (text or
//! JSON) and a map of typed questions, and returns one answer per question
//! with probabilities the caller can branch on: a yes/no probability
//! (`noul`), a chosen option with its distribution and confidence
//! (`choice`), or a probability-weighted level on an ordered rubric
//! (`score`). There is no generated text to parse and no job to poll.
//!
//! The first such model is TypeSafe's Jev, reachable directly
//! (`typesafe:jev-latest`) or through Cloudflare AI Gateway unified billing
//! (`cloudflare:typesafe/jev`). Both speak the same payload; only the
//! envelope differs.

pub mod errors;
pub mod factory;
pub mod providers;
pub mod traits;
pub mod types;

pub use errors::{EvaluationError, EvaluationResult};
pub use factory::EvaluationProviderFactory;
pub use providers::{CloudflareEvaluationProvider, TypeSafeEvaluationProvider};
pub use traits::EvaluationProvider;
pub use types::{
    Answer, EvaluationPricing, EvaluationRequest, EvaluationResponse, EvaluationUsage,
    NoulCriteria, Question,
};

/// Evaluate `request` on `provider:model`, e.g. `typesafe:jev-latest`.
pub async fn evaluate(
    provider_model: &str,
    mut request: EvaluationRequest,
) -> EvaluationResult<EvaluationResponse> {
    let (provider, model) = EvaluationProviderFactory::get_provider_for_model(provider_model)?;
    if !request.model.is_empty() && request.model != model {
        return Err(EvaluationError::InvalidRequest(format!(
            "request model '{}' does not match '{provider_model}'",
            request.model
        )));
    }
    request.model = model;
    provider.evaluate(request).await
}
