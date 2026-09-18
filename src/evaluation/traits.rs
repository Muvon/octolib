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

use super::errors::EvaluationResult;
use super::types::{EvaluationPricing, EvaluationRequest, EvaluationResponse};

/// A System One style evaluator: one state, a map of typed questions, one
/// calibrated answer per question. No text generation and no job lifecycle;
/// every call completes on return.
#[async_trait::async_trait]
pub trait EvaluationProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;
    fn get_model_pricing(&self, model: &str) -> Option<EvaluationPricing>;
    async fn evaluate(&self, request: EvaluationRequest) -> EvaluationResult<EvaluationResponse>;
}
