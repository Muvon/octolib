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

//! Compatibility facade for reference model pricing.
//!
//! The authoritative reference data lives in [`crate::llm::reference_models`]
//! so pricing, capabilities, and proxy schema policy use the same model match.

use crate::llm::types::ModelPricing;

/// Look up baseline cloud-equivalent pricing for a model by fuzzy name matching.
pub fn get_reference_pricing(model: &str) -> Option<ModelPricing> {
    crate::llm::reference_models::get_reference_pricing(model)
}

/// Calculate cost using reference pricing.
pub fn calculate_reference_cost(
    model: &str,
    input_tokens: u64,
    cache_read_tokens: u64,
    output_tokens: u64,
) -> Option<f64> {
    crate::llm::reference_models::calculate_reference_cost(
        model,
        input_tokens,
        cache_read_tokens,
        output_tokens,
    )
}

#[cfg(test)]
#[path = "reference_pricing_tests.rs"]
mod tests;
