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

//! Compatibility facade for reference model capabilities.
//!
//! The authoritative reference data lives in [`crate::llm::reference_models`]
//! so pricing, capabilities, and proxy schema policy use the same model match.

pub use crate::llm::reference_models::ModelCapabilities;

/// Look up reference capabilities for a model by fuzzy name matching.
pub fn get_reference_capabilities(model: &str) -> Option<ModelCapabilities> {
    crate::llm::reference_models::get_reference_capabilities(model)
}

#[cfg(test)]
#[path = "reference_capabilities_tests.rs"]
mod tests;
