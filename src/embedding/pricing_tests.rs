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

fn approx(got: Option<f64>, want: f64) -> bool {
    got.is_some_and(|x| (x - want).abs() < 1e-9)
}

#[test]
fn prices_known_models() {
    // 1M tokens of voyage-3.5 at $0.06/M = $0.06
    assert!(approx(
        calculate_embedding_cost("voyage-3.5", 1_000_000),
        0.06
    ));
    // resolved names match case-insensitively
    assert!(approx(
        calculate_embedding_cost("VOYAGE-3.5", 1_000_000),
        0.06
    ));
    // 500k tokens of text-embedding-3-small at $0.02/M = $0.01
    assert!(approx(
        calculate_embedding_cost("text-embedding-3-small", 500_000),
        0.01
    ));
    // zero tokens = zero cost, still priced (Some)
    assert!(approx(calculate_embedding_cost("voyage-code-3", 0), 0.0));
}

#[test]
fn unknown_or_local_model_is_unpriced() {
    assert_eq!(
        calculate_embedding_cost("nomic-embed-text", 1_000_000),
        None
    );
    assert_eq!(
        calculate_embedding_cost("sentence-transformers/all-MiniLM-L6-v2", 1_000_000),
        None
    );
    // multimodal / colbert intentionally omitted → unpriced
    assert_eq!(calculate_embedding_cost("jina-clip-v2", 1_000_000), None);
    assert_eq!(
        calculate_embedding_cost("voyage-multimodal-3.5", 1_000_000),
        None
    );
}
