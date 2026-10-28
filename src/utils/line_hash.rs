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

//! Content-based line IDs shared by source readers and filesystem editors.

/// FNV-1a hash of line content folded to 8 bits (2 hex chars).
/// Content-only — position lives in the id's number part, so a line keeps its
/// hash when it moves and a stale-id error can report where the content went.
/// 8 bits means a changed line keeps its hash with probability 1/256 — accepted
/// trade-off for id brevity (the position check catches gross drift anyway).
fn fnv1a_8(content: &str) -> u8 {
	const FNV_OFFSET: u32 = 2166136261;
	const FNV_PRIME: u32 = 16777619;

	let mut hash = FNV_OFFSET;
	for byte in content.bytes() {
		hash ^= byte as u32;
		hash = hash.wrapping_mul(FNV_PRIME);
	}

	// Fold 32 -> 16 -> 8 via XOR
	let h16 = ((hash >> 16) ^ (hash & 0xFFFF)) as u16;
	((h16 >> 8) ^ (h16 & 0xFF)) as u8
}

/// 2-char hex content hash of a single line.
pub fn line_hash(content: &str) -> String {
	format!("{:02x}", fnv1a_8(content))
}

/// Composite id "N:hh" for a 1-indexed line with the given content.
pub fn line_id(line_1idx: usize, content: &str) -> String {
	format!("{}:{}", line_1idx, line_hash(content))
}

/// Render an original source line without changing its content or indentation.
pub fn format_line(line_1idx: usize, content: &str) -> String {
	format!("{}|{}", line_id(line_1idx, content), content)
}

#[cfg(test)]
#[path = "line_hash_tests.rs"]
mod tests;
