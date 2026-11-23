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

/// What stands between a line id and the line's content in rendered output. One
/// space, because a tokenizer folds it into the indentation that follows, where a
/// visible mark such as `|` costs a token of its own on every line. The id ends at
/// the first space, so the content, indentation included, is everything after it.
pub const LINE_ID_SEPARATOR: char = ' ';

/// Render an original source line without changing its content or indentation.
pub fn format_line(line_1idx: usize, content: &str) -> String {
	format!("{}{LINE_ID_SEPARATOR}{}", line_id(line_1idx, content), content)
}

/// The line number and content of a line [`format_line`] rendered, or `None` for
/// any other text. The hash has to be the content's own: a numbered line another
/// program printed (`7:be careful` from `grep -n`) has the same shape, and only
/// the hash tells it from an id line. A blank line whose trailing separator was
/// trimmed away still parses.
pub fn parse_line(rendered: &str) -> Option<(usize, &str)> {
	let (id, content) = rendered.split_once(LINE_ID_SEPARATOR).unwrap_or((rendered, ""));
	let (number, hash) = id.split_once(':')?;
	// `usize::from_str` takes a leading `+`, which an id never has.
	if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
		return None;
	}
	let line_1idx: usize = number.parse().ok().filter(|n| *n > 0)?;
	(line_hash(content) == hash).then_some((line_1idx, content))
}

#[cfg(test)]
#[path = "line_hash_tests.rs"]
mod tests;
