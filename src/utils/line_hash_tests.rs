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

#[test]
fn hashes_preserve_the_octofs_wire_format() {
	assert_eq!(line_hash(""), "Al");
	assert_eq!(line_hash("export enum EffectFlags {"), "Hc");
	assert_eq!(line_hash("  ALLOW_RECURSE = 1 << 7,"), "KD");
	assert_eq!(line_hash("\tlet café = \"界\";"), "hN");
	assert_eq!(line_hash("}"), "Yu");
}

#[test]
fn hashes_are_two_ascii_letters() {
	for text in ["", "}", "fn main() {", "\tlet x = a | b;", "12:30"] {
		let hash = line_hash(text);
		assert_eq!(hash.len(), LINE_HASH_LEN, "{text:?} -> {hash}");
		assert!(hash.bytes().all(|b| b.is_ascii_alphabetic()), "{text:?} -> {hash}");
	}
}

#[test]
fn line_ids_hash_the_whole_physical_line_and_keep_its_position_separate() {
	let text = "  export enum EffectFlags {";
	assert_ne!(line_hash(text), line_hash(text.trim_start()));
	assert_eq!(line_id(51, text), format!("51:{}", line_hash(text)));
	assert_eq!(format_line(51, text), format!("{} {text}", line_id(51, text)));
	// The separator is one space: the line's own indentation follows it untouched.
	assert_eq!(
		format_line(51, text),
		"51:".to_string() + &line_hash(text) + "   export enum EffectFlags {"
	);
	assert_eq!(format_line(7, ""), "7:Al ");
}

#[test]
fn parse_line_reads_back_what_format_line_rendered() {
	for (number, text) in [
		(1, "fn main() {"),
		(51, "  export enum EffectFlags {"),
		(9, "\tlet x = a | b;"),
		(7, ""),
	] {
		assert_eq!(parse_line(&format_line(number, text)), Some((number, text)));
	}
	// A blank line keeps its id when the trailing separator was trimmed.
	assert_eq!(parse_line("7:Al"), Some((7, "")));
}

#[test]
fn parse_line_id_checks_the_shape_only() {
	assert_eq!(parse_line_id("12:aK"), Some((12, "aK")));
	assert_eq!(parse_line_id("12:Zz"), Some((12, "Zz")));
	// Digits, a wrong length, line 0 and a sign are not ids.
	assert_eq!(parse_line_id("12:30"), None);
	assert_eq!(parse_line_id("12:a3"), None);
	assert_eq!(parse_line_id("12:a"), None);
	assert_eq!(parse_line_id("12:abc"), None);
	assert_eq!(parse_line_id("0:aK"), None);
	assert_eq!(parse_line_id("+7:aK"), None);
	assert_eq!(parse_line_id(":aK"), None);
	assert_eq!(parse_line_id("aK"), None);
}

#[test]
fn parse_line_rejects_text_that_only_looks_like_an_id_line() {
	// `grep -n` output: "be" has the shape of a hash, but is not the hash of "careful".
	assert_eq!(parse_line("7:be careful"), None);
	// Line 0 does not exist, a sign is not part of an id, and a hash alone names no line.
	assert_eq!(parse_line("0:Al "), None);
	assert_eq!(parse_line("+7:Al "), None);
	assert_eq!(parse_line("Al "), None);
	// The old hex form no longer parses.
	assert_eq!(parse_line("7:c5 "), None);
	// Edit-result rows that carry ids but no line content.
	assert_eq!(parse_line("2:cd..3:ef (2 lines as sent)"), None);
	assert_eq!(parse_line("[... 4 lines omitted]"), None);
}

#[test]
fn lf_and_crlf_source_lines_produce_identical_ids() {
	let lf = "alpha\n  beta\n";
	let crlf = "alpha\r\n  beta\r\n";
	let render = |source: &str| {
		source
			.lines()
			.enumerate()
			.map(|(i, line)| format_line(i + 1, line))
			.collect::<Vec<_>>()
	};
	assert_eq!(render(lf), render(crlf));
}
