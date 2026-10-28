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
	assert_eq!(line_hash(""), "c5");
	assert_eq!(line_hash("export enum EffectFlags {"), "9f");
	assert_eq!(line_hash("  ALLOW_RECURSE = 1 << 7,"), "30");
	assert_eq!(line_hash("\tlet café = \"界\";"), "2b");
	assert_eq!(line_hash("}"), "14");
}

#[test]
fn line_ids_hash_the_whole_physical_line_and_keep_its_position_separate() {
	let text = "  export enum EffectFlags {";
	assert_ne!(line_hash(text), line_hash(text.trim_start()));
	assert_eq!(line_id(51, text), format!("51:{}", line_hash(text)));
	assert_eq!(format_line(51, text), format!("{}|{text}", line_id(51, text)));
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
