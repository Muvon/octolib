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
fn splits_bare_repo() {
	let (repo, revision, file) = split_model_string("muvon/octomind-embed");
	assert_eq!(repo, "muvon/octomind-embed");
	assert_eq!(revision, None);
	assert_eq!(file, None);
}

#[test]
fn splits_pinned_graph() {
	let (repo, revision, file) = split_model_string("muvon/octomind-embed#onnx/model_quantized.onnx");
	assert_eq!(repo, "muvon/octomind-embed");
	assert_eq!(revision, None);
	assert_eq!(file.as_deref(), Some("onnx/model_quantized.onnx"));
}

#[test]
fn splits_pinned_revision() {
	let (repo, revision, file) = split_model_string("muvon/octomind-embed@0751f83a");
	assert_eq!(repo, "muvon/octomind-embed");
	assert_eq!(revision.as_deref(), Some("0751f83a"));
	assert_eq!(file, None);
}

#[test]
fn splits_pinned_revision_and_graph() {
	let (repo, revision, file) = split_model_string("muvon/octomind-embed@0751f83a#onnx/model.onnx");
	assert_eq!(repo, "muvon/octomind-embed");
	assert_eq!(revision.as_deref(), Some("0751f83a"));
	assert_eq!(file.as_deref(), Some("onnx/model.onnx"));
}

#[test]
fn trims_whitespace_around_parts() {
	let (repo, revision, file) = split_model_string("  muvon/octomind-embed @ 0751f83a # onnx/model.onnx ");
	assert_eq!(repo, "muvon/octomind-embed");
	assert_eq!(revision.as_deref(), Some("0751f83a"));
	assert_eq!(file.as_deref(), Some("onnx/model.onnx"));
}

#[test]
fn quantized_graph_is_preferred_over_fp32() {
	// Probe order decides which graph a repo publishing both ends up using.
	let quantized = GRAPH_CANDIDATES
		.iter()
		.position(|c| *c == "onnx/model_quantized.onnx")
		.expect("quantized candidate present");
	let fp32 = GRAPH_CANDIDATES
		.iter()
		.position(|c| *c == "onnx/model.onnx")
		.expect("fp32 candidate present");
	assert!(quantized < fp32, "quantized graph must be probed before fp32");
}
