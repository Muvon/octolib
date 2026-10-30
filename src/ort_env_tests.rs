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
fn cache_identity_tracks_graph_and_external_weight_changes() {
	let original = model_hash(std::io::Cursor::new(b"graph"), Some(("weights", b"original"))).unwrap();
	let changed = model_hash(std::io::Cursor::new(b"graph"), Some(("weights", b"updated"))).unwrap();
	assert_ne!(original, changed);
	assert_ne!(
		original,
		model_hash(std::io::Cursor::new(b"other-graph"), Some(("weights", b"original"))).unwrap()
	);
	assert_ne!(
		original,
		model_hash(std::io::Cursor::new(b"graph"), Some(("other-name", b"original"))).unwrap()
	);
	assert_ne!(original, model_hash(std::io::Cursor::new(b"graph"), None).unwrap());
	assert_eq!(
		original,
		model_hash(std::io::Cursor::new(b"graph"), Some(("weights", b"original"))).unwrap()
	);
}

#[test]
fn compiled_cache_is_separated_by_runtime_and_architecture() {
	let identity = model_hash(std::io::Cursor::new(b"same-model"), None).unwrap();
	let key = cache_key(&identity, "runtime-a", "arm64");
	assert_ne!(key, cache_key(&identity, "runtime-b", "arm64"));
	assert_ne!(key, cache_key(&identity, "runtime-a", "x86_64"));
	assert_eq!(key, cache_key(&identity, "runtime-a", "arm64"));
	assert_eq!(key.len(), 64);
	assert!(key.chars().all(|ch| ch.is_ascii_hexdigit()));
}
