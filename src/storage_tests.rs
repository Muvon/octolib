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
fn test_get_model_cache_dir_creates_and_returns_path() {
	let dir = get_model_cache_dir().unwrap();
	assert!(dir.ends_with("octolib/huggingface"));
	assert!(dir.is_dir());
}

#[test]
fn test_deprecated_aliases_return_same_dir() {
	let canonical = get_model_cache_dir().unwrap();
	#[allow(deprecated)]
	let fastembed = get_fastembed_cache_dir().unwrap();
	#[allow(deprecated)]
	let huggingface = get_huggingface_cache_dir().unwrap();
	assert_eq!(canonical, fastembed);
	assert_eq!(canonical, huggingface);
}
