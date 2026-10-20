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
use std::path::PathBuf;

struct TempDir(PathBuf);

impl TempDir {
	fn new() -> Self {
		let path = std::env::temp_dir().join(format!("octolib-scope-{}", uuid::Uuid::new_v4()));
		std::fs::create_dir_all(&path).unwrap();
		Self(path)
	}
}

impl Drop for TempDir {
	fn drop(&mut self) {
		let _ = std::fs::remove_dir_all(&self.0);
	}
}

fn repo_with_origin(path: &Path, origin: &str) {
	std::fs::create_dir_all(path).unwrap();
	for args in [vec!["init", "-q"], vec!["remote", "add", "origin", origin]] {
		let status = Command::new("git").args(args).current_dir(path).status().unwrap();
		assert!(status.success());
	}
}

#[test]
fn clones_of_one_repo_share_readable_scope_and_id() {
	let root = TempDir::new();
	let https = root.0.join("https");
	let ssh = root.0.join("ssh");
	repo_with_origin(&https, "https://x-access-token:secret@github.com/Muvon/Octolib.git");
	repo_with_origin(&ssh, "git@github.com:muvon/octolib.git");

	assert_eq!(path_to_scope(&https), "github.com/muvon/octolib");
	assert_eq!(path_to_scope(&ssh), "github.com/muvon/octolib");
	assert_eq!(path_to_id(&https), path_to_id(&ssh));
}

#[test]
fn directory_without_remote_gets_local_scope() {
	let root = TempDir::new();
	let workdir = root.0.join("Plain-Work");
	std::fs::create_dir(&workdir).unwrap();
	let parent = root
		.0
		.canonicalize()
		.unwrap()
		.file_name()
		.unwrap()
		.to_string_lossy()
		.to_lowercase();

	assert_eq!(path_to_scope(&workdir), format!("local/{}/plain-work", parent));
}
