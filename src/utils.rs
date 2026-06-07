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

//! Utility functions shared across Octo projects.
//!
//! Currently includes git project ID derivation. More utilities may be added here.

use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

/// Normalize a git remote URL to a stable `host/org/repo` key (lowercased,
/// no scheme, no credentials, no `.git` suffix).
///
/// Returns `None` when the input cannot be parsed into a recognizable form.
pub fn normalize_git_url(url: &str) -> String {
    let url = url.trim().strip_suffix(".git").unwrap_or(url.trim());

    // Scheme-based URLs (https://, http://, ssh://, git://)
    if let Some(scheme_end) = url.find("://") {
        let authority_and_path = &url[scheme_end + 3..];
        return strip_userinfo(authority_and_path).to_lowercase();
    }

    // SCP-style SSH: [user@]host:path
    if let Some(at) = url.find('@') {
        if let Some(colon_rel) = url[at..].find(':') {
            let host = &url[at + 1..at + colon_rel];
            let path = &url[at + colon_rel + 1..];
            return format!("{}/{}", host, path).to_lowercase();
        }
    }

    url.to_lowercase()
}

/// Strip a leading `userinfo@` from an `[authority]/path` string.
/// Only the authority segment (before the first `/`) is examined so that
/// `@` characters in the path are left untouched.
fn strip_userinfo(authority_and_path: &str) -> &str {
    let auth_end = authority_and_path
        .find('/')
        .unwrap_or(authority_and_path.len());
    match authority_and_path[..auth_end].rfind('@') {
        Some(at) => &authority_and_path[at + 1..],
        None => authority_and_path,
    }
}

/// Derive a stable 16-hex-char project ID from a directory path.
///
/// Always canonicalizes the path first. If the path is inside a git repo with
/// a remote origin, the normalized URL is hashed (credentials stripped,
/// lowercased, scheme and .git suffix removed) so SSH and token-auth HTTPS
/// clones of the same repo produce the same ID. Falls back to the canonical
/// absolute path when no git remote is found.
pub fn path_to_id(path: &Path) -> String {
    let source = Command::new("git")
        .args(["remote", "get-url", "origin"])
        .current_dir(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| normalize_git_url(s.trim()))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            path.canonicalize()
                .unwrap_or_else(|_| path.to_path_buf())
                .to_string_lossy()
                .into_owned()
        });

    let hash = Sha256::digest(source.as_bytes());
    hex::encode(hash)[..16].to_string()
}

/// Derive a stable 16-hex-char project ID from the current working directory.
pub fn path_to_id_cwd() -> String {
    let path = std::env::current_dir().unwrap_or_default();
    path_to_id(&path)
}

/// Returns true if `path` itself is a git repository root (has a `.git` entry).
/// Does NOT walk up to parent directories.
pub fn is_git_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

/// Returns true if the current working directory is inside a git repository.
pub fn is_git_repo_cwd() -> bool {
    let path = std::env::current_dir().unwrap_or_default();
    is_git_repo(&path)
}

/// Extract the `org/repo` portion (lowercased) from a normalized URL or raw remote URL.
///
/// `github.com/muvon/octomind` → `muvon/octomind`
/// `git@github.com:Muvon/octomind.git` → `muvon/octomind`
pub fn org_repo_from_url(url: &str) -> String {
    let normalized = normalize_git_url(url);
    // normalized is `host/org/repo` — drop the host segment
    normalized
        .split_once('/')
        .map(|(_, rest)| rest)
        .unwrap_or(&normalized)
        .to_string()
}
