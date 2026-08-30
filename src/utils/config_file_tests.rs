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
        let path = std::env::temp_dir().join(format!("octolib-cfg-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn file(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn migration() -> Migration {
    Migration {
        content: "version = 2\n".to_string(),
        from_version: 1,
        to_version: 2,
    }
}

#[test]
fn parent_directory_maps_bare_names_to_cwd() {
    assert_eq!(
        parent_directory(Path::new("config.toml")).unwrap(),
        Path::new(".")
    );
    assert_eq!(
        parent_directory(Path::new("/a/b/config.toml")).unwrap(),
        Path::new("/a/b")
    );
    assert!(parent_directory(Path::new("/")).is_err());
}

#[test]
fn atomic_write_replaces_and_leaves_no_temp_files() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");
    fs::write(&path, "old").unwrap();

    atomic_write(&path, b"new", None).unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), "new");
    let leftovers: Vec<_> = fs::read_dir(&dir.0)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(
        leftovers.is_empty(),
        "temp files left behind: {leftovers:?}"
    );
}

#[test]
fn atomic_write_creates_missing_parent_directories() {
    let dir = TempDir::new();
    let path = dir.0.join("nested/deeper/config.toml");

    atomic_write(&path, b"fresh", None).unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), "fresh");
}

fn backup_name(version: u32, content: &[u8]) -> String {
    let digest = hex::encode(&Sha256::digest(content)[..4]);
    format!("config.toml.v{version}.{digest}.bak")
}

#[test]
fn backup_is_written_once_and_is_idempotent() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");
    fs::write(&path, "v1 body").unwrap();
    let permissions = fs::metadata(&path).unwrap().permissions();

    write_backup_if_missing(&path, 1, b"v1 body", permissions.clone()).unwrap();
    // second identical call must not fail
    write_backup_if_missing(&path, 1, b"v1 body", permissions).unwrap();

    assert_eq!(
        fs::read_to_string(dir.file(&backup_name(1, b"v1 body"))).unwrap(),
        "v1 body"
    );
}

#[test]
fn re_migrating_an_edited_config_keeps_both_backups() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");
    fs::write(&path, "edited").unwrap();
    let permissions = fs::metadata(&path).unwrap().permissions();

    write_backup_if_missing(&path, 1, b"original", permissions.clone()).unwrap();
    write_backup_if_missing(&path, 1, b"edited", permissions).unwrap();

    assert_eq!(
        fs::read_to_string(dir.file(&backup_name(1, b"original"))).unwrap(),
        "original"
    );
    assert_eq!(
        fs::read_to_string(dir.file(&backup_name(1, b"edited"))).unwrap(),
        "edited"
    );
}

#[test]
fn apply_migration_backs_up_then_replaces() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");
    fs::write(&path, "version = 1\n").unwrap();

    apply_migration(&path, b"version = 1\n", &migration()).unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), "version = 2\n");
    assert_eq!(
        fs::read_to_string(dir.file(&backup_name(1, b"version = 1\n"))).unwrap(),
        "version = 1\n"
    );
}

#[test]
fn lock_is_reentrant_across_sequential_calls_and_keeps_the_lock_file() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");

    for _ in 0..2 {
        with_lock(&path, || Ok(())).unwrap();
    }
    assert!(dir.file(".config.toml.lock").exists());
}

#[test]
fn lock_propagates_the_operation_error() {
    let dir = TempDir::new();
    let path = dir.file("config.toml");

    let error = with_lock(&path, || -> Result<()> { anyhow::bail!("inner failed") }).unwrap_err();
    assert!(error.to_string().contains("inner failed"));
}
