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

//! Crash-safe configuration file writes.
//!
//! A configuration upgrade rewrites a file the user owns and may be running
//! several processes against. Three properties are non-negotiable and are
//! implemented here once for every Octo product:
//!
//! 1. **Atomic** — write a sibling temp file, fsync it, rename over the target.
//!    A crash mid-write leaves either the old file or the new one, never a
//!    truncated config.
//! 2. **Locked** — an exclusive advisory lock on a sibling `.<name>.lock`
//!    serialises concurrent starts so two processes cannot both migrate.
//! 3. **Backed up** — the pre-migration bytes land in `<name>.v<N>.<digest>.bak`
//!    before the rewrite. The digest names the content, so repeated migrations
//!    never overwrite an earlier backup.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions, Permissions};
use std::io::Write;
use std::path::{Path, PathBuf};

use super::config_migration::Migration;

/// Parent directory of `path`, mapping the "bare file name" case (parent is
/// `""`) to `.` so a relative config path still resolves.
pub fn parent_directory(path: &Path) -> Result<&Path> {
    match path.parent() {
        Some(parent) if parent.as_os_str().is_empty() => Ok(Path::new(".")),
        Some(parent) => Ok(parent),
        None => anyhow::bail!("configuration path has no parent: {}", path.display()),
    }
}

/// Run `operation` holding an exclusive lock on a sibling of `config_path`.
///
/// The lock file is `.<file name>.lock` and is intentionally never deleted:
/// unlinking it would let a second process lock a different inode and defeat
/// the mutual exclusion.
pub fn with_lock<T>(config_path: &Path, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    let parent = parent_directory(config_path)?;
    fs::create_dir_all(parent)?;

    let file_name = file_name_of(config_path)?;
    let lock_path = config_path.with_file_name(format!(".{file_name}.lock"));
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("failed to open configuration lock {}", lock_path.display()))?;
    fs4::FileExt::lock_exclusive(&lock_file)
        .with_context(|| format!("failed to lock configuration at {}", config_path.display()))?;

    operation()
}

/// Replace `path` with `content` atomically, optionally carrying permissions
/// over from the file being replaced.
pub fn atomic_write(path: &Path, content: &[u8], permissions: Option<Permissions>) -> Result<()> {
    let parent = parent_directory(path)?;
    fs::create_dir_all(parent)?;

    let file_name = file_name_of(path)?;
    let temporary_path = parent.join(format!(".{file_name}.{}.tmp", uuid::Uuid::new_v4()));

    let result = (|| -> Result<()> {
        let mut temporary = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)?;
        temporary.write_all(content)?;
        temporary.sync_all()?;
        if let Some(permissions) = permissions {
            temporary.set_permissions(permissions)?;
            temporary.sync_all()?;
        }
        drop(temporary);

        fs::rename(&temporary_path, path)
            .with_context(|| format!("failed to replace configuration at {}", path.display()))?;

        // Durable rename: without fsyncing the directory the entry can be lost
        // on power failure even though the file contents were synced.
        #[cfg(unix)]
        OpenOptions::new().read(true).open(parent)?.sync_all()?;

        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_path);
    }

    result
}

/// Write `content` to `<config file name>.v<version>.<digest>.bak`.
///
/// The name carries a digest of the content, so an existing file at that path
/// already holds these bytes: re-running a migration is a no-op, and a config
/// the user edited between runs gets its own backup instead of clobbering the
/// previous one.
pub fn write_backup_if_missing(
    config_path: &Path,
    version: u32,
    content: &[u8],
    permissions: Permissions,
) -> Result<PathBuf> {
    let file_name = file_name_of(config_path)?;
    let digest = hex::encode(&Sha256::digest(content)[..4]);
    let backup_path = config_path.with_file_name(format!("{file_name}.v{version}.{digest}.bak"));

    let mut backup = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&backup_path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Ok(backup_path),
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "failed to create configuration backup {}",
                    backup_path.display()
                )
            })
        }
    };

    let result = (|| -> Result<()> {
        backup.write_all(content)?;
        backup.sync_all()?;
        backup.set_permissions(permissions)?;
        backup.sync_all()?;
        Ok(())
    })();

    if result.is_err() {
        drop(backup);
        let _ = fs::remove_file(&backup_path);
    }

    result.map(|()| backup_path)
}

/// Persist a migration: back up `original` under its old version, then write
/// the migrated content atomically, preserving the file's permissions.
/// Returns the backup path.
///
/// The caller MUST have validated that the migrated content actually
/// deserializes before calling this — the user's file is only replaced once we
/// know the result is loadable.
pub fn apply_migration(
    config_path: &Path,
    original: &[u8],
    migration: &Migration,
) -> Result<PathBuf> {
    let permissions = fs::metadata(config_path)
        .with_context(|| {
            format!(
                "failed to read configuration metadata at {}",
                config_path.display()
            )
        })?
        .permissions();

    let backup_path = write_backup_if_missing(
        config_path,
        migration.from_version,
        original,
        permissions.clone(),
    )?;
    atomic_write(config_path, migration.content.as_bytes(), Some(permissions))?;
    Ok(backup_path)
}

fn file_name_of(path: &Path) -> Result<String> {
    Ok(path
        .file_name()
        .with_context(|| format!("configuration path has no file name: {}", path.display()))?
        .to_string_lossy()
        .into_owned())
}

#[cfg(test)]
mod tests {
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

        let error =
            with_lock(&path, || -> Result<()> { anyhow::bail!("inner failed") }).unwrap_err();
        assert!(error.to_string().contains("inner failed"));
    }
}
