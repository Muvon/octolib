# Versioned Configuration Migration

Octolib provides reusable building blocks for upgrading application-owned TOML configuration files without discarding user values, comments, unknown fields, or formatting.

The API is split into two modules:

- `octolib::utils::config_migration` parses versions, runs an ordered migration chain, and provides helpers for copying fields from an embedded template.
- `octolib::utils::config_file` serializes concurrent upgrades, creates a versioned backup, and replaces the configuration atomically.

Your application still owns its configuration schema, embedded default template, migration steps, and final typed validation.

## Define the version contract

Keep a root-level integer `version` in the default template and embed that template in the binary:

```toml
version = 2

[model]
name = "openai:gpt-5.6"

# Added in version 2.
[cache]
enabled = true
```

```rust
const DEFAULT_CONFIG: &str = include_str!("../config-templates/default.toml");
```

Embedding is important: a released binary must migrate against the template it was built with, not a checkout-relative file that may belong to another release.

If early releases had no `version` field, decide which released schema they represent and configure that explicitly with `with_missing_version`. Do not treat an absent version as the newest schema.

## Define sequential migration steps

Each step changes one schema version only. The migration driver runs every required step in order and updates the root `version` field after each successful step.

```rust
use anyhow::Result;
use octolib::utils::config_migration::{
    merge_missing,
    toml_edit::DocumentMut,
    MigrationPlan,
    VersionMigration,
};

fn migrate_v1_to_v2(
    document: &mut DocumentMut,
    template: &DocumentMut,
) -> Result<()> {
    // Version 2 introduced the whole [cache] section. Existing user values win.
    merge_missing(document.as_table_mut(), template.as_table(), "cache")
}

fn migration_plan() -> MigrationPlan {
    MigrationPlan::new(
        "myapp",
        vec![VersionMigration {
            from: 1,
            to: 2,
            apply: migrate_v1_to_v2,
        }],
    )
    // Use only when released unversioned configs are known to be schema v1.
    .with_missing_version(1)
}
```

When a later release introduces version 3, append a `2 -> 3` step. Do not fold the new behavior into `1 -> 2`: users may already be on version 2, and the chain must work from every supported starting version.

Useful helpers are:

- `copy_missing_item`: copy one template field only when the user does not have it.
- `merge_missing`: recursively fill missing fields beneath a key while preserving user values.
- `ensure_table`: copy a missing table from the template, then return it for more targeted edits.
- `required_table` and `required_table_mut`: access a required table with a contextual error.
- `copy_item`: deliberately replace or insert an item from the template. Prefer the non-overwriting helpers for ordinary additions.

Migration callbacks must not set `version`; `MigrationPlan` owns that field. Keep each callback narrow, deterministic, and safe when user-defined or unknown fields are present.

## Load, validate, and persist safely

The safe order for an existing outdated file is:

1. Read its exact bytes.
2. Build the migrated TOML in memory.
3. Deserialize and validate the migrated result with the application's normal parser.
4. Write the original bytes to `config.toml.v<N>.bak`.
5. Atomically replace `config.toml`, preserving its permissions.

The following example also avoids creating a lock file for configurations that are already current. It re-reads and re-runs the plan after acquiring the lock because another process may have completed the migration meanwhile.

```rust
use anyhow::{Context, Result};
use octolib::utils::{
    config_file::{apply_migration, atomic_write, with_lock},
    config_migration::MigrationPlan,
};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, Deserialize)]
struct AppConfig {
    version: u32,
    model: ModelConfig,
    cache: CacheConfig,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    name: String,
}

#[derive(Debug, Deserialize)]
struct CacheConfig {
    enabled: bool,
}

fn parse_and_validate(content: &str) -> Result<AppConfig> {
    let config: AppConfig = toml::from_str(content).context("invalid configuration")?;

    if config.model.name.trim().is_empty() {
        anyhow::bail!("model.name must not be empty");
    }

    Ok(config)
}

fn load_existing_locked(path: &Path, plan: &MigrationPlan) -> Result<AppConfig> {
    let original = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    let Some(migration) = plan.migrate(&original, DEFAULT_CONFIG)? else {
        // Another process may have migrated the file before this lock was acquired.
        return parse_and_validate(&original);
    };

    // Validate before touching the user's file.
    let config = parse_and_validate(&migration.content)?;
    apply_migration(path, original.as_bytes(), &migration)?;
    Ok(config)
}

fn load_config(path: &Path) -> Result<AppConfig> {
    let plan = migration_plan();

    if !path.exists() {
        return with_lock(path, || {
            // Another process may have created it while this process waited.
            if path.exists() {
                return load_existing_locked(path, &plan);
            }

            let config = parse_and_validate(DEFAULT_CONFIG)?;
            atomic_write(path, DEFAULT_CONFIG.as_bytes(), None)?;
            Ok(config)
        });
    }

    let original = fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    if plan.migrate(&original, DEFAULT_CONFIG)?.is_none() {
        // Current configurations are parsed but not rewritten, backed up, or locked.
        return parse_and_validate(&original);
    }

    with_lock(path, || load_existing_locked(path, &plan))
}
```

This example uses `toml` and `serde` in the application for typed validation. Octolib deliberately does not choose the application's configuration type or validation rules.

## File behavior

For `config.toml`, the persistence helpers use these sibling files:

- `.config.toml.lock`: the advisory lock. It is intentionally retained because deleting it can let processes lock different inodes.
- `config.toml.v1.bak`: the exact pre-migration content for a migration starting at version 1.
- `.config.toml.<uuid>.tmp`: a same-directory temporary file used during atomic replacement and removed after success or a handled failure.

An existing backup with identical bytes is accepted, which makes retrying safe. An existing same-version backup with different bytes aborts the write rather than destroying the earlier backup.

`atomic_write` syncs the temporary file before rename and, on Unix, syncs the parent directory afterward. `apply_migration` also carries the original file permissions to the replacement.

## Failure and compatibility rules

- A configuration newer than the embedded template is rejected. This prevents an older binary from silently downgrading a newer file.
- A missing migration step, a non-advancing step, or a step that overshoots the template version is rejected.
- Invalid source TOML and invalid migrated application configuration fail before replacement.
- `Ok(None)` from `MigrationPlan::migrate` means the file is already current; leave it byte-for-byte unchanged.
- Missing configurations should be created from the embedded template exactly. Migration helpers are for existing user files, not for synthesizing a new default.
- Do not deserialize and reserialize merely to migrate. That loses the comments and formatting these utilities are designed to preserve.
- Do not merge every new default into every existing config automatically. Add only fields required by the schema transition, so changed defaults do not overwrite or unexpectedly expand user configuration.

For API-level details, see the rustdoc for `utils::config_migration` and `utils::config_file`.
