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

//! Versioned TOML configuration migration.
//!
//! Every Octo product ships an embedded default template that is the single
//! source of truth for its config schema, and stamps a `version` integer into
//! it. When a user's on-disk config lags behind, it must be upgraded in place
//! WITHOUT losing their values, comments or formatting — which rules out
//! deserialize/reserialize and mandates `toml_edit`.
//!
//! This module owns the parts that are identical in every product: reading the
//! version off both documents, walking the version chain one step at a time,
//! the safety guards (future version, gap in the chain, non-advancing step),
//! and the table helpers each step is written in terms of.
//!
//! A product supplies only its own [`VersionMigration`] steps:
//!
//! ```no_run
//! use octolib::utils::config_migration::{MigrationPlan, VersionMigration, merge_missing};
//!
//! const TEMPLATE: &str = "version = 2\n";
//!
//! let plan = MigrationPlan::new("myapp", vec![VersionMigration {
//!     from: 1,
//!     to: 2,
//!     apply: |document, template| {
//!         // v2 added a whole new section; existing user values always win.
//!         merge_missing(document.as_table_mut(), template.as_table(), "newsection")
//!     },
//! }]);
//!
//! let migration = plan.migrate("version = 1\n", TEMPLATE)?;
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{bail, Context, Result};
use toml_edit::{value, DocumentMut, Item, Table};

/// Re-exported so products can spell the `apply` signature's types without
/// taking their own `toml_edit` dependency (and risking a version skew that
/// would make the function pointer type-mismatch).
pub use toml_edit;

/// A completed upgrade: the rewritten document plus the version span it covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Migration {
    /// Full migrated document, ready to be written back verbatim.
    pub content: String,
    pub from_version: u32,
    pub to_version: u32,
}

/// One step of the version chain. `apply` must upgrade `document` from
/// `from` to `to` and nothing else — the driver stamps the new `version` and
/// advances, so a step never touches the version field itself.
///
/// `template` is the embedded default document, the source for any field the
/// step needs to add. Steps must be additive and idempotent-friendly: a user
/// value that already exists always wins over the template's.
#[derive(Clone)]
pub struct VersionMigration {
    pub from: u32,
    pub to: u32,
    pub apply: fn(document: &mut DocumentMut, template: &DocumentMut) -> Result<()>,
}

impl std::fmt::Debug for VersionMigration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VersionMigration")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish_non_exhaustive()
    }
}

/// The ordered set of migrations one product knows how to perform.
#[derive(Debug, Clone)]
pub struct MigrationPlan {
    /// Product name, used only in error messages ("… this octomind binary …").
    product: &'static str,
    steps: Vec<VersionMigration>,
    missing_version: Option<u32>,
}

impl MigrationPlan {
    /// A plan that REQUIRES an explicit `version` in the user document. Use
    /// this when the product has always stamped one.
    pub fn new(product: &'static str, steps: Vec<VersionMigration>) -> Self {
        Self {
            product,
            steps,
            missing_version: None,
        }
    }

    /// Treat a document with no `version` field as being at `version` instead
    /// of failing. Needed by products whose earliest releases predate the
    /// version stamp — for those, "no version" is a real, migratable state.
    pub fn with_missing_version(mut self, version: u32) -> Self {
        self.missing_version = Some(version);
        self
    }

    /// Version this plan upgrades to — the version of the embedded template.
    pub fn target_version(&self, template: &str) -> Result<u32> {
        let template = parse_document(template, "embedded default configuration")?;
        self.document_version(&template, "embedded default configuration", None)
    }

    /// Read the declared version of an on-disk configuration, applying the
    /// `missing_version` fallback. Cheap: no migration is attempted.
    pub fn version_of(&self, existing: &str) -> Result<u32> {
        let document = parse_document(existing, "user configuration")?;
        self.document_version(&document, "user configuration", self.missing_version)
    }

    /// Upgrade `existing` to the version declared by `template`.
    ///
    /// `Ok(None)` means the config is already current — the caller must then
    /// leave the file completely untouched (no rewrite, no backup, no lock).
    ///
    /// Errors instead of guessing when the config is newer than this binary,
    /// or when no step exists for a version in the chain.
    pub fn migrate(&self, existing: &str, template: &str) -> Result<Option<Migration>> {
        let mut document = parse_document(existing, "user configuration")?;
        let template = parse_document(template, "embedded default configuration")?;

        let from_version =
            self.document_version(&document, "user configuration", self.missing_version)?;
        let target_version =
            self.document_version(&template, "embedded default configuration", None)?;

        if from_version > target_version {
            bail!(
				"configuration version {from_version} is newer than this {} binary supports ({target_version})",
				self.product
			);
        }

        if from_version == target_version {
            return Ok(None);
        }

        let mut version = from_version;
        while version < target_version {
            let step = self
                .steps
                .iter()
                .find(|step| step.from == version)
                .with_context(|| {
                    format!("no configuration migration exists from version {version}")
                })?;

            // A step that does not advance would spin the loop forever; a step
            // that overshoots the template would leave the file claiming a
            // version this binary cannot actually produce.
            if step.to <= version || step.to > target_version {
                bail!(
                    "invalid configuration migration {} -> {} (target version is {target_version})",
                    step.from,
                    step.to
                );
            }

            (step.apply)(&mut document, &template)
                .with_context(|| format!("migrating configuration {} -> {}", step.from, step.to))?;

            version = step.to;
            document["version"] = value(i64::from(version));
        }

        Ok(Some(Migration {
            content: document.to_string(),
            from_version,
            to_version: target_version,
        }))
    }

    fn document_version(
        &self,
        document: &DocumentMut,
        description: &str,
        missing: Option<u32>,
    ) -> Result<u32> {
        let Some(item) = document.get("version") else {
            return missing
                .with_context(|| format!("{description} must contain an integer 'version' field"));
        };

        let version = item
            .as_integer()
            .with_context(|| format!("{description} must contain an integer 'version' field"))?;

        u32::try_from(version)
            .with_context(|| format!("{description} contains invalid version {version}"))
    }
}

fn parse_document(content: &str, description: &str) -> Result<DocumentMut> {
    content
        .parse::<DocumentMut>()
        .with_context(|| format!("failed to parse {description}"))
}

/// Borrow a required table, failing with a locating message when it is absent
/// or is not a table.
pub fn required_table<'a>(table: &'a Table, key: &str, description: &str) -> Result<&'a Table> {
    table
        .get(key)
        .and_then(Item::as_table)
        .with_context(|| format!("{description} must contain a '{key}' table"))
}

/// Mutable [`required_table`].
pub fn required_table_mut<'a>(
    table: &'a mut Table,
    key: &str,
    description: &str,
) -> Result<&'a mut Table> {
    table
        .get_mut(key)
        .and_then(Item::as_table_mut)
        .with_context(|| format!("{description} must contain a '{key}' table"))
}

/// Copy `key` from the template, preserving its formatting and leading
/// comments (that is what carries the documentation into the user's file).
pub fn copy_item(target: &mut Table, source: &Table, key: &str) -> Result<()> {
    let (formatted_key, item) = source
        .get_key_value(key)
        .with_context(|| format!("embedded default configuration is missing '{key}'"))?;
    target.insert_formatted(formatted_key, item.clone());
    Ok(())
}

/// [`copy_item`], but a key the user already set is never overwritten.
pub fn copy_missing_item(target: &mut Table, source: &Table, key: &str) -> Result<()> {
    if target.contains_key(key) {
        return Ok(());
    }
    copy_item(target, source, key)
}

/// Recursively add everything the template has under `key` that the user does
/// not. Absent subtable ⇒ copied whole (comments included); present subtable ⇒
/// descended into, so a user who customised two of five keys keeps both and
/// gains the other three.
///
/// A user value NEVER loses to the template, at any depth. If the user turned
/// `key` into a non-table (or the template's `key` is not a table), it is left
/// exactly as-is: guessing at a type conflict is worse than the stale field.
pub fn merge_missing(target: &mut Table, source: &Table, key: &str) -> Result<()> {
    if !target.contains_key(key) {
        return copy_item(target, source, key);
    }

    let (Some(source_table), Some(target_table)) = (
        source.get(key).and_then(Item::as_table),
        target.get_mut(key).and_then(Item::as_table_mut),
    ) else {
        return Ok(());
    };

    let keys: Vec<String> = source_table.iter().map(|(k, _)| k.to_string()).collect();
    for child in keys {
        merge_missing(target_table, source_table, &child)?;
    }
    Ok(())
}

/// Borrow `key` as a table, creating it from the template when the user does
/// not have it. Use when a step must then edit individual fields inside it.
pub fn ensure_table<'a>(
    target: &'a mut Table,
    source: &Table,
    key: &str,
    description: &str,
) -> Result<&'a mut Table> {
    if !target.contains_key(key) {
        copy_item(target, source, key)?;
    }
    required_table_mut(target, key, description)
}

#[cfg(test)]
#[path = "config_migration_tests.rs"]
mod tests;
