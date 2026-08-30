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

const TEMPLATE: &str = r#"version = 2

[alpha]
kept = 1

# documented section
[beta]
added = "from-template"
nested_missing = true

[beta.deep]
value = 9
"#;

fn add_beta(document: &mut DocumentMut, template: &DocumentMut) -> Result<()> {
    merge_missing(document.as_table_mut(), template.as_table(), "beta")
}

fn plan() -> MigrationPlan {
    MigrationPlan::new(
        "testapp",
        vec![VersionMigration {
            from: 1,
            to: 2,
            apply: add_beta,
        }],
    )
}

#[test]
fn current_version_is_never_rewritten() {
    assert!(plan().migrate(TEMPLATE, TEMPLATE).unwrap().is_none());
}

#[test]
fn adds_missing_section_with_its_comments() {
    let migration = plan()
        .migrate("version = 1\n\n[alpha]\nkept = 42\n", TEMPLATE)
        .unwrap()
        .expect("v1 must migrate");

    assert_eq!(migration.from_version, 1);
    assert_eq!(migration.to_version, 2);
    assert!(migration.content.contains("# documented section"));

    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();
    assert_eq!(migrated["version"].as_integer(), Some(2));
    assert_eq!(migrated["alpha"]["kept"].as_integer(), Some(42));
    assert_eq!(migrated["beta"]["added"].as_str(), Some("from-template"));
}

#[test]
fn user_values_always_win_and_gaps_are_filled() {
    let existing = "version = 1\n\n[beta]\nadded = \"mine\"\n\n[beta.deep]\nvalue = 1\n";
    let migration = plan().migrate(existing, TEMPLATE).unwrap().unwrap();
    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();

    assert_eq!(migrated["beta"]["added"].as_str(), Some("mine"));
    assert_eq!(migrated["beta"]["deep"]["value"].as_integer(), Some(1));
    // untouched-by-user key gained from the template, at both levels
    assert_eq!(migrated["beta"]["nested_missing"].as_bool(), Some(true));
}

#[test]
fn merge_missing_leaves_type_conflicts_alone() {
    // user turned a table into a scalar — we do not "fix" that silently
    let existing = "version = 1\nbeta = 5\n";
    let migration = plan().migrate(existing, TEMPLATE).unwrap().unwrap();
    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();
    assert_eq!(migrated["beta"].as_integer(), Some(5));
}

#[test]
fn comments_and_unknown_user_keys_survive() {
    let existing = "# my notes\nversion = 1\n\n[custom]\nmine = true\n";
    let migration = plan().migrate(existing, TEMPLATE).unwrap().unwrap();

    assert!(migration.content.contains("# my notes"));
    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();
    assert_eq!(migrated["custom"]["mine"].as_bool(), Some(true));
}

#[test]
fn walks_a_multi_step_chain_in_order() {
    const V3: &str = "version = 3\n\n[beta]\nadded = \"t\"\n\n[gamma]\ng = 1\n";
    fn add_gamma(document: &mut DocumentMut, template: &DocumentMut) -> Result<()> {
        merge_missing(document.as_table_mut(), template.as_table(), "gamma")
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![
            VersionMigration {
                from: 1,
                to: 2,
                apply: add_beta,
            },
            VersionMigration {
                from: 2,
                to: 3,
                apply: add_gamma,
            },
        ],
    );

    let migration = plan.migrate("version = 1\n", V3).unwrap().unwrap();
    assert_eq!(migration.from_version, 1);
    assert_eq!(migration.to_version, 3);
    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();
    assert_eq!(migrated["version"].as_integer(), Some(3));
    assert_eq!(migrated["beta"]["added"].as_str(), Some("t"));
    assert_eq!(migrated["gamma"]["g"].as_integer(), Some(1));
}

#[test]
fn starts_mid_chain_without_replaying_earlier_steps() {
    const V3: &str = "version = 3\n\n[gamma]\ng = 1\n";
    fn boom(_: &mut DocumentMut, _: &DocumentMut) -> Result<()> {
        panic!("earlier step must not run");
    }
    fn add_gamma(document: &mut DocumentMut, template: &DocumentMut) -> Result<()> {
        merge_missing(document.as_table_mut(), template.as_table(), "gamma")
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![
            VersionMigration {
                from: 1,
                to: 2,
                apply: boom,
            },
            VersionMigration {
                from: 2,
                to: 3,
                apply: add_gamma,
            },
        ],
    );

    let migration = plan.migrate("version = 2\n", V3).unwrap().unwrap();
    assert_eq!(migration.from_version, 2);
}

#[test]
fn rejects_future_versions_without_touching_them() {
    let future = TEMPLATE.replacen("version = 2", "version = 9", 1);
    let error = plan().migrate(&future, TEMPLATE).unwrap_err();
    assert!(error.to_string().contains("newer than this testapp binary"));
}

#[test]
fn rejects_a_gap_in_the_chain() {
    // The chain knows 1 -> 2 only, so a template at 3 leaves 2 -> 3 unmet.
    const V3: &str = "version = 3\n";
    let error = plan().migrate("version = 2\n", V3).unwrap_err();
    assert!(error
        .to_string()
        .contains("no configuration migration exists from version 2"));
}

#[test]
fn rejects_a_step_that_does_not_advance() {
    fn noop(_: &mut DocumentMut, _: &DocumentMut) -> Result<()> {
        Ok(())
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![VersionMigration {
            from: 1,
            to: 1,
            apply: noop,
        }],
    );
    let error = plan.migrate("version = 1\n", TEMPLATE).unwrap_err();
    assert!(error
        .to_string()
        .contains("invalid configuration migration 1 -> 1"));
}

#[test]
fn rejects_a_step_that_overshoots_the_template() {
    fn noop(_: &mut DocumentMut, _: &DocumentMut) -> Result<()> {
        Ok(())
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![VersionMigration {
            from: 1,
            to: 7,
            apply: noop,
        }],
    );
    let error = plan.migrate("version = 1\n", TEMPLATE).unwrap_err();
    assert!(error
        .to_string()
        .contains("invalid configuration migration 1 -> 7"));
}

#[test]
fn a_failing_step_aborts_the_whole_migration() {
    fn fails(_: &mut DocumentMut, _: &DocumentMut) -> Result<()> {
        bail!("step exploded")
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![VersionMigration {
            from: 1,
            to: 2,
            apply: fails,
        }],
    );
    let error = plan.migrate("version = 1\n", TEMPLATE).unwrap_err();
    assert!(format!("{error:#}").contains("migrating configuration 1 -> 2"));
    assert!(format!("{error:#}").contains("step exploded"));
}

#[test]
fn missing_version_is_an_error_by_default() {
    let error = plan().migrate("[alpha]\nkept = 1\n", TEMPLATE).unwrap_err();
    assert!(error.to_string().contains("integer 'version' field"));
}

#[test]
fn missing_version_fallback_makes_unversioned_configs_migratable() {
    fn stamp(_: &mut DocumentMut, _: &DocumentMut) -> Result<()> {
        Ok(())
    }
    let plan = MigrationPlan::new(
        "testapp",
        vec![
            VersionMigration {
                from: 0,
                to: 1,
                apply: stamp,
            },
            VersionMigration {
                from: 1,
                to: 2,
                apply: add_beta,
            },
        ],
    )
    .with_missing_version(0);

    let migration = plan
        .migrate("[alpha]\nkept = 3\n", TEMPLATE)
        .unwrap()
        .unwrap();
    assert_eq!(migration.from_version, 0);
    let migrated: toml::Value = toml::from_str(&migration.content).unwrap();
    assert_eq!(migrated["version"].as_integer(), Some(2));
    assert_eq!(migrated["alpha"]["kept"].as_integer(), Some(3));
}

#[test]
fn non_integer_version_is_rejected_even_with_a_fallback() {
    let plan = plan().with_missing_version(0);
    let error = plan.migrate("version = \"1\"\n", TEMPLATE).unwrap_err();
    assert!(error.to_string().contains("integer 'version' field"));
}

#[test]
fn negative_version_is_rejected() {
    let error = plan().migrate("version = -1\n", TEMPLATE).unwrap_err();
    assert!(error.to_string().contains("invalid version -1"));
}

#[test]
fn malformed_toml_is_rejected_before_anything_is_read() {
    let error = plan()
        .migrate("version = 1\n[unclosed\n", TEMPLATE)
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("failed to parse user configuration"));
}

#[test]
fn version_of_and_target_version_report_the_chain_ends() {
    let plan = plan().with_missing_version(0);
    assert_eq!(plan.version_of("version = 1\n").unwrap(), 1);
    assert_eq!(plan.version_of("[alpha]\n").unwrap(), 0);
    assert_eq!(plan.target_version(TEMPLATE).unwrap(), 2);
}

#[test]
fn copy_helpers_respect_existing_values() {
    let template: DocumentMut = TEMPLATE.parse().unwrap();
    let mut document: DocumentMut = "[beta]\nadded = \"mine\"\n".parse().unwrap();

    let source = required_table(template.as_table(), "beta", "template").unwrap();
    let target = required_table_mut(document.as_table_mut(), "beta", "user").unwrap();

    copy_missing_item(target, source, "added").unwrap();
    assert_eq!(target["added"].as_str(), Some("mine"));

    copy_item(target, source, "added").unwrap();
    assert_eq!(target["added"].as_str(), Some("from-template"));

    assert!(copy_item(target, source, "nope")
        .unwrap_err()
        .to_string()
        .contains("missing 'nope'"));
}

#[test]
fn ensure_table_creates_then_reuses() {
    let template: DocumentMut = TEMPLATE.parse().unwrap();
    let mut document: DocumentMut = "version = 1\n".parse().unwrap();

    {
        let beta =
            ensure_table(document.as_table_mut(), template.as_table(), "beta", "user").unwrap();
        beta["added"] = value("mine");
    }
    let beta = ensure_table(document.as_table_mut(), template.as_table(), "beta", "user").unwrap();
    assert_eq!(beta["added"].as_str(), Some("mine"));
}
