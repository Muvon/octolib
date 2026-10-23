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

//! Scoped environment overrides for tests.
//!
//! Tests point providers at a mock through `<PROVIDER>_API_URL` and fake keys.
//! Clearing those by hand at the end of a test leaks them whenever an assertion
//! fails first, and every later test then talks to a dead mock. [`EnvGuard`]
//! restores each variable it touched when it drops, pass or fail.
//!
//! The guard only restores state; it does not stop a concurrent test from
//! reading it. Every test that touches a shared variable still needs
//! `#[serial_test::serial]`.

use std::ffi::{OsStr, OsString};

/// Environment changes undone, newest first, when the guard drops.
#[derive(Default)]
#[must_use = "dropping the guard restores the environment immediately"]
pub(crate) struct EnvGuard {
	saved: Vec<(OsString, Option<OsString>)>,
}

impl EnvGuard {
	pub(crate) fn new() -> Self {
		Self::default()
	}

	/// Set `key` to `value` until the guard drops.
	pub(crate) fn set(mut self, key: impl AsRef<OsStr>, value: impl AsRef<OsStr>) -> Self {
		let key = key.as_ref();
		self.saved.push((key.to_os_string(), std::env::var_os(key)));
		std::env::set_var(key, value);
		self
	}

	/// Unset `key` until the guard drops.
	pub(crate) fn remove(mut self, key: impl AsRef<OsStr>) -> Self {
		let key = key.as_ref();
		self.saved.push((key.to_os_string(), std::env::var_os(key)));
		std::env::remove_var(key);
		self
	}
}

impl Drop for EnvGuard {
	fn drop(&mut self) {
		for (key, previous) in self.saved.drain(..).rev() {
			match previous {
				Some(value) => std::env::set_var(&key, value),
				None => std::env::remove_var(&key),
			}
		}
	}
}

#[cfg(test)]
#[path = "env_tests.rs"]
mod tests;
