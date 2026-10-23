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

// Keys are unique to this file, so these tests need no serialization.
const SET_KEY: &str = "OCTOLIB_ENV_GUARD_TEST_SET";
const REMOVE_KEY: &str = "OCTOLIB_ENV_GUARD_TEST_REMOVE";
const PANIC_KEY: &str = "OCTOLIB_ENV_GUARD_TEST_PANIC";

#[test]
fn drop_restores_the_previous_value_after_repeated_sets() {
	std::env::set_var(SET_KEY, "original");
	{
		let _env = EnvGuard::new().set(SET_KEY, "first").set(SET_KEY, "second");
		assert_eq!(std::env::var(SET_KEY).as_deref(), Ok("second"));
	}
	assert_eq!(std::env::var(SET_KEY).as_deref(), Ok("original"));
	std::env::remove_var(SET_KEY);
}

#[test]
fn drop_restores_removed_and_unset_variables() {
	std::env::set_var(REMOVE_KEY, "kept");
	{
		let _env = EnvGuard::new().remove(REMOVE_KEY);
		assert!(std::env::var_os(REMOVE_KEY).is_none());
	}
	assert_eq!(std::env::var(REMOVE_KEY).as_deref(), Ok("kept"));

	std::env::remove_var(REMOVE_KEY);
	{
		let _env = EnvGuard::new().set(REMOVE_KEY, "temporary");
	}
	assert!(std::env::var_os(REMOVE_KEY).is_none());
}

#[test]
fn drop_restores_the_environment_when_the_test_panics() {
	let result = std::panic::catch_unwind(|| {
		let _env = EnvGuard::new().set(PANIC_KEY, "leaked?");
		panic!("assertion failed mid-test");
	});
	assert!(result.is_err());
	assert!(std::env::var_os(PANIC_KEY).is_none());
}
