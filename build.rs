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

use std::env;

fn main() {
	println!("cargo:rerun-if-changed=build.rs");

	// The prebuilt Apple ONNX Runtime ships the CoreML EP, whose `@available`
	// checks call `___isPlatformVersionAtLeast` from clang's compiler-rt.
	// rustc links with `-nodefaultlibs`, so that runtime is dropped unless
	// asked for explicitly. Applies to this package's tests and examples only;
	// downstream binaries need the same flag in their own build script.
	let apple = env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|v| v == "apple");
	if apple && env::var_os("CARGO_FEATURE_ONNX").is_some() {
		println!("cargo:rustc-link-arg=-fapple-link-rtlib");
	}
}
