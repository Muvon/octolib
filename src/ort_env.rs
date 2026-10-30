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

//! The one ONNX Runtime environment for the whole process.
//!
//! ort 2.0.0-rc.15 refuses to build a second `Environment` in a process, and
//! fastembed builds its own whenever an init option carries none. Every ORT
//! session in this crate (fastembed embeddings, fastembed reranking, the user
//! defined ONNX provider, hindsight) therefore takes its environment from
//! here, so any number of models can coexist.

use anyhow::{anyhow, Result};
use ort::environment::Environment;
use std::sync::LazyLock;

// ort's error carries a non-`Send` payload, so only the message is kept.
static ENVIRONMENT: LazyLock<std::result::Result<Environment, String>> =
	LazyLock::new(|| ort::init().with_name("octolib").build().map_err(|e| e.to_string()));

/// The shared environment. Cloning is an `Arc` bump. Fails only if ONNX
/// Runtime could not be initialised, or if the host process already built
/// an environment of its own before this crate did.
pub(crate) fn environment() -> Result<Environment> {
	ENVIRONMENT
		.as_ref()
		.cloned()
		.map_err(|e| anyhow!("onnxruntime environment: {e}"))
}

/// CoreML's default writes a fresh temporary model per session. Immutable cache
/// identities let independent processes reuse the compiled graph after restart.
pub(crate) fn execution_providers(identity: &str) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
	#[cfg(target_os = "macos")]
	{
		use sha2::{Digest, Sha256};
		let key = hex::encode(Sha256::digest(format!(
			"coreml-v1:{}:{}:{identity}",
			ort::info(),
			std::env::consts::ARCH
		)));
		let root = crate::storage::get_model_cache_dir()?.with_file_name("coreml");
		let path = root.join(key);
		std::fs::create_dir_all(&path)?;
		let path = path.to_str().ok_or_else(|| anyhow!("CoreML cache path is not UTF-8"))?;
		Ok(vec![ort::ep::CoreML::default()
			.with_model_cache_dir(path)
			.build()
			.error_on_failure()])
	}
	#[cfg(not(target_os = "macos"))]
	{
		let _ = identity;
		Ok(Vec::new())
	}
}

pub(crate) fn file_execution_providers(path: &std::path::Path) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
	#[cfg(target_os = "macos")]
	{
		use sha2::{Digest, Sha256};
		use std::io::Read;
		let mut file = std::fs::File::open(path)?;
		let mut hash = Sha256::new();
		let mut buffer = [0u8; 64 * 1024];
		loop {
			let count = file.read(&mut buffer)?;
			if count == 0 {
				break;
			}
			hash.update(&buffer[..count]);
		}
		execution_providers(&hex::encode(hash.finalize()))
	}
	#[cfg(not(target_os = "macos"))]
	{
		let _ = path;
		execution_providers("")
	}
}
