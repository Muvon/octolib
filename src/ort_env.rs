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
/// The GPU unit runs graphs through MPSGraph, which leaks its context and
/// prints "Context leak detected" straight to stderr; the Neural Engine and
/// CPU keep the CoreML speedup without that path.
///
/// CoreML takes only the nodes whose input shapes are fixed. A model it compiles
/// for a variable batch or sequence length fails on any batch above one
/// ("Unable to compute the prediction using a neural network model") and runs
/// about three times slower than ONNX Runtime's own CPU kernels at batch one,
/// so text encoders stay on the CPU and fixed-shape graphs still go to CoreML.
#[cfg(any(feature = "fastembed", feature = "hindsight"))]
pub(crate) fn execution_providers(identity: &str) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
	#[cfg(target_os = "macos")]
	{
		let key = cache_key(identity, ort::info(), std::env::consts::ARCH);
		let root = crate::storage::get_model_cache_dir()?.with_file_name("coreml");
		let path = root.join(key);
		std::fs::create_dir_all(&path)?;
		let path = path.to_str().ok_or_else(|| anyhow!("CoreML cache path is not UTF-8"))?;
		Ok(vec![ort::ep::CoreML::default()
			.with_compute_units(ort::ep::coreml::ComputeUnits::CPUAndNeuralEngine)
			.with_static_input_shapes(true)
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

#[cfg(feature = "hindsight")]
pub(crate) fn file_execution_providers(path: &std::path::Path) -> Result<Vec<ort::ep::ExecutionProviderDispatch>> {
	#[cfg(target_os = "macos")]
	{
		execution_providers(&model_hash(std::fs::File::open(path)?)?)
	}
	#[cfg(not(target_os = "macos"))]
	{
		let _ = path;
		execution_providers("")
	}
}

#[cfg(any(all(target_os = "macos", any(feature = "fastembed", feature = "hindsight")), test))]
fn cache_key(identity: &str, runtime: &str, arch: &str) -> String {
	use sha2::{Digest, Sha256};
	// v2: compiled models depend on the compute units, which v1 left at ALL.
	// v3: static-shape-only partitioning changes which nodes CoreML compiles.
	hex::encode(Sha256::digest(format!("coreml-v3:{runtime}:{arch}:{identity}")))
}

#[cfg(any(all(target_os = "macos", feature = "hindsight"), test))]
fn model_hash(mut reader: impl std::io::Read) -> Result<String> {
	use sha2::{Digest, Sha256};
	let mut hash = Sha256::new();
	let mut buffer = [0u8; 64 * 1024];
	loop {
		let count = reader.read(&mut buffer)?;
		if count == 0 {
			break;
		}
		hash.update(&buffer[..count]);
	}
	Ok(hex::encode(hash.finalize()))
}

#[cfg(test)]
#[path = "ort_env_tests.rs"]
mod tests;
