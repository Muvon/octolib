// Copyright 2025 Muvon Un Limited
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

//! Storage utilities for embedding models

use std::path::PathBuf;

/// The one shared on-disk cache for EVERY locally downloaded model.
///
/// fastembed and the hf-hub (huggingface) providers download the same models
/// in the same hf-hub layout (`models--org--name/{blobs,snapshots,refs}`), so
/// per-provider cache dirs just stored every model twice (~3.7GB doubled on
/// one machine — enough to fill a 5GB working disk and crash anything that
/// writes locally with ENOSPC). Every provider resolves here.
///
/// The directory keeps the historical `huggingface` name on purpose: renaming
/// it would orphan gigabytes of already-downloaded weights on existing
/// machines and force a re-download.
pub fn get_model_cache_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("octolib")
        .join("huggingface");

    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

#[deprecated(note = "all model caches share one dir — use get_model_cache_dir()")]
pub fn get_fastembed_cache_dir() -> anyhow::Result<PathBuf> {
    get_model_cache_dir()
}

#[deprecated(note = "all model caches share one dir — use get_model_cache_dir()")]
pub fn get_huggingface_cache_dir() -> anyhow::Result<PathBuf> {
    get_model_cache_dir()
}
