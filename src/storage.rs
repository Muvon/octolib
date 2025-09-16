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

/// Get cache directory for FastEmbed models
pub fn get_fastembed_cache_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("octolib")
        .join("fastembed");

    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Get cache directory for HuggingFace models
pub fn get_huggingface_cache_dir() -> anyhow::Result<PathBuf> {
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
        .join("octolib")
        .join("huggingface");

    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}
