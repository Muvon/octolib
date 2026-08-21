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

use super::types::{GenerationFailure, JobHandle, MediaTask};
use thiserror::Error;

pub type MediaResult<T> = Result<T, MediaError>;

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("invalid media model '{0}'; expected provider:model")]
    InvalidModelFormat(String),
    #[error("unsupported media provider: {0}")]
    UnsupportedProvider(String),
    #[error("provider {provider} does not support {task:?} for model {model}")]
    UnsupportedTask {
        provider: String,
        model: String,
        task: MediaTask,
    },
    #[error("unsupported parameter '{parameter}' for provider {provider}: {reason}")]
    UnsupportedParameter {
        provider: String,
        parameter: String,
        reason: String,
    },
    #[error("invalid media request: {0}")]
    InvalidRequest(String),
    #[error("API key not found in environment variable {0}")]
    MissingApiKey(String),
    #[error("authentication failed for provider {provider}: {message}")]
    Authentication { provider: String, message: String },
    #[error("permission denied by provider {provider}: {message}")]
    Permission { provider: String, message: String },
    #[error("insufficient credits at provider {provider}: {message}")]
    InsufficientCredits { provider: String, message: String },
    #[error("rate limited by provider {provider}: {message}")]
    RateLimit {
        provider: String,
        message: String,
        retry_after_secs: Option<u64>,
    },
    #[error("{provider} API error ({status}): {message}")]
    Api {
        provider: String,
        status: u16,
        message: String,
    },
    #[error("invalid {provider} response: {message}")]
    InvalidResponse { provider: String, message: String },
    #[error(
        "job handle does not belong to provider {expected_provider} and task {expected_task:?}"
    )]
    WrongJobHandle {
        expected_provider: String,
        expected_task: MediaTask,
    },
    #[error("remote media operation failed: {0:?}")]
    RemoteFailure(GenerationFailure),
    #[error("local wait timed out; the remote operation can be resumed")]
    WaitTimeout { handle: Box<JobHandle> },
    #[error("local wait was cancelled; the remote operation was not cancelled")]
    LocalWaitCancelled { handle: Box<JobHandle> },
    #[error("media source is too large: {actual} bytes exceeds {maximum} bytes")]
    SourceTooLarge { actual: u64, maximum: usize },
    #[error("artifact is too large: more than {maximum} bytes")]
    ArtifactTooLarge { maximum: usize },
    #[error("provider response is too large: more than {maximum} bytes")]
    ResponseTooLarge { maximum: usize },
    #[error("media I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("media transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("media JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid base64 media: {0}")]
    Base64(#[from] base64::DecodeError),
}
