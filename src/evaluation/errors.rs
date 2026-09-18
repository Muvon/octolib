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

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("invalid evaluation model '{0}'; expected provider:model")]
    InvalidModelFormat(String),
    #[error("unsupported evaluation provider: {0}")]
    UnsupportedProvider(String),
    #[error("provider {provider} does not serve evaluation model {model}")]
    UnsupportedModel { provider: String, model: String },
    #[error("invalid evaluation request: {0}")]
    InvalidRequest(String),
    #[error("API key not found in environment variable {0}")]
    MissingApiKey(String),
    #[error("authentication failed for provider {provider}: {message}")]
    Authentication { provider: String, message: String },
    #[error("permission denied by provider {provider}: {message}")]
    Permission { provider: String, message: String },
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
    #[error("evaluation transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("evaluation JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type EvaluationResult<T> = Result<T, EvaluationError>;
