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

#[test]
fn test_error_context() {
    let result: Result<(), std::io::Error> = Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "file not found",
    ));

    let with_context = result.with_context("Failed to read config");
    assert!(with_context.is_err());

    if let Err(ProviderError::ConfigurationError { message }) = with_context {
        assert!(message.contains("Failed to read config"));
        assert!(message.contains("file not found"));
    } else {
        panic!("Expected ConfigurationError");
    }
}

#[test]
fn test_provider_context() {
    let result: Result<(), std::io::Error> = Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "connection timeout",
    ));

    let with_context = result.with_provider_context("openai");
    assert!(with_context.is_err());

    if let Err(ProviderError::ApiError {
        provider, message, ..
    }) = with_context
    {
        assert_eq!(provider, "openai");
        assert!(message.contains("connection timeout"));
    } else {
        panic!("Expected ApiError");
    }
}

#[test]
fn test_api_error() {
    let error = api_error("anthropic", 400, "Bad Request");

    if let ProviderError::ApiError {
        provider,
        status,
        message,
    } = error
    {
        assert_eq!(provider, "anthropic");
        assert_eq!(status, 400);
        assert_eq!(message, "Bad Request");
    } else {
        panic!("Expected ApiError");
    }
}
