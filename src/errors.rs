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

//! Comprehensive error handling for octolib operations

use thiserror::Error;

/// Errors that can occur during message operations
#[derive(Debug, Error)]
pub enum MessageError {
    #[error("Tool message missing required field: {field}")]
    MissingToolField { field: String },

    #[error("Invalid role: {role}")]
    InvalidRole { role: String },

    #[error("Missing required content for message")]
    MissingContent,

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(#[from] std::time::SystemTimeError),

    #[error("Image conversion failed: {0}")]
    ImageConversionError(String),

    #[error("Cache marker application failed: {reason}")]
    CacheMarkerError { reason: String },

    #[error("Tool calls deserialization failed: {0}")]
    ToolCallsError(#[from] serde_json::Error),
}

/// Errors that can occur during tool call operations
#[derive(Debug, Error)]
pub enum ToolCallError {
    #[error("Failed to deserialize tool calls: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("Tool call missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid tool call format for provider {provider}: {reason}")]
    InvalidFormat { provider: String, reason: String },

    #[error("No tool calls found in provider response")]
    NoToolCalls,

    #[error("Unsupported provider format: {provider}")]
    UnsupportedProvider { provider: String },

    #[error("Invalid JSON in tool call arguments: {0}")]
    InvalidArguments(serde_json::Error),
}

/// Errors that can occur during provider operations
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("Provider not found: {provider}")]
    ProviderNotFound { provider: String },

    #[error("Model not supported by provider {provider}: {model}")]
    ModelNotSupported { provider: String, model: String },

    #[error("API key not found for provider: {provider}")]
    ApiKeyNotFound { provider: String },

    #[error("Invalid API key for provider: {provider}")]
    InvalidApiKey { provider: String },

    #[error("Rate limit exceeded for provider: {provider}")]
    RateLimitExceeded { provider: String },

    #[error("Provider API error: {provider} - {status}: {message}")]
    ApiError {
        provider: String,
        status: u16,
        message: String,
    },

    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    #[error("Timeout error for provider: {provider}")]
    TimeoutError { provider: String },

    #[error("Message processing failed: {0}")]
    MessageError(#[from] MessageError),

    #[error("Tool call processing failed: {0}")]
    ToolCallError(#[from] ToolCallError),

    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Unsupported operation for provider {provider}: {operation}")]
    UnsupportedOperation { provider: String, operation: String },

    #[error("Request cancelled")]
    Cancelled,

    #[error("Response parsing failed: {0}")]
    ResponseParsingError(#[from] serde_json::Error),
}

/// Errors that can occur during configuration operations
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Invalid cache TTL value: {value}")]
    InvalidCacheTTL { value: String },

    #[error("Invalid duration format: {format}")]
    InvalidDurationFormat { format: String },

    #[error("Configuration validation failed: {field} - {reason}")]
    ValidationFailed { field: String, reason: String },

    #[error("Missing required configuration: {field}")]
    MissingRequired { field: String },

    #[error("Invalid configuration value for {field}: {value}")]
    InvalidValue { field: String, value: String },
}

/// Result type for provider operations
pub type ProviderResult<T> = Result<T, ProviderError>;

/// Result type for message operations
pub type MessageResult<T> = Result<T, MessageError>;

/// Result type for tool call operations
pub type ToolCallResult<T> = Result<T, ToolCallError>;

/// Result type for configuration operations
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Extension trait for adding context to errors
pub trait ErrorContext<T> {
    /// Add general context to an error
    fn with_context(self, context: &str) -> ProviderResult<T>;

    /// Add provider-specific context to an error
    fn with_provider_context(self, provider: &str) -> ProviderResult<T>;

    /// Add operation context to an error
    fn with_operation_context(self, operation: &str) -> ProviderResult<T>;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn with_context(self, context: &str) -> ProviderResult<T> {
        self.map_err(|e| ProviderError::ConfigurationError {
            message: format!("{}: {}", context, e),
        })
    }

    fn with_provider_context(self, provider: &str) -> ProviderResult<T> {
        self.map_err(|e| ProviderError::ApiError {
            provider: provider.to_string(),
            status: 0, // Unknown status
            message: e.to_string(),
        })
    }

    fn with_operation_context(self, operation: &str) -> ProviderResult<T> {
        self.map_err(|e| ProviderError::UnsupportedOperation {
            provider: "unknown".to_string(),
            operation: format!("{}: {}", operation, e),
        })
    }
}

/// Helper function to create API errors with status codes
pub fn api_error(provider: &str, status: u16, message: &str) -> ProviderError {
    ProviderError::ApiError {
        provider: provider.to_string(),
        status,
        message: message.to_string(),
    }
}

/// Helper function to create configuration errors
pub fn config_error(message: &str) -> ProviderError {
    ProviderError::ConfigurationError {
        message: message.to_string(),
    }
}

/// Helper function to create tool call errors
pub fn tool_call_error(provider: &str, reason: &str) -> ToolCallError {
    ToolCallError::InvalidFormat {
        provider: provider.to_string(),
        reason: reason.to_string(),
    }
}

#[cfg(test)]
mod tests {
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
    fn test_api_error_helper() {
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
}
