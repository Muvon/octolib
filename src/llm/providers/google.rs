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

//! Google Vertex AI provider implementation
//!
//! Authentication: Uses service account JSON key file for authentication.
//! Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CREDENTIAL_FILE to the path of your service account JSON file.
//!
//! To create a service account:
//! 1. Go to Google Cloud Console → IAM & Admin → Service Accounts
//! 2. Create a service account with "Vertex AI User" role
//! 3. Create and download a JSON key file
//! 4. Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::normalize_model_name;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::env;

/// Google Vertex AI provider
#[derive(Debug, Clone)]
pub struct GoogleVertexProvider;

impl Default for GoogleVertexProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl GoogleVertexProvider {
    pub fn new() -> Self {
        Self
    }
}

const GOOGLE_CREDENTIAL_FILE_ENV: &str = "GOOGLE_CREDENTIAL_FILE";
const GOOGLE_APPLICATION_CREDENTIALS_ENV: &str = "GOOGLE_APPLICATION_CREDENTIALS";
const GOOGLE_CLOUD_PROJECT_ID_ENV: &str = "GOOGLE_CLOUD_PROJECT_ID";
const GOOGLE_CLOUD_LOCATION_ENV: &str = "GOOGLE_CLOUD_LOCATION";
const GOOGLE_API_URL_ENV: &str = "GOOGLE_API_URL";
const GOOGLE_VERTEX_API_URL_TEMPLATE: &str =
    "https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions";

fn default_vertex_api_url(project: &str, location: &str) -> String {
    GOOGLE_VERTEX_API_URL_TEMPLATE
        .replace("{project}", project)
        .replace("{location}", location)
}

#[derive(Debug, Deserialize)]
struct GoogleServiceAccountFile {
    project_id: Option<String>,
}

/// Resolve the path to the Google service account credentials file
fn resolve_credentials_file() -> Result<String> {
    // Try GOOGLE_CREDENTIAL_FILE first (our preferred env var)
    if let Ok(path) = env::var(GOOGLE_CREDENTIAL_FILE_ENV) {
        let path = path.trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    // Fall back to standard GOOGLE_APPLICATION_CREDENTIALS
    if let Ok(path) = env::var(GOOGLE_APPLICATION_CREDENTIALS_ENV) {
        let path = path.trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    Err(anyhow::anyhow!(
        "Google service account credentials file not found. Set {} (preferred) or {}. \
        Download a service account JSON key from Google Cloud Console → IAM & Admin → Service Accounts.",
        GOOGLE_CREDENTIAL_FILE_ENV,
        GOOGLE_APPLICATION_CREDENTIALS_ENV
    ))
}

/// Generate an access token from service account JSON file using JWT authentication
async fn generate_access_token(credentials_file: &str) -> Result<String> {
    // Read service account JSON file
    let client_json = std::fs::read_to_string(credentials_file).context(format!(
        "Failed to read service account file '{}'",
        credentials_file
    ))?;

    // Build AuthConfig with Cloud Platform scope
    let config = google_jwt_auth::AuthConfig::build(
        &client_json,
        &google_jwt_auth::usage::Usage::CloudPlatform,
    )
    .context("Failed to build auth config from service account JSON")?;

    // Generate token with 3600 seconds (1 hour) lifetime
    let token = config
        .generate_auth_token(3600)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to generate OAuth access token: {}", e))?;

    Ok(token)
}

/// Extract project ID from service account JSON file or environment variable
fn resolve_vertex_project_id(credentials_file: &str) -> Result<String> {
    // Try environment variable first
    if let Ok(project) = env::var(GOOGLE_CLOUD_PROJECT_ID_ENV) {
        let project = project.trim().to_string();
        if !project.is_empty() {
            return Ok(project);
        }
    }

    // Parse service account JSON to extract project_id
    let file_content = std::fs::read_to_string(credentials_file).context(format!(
        "Failed to read Google service account file '{}'",
        credentials_file
    ))?;

    let creds: GoogleServiceAccountFile = serde_json::from_str(&file_content).context(format!(
        "Failed to parse Google service account JSON file '{}'",
        credentials_file
    ))?;

    if let Some(project) = creds.project_id {
        let project = project.trim().to_string();
        if !project.is_empty() {
            return Ok(project);
        }
    }

    Err(anyhow::anyhow!(
        "Google Cloud project ID not found. Set {} or ensure 'project_id' field exists in service account file '{}'.",
        GOOGLE_CLOUD_PROJECT_ID_ENV,
        credentials_file
    ))
}

#[async_trait::async_trait]
impl AiProvider for GoogleVertexProvider {
    fn name(&self) -> &str {
        "google"
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.is_empty()
    }

    fn get_api_key(&self) -> Result<String> {
        // For Google Vertex AI, we just validate that credentials file exists
        // The actual token generation happens in chat_completion (async)
        resolve_credentials_file()?;
        Ok(String::new()) // Return empty string as placeholder
    }

    fn supports_caching(&self, _model: &str) -> bool {
        false
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Google Vertex AI vision (case-insensitive)
        normalize_model_name(model).contains("gemini")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Google Vertex AI model context window limits (case-insensitive)
        let normalized = normalize_model_name(model);
        if normalized.contains("gemini-3") {
            1_048_576 // Gemini 3.0 has ~1M context
        } else if normalized.contains("gemini-2") {
            2_000_000 // Gemini 2.0 has 2M context
        } else if normalized.contains("gemini-1.5") {
            1_000_000 // Gemini 1.5 has 1M context
        } else if normalized.contains("gemini-1.0") || normalized.contains("bison-32k") {
            32_768 // Gemini 1.0 and 32K variants have 32K context
        } else if normalized.contains("bison") {
            8_192 // Standard Bison models
        } else {
            32_768 // Conservative default
        }
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        // Generate access token from service account
        let credentials_file = resolve_credentials_file()?;
        let api_key = generate_access_token(&credentials_file).await?;

        let api_url = if let Ok(url) = env::var(GOOGLE_API_URL_ENV) {
            url
        } else {
            let project = resolve_vertex_project_id(&credentials_file)?;
            let location = env::var(GOOGLE_CLOUD_LOCATION_ENV)
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "us-central1".to_string());
            default_vertex_api_url(&project, &location)
        };

        openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "google",
                usage_fallback_cost: None,
                use_response_cost: true,
            },
            api_key,
            api_url,
            params,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_model() {
        let provider = GoogleVertexProvider::new();

        // Generic provider: accept any non-empty model identifier
        assert!(provider.supports_model("gemini-1.5-pro"));
        assert!(provider.supports_model("gemini-2.0-flash"));
        assert!(provider.supports_model("gemini-1.0-pro"));
        assert!(provider.supports_model("text-bison"));
        assert!(provider.supports_model("gpt-4"));
        assert!(provider.supports_model("claude-3"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_supports_model_case_insensitive() {
        let provider = GoogleVertexProvider::new();

        // Test uppercase
        assert!(provider.supports_model("GEMINI-1.5-PRO"));
        assert!(provider.supports_model("GEMINI-2.0-FLASH"));
        // Test mixed case
        assert!(provider.supports_model("Gemini-1.5-Pro"));
        assert!(provider.supports_model("GEMINI-1.0-pro"));
    }

    #[test]
    fn test_supports_caching_default_false() {
        let provider = GoogleVertexProvider::new();
        assert!(!provider.supports_caching("gemini-1.5-pro"));
        assert!(!provider.supports_caching("Gemini-2.0-Flash"));
    }
}
