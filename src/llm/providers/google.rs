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
//!
//! Model discovery: Available models are lazy-loaded from the Vertex AI API on first
//! chat_completion() call. The list is cached for the lifetime of the process.

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ProviderResponse};
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::{Context, Result};
use jsonwebtoken::{Algorithm, EncodingKey, Header};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::OnceCell;

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

/// Google Vertex AI / Gemini API pricing (per 1M tokens in USD)
/// Source: https://cloud.google.com/vertex-ai/generative-ai/pricing (verified Apr 6, 2026)
/// Using ≤200K context tier prices. Format: (model, input, output, cache_write, cache_read)
const PRICING: &[PricingTuple] = &[
    // Gemini 3.x series
    ("gemini-3.1-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3.1-flash", 0.50, 3.00, 0.50, 0.05),
    ("gemini-3.1-flash-lite", 0.25, 1.50, 0.25, 0.025),
    ("gemini-3-pro", 2.00, 12.00, 2.00, 0.20),
    ("gemini-3-flash", 0.50, 3.00, 0.50, 0.05),
    // Gemini 2.5 series
    ("gemini-2.5-flash-lite", 0.10, 0.40, 0.10, 0.01),
    ("gemini-2.5-flash", 0.30, 2.50, 0.30, 0.03),
    ("gemini-2.5-pro", 1.25, 10.00, 1.25, 0.125),
    // Gemini 2.0 series
    ("gemini-2.0-flash", 0.10, 0.40, 0.10, 0.025),
];

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

// --- Lazy model discovery ---

/// Cached model from the API
#[derive(Debug, Clone)]
struct CachedModel {
    id: String,
    input_token_limit: Option<usize>,
}

/// Process-wide cache of available models, populated on first chat_completion()
static MODELS_CACHE: OnceCell<Vec<CachedModel>> = OnceCell::const_new();

/// OpenAI-compat /models response
#[derive(Deserialize)]
struct ModelsListResponse {
    #[serde(default)]
    data: Vec<ApiModelEntry>,
}

#[derive(Deserialize)]
struct ApiModelEntry {
    id: String,
    #[serde(default)]
    input_token_limit: Option<usize>,
}

/// Fetch available models from the OpenAI-compat /models endpoint.
/// Derives the URL from the chat completions URL by replacing the path suffix.
async fn fetch_available_models(access_token: &str, chat_url: &str) -> Result<Vec<CachedModel>> {
    let models_url = chat_url.replace("/chat/completions", "/models");

    let response = super::shared::http_client()
        .get(&models_url)
        .header("Authorization", format!("Bearer {}", access_token))
        .send()
        .await
        .context("Failed to fetch models list from Google API")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(anyhow::anyhow!(
            "Google models API error {}: {}",
            status,
            text
        ));
    }

    let list: ModelsListResponse = response
        .json()
        .await
        .context("Failed to parse models list response")?;

    Ok(list
        .data
        .into_iter()
        .map(|m| CachedModel {
            id: m.id,
            input_token_limit: m.input_token_limit,
        })
        .collect())
}

/// Check if a model exists in the cached model list (case-insensitive)
fn is_model_cached(model: &str) -> Option<bool> {
    let models = MODELS_CACHE.get()?;
    let normalized = normalize_model_name(model);
    Some(
        models
            .iter()
            .any(|m| normalize_model_name(&m.id) == normalized),
    )
}

/// Get cached input token limit for a model
fn get_cached_input_limit(model: &str) -> Option<usize> {
    let models = MODELS_CACHE.get()?;
    let normalized = normalize_model_name(model);
    models
        .iter()
        .find(|m| normalize_model_name(&m.id) == normalized)
        .and_then(|m| m.input_token_limit)
}

// --- Auth ---

#[derive(Debug, Deserialize)]
struct GoogleServiceAccountFile {
    project_id: Option<String>,
    client_email: String,
    private_key: String,
    private_key_id: String,
}

#[derive(Serialize)]
struct JwtClaims {
    iss: String,
    sub: String,
    aud: String,
    scope: String,
    iat: u64,
    exp: u64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
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
    let client_json = std::fs::read_to_string(credentials_file).context(format!(
        "Failed to read service account file '{}'",
        credentials_file
    ))?;

    let creds: GoogleServiceAccountFile =
        serde_json::from_str(&client_json).context("Failed to parse service account JSON")?;

    let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let claims = JwtClaims {
        iss: creds.client_email.clone(),
        sub: creds.client_email,
        aud: "https://oauth2.googleapis.com/token".to_string(),
        scope: "https://www.googleapis.com/auth/cloud-platform".to_string(),
        iat: now,
        exp: now + 3600,
    };

    let mut header = Header::new(Algorithm::RS256);
    header.kid = Some(creds.private_key_id);
    let key = EncodingKey::from_rsa_pem(creds.private_key.as_bytes())
        .context("Failed to parse RSA private key from service account")?;
    let jwt = jsonwebtoken::encode(&header, &claims, &key).context("Failed to sign JWT")?;

    // Exchange JWT for OAuth2 access token using octolib's shared HTTP client
    let body = format!(
        "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer&assertion={}",
        jwt
    );
    let resp: TokenResponse = super::shared::http_client()
        .post("https://oauth2.googleapis.com/token")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .context("Failed to send token request to Google OAuth2 endpoint")?
        .json()
        .await
        .context("Failed to parse token response from Google OAuth2 endpoint")?;

    Ok(resp.access_token)
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
        if model.is_empty() {
            return false;
        }
        // Use cached model list if available (populated on first chat_completion)
        is_model_cached(model).unwrap_or(true)
    }

    fn get_api_key(&self) -> Result<String> {
        // For Google Vertex AI, we just validate that credentials file exists
        // The actual token generation happens in chat_completion (async)
        resolve_credentials_file()?;
        Ok(String::new()) // Return empty string as placeholder
    }

    fn supports_caching(&self, model: &str) -> bool {
        let normalized = normalize_model_name(model);
        normalized.contains("gemini-3") || normalized.contains("gemini-2.5")
    }

    fn supports_vision(&self, model: &str) -> bool {
        // Google Vertex AI vision (case-insensitive)
        normalize_model_name(model).contains("gemini")
    }

    fn supports_structured_output(&self, _model: &str) -> bool {
        true
    }

    fn get_model_pricing(&self, model: &str) -> Option<crate::llm::types::ModelPricing> {
        let (input_price, output_price, cache_write_price, cache_read_price) =
            get_model_pricing(model, PRICING)?;
        Some(crate::llm::types::ModelPricing::new(
            input_price,
            output_price,
            cache_write_price,
            cache_read_price,
        ))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        // Prefer cached value from API if available
        if let Some(limit) = get_cached_input_limit(model) {
            return limit;
        }
        // Fallback to hardcoded limits
        let normalized = normalize_model_name(model);
        if normalized.contains("gemini-3") || normalized.contains("gemini-2") {
            1_048_576 // Gemini 2.x/3.x has ~1M context
        } else if normalized.contains("gemini-1.5") {
            1_000_000 // Gemini 1.5 has 1M context
        } else if normalized.contains("gemini-1.0") || normalized.contains("bison-32k") {
            32_768
        } else if normalized.contains("bison") {
            8_192
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

        // Lazy-load available models on first call (errors silently ignored; retries next call)
        let token = api_key.clone();
        let url = api_url.clone();
        let _ = MODELS_CACHE
            .get_or_try_init(|| async move { fetch_available_models(&token, &url).await })
            .await;

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
    fn test_supports_model_before_cache() {
        let provider = GoogleVertexProvider::new();

        // Before cache is populated, accept any non-empty model
        assert!(provider.supports_model("gemini-1.5-pro"));
        assert!(provider.supports_model("gemini-2.0-flash"));
        assert!(provider.supports_model("anything-goes"));
        assert!(!provider.supports_model(""));
    }

    #[test]
    fn test_supports_caching() {
        let provider = GoogleVertexProvider::new();
        assert!(provider.supports_caching("gemini-3-flash"));
        assert!(provider.supports_caching("gemini-2.5-pro"));
        assert!(provider.supports_caching("gemini-2.5-flash"));
        assert!(!provider.supports_caching("gemini-2.0-flash"));
        assert!(!provider.supports_caching("gemini-1.5-pro"));
    }

    #[test]
    fn test_model_pricing() {
        let provider = GoogleVertexProvider::new();

        let p = provider.get_model_pricing("gemini-3.1-pro").unwrap();
        assert_eq!(p.input_price_per_1m, 2.00);
        assert_eq!(p.output_price_per_1m, 12.00);

        let p = provider.get_model_pricing("gemini-2.5-flash").unwrap();
        assert_eq!(p.input_price_per_1m, 0.30);
        assert_eq!(p.output_price_per_1m, 2.50);

        // Unknown models return None (no fallback to zero)
        assert!(provider.get_model_pricing("gemma-3-27b").is_none());
    }

    #[test]
    fn test_max_input_tokens_fallback() {
        let provider = GoogleVertexProvider::new();
        assert_eq!(provider.get_max_input_tokens("gemini-3-flash"), 1_048_576);
        assert_eq!(provider.get_max_input_tokens("gemini-2.5-pro"), 1_048_576);
        assert_eq!(provider.get_max_input_tokens("gemini-1.5-pro"), 1_000_000);
        assert_eq!(provider.get_max_input_tokens("text-bison"), 8_192);
    }
}
