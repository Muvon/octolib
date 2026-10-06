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

//! Sign in with ChatGPT for open-source, locally run apps.
//!
//! OAuth 2.0 authorization code + PKCE against a loopback redirect. There is no
//! client secret: the first sign-in registers the app dynamically and the
//! callback returns its issued `client_id`, which later sign-ins and token
//! refreshes reuse. Each host also carries a stable `ext_agent_host_id`, chosen
//! before its first sign-in.
//! See <https://developers.openai.com/siwc/token-sharing-open-source/sign-in>

use crate::llm::providers::shared;
use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
const MODELS_URL: &str = "https://api.openai.com/v1/models";
const RESOURCE: &str = "https://api.openai.com/v1";
const SCOPE: &str = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
/// The granted scope that bills requests to the user's ChatGPT plan.
const PLAN_USAGE_SCOPE: &str = "chatgpt.tokens.use.direct";
/// First-sign-in registration entrypoint, never a client id to keep.
const REGISTRATION_CLIENT_ID: &str = "dynamic_agent_client";
const CALLBACK_PATH: &str = "/callback";
/// Upper bound on the callback request line; the real one is a few hundred bytes.
const MAX_REQUEST_LINE_BYTES: usize = 8192;
/// Refresh this early so a request and its retries never start on a dying token.
const REFRESH_MARGIN_SECS: u64 = 300;
const AUTH_FILE: &str = "auth.json";
const HOST_ID_FILE: &str = "host-id";

const CALLBACK_OK_PAGE: &str = "<!doctype html><title>ChatGPT sign-in</title>\
    <p>Access approved. Return to the terminal.</p>";
const CALLBACK_FAILED_PAGE: &str = "<!doctype html><title>ChatGPT sign-in</title>\
    <p>Sign-in failed. See the terminal for details.</p>";

/// The ChatGPT account a host is signed in as.
#[derive(Debug, Clone)]
pub struct Account {
    /// Absent when the ID token carries no email claim.
    pub email: Option<String>,
}

/// A model the signed-in account can use on its plan.
#[derive(Debug, Clone)]
pub struct ChatGptModel {
    /// Model id for requests: `chatgpt:<slug>`.
    pub slug: String,
    pub display_name: String,
}

/// A signed-in registration: the issued client and its tokens.
#[derive(Serialize, Deserialize)]
struct Registration {
    client_id: String,
    subject: String,
    email: Option<String>,
    id_token: String,
    access_token: String,
    refresh_token: String,
    /// Access-token expiry, unix seconds.
    expires_at: u64,
    scope: String,
}

impl Registration {
    fn expires_soon(&self, now: u64) -> bool {
        self.expires_at <= now + REFRESH_MARGIN_SECS
    }
}

#[derive(Deserialize)]
struct CodeExchangeResponse {
    access_token: String,
    refresh_token: String,
    id_token: String,
    expires_in: u64,
    scope: String,
}

#[derive(Deserialize)]
struct RefreshResponse {
    access_token: String,
    refresh_token: String,
    expires_in: u64,
}

#[derive(Deserialize)]
struct IdTokenClaims {
    sub: String,
    nonce: String,
    email: Option<String>,
}

/// A sign-in waiting for the user to approve access in the browser.
pub struct PendingLogin {
    listener: TcpListener,
    authorize_url: String,
    redirect_uri: String,
    state: String,
    nonce: String,
    code_verifier: String,
    /// Issued client id of an earlier registration; `None` on first sign-in.
    saved_client_id: Option<String>,
    auth_path: PathBuf,
}

/// The account this host is signed in as, or `None` before the first sign-in.
pub fn current_account() -> Result<Option<Account>> {
    Ok(
        load_registration(&auth_dir()?.join(AUTH_FILE))?.map(|registration| Account {
            email: registration.email,
        }),
    )
}

/// Begin Sign in with ChatGPT. `app_name` is suggested as the agent name on the
/// consent page. Open [`PendingLogin::authorize_url`] in a browser, then await
/// [`PendingLogin::finish`].
pub async fn start_login(app_name: &str) -> Result<PendingLogin> {
    let dir = auth_dir()?;
    let host_id = host_id(&dir)?;
    let auth_path = dir.join(AUTH_FILE);
    let saved_client_id = load_registration(&auth_path)?.map(|registration| registration.client_id);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .context("Could not open the sign-in callback listener")?;
    let redirect_uri = format!(
        "http://127.0.0.1:{}{}",
        listener.local_addr()?.port(),
        CALLBACK_PATH
    );
    let state = random_token();
    let nonce = random_token();
    let code_verifier = format!("{}{}", random_token(), random_token());
    let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(code_verifier.as_bytes()));

    let authorize_url = reqwest::Url::parse_with_params(
        AUTHORIZE_URL,
        [
            (
                "client_id",
                saved_client_id.as_deref().unwrap_or(REGISTRATION_CLIENT_ID),
            ),
            ("agent_name_hint", app_name),
            ("ext_agent_host_id", host_id.as_str()),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri.as_str()),
            ("scope", SCOPE),
            ("resource", RESOURCE),
            ("state", state.as_str()),
            ("nonce", nonce.as_str()),
            ("code_challenge_method", "S256"),
            ("code_challenge", code_challenge.as_str()),
        ],
    )?
    .to_string();

    Ok(PendingLogin {
        listener,
        authorize_url,
        redirect_uri,
        state,
        nonce,
        code_verifier,
        saved_client_id,
        auth_path,
    })
}

impl PendingLogin {
    pub fn authorize_url(&self) -> &str {
        &self.authorize_url
    }

    /// Wait for the browser callback, exchange the code, and persist the session.
    pub async fn finish(self) -> Result<Account> {
        let callback = wait_for_callback(&self.listener).await?;
        if let Some(error) = callback.get("error") {
            anyhow::bail!(
                "ChatGPT sign-in was not approved: {} {}",
                error,
                callback
                    .get("error_description")
                    .map(String::as_str)
                    .unwrap_or_default()
            );
        }
        anyhow::ensure!(
            callback.get("state") == Some(&self.state),
            "ChatGPT sign-in callback state does not match; start the sign-in again"
        );
        let code = callback
            .get("code")
            .context("ChatGPT sign-in callback has no authorization code")?;
        // A first sign-in registers the app; the callback carries the issued id.
        let client_id = match callback.get("client_id") {
            Some(issued) => issued.clone(),
            None => self
                .saved_client_id
                .context("ChatGPT sign-in callback has no client id")?,
        };

        let tokens: CodeExchangeResponse = token_request(&[
            ("grant_type", "authorization_code"),
            ("client_id", client_id.as_str()),
            ("code", code.as_str()),
            ("code_verifier", self.code_verifier.as_str()),
            ("redirect_uri", self.redirect_uri.as_str()),
            ("resource", RESOURCE),
        ])
        .await?;
        anyhow::ensure!(
            tokens
                .scope
                .split_whitespace()
                .any(|scope| scope == PLAN_USAGE_SCOPE),
            "ChatGPT plan usage was not granted (granted scopes: {})",
            tokens.scope
        );
        let claims = id_token_claims(&tokens.id_token)?;
        anyhow::ensure!(
            claims.nonce == self.nonce,
            "ChatGPT ID token nonce does not match; start the sign-in again"
        );

        let registration = Registration {
            client_id,
            subject: claims.sub,
            email: claims.email,
            id_token: tokens.id_token,
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token,
            expires_at: unix_now()? + tokens.expires_in,
            scope: tokens.scope,
        };
        save_registration(&self.auth_path, &registration)?;

        Ok(Account {
            email: registration.email,
        })
    }
}

/// Models the signed-in account can use on its plan.
pub async fn list_models() -> Result<Vec<ChatGptModel>> {
    #[derive(Deserialize)]
    struct Catalog {
        models: Vec<CatalogModel>,
    }

    #[derive(Deserialize)]
    struct CatalogModel {
        slug: String,
        display_name: String,
        visibility: String,
    }

    let response = shared::http_client()
        .get(MODELS_URL)
        .bearer_auth(access_token().await?)
        .send()
        .await
        .context("ChatGPT model catalog request failed")?;
    let catalog: Catalog = read_json(response, "ChatGPT model catalog").await?;

    Ok(catalog
        .models
        .into_iter()
        .filter(|model| model.visibility == "list")
        .map(|model| ChatGptModel {
            slug: model.slug,
            display_name: model.display_name,
        })
        .collect())
}

/// A valid access token for plan-usage requests, refreshed near expiry.
pub(super) async fn access_token() -> Result<String> {
    let path = auth_dir()?.join(AUTH_FILE);
    let registration = require_registration(&path)?;
    if !registration.expires_soon(unix_now()?) {
        return Ok(registration.access_token);
    }

    let _lock = lock_refresh(&path).await?;
    // Another process may have refreshed while this one waited for the lock.
    let mut registration = require_registration(&path)?;
    if registration.expires_soon(unix_now()?) {
        let tokens: RefreshResponse = token_request(&[
            ("grant_type", "refresh_token"),
            ("client_id", registration.client_id.as_str()),
            ("refresh_token", registration.refresh_token.as_str()),
            ("resource", RESOURCE),
        ])
        .await
        .context("Refreshing the ChatGPT session failed; sign in again")?;
        registration.access_token = tokens.access_token;
        registration.refresh_token = tokens.refresh_token;
        registration.expires_at = unix_now()? + tokens.expires_in;
        save_registration(&path, &registration)?;
    }

    Ok(registration.access_token)
}

/// The stored access token without refreshing it; proves a sign-in exists.
pub(super) fn stored_access_token() -> Result<String> {
    Ok(require_registration(&auth_dir()?.join(AUTH_FILE))?.access_token)
}

fn auth_dir() -> Result<PathBuf> {
    Ok(dirs::config_dir()
        .context("Could not determine config directory")?
        .join("octolib")
        .join("chatgpt"))
}

fn unix_now() -> Result<u64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())
}

fn random_token() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// This host's `ext_agent_host_id`, created and persisted before its first sign-in.
fn host_id(dir: &Path) -> Result<String> {
    let path = dir.join(HOST_ID_FILE);
    match fs::read_to_string(&path) {
        Ok(id) => Ok(id.trim().to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let id = format!("urn:uuid:{}", uuid::Uuid::new_v4());
            write_private(&path, id.as_bytes())?;
            Ok(id)
        }
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn load_registration(path: &Path) -> Result<Option<Registration>> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map(Some)
            .with_context(|| format!("invalid ChatGPT credentials at {}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

fn require_registration(path: &Path) -> Result<Registration> {
    load_registration(path)?.with_context(|| {
        format!(
            "Not signed in with ChatGPT (no credentials at {})",
            path.display()
        )
    })
}

fn save_registration(path: &Path, registration: &Registration) -> Result<()> {
    write_private(path, &serde_json::to_vec_pretty(registration)?)
}

/// Atomically write `content` readable by the owner only. The directory is
/// owner-only too, so the temp file staged before the rename never leaks.
fn write_private(path: &Path, content: &[u8]) -> Result<()> {
    let dir = crate::utils::config_file::parent_directory(path)?;
    fs::create_dir_all(dir)?;

    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        Some(fs::Permissions::from_mode(0o600))
    };
    #[cfg(not(unix))]
    let permissions = None;

    crate::utils::config_file::atomic_write(path, content, permissions)
}

/// Exclusive cross-process lock serializing token refreshes. Refresh tokens
/// rotate, so two processes refreshing the same one would race and one would
/// keep a revoked token. Released when the returned file is dropped.
async fn lock_refresh(path: &Path) -> Result<fs::File> {
    let lock_path = path.with_file_name(format!(".{AUTH_FILE}.lock"));
    tokio::task::spawn_blocking(move || -> Result<fs::File> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .with_context(|| format!("failed to open {}", lock_path.display()))?;
        fs4::FileExt::lock_exclusive(&file)
            .with_context(|| format!("failed to lock {}", lock_path.display()))?;
        Ok(file)
    })
    .await?
}

async fn token_request<T: DeserializeOwned>(form: &[(&str, &str)]) -> Result<T> {
    let response = shared::http_client()
        .post(TOKEN_URL)
        .form(form)
        .send()
        .await
        .context("ChatGPT token request failed")?;
    read_json(response, "ChatGPT token endpoint").await
}

async fn read_json<T: DeserializeOwned>(response: reqwest::Response, source: &str) -> Result<T> {
    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        anyhow::bail!("{} error {}: {}", source, status, body);
    }
    // The body may hold tokens, so it stays out of the parse error.
    serde_json::from_str(&body)
        .with_context(|| format!("{} returned an unexpected response", source))
}

/// Claims of the ID token. Its signature is not checked: the token comes
/// straight from the token endpoint over TLS, which OIDC Core §3.1.3.7 accepts
/// in place of signature validation.
fn id_token_claims(id_token: &str) -> Result<IdTokenClaims> {
    let payload = id_token
        .split('.')
        .nth(1)
        .context("ChatGPT ID token is not a JWT")?;
    let bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .context("ChatGPT ID token payload is not base64url")?;
    serde_json::from_slice(&bytes).context("ChatGPT ID token payload is malformed")
}

/// Serve the loopback redirect until the browser lands on the callback path,
/// and return the callback's query parameters.
async fn wait_for_callback(listener: &TcpListener) -> Result<HashMap<String, String>> {
    // Browsers open speculative connections that may never send a request;
    // reading each connection on its own task keeps an idle one from blocking
    // the real callback. Dropping the set aborts the stragglers.
    let mut connections = tokio::task::JoinSet::new();

    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let (mut stream, _) = accepted.context("Sign-in callback listener failed")?;
                connections.spawn(async move {
                    let url = read_request(&mut stream).await?;
                    Ok::<_, anyhow::Error>((url, stream))
                });
            }
            Some(joined) = connections.join_next() => {
                let (url, mut stream) = match joined? {
                    Ok(request) => request,
                    Err(error) => {
                        tracing::debug!("ignoring sign-in callback connection: {}", error);
                        continue;
                    }
                };

                if url.path() != CALLBACK_PATH {
                    let not_found = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    if let Err(error) = stream.write_all(not_found).await {
                        tracing::debug!("sign-in callback 404 not delivered: {}", error);
                    }
                    continue;
                }

                let params: HashMap<String, String> = url.query_pairs().into_owned().collect();
                let page = if params.contains_key("error") {
                    CALLBACK_FAILED_PAGE
                } else {
                    CALLBACK_OK_PAGE
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    page.len(),
                    page
                );
                // The parameters are in hand either way; a browser that hung up
                // early only misses the confirmation page.
                if let Err(error) = stream.write_all(response.as_bytes()).await {
                    tracing::debug!("sign-in callback page not delivered: {}", error);
                }
                return Ok(params);
            }
        }
    }
}

/// Read the request line (`GET /callback?code=… HTTP/1.1`) and return its target as a URL.
async fn read_request(stream: &mut TcpStream) -> Result<reqwest::Url> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    let line_end = loop {
        if let Some(end) = buffer.windows(2).position(|pair| pair == b"\r\n") {
            break end;
        }
        anyhow::ensure!(
            buffer.len() <= MAX_REQUEST_LINE_BYTES,
            "request line too long"
        );
        let read = stream.read(&mut chunk).await?;
        anyhow::ensure!(read > 0, "connection closed before the request line");
        buffer.extend_from_slice(&chunk[..read]);
    };

    let line = std::str::from_utf8(&buffer[..line_end])?;
    let target = line.split(' ').nth(1).context("malformed request line")?;
    Ok(reqwest::Url::parse(&format!("http://127.0.0.1{target}"))?)
}

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
