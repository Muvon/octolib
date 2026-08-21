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

use crate::media::errors::{MediaError, MediaResult};
use crate::media::types::{JobHandle, MediaSource, MediaTask, ProviderOptions, RequestOptions};
use base64::Engine;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Duration;

/// Separate process-wide client for explicit artifact downloads. Redirects are
/// disabled so a provider URL cannot silently redirect into another trust zone.
static ARTIFACT_HTTP_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build media artifact HTTP client")
});

pub(crate) struct CapturedResponse {
    pub status: reqwest::StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

pub(crate) fn artifact_http_client() -> reqwest::Client {
    ARTIFACT_HTTP_CLIENT.clone()
}

pub(crate) fn validate_https_url(value: &str, field: &str) -> MediaResult<()> {
    let url = reqwest::Url::parse(value)
        .map_err(|error| MediaError::InvalidRequest(format!("invalid {field}: {error}")))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(MediaError::InvalidRequest(format!(
            "{field} must be an HTTPS URL without embedded credentials"
        )));
    }
    Ok(())
}

pub(crate) fn api_key(environment_variable: &str) -> MediaResult<String> {
    std::env::var(environment_variable)
        .map_err(|_| MediaError::MissingApiKey(environment_variable.to_string()))
}

pub(crate) fn provider_options(
    options: &ProviderOptions,
    provider: &str,
) -> MediaResult<Map<String, Value>> {
    for namespace in options.keys() {
        if namespace != provider {
            return Err(MediaError::InvalidRequest(format!(
                "provider option namespace '{namespace}' cannot be used with {provider}"
            )));
        }
    }
    match options.get(provider) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(value)) => Ok(value.clone()),
        Some(_) => Err(MediaError::InvalidRequest(format!(
            "provider_options.{provider} must be a JSON object"
        ))),
    }
}

pub(crate) fn validate_handle(
    handle: &JobHandle,
    provider: &str,
    allowed_tasks: &[MediaTask],
) -> MediaResult<()> {
    if handle.provider != provider || !allowed_tasks.contains(&handle.task) {
        return Err(MediaError::WrongJobHandle {
            expected_provider: provider.to_string(),
            expected_task: allowed_tasks[0],
        });
    }
    if handle.remote_id.is_empty() {
        return Err(MediaError::InvalidRequest(
            "job handle remote_id must not be empty".to_string(),
        ));
    }
    if !handle
        .remote_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(MediaError::InvalidRequest(
            "job handle remote_id contains unsupported URL path characters".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn source_to_data_or_uri(
    source: &MediaSource,
    maximum_bytes: usize,
) -> MediaResult<String> {
    match source {
        MediaSource::Bytes { data, media_type } => {
            validate_media_type(media_type)?;
            check_size(data.len() as u64, maximum_bytes)?;
            Ok(data_url(data, media_type))
        }
        MediaSource::Url { url, .. } => validate_uri(url),
        MediaSource::File { path, media_type } => {
            let metadata = std::fs::metadata(path)?;
            check_size(metadata.len(), maximum_bytes)?;
            let data = std::fs::read(path)?;
            let media_type = media_type.as_deref().ok_or_else(|| {
                MediaError::InvalidRequest(format!(
                    "media_type is required for local file {}",
                    path.display()
                ))
            })?;
            validate_media_type(media_type)?;
            Ok(data_url(&data, media_type))
        }
        MediaSource::ProviderFile { .. } => Err(MediaError::InvalidRequest(
            "provider file IDs require a provider-specific upload contract".to_string(),
        )),
        MediaSource::ObjectStorage { uri, .. } => validate_uri(uri),
    }
}

pub(crate) fn source_to_base64(
    source: &MediaSource,
    maximum_bytes: usize,
) -> MediaResult<(String, String)> {
    match source {
        MediaSource::Bytes { data, media_type } => {
            validate_media_type(media_type)?;
            check_size(data.len() as u64, maximum_bytes)?;
            Ok((
                base64::engine::general_purpose::STANDARD.encode(data),
                media_type.clone(),
            ))
        }
        MediaSource::File { path, media_type } => {
            let metadata = std::fs::metadata(path)?;
            check_size(metadata.len(), maximum_bytes)?;
            let media_type = media_type.clone().ok_or_else(|| {
                MediaError::InvalidRequest(format!(
                    "media_type is required for local file {}",
                    path.display()
                ))
            })?;
            validate_media_type(&media_type)?;
            let data = std::fs::read(path)?;
            Ok((
                base64::engine::general_purpose::STANDARD.encode(data),
                media_type,
            ))
        }
        _ => Err(MediaError::InvalidRequest(
            "this endpoint requires inline bytes or a local file; URL download is intentionally opt-in"
                .to_string(),
        )),
    }
}

pub(crate) fn decode_data_url(value: &str) -> MediaResult<(Vec<u8>, String)> {
    let rest = value
        .strip_prefix("data:")
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: "media".to_string(),
            message: "expected a data URL".to_string(),
        })?;
    let (header, encoded) = rest
        .split_once(',')
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: "media".to_string(),
            message: "malformed data URL".to_string(),
        })?;
    if !header.ends_with(";base64") {
        return Err(MediaError::InvalidResponse {
            provider: "media".to_string(),
            message: "only base64 data URLs are supported".to_string(),
        });
    }
    let media_type = header.trim_end_matches(";base64").to_string();
    let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
    Ok((bytes, media_type))
}

pub(crate) fn media_type_from_url(url: &str, kind: &str) -> String {
    let path = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
    let inferred = if path.ends_with(".png") {
        Some("image/png")
    } else if path.ends_with(".jpg") || path.ends_with(".jpeg") {
        Some("image/jpeg")
    } else if path.ends_with(".webp") {
        Some("image/webp")
    } else if path.ends_with(".svg") {
        Some("image/svg+xml")
    } else if path.ends_with(".mp4") {
        Some("video/mp4")
    } else if path.ends_with(".webm") {
        Some("video/webm")
    } else if path.ends_with(".mov") {
        Some("video/quicktime")
    } else if path.ends_with(".mp3") {
        Some("audio/mpeg")
    } else if path.ends_with(".wav") {
        Some("audio/wav")
    } else if path.ends_with(".flac") {
        Some("audio/flac")
    } else if path.ends_with(".ogg") {
        Some("audio/ogg")
    } else {
        None
    };
    inferred.unwrap_or(kind).to_string()
}

pub(crate) async fn send(
    provider: &str,
    options: &RequestOptions,
    build: impl Fn() -> reqwest::RequestBuilder,
) -> MediaResult<CapturedResponse> {
    send_with_retry_policy(provider, options, false, build).await
}

/// Retry only requests whose upstream semantics are idempotent, such as GET
/// polling and schema discovery. Media creation POSTs use `send` so transport
/// ambiguity cannot create duplicate paid jobs.
pub(crate) async fn send_idempotent(
    provider: &str,
    options: &RequestOptions,
    build: impl Fn() -> reqwest::RequestBuilder,
) -> MediaResult<CapturedResponse> {
    send_with_retry_policy(provider, options, true, build).await
}

async fn send_with_retry_policy(
    provider: &str,
    options: &RequestOptions,
    retry_safe: bool,
    build: impl Fn() -> reqwest::RequestBuilder,
) -> MediaResult<CapturedResponse> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy
            == crate::media::types::UnsupportedParameterPolicy::Error
    {
        return Err(MediaError::UnsupportedParameter {
            provider: provider.to_string(),
            parameter: "request_options.idempotency_key".to_string(),
            reason: "the provider contract does not document idempotency keys".to_string(),
        });
    }

    let headers = strict_extra_headers(options.extra_headers.as_ref())?;
    let mut attempt = 0_u32;
    loop {
        if cancellation_requested(options) {
            return Err(MediaError::InvalidRequest(
                "request cancelled before remote submission".to_string(),
            ));
        }
        let mut request = build();
        if !headers.is_empty() {
            request = request.headers(headers.clone());
        }
        if let Some(timeout) = options.submit_timeout {
            request = request.timeout(timeout);
        }
        match request.send().await {
            Ok(response) => {
                let status = response.status();
                let response_headers = response.headers().clone();
                let body = read_bounded(response, options.max_response_bytes).await?;
                if retry_safe
                    && (status.as_u16() == 429 || status.is_server_error())
                    && attempt < options.max_retries
                {
                    tokio::time::sleep(backoff(options.retry_backoff, attempt)).await;
                    attempt += 1;
                    continue;
                }
                return Ok(CapturedResponse {
                    status,
                    headers: response_headers,
                    body,
                });
            }
            Err(error) if retry_safe && attempt < options.max_retries && !error.is_timeout() => {
                crate::llm::providers::shared::refresh_http_client();
                tracing::warn!(provider, attempt, error = %error, "retrying media request");
                tokio::time::sleep(backoff(options.retry_backoff, attempt)).await;
                attempt += 1;
            }
            Err(error) => return Err(MediaError::Transport(error)),
        }
    }
}

/// Send a request while leaving a successful body unbuffered for genuine
/// provider streaming. Error bodies are bounded and consumed for diagnostics.
pub(crate) async fn send_stream(
    provider: &str,
    options: &RequestOptions,
    build: impl Fn() -> reqwest::RequestBuilder,
) -> MediaResult<reqwest::Response> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy
            == crate::media::types::UnsupportedParameterPolicy::Error
    {
        return Err(MediaError::UnsupportedParameter {
            provider: provider.to_string(),
            parameter: "request_options.idempotency_key".to_string(),
            reason: "the provider contract does not document idempotency keys".to_string(),
        });
    }
    let headers = strict_extra_headers(options.extra_headers.as_ref())?;
    if cancellation_requested(options) {
        return Err(MediaError::InvalidRequest(
            "request cancelled before remote submission".to_string(),
        ));
    }
    let mut request = build();
    if !headers.is_empty() {
        request = request.headers(headers);
    }
    let sent = match options.submit_timeout {
        Some(timeout) => match tokio::time::timeout(timeout, request.send()).await {
            Ok(result) => result,
            Err(_) => {
                return Err(MediaError::InvalidRequest(
                    "stream submission timed out before response headers".to_string(),
                ))
            }
        },
        None => request.send().await,
    };
    match sent {
        Ok(response) if response.status().is_success() => Ok(response),
        Ok(response) => {
            let status = response.status();
            let response_headers = response.headers().clone();
            let body = read_bounded(response, 64 * 1024)
                .await
                .unwrap_or_else(|_| b"provider error body exceeded 64 KiB".to_vec());
            classified_api_error(provider, status, &response_headers, &body)?;
            Err(MediaError::InvalidResponse {
                provider: provider.to_string(),
                message: "non-success response was not classified".to_string(),
            })
        }
        Err(error) => Err(MediaError::Transport(error)),
    }
}

pub(crate) fn require_success(provider: &str, response: &CapturedResponse) -> MediaResult<()> {
    if response.status.is_success() {
        return Ok(());
    }
    classified_api_error(provider, response.status, &response.headers, &response.body)
}

pub(crate) fn parse_json(provider: &str, response: &CapturedResponse) -> MediaResult<Value> {
    require_success(provider, response)?;
    serde_json::from_slice(&response.body).map_err(|error| MediaError::InvalidResponse {
        provider: provider.to_string(),
        message: format!("response was not valid JSON: {error}"),
    })
}

fn strict_extra_headers(extra: Option<&HashMap<String, String>>) -> MediaResult<HeaderMap> {
    let mut headers = HeaderMap::new();
    let Some(extra) = extra else {
        return Ok(headers);
    };
    for (name, value) in extra {
        if name.eq_ignore_ascii_case("authorization") || name.eq_ignore_ascii_case("content-type") {
            return Err(MediaError::InvalidRequest(format!(
                "extra header '{name}' cannot override provider authentication or content type"
            )));
        }
        let name = name.parse::<HeaderName>().map_err(|error| {
            MediaError::InvalidRequest(format!("invalid extra header name: {error}"))
        })?;
        let value = value.parse::<HeaderValue>().map_err(|error| {
            MediaError::InvalidRequest(format!("invalid extra header value: {error}"))
        })?;
        headers.insert(name, value);
    }
    Ok(headers)
}

fn cancellation_requested(options: &RequestOptions) -> bool {
    options
        .cancellation_token
        .as_ref()
        .is_some_and(|receiver| *receiver.borrow())
}

fn backoff(base: Duration, attempt: u32) -> Duration {
    base.saturating_mul(2_u32.saturating_pow(attempt))
        .min(Duration::from_secs(30))
}

fn check_size(actual: u64, maximum: usize) -> MediaResult<()> {
    if actual > maximum as u64 {
        return Err(MediaError::SourceTooLarge { actual, maximum });
    }
    Ok(())
}

fn data_url(data: &[u8], media_type: &str) -> String {
    format!(
        "data:{media_type};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(data)
    )
}

async fn read_bounded(mut response: reqwest::Response, maximum: usize) -> MediaResult<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum as u64)
    {
        return Err(MediaError::ResponseTooLarge { maximum });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len().saturating_add(chunk.len()) > maximum {
            return Err(MediaError::ResponseTooLarge { maximum });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn validate_media_type(media_type: &str) -> MediaResult<()> {
    let valid = !media_type.is_empty()
        && media_type.len() <= 127
        && media_type.contains('/')
        && media_type
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'+' | b'.'));
    if !valid {
        return Err(MediaError::InvalidRequest(format!(
            "invalid media type '{media_type}'"
        )));
    }
    Ok(())
}

fn validate_uri(value: &str) -> MediaResult<String> {
    if value.len() > 8192 || value.contains(['\r', '\n']) {
        return Err(MediaError::InvalidRequest(
            "media URI has an unsupported scheme or unsafe characters".to_string(),
        ));
    }
    if value.starts_with("http://") || value.starts_with("https://") {
        let url = reqwest::Url::parse(value).map_err(|_| {
            MediaError::InvalidRequest("media URL is not a valid HTTP(S) URL".to_string())
        })?;
        if url.host_str().is_none() || !url.username().is_empty() || url.password().is_some() {
            return Err(MediaError::InvalidRequest(
                "media URL must have a host and no embedded credentials".to_string(),
            ));
        }
    } else if let Some(path) = value
        .strip_prefix("s3://")
        .or_else(|| value.strip_prefix("gs://"))
    {
        if !path
            .split_once('/')
            .is_some_and(|(bucket, object)| !bucket.is_empty() && !object.is_empty())
            || path.bytes().any(|byte| byte.is_ascii_whitespace())
        {
            return Err(MediaError::InvalidRequest(
                "object-storage URI must contain a bucket and object path".to_string(),
            ));
        }
    } else {
        return Err(MediaError::InvalidRequest(
            "media URI has an unsupported scheme".to_string(),
        ));
    }
    Ok(value.to_string())
}

fn error_message(body: &[u8]) -> String {
    const MAX: usize = 16 * 1024;
    let body = &body[..body.len().min(MAX)];
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        for pointer in ["/error/message", "/detail", "/error"] {
            if let Some(message) = value.pointer(pointer).and_then(Value::as_str) {
                return message.to_string();
            }
        }
        return value.to_string();
    }
    String::from_utf8_lossy(body).into_owned()
}

fn classified_api_error(
    provider: &str,
    status: reqwest::StatusCode,
    headers: &HeaderMap,
    body: &[u8],
) -> MediaResult<()> {
    let message = error_message(body);
    match status.as_u16() {
        401 => Err(MediaError::Authentication {
            provider: provider.to_string(),
            message,
        }),
        402 => Err(MediaError::InsufficientCredits {
            provider: provider.to_string(),
            message,
        }),
        403 => Err(MediaError::Permission {
            provider: provider.to_string(),
            message,
        }),
        429 => Err(MediaError::RateLimit {
            provider: provider.to_string(),
            message,
            retry_after_secs: headers
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.parse().ok()),
        }),
        _ => Err(MediaError::Api {
            provider: provider.to_string(),
            status: status.as_u16(),
            message,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn rejects_cross_provider_options() {
        let mut options = ProviderOptions::new();
        options.insert("replicate".to_string(), serde_json::json!({}));
        assert!(provider_options(&options, "openrouter").is_err());
    }

    #[test]
    fn round_trips_data_urls() {
        let encoded = data_url(b"image", "image/png");
        let (decoded, media_type) = decode_data_url(&encoded).unwrap();
        assert_eq!(decoded, b"image");
        assert_eq!(media_type, "image/png");
    }

    #[test]
    fn classifies_credit_and_rate_limit_errors() {
        assert!(matches!(
            classified_api_error(
                "provider",
                reqwest::StatusCode::PAYMENT_REQUIRED,
                &HeaderMap::new(),
                b"credits"
            ),
            Err(MediaError::InsufficientCredits { .. })
        ));
        assert!(matches!(
            classified_api_error(
                "provider",
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                &HeaderMap::new(),
                b"slow down"
            ),
            Err(MediaError::RateLimit { .. })
        ));
    }

    #[test]
    fn rejects_wrong_or_path_shaped_job_handles() {
        let wrong_provider = JobHandle {
            provider: "replicate".to_string(),
            model: "owner/model".to_string(),
            task: MediaTask::TextToImage,
            remote_id: "job".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        assert!(matches!(
            validate_handle(&wrong_provider, "openrouter", &[MediaTask::TextToImage]),
            Err(MediaError::WrongJobHandle { .. })
        ));

        let unsafe_id = JobHandle {
            provider: "replicate".to_string(),
            remote_id: "../job".to_string(),
            ..wrong_provider
        };
        assert!(matches!(
            validate_handle(&unsafe_id, "replicate", &[MediaTask::TextToImage]),
            Err(MediaError::InvalidRequest(_))
        ));
    }

    #[test]
    fn rejects_oversized_inline_source_before_encoding() {
        let source = MediaSource::Bytes {
            data: vec![0; 5],
            media_type: "image/png".to_string(),
        };
        assert!(matches!(
            source_to_data_or_uri(&source, 4),
            Err(MediaError::SourceTooLarge {
                actual: 5,
                maximum: 4
            })
        ));
    }

    #[test]
    fn rejects_oversized_file_before_reading_it() {
        let path = std::env::temp_dir().join(format!(
            "octolib-media-size-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"12345").unwrap();
        let source = MediaSource::File {
            path: path.clone(),
            media_type: Some("audio/mpeg".to_string()),
        };
        let result = source_to_base64(&source, 4);
        std::fs::remove_file(path).unwrap();
        assert!(matches!(result, Err(MediaError::SourceTooLarge { .. })));
    }

    #[test]
    fn media_uris_require_valid_hosts_or_object_paths() {
        assert!(validate_uri("https://cdn.example.test/image.png").is_ok());
        assert!(validate_uri("https://user:secret@example.test/image.png").is_err());
        assert!(validate_uri("https://").is_err());
        assert!(validate_uri("s3://bucket/path/image.png").is_ok());
        assert!(validate_uri("s3://bucket").is_err());
    }

    #[tokio::test]
    async fn retries_queries_but_never_ambiguous_media_creation() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        drop(listener);
        let url = format!("http://{address}");
        let options = RequestOptions {
            max_retries: 2,
            retry_backoff: Duration::ZERO,
            submit_timeout: Some(Duration::from_secs(1)),
            ..RequestOptions::default()
        };

        let post_attempts = Arc::new(AtomicUsize::new(0));
        let post_counter = Arc::clone(&post_attempts);
        let _ = send("provider", &options, || {
            post_counter.fetch_add(1, Ordering::Relaxed);
            crate::llm::providers::shared::http_client().post(&url)
        })
        .await;
        assert_eq!(post_attempts.load(Ordering::Relaxed), 1);

        let get_attempts = Arc::new(AtomicUsize::new(0));
        let get_counter = Arc::clone(&get_attempts);
        let _ = send_idempotent("provider", &options, || {
            get_counter.fetch_add(1, Ordering::Relaxed);
            crate::llm::providers::shared::http_client().get(&url)
        })
        .await;
        assert_eq!(get_attempts.load(Ordering::Relaxed), 3);
    }
}
