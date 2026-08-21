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

//! Unified media generation and transcription.
//!
//! Media uses task-specific traits instead of extending the chat-oriented
//! `AiProvider`. Requests retain a strict portable core while namespaced JSON
//! carries verified provider extensions. Long-running jobs can be serialized,
//! persisted, polled, and cancelled after process restart.

pub mod errors;
pub mod factory;
pub mod providers;
pub mod traits;
pub mod types;

pub use errors::{MediaError, MediaResult};
pub use factory::MediaProviderFactory;
pub use providers::{
    OpenRouterMediaOptions, OpenRouterMediaProvider, ReplicateMediaOptions, ReplicateMediaProvider,
};
pub use traits::*;
pub use types::*;

use std::time::{Duration, Instant};

/// Generate an image and wait for completion. A local timeout returns a
/// resumable `JobHandle` and never implies remote cancellation.
pub async fn generate_image(
    provider_model: &str,
    mut request: ImageGenerationRequest,
) -> MediaResult<ImageGenerationResult> {
    let (provider, model) = MediaProviderFactory::get_image_provider_for_model(provider_model)?;
    set_request_model(&mut request.model, model)?;
    let controls = request.request_options.clone();
    let operation = provider.submit_image(request).await?;
    wait_image(provider.as_ref(), operation, &controls).await
}

pub async fn generate_video(
    provider_model: &str,
    mut request: VideoGenerationRequest,
) -> MediaResult<VideoGenerationResult> {
    let (provider, model) = MediaProviderFactory::get_video_provider_for_model(provider_model)?;
    set_request_model(&mut request.model, model)?;
    let controls = request.request_options.clone();
    let operation = provider.submit_video(request).await?;
    wait_video(provider.as_ref(), operation, &controls).await
}

pub async fn synthesize_speech(
    provider_model: &str,
    mut request: SpeechSynthesisRequest,
) -> MediaResult<SpeechSynthesisResult> {
    let (provider, model) = MediaProviderFactory::get_speech_provider_for_model(provider_model)?;
    set_request_model(&mut request.model, model)?;
    let controls = request.request_options.clone();
    let operation = provider.submit_speech(request).await?;
    wait_speech(provider.as_ref(), operation, &controls).await
}

pub async fn transcribe(
    provider_model: &str,
    mut request: TranscriptionRequest,
) -> MediaResult<TranscriptionResult> {
    let (provider, model) =
        MediaProviderFactory::get_transcription_provider_for_model(provider_model)?;
    set_request_model(&mut request.model, model)?;
    let controls = request.request_options.clone();
    let operation = provider.submit_transcription(request).await?;
    wait_transcription(provider.as_ref(), operation, &controls).await
}

/// Explicitly download a URL artifact with a hard byte limit. Inline artifacts
/// are copied after the same limit check. Provider authentication is inferred
/// only from adapter-owned artifact metadata.
pub async fn download_artifact(
    artifact: &MediaArtifact,
    maximum_bytes: usize,
    timeout: Option<Duration>,
) -> MediaResult<MediaSource> {
    match &artifact.source {
        ArtifactSource::Inline(data) => {
            if data.len() > maximum_bytes {
                return Err(MediaError::ArtifactTooLarge {
                    maximum: maximum_bytes,
                });
            }
            Ok(MediaSource::Bytes {
                data: data.clone(),
                media_type: artifact.media_type.clone(),
            })
        }
        ArtifactSource::Url(url) => {
            validate_artifact_url(url)?;
            let mut request = providers::shared::artifact_http_client().get(url);
            if let Some(timeout) = timeout {
                request = request.timeout(timeout);
            }
            if artifact
                .metadata
                .get("requires_auth")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                let environment = match artifact
                    .metadata
                    .get("provider")
                    .and_then(serde_json::Value::as_str)
                {
                    Some("openrouter") => "OPENROUTER_API_KEY",
                    Some("replicate") => "REPLICATE_API_TOKEN",
                    _ => {
                        return Err(MediaError::InvalidRequest(
                            "authenticated artifact has no recognized provider".to_string(),
                        ))
                    }
                };
                let key = std::env::var(environment)
                    .map_err(|_| MediaError::MissingApiKey(environment.to_string()))?;
                request = request.bearer_auth(key);
            }
            let mut response = request.send().await?;
            if response.status().is_redirection() {
                return Err(MediaError::InvalidResponse {
                    provider: "artifact".to_string(),
                    message: "artifact redirects are not followed; download the redirected URL only after validating its trust boundary".to_string(),
                });
            }
            if !response.status().is_success() {
                return Err(MediaError::Api {
                    provider: "artifact".to_string(),
                    status: response.status().as_u16(),
                    message: "artifact download failed".to_string(),
                });
            }
            if response
                .content_length()
                .is_some_and(|length| length > maximum_bytes as u64)
            {
                return Err(MediaError::ArtifactTooLarge {
                    maximum: maximum_bytes,
                });
            }
            let media_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(';').next())
                .unwrap_or(&artifact.media_type)
                .to_string();
            if !compatible_media_types(&artifact.media_type, &media_type) {
                return Err(MediaError::InvalidResponse {
                    provider: "artifact".to_string(),
                    message: format!(
                        "artifact Content-Type '{media_type}' does not match expected '{}'",
                        artifact.media_type
                    ),
                });
            }
            let mut data = Vec::new();
            while let Some(chunk) = response.chunk().await? {
                if data.len().saturating_add(chunk.len()) > maximum_bytes {
                    return Err(MediaError::ArtifactTooLarge {
                        maximum: maximum_bytes,
                    });
                }
                data.extend_from_slice(&chunk);
            }
            Ok(MediaSource::Bytes { data, media_type })
        }
        ArtifactSource::ProviderFileId(_) | ArtifactSource::ObjectStorageUri(_) => {
            Err(MediaError::InvalidRequest(
                "provider-file and object-storage artifacts require an application-owned fetcher"
                    .to_string(),
            ))
        }
    }
}

fn set_request_model(target: &mut String, resolved: String) -> MediaResult<()> {
    if !target.is_empty() && target != &resolved {
        return Err(MediaError::InvalidRequest(format!(
            "request model '{}' conflicts with resolved model '{resolved}'",
            target
        )));
    }
    *target = resolved;
    Ok(())
}

fn compatible_media_types(expected: &str, actual: &str) -> bool {
    if expected == actual || expected == "application/octet-stream" {
        return true;
    }
    let Some((expected_kind, expected_subtype)) = expected.split_once('/') else {
        return false;
    };
    let Some((actual_kind, _)) = actual.split_once('/') else {
        return false;
    };
    expected_kind == actual_kind && expected_subtype == "*"
}

fn validate_artifact_url(value: &str) -> MediaResult<()> {
    let url = reqwest::Url::parse(value)
        .map_err(|error| MediaError::InvalidRequest(format!("invalid artifact URL: {error}")))?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err(MediaError::InvalidRequest(
            "artifact URL must use HTTPS and contain no embedded credentials".to_string(),
        ));
    }
    let host = url.host_str().unwrap_or_default();
    if host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host.to_ascii_lowercase().ends_with(".local")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| match address {
                std::net::IpAddr::V4(address) => {
                    address.is_private()
                        || address.is_loopback()
                        || address.is_link_local()
                        || address.is_unspecified()
                        || address.is_broadcast()
                        || address.is_multicast()
                }
                std::net::IpAddr::V6(address) => {
                    address.is_loopback()
                        || address.is_unspecified()
                        || address.is_unique_local()
                        || address.is_unicast_link_local()
                        || address.is_multicast()
                }
            })
    {
        return Err(MediaError::InvalidRequest(
            "artifact URL targets a local or private network address".to_string(),
        ));
    }
    Ok(())
}

async fn wait_image(
    provider: &dyn ImageGenerationProvider,
    mut operation: Operation<ImageGenerationResult>,
    controls: &RequestOptions,
) -> MediaResult<ImageGenerationResult> {
    let started = Instant::now();
    loop {
        if let Some(result) = take_terminal_result(&mut operation)? {
            return Ok(result);
        }
        let handle = resumable_handle(&operation)?;
        check_wait_control(controls, started, &handle)?;
        wait_before_poll(controls, started, &handle).await?;
        operation = provider.poll_image(&handle).await?;
    }
}

async fn wait_video(
    provider: &dyn VideoGenerationProvider,
    mut operation: Operation<VideoGenerationResult>,
    controls: &RequestOptions,
) -> MediaResult<VideoGenerationResult> {
    let started = Instant::now();
    loop {
        if let Some(result) = take_terminal_result(&mut operation)? {
            return Ok(result);
        }
        let handle = resumable_handle(&operation)?;
        check_wait_control(controls, started, &handle)?;
        wait_before_poll(controls, started, &handle).await?;
        operation = provider.poll_video(&handle).await?;
    }
}

async fn wait_speech(
    provider: &dyn SpeechSynthesisProvider,
    mut operation: Operation<SpeechSynthesisResult>,
    controls: &RequestOptions,
) -> MediaResult<SpeechSynthesisResult> {
    let started = Instant::now();
    loop {
        if let Some(result) = take_terminal_result(&mut operation)? {
            return Ok(result);
        }
        let handle = resumable_handle(&operation)?;
        check_wait_control(controls, started, &handle)?;
        wait_before_poll(controls, started, &handle).await?;
        operation = provider.poll_speech(&handle).await?;
    }
}

async fn wait_transcription(
    provider: &dyn TranscriptionProvider,
    mut operation: Operation<TranscriptionResult>,
    controls: &RequestOptions,
) -> MediaResult<TranscriptionResult> {
    let started = Instant::now();
    loop {
        if let Some(result) = take_terminal_result(&mut operation)? {
            return Ok(result);
        }
        let handle = resumable_handle(&operation)?;
        check_wait_control(controls, started, &handle)?;
        wait_before_poll(controls, started, &handle).await?;
        operation = provider.poll_transcription(&handle).await?;
    }
}

fn take_terminal_result<T>(operation: &mut Operation<T>) -> MediaResult<Option<T>> {
    match operation.status {
        OperationStatus::Succeeded => {
            operation
                .result
                .take()
                .map(Some)
                .ok_or_else(|| MediaError::InvalidResponse {
                    provider: operation
                        .handle
                        .as_ref()
                        .map_or_else(|| "media".to_string(), |handle| handle.provider.clone()),
                    message: "succeeded operation has no result".to_string(),
                })
        }
        OperationStatus::Failed | OperationStatus::Cancelled | OperationStatus::Expired => Err(
            MediaError::RemoteFailure(operation.error.clone().unwrap_or(GenerationFailure {
                category: FailureCategory::RemoteJob,
                message: format!("operation ended in {:?}", operation.status),
                provider_code: None,
                http_status: None,
                request_id: None,
                provider_metadata: serde_json::Value::Null,
            })),
        ),
        _ => Ok(None),
    }
}

fn resumable_handle<T>(operation: &Operation<T>) -> MediaResult<JobHandle> {
    operation
        .handle
        .clone()
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: "media".to_string(),
            message: "non-terminal operation has no resumable handle".to_string(),
        })
}

fn check_wait_control(
    controls: &RequestOptions,
    started: Instant,
    handle: &JobHandle,
) -> MediaResult<()> {
    if controls
        .cancellation_token
        .as_ref()
        .is_some_and(|receiver| *receiver.borrow())
    {
        return Err(MediaError::LocalWaitCancelled {
            handle: Box::new(handle.clone()),
        });
    }
    if controls
        .wait_timeout
        .is_some_and(|timeout| started.elapsed() >= timeout)
    {
        return Err(MediaError::WaitTimeout {
            handle: Box::new(handle.clone()),
        });
    }
    Ok(())
}

async fn wait_before_poll(
    controls: &RequestOptions,
    started: Instant,
    handle: &JobHandle,
) -> MediaResult<()> {
    check_wait_control(controls, started, handle)?;
    let delay = controls
        .wait_timeout
        .map_or(controls.polling_interval, |timeout| {
            controls
                .polling_interval
                .min(timeout.saturating_sub(started.elapsed()))
        });
    if let Some(mut cancellation) = controls.cancellation_token.clone() {
        tokio::select! {
            _ = tokio::time::sleep(delay) => {}
            changed = cancellation.changed() => {
                if changed.is_ok() && *cancellation.borrow() {
                    return Err(MediaError::LocalWaitCancelled { handle: Box::new(handle.clone()) });
                }
            }
        }
    } else {
        tokio::time::sleep(delay).await;
    }
    check_wait_control(controls, started, handle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    #[test]
    fn timeout_keeps_the_exact_handle_resumable() {
        let handle = JobHandle {
            provider: "replicate".to_string(),
            model: "owner/model".to_string(),
            task: MediaTask::TextToVideo,
            remote_id: "job".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        let options = RequestOptions {
            wait_timeout: Some(Duration::ZERO),
            ..RequestOptions::default()
        };
        let error = check_wait_control(&options, Instant::now(), &handle).unwrap_err();
        assert!(
            matches!(error, MediaError::WaitTimeout { handle: returned } if *returned == handle)
        );
    }

    #[test]
    fn high_level_model_resolution_rejects_conflicts() {
        let mut target = "owner/model-a".to_string();
        assert!(set_request_model(&mut target, "owner/model-b".to_string()).is_err());
        assert_eq!(target, "owner/model-a");
    }

    #[test]
    fn artifact_content_type_compatibility_is_strict_by_media_kind() {
        assert!(compatible_media_types("image/*", "image/png"));
        assert!(!compatible_media_types("image/*", "text/html"));
        assert!(!compatible_media_types("image/png", "image/jpeg"));
    }

    #[test]
    fn artifact_download_urls_fail_closed() {
        assert!(validate_artifact_url("https://cdn.example.test/file.png").is_ok());
        assert!(validate_artifact_url("http://cdn.example.test/file.png").is_err());
        assert!(validate_artifact_url("https://user:secret@example.test/file.png").is_err());
        assert!(validate_artifact_url("https://127.0.0.1/file.png").is_err());
        assert!(validate_artifact_url("https://metadata.local/file.png").is_err());
    }

    #[test]
    fn succeeded_operation_without_result_fails_closed() {
        let mut operation: Operation<ImageGenerationResult> = Operation {
            handle: None,
            status: OperationStatus::Succeeded,
            progress: Some(1.0),
            result: None,
            error: None,
            created_at: None,
            started_at: None,
            completed_at: None,
            provider_metadata: serde_json::Value::Null,
        };
        assert!(matches!(
            take_terminal_result(&mut operation),
            Err(MediaError::InvalidResponse { .. })
        ));
    }

    struct MockVideoProvider {
        polls: Mutex<VecDeque<Operation<VideoGenerationResult>>>,
    }

    #[async_trait::async_trait]
    impl VideoGenerationProvider for MockVideoProvider {
        fn name(&self) -> &str {
            "mock"
        }

        fn supports_model(&self, _model: &str) -> bool {
            true
        }

        fn capabilities(&self, model: &str) -> VideoCapabilities {
            MediaModelDescriptor {
                provider: "mock".to_string(),
                model: model.to_string(),
                tasks: vec![MediaTask::TextToVideo],
                input_modalities: vec![Modality::Text],
                output_modalities: vec![Modality::Video],
                execution: ExecutionCapabilities {
                    immediate: CapabilitySupport::Unsupported,
                    persistent_jobs: CapabilitySupport::Supported,
                    polling: CapabilitySupport::Supported,
                    cancellation: CapabilitySupport::Supported,
                    progress: CapabilitySupport::Unknown,
                    webhooks: CapabilitySupport::Unsupported,
                    binary_streaming: CapabilitySupport::Unsupported,
                    resumable: CapabilitySupport::Supported,
                },
                parameters: ParameterCapabilities {
                    count: CapabilitySupport::Unknown,
                    seed: CapabilitySupport::Unknown,
                    dimensions: CapabilitySupport::Unknown,
                    aspect_ratio: CapabilitySupport::Unknown,
                    duration: CapabilitySupport::Unknown,
                    mask: CapabilitySupport::Unsupported,
                    negative_prompt: CapabilitySupport::Unknown,
                    output_format: CapabilitySupport::Unknown,
                },
                limits: MediaLimits::default(),
                provider_options_schema: None,
            }
        }

        async fn submit_video(
            &self,
            _request: VideoGenerationRequest,
        ) -> MediaResult<Operation<VideoGenerationResult>> {
            Err(MediaError::InvalidRequest("not used".to_string()))
        }

        async fn poll_video(
            &self,
            _handle: &JobHandle,
        ) -> MediaResult<Operation<VideoGenerationResult>> {
            self.polls
                .lock()
                .map_err(|_| MediaError::InvalidRequest("mock lock poisoned".to_string()))?
                .pop_front()
                .ok_or_else(|| MediaError::InvalidRequest("no mock poll response".to_string()))
        }

        async fn cancel_video(&self, _handle: &JobHandle) -> MediaResult<()> {
            Ok(())
        }
    }

    fn mock_handle() -> JobHandle {
        JobHandle {
            provider: "mock".to_string(),
            model: "video".to_string(),
            task: MediaTask::TextToVideo,
            remote_id: "job".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        }
    }

    fn pending_video(status: OperationStatus) -> Operation<VideoGenerationResult> {
        Operation {
            handle: Some(mock_handle()),
            status,
            progress: None,
            result: None,
            error: None,
            created_at: None,
            started_at: None,
            completed_at: None,
            provider_metadata: serde_json::Value::Null,
        }
    }

    #[tokio::test]
    async fn shared_waiter_normalizes_queued_running_succeeded_lifecycle() {
        let expected = VideoGenerationResult {
            artifacts: vec![MediaArtifact {
                kind: MediaKind::Video,
                media_type: "video/mp4".to_string(),
                source: ArtifactSource::Url("https://example.test/video.mp4".to_string()),
                size_bytes: None,
                dimensions: None,
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: None,
                channels: None,
                expires_at: None,
                metadata: serde_json::Value::Null,
            }],
            usage: None,
            warnings: Vec::new(),
            safety: SafetyReport::default(),
            provider_metadata: serde_json::Value::Null,
        };
        let provider = MockVideoProvider {
            polls: Mutex::new(VecDeque::from([
                pending_video(OperationStatus::Running),
                Operation {
                    handle: Some(mock_handle()),
                    status: OperationStatus::Succeeded,
                    progress: Some(1.0),
                    result: Some(expected.clone()),
                    error: None,
                    created_at: None,
                    started_at: None,
                    completed_at: None,
                    provider_metadata: serde_json::Value::Null,
                },
            ])),
        };
        let options = RequestOptions {
            polling_interval: Duration::from_millis(1),
            wait_timeout: Some(Duration::from_secs(1)),
            ..RequestOptions::default()
        };
        let actual = wait_video(&provider, pending_video(OperationStatus::Queued), &options)
            .await
            .unwrap();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn local_cancellation_does_not_call_remote_cancel() {
        let (sender, receiver) = tokio::sync::watch::channel(true);
        let provider = MockVideoProvider {
            polls: Mutex::new(VecDeque::new()),
        };
        let options = RequestOptions {
            cancellation_token: Some(receiver),
            ..RequestOptions::default()
        };
        let error = wait_video(
            &provider,
            pending_video(OperationStatus::CancellationRequested),
            &options,
        )
        .await
        .unwrap_err();
        drop(sender);
        assert!(matches!(error, MediaError::LocalWaitCancelled { .. }));
        assert!(provider.polls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn speech_stream_yields_before_the_server_finishes() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n",
                )
                .unwrap();
            stream.flush().unwrap();
            std::thread::sleep(Duration::from_millis(1000));
            stream.write_all(b"5\r\nworld\r\n0\r\n\r\n").unwrap();
        });
        let response = reqwest::get(format!("http://{address}")).await.unwrap();
        let mut stream = SpeechStream::new("audio/mpeg".to_string(), None, response);
        let started = Instant::now();
        assert_eq!(stream.next_chunk().await.unwrap(), Some(b"hello".to_vec()));
        assert!(started.elapsed() < Duration::from_millis(800));
        server.join().unwrap();
    }
}
