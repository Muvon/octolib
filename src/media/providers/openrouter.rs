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

//! OpenRouter's dedicated image, video, speech, and transcription APIs.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use base64::Engine;
use serde_json::{json, Map, Value};

const PROVIDER: &str = "openrouter";
const API_KEY_ENV: &str = "OPENROUTER_API_KEY";
const API_BASE_ENV: &str = "OPENROUTER_MEDIA_API_URL";
const API_BASE: &str = "https://openrouter.ai/api/v1";

#[derive(Debug, Clone, Default)]
pub struct OpenRouterMediaProvider;

/// Stable OpenRouter endpoint extensions. Routing and upstream passthrough
/// settings use the native `provider` object documented by OpenRouter.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenRouterMediaOptions {
    pub provider: Option<Value>,
    pub quality: Option<String>,
    pub background: Option<String>,
    pub output_compression: Option<u8>,
    pub generate_audio: Option<bool>,
    pub callback_url: Option<String>,
}

impl OpenRouterMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_openrouter_options(&self)?;
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.retain(|_, value| !value.is_null());
        }
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), value);
        Ok(options)
    }
}

impl OpenRouterMediaProvider {
    pub fn new() -> Self {
        Self
    }

    fn api_base(&self) -> String {
        std::env::var(API_BASE_ENV)
            .unwrap_or_else(|_| API_BASE.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    fn key(&self) -> MediaResult<String> {
        shared::api_key(API_KEY_ENV)
    }

    fn descriptor(&self, model: &str, task: MediaTask) -> MediaModelDescriptor {
        let (inputs, outputs, execution, parameters, formats) = match task {
            MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::ImageVariation => (
                vec![Modality::Text, Modality::Image],
                vec![Modality::Image],
                execution(
                    CapabilitySupport::Supported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unknown,
                ),
                parameters(
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unknown,
                ),
                vec!["png", "jpeg", "webp", "svg"],
            ),
            MediaTask::TextToVideo | MediaTask::ImageToVideo | MediaTask::ReferenceToVideo => (
                vec![
                    Modality::Text,
                    Modality::Image,
                    Modality::Video,
                    Modality::Audio,
                ],
                vec![Modality::Video, Modality::Audio],
                execution(
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Supported,
                    CapabilitySupport::Supported,
                    CapabilitySupport::Supported,
                ),
                parameters(
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                ),
                Vec::new(),
            ),
            MediaTask::TextToSpeech => (
                vec![Modality::Text],
                vec![Modality::Audio],
                execution(
                    CapabilitySupport::Supported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Supported,
                ),
                parameters(
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unknown,
                ),
                vec!["mp3", "pcm"],
            ),
            MediaTask::SpeechToText => (
                vec![Modality::Audio],
                vec![Modality::Text],
                execution(
                    CapabilitySupport::Supported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                ),
                parameters(
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                    CapabilitySupport::Unsupported,
                ),
                Vec::new(),
            ),
            _ => (
                Vec::new(),
                Vec::new(),
                execution(
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                    CapabilitySupport::Unknown,
                ),
                unknown_parameters(),
                Vec::new(),
            ),
        };
        MediaModelDescriptor {
            provider: PROVIDER.to_string(),
            model: model.to_string(),
            tasks: vec![task],
            input_modalities: inputs,
            output_modalities: outputs,
            execution,
            parameters,
            limits: MediaLimits {
                max_input_bytes: None,
                max_outputs: (task == MediaTask::TextToImage).then_some(10),
                max_duration_secs: None,
                supported_formats: formats.into_iter().map(str::to_string).collect(),
            },
            provider_options_schema: Some(openrouter_options_schema(task)),
        }
    }

    async fn fetch_generation_cost(
        &self,
        generation_id: &str,
        key: &str,
        request_options: &RequestOptions,
    ) -> Option<f64> {
        let mut options = request_options.clone();
        options.idempotency_key = None;
        options.max_retries = options.max_retries.min(1);
        let url = format!("{}/generation", self.api_base());
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .bearer_auth(key)
                .query(&[("id", generation_id)])
        })
        .await
        .ok()?;
        let value = shared::parse_json(PROVIDER, &response).ok()?;
        value.pointer("/data/total_cost").and_then(Value::as_f64)
    }
}

#[async_trait::async_trait]
impl ImageGenerationProvider for OpenRouterMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.trim().is_empty()
    }

    fn capabilities(&self, model: &str) -> ImageCapabilities {
        self.descriptor(model, MediaTask::TextToImage)
    }

    async fn submit_image(
        &self,
        request: ImageGenerationRequest,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        validate_text(&request.model, &request.prompt)?;
        let mut warnings = request_warnings(&request.request_options);
        let mut body = Map::new();
        body.insert("model".to_string(), json!(request.model));
        body.insert("prompt".to_string(), json!(request.prompt));

        if request.mode == ImageGenerationMode::Inpaint {
            return Err(MediaError::UnsupportedTask {
                provider: PROVIDER.to_string(),
                model: request.model,
                task: MediaTask::Inpainting,
            });
        }
        if request.mask.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "mask",
                "OpenRouter's dedicated Image API does not document masks",
            )?;
        }
        if request.mode == ImageGenerationMode::Variation && request.source_images.is_empty() {
            return Err(MediaError::InvalidRequest(
                "image variation requires at least one source image".to_string(),
            ));
        }
        if request.mode == ImageGenerationMode::Edit && request.source_images.is_empty() {
            return Err(MediaError::InvalidRequest(
                "image edit requires at least one source image".to_string(),
            ));
        }
        if let Some(count) = request.count {
            if !(1..=10).contains(&count) {
                return Err(MediaError::InvalidRequest(
                    "image count must be between 1 and 10".to_string(),
                ));
            }
            body.insert("n".to_string(), json!(count));
        }
        if let Some(seed) = request.seed {
            body.insert("seed".to_string(), json!(seed));
        }
        if let Some(geometry) = request.geometry {
            match geometry {
                OutputGeometry::Dimensions { .. } => {
                    body.insert("size".to_string(), json!(geometry.as_api_string()));
                }
                OutputGeometry::AspectRatio { .. } => {
                    body.insert("aspect_ratio".to_string(), json!(geometry.as_api_string()));
                }
            }
        }
        if request.negative_prompt.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "negative_prompt",
                "it is not a portable OpenRouter Image API parameter",
            )?;
        }
        if let Some(format) = request.output_format {
            body.insert("output_format".to_string(), json!(format.as_str()));
        }
        if !request.source_images.is_empty() {
            let references = request
                .source_images
                .iter()
                .map(|source| {
                    shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)
                        .map(|url| json!({"type":"image_url","image_url":{"url":url}}))
                })
                .collect::<MediaResult<Vec<_>>>()?;
            body.insert("input_references".to_string(), Value::Array(references));
        }
        let extensions = openrouter_extensions(
            &request.provider_options,
            &["provider", "quality", "background", "output_compression"],
            &request.request_options,
            &mut warnings,
        )?;
        if extensions.get("background").and_then(Value::as_str) == Some("transparent")
            && request
                .output_format
                .is_some_and(|format| !matches!(format, ImageFormat::Png | ImageFormat::Webp))
        {
            return Err(MediaError::InvalidRequest(
                "OpenRouter transparent background requires png or webp output".to_string(),
            ));
        }
        merge_extensions(&mut body, extensions)?;

        let key = self.key()?;
        let url = format!("{}/images", self.api_base());
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        let result = parse_image_response(&value, request.count, warnings)?;
        Ok(Operation::completed(result))
    }

    async fn poll_image(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }

    async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }
}

#[async_trait::async_trait]
impl VideoGenerationProvider for OpenRouterMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.trim().is_empty()
    }

    fn capabilities(&self, model: &str) -> VideoCapabilities {
        self.descriptor(model, MediaTask::TextToVideo)
    }

    async fn submit_video(
        &self,
        request: VideoGenerationRequest,
    ) -> MediaResult<Operation<VideoGenerationResult>> {
        validate_text(&request.model, &request.prompt)?;
        let mut warnings = request_warnings(&request.request_options);
        if matches!(request.mode, VideoGenerationMode::Extend) {
            return Err(MediaError::UnsupportedTask {
                provider: PROVIDER.to_string(),
                model: request.model,
                task: MediaTask::VideoExtend,
            });
        }
        if matches!(request.mode, VideoGenerationMode::Edit) {
            return Err(MediaError::UnsupportedTask {
                provider: PROVIDER.to_string(),
                model: request.model,
                task: MediaTask::VideoEdit,
            });
        }
        if request.mode == VideoGenerationMode::ImageToVideo && request.first_frame.is_none() {
            return Err(MediaError::InvalidRequest(
                "image-to-video requires first_frame".to_string(),
            ));
        }
        if request.mode == VideoGenerationMode::ReferenceToVideo
            && request.reference_images.is_empty()
        {
            return Err(MediaError::InvalidRequest(
                "reference-to-video requires reference_images".to_string(),
            ));
        }
        if request.count.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "count",
                "OpenRouter's video request has no output-count parameter",
            )?;
        }
        if request.negative_prompt.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "negative_prompt",
                "use a documented upstream option under provider.options",
            )?;
        }
        if request.output_format.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "output_format",
                "OpenRouter's video endpoint does not expose output format",
            )?;
        }
        if request.source_video.is_some() {
            return Err(MediaError::UnsupportedParameter {
                provider: PROVIDER.to_string(),
                parameter: "source_video".to_string(),
                reason: "OpenRouter does not document video input for generation".to_string(),
            });
        }

        let mut body = Map::new();
        body.insert("model".to_string(), json!(request.model));
        body.insert("prompt".to_string(), json!(request.prompt));
        if let Some(seed) = request.seed {
            body.insert("seed".to_string(), json!(seed));
        }
        if let Some(duration) = request.duration_secs {
            if duration.fract() != 0.0 || duration < 1.0 {
                return Err(MediaError::InvalidRequest(
                    "OpenRouter video duration must be a positive whole number of seconds"
                        .to_string(),
                ));
            }
            body.insert("duration".to_string(), json!(duration as u64));
        }
        if let Some(geometry) = request.geometry {
            match geometry {
                OutputGeometry::Dimensions { .. } => {
                    body.insert("size".to_string(), json!(geometry.as_api_string()));
                }
                OutputGeometry::AspectRatio { .. } => {
                    body.insert("aspect_ratio".to_string(), json!(geometry.as_api_string()));
                }
            }
        }
        let mut frames = Vec::new();
        if let Some(source) = request.first_frame.as_ref() {
            frames.push(video_image_reference(
                source,
                Some("first_frame"),
                request.request_options.max_source_bytes,
            )?);
        }
        if let Some(source) = request.last_frame.as_ref() {
            frames.push(video_image_reference(
                source,
                Some("last_frame"),
                request.request_options.max_source_bytes,
            )?);
        }
        if !frames.is_empty() {
            body.insert("frame_images".to_string(), Value::Array(frames));
        }
        if !request.reference_images.is_empty() {
            let references = request
                .reference_images
                .iter()
                .map(|source| {
                    video_image_reference(source, None, request.request_options.max_source_bytes)
                })
                .collect::<MediaResult<Vec<_>>>()?;
            body.insert("input_references".to_string(), Value::Array(references));
        }
        let extensions = openrouter_extensions(
            &request.provider_options,
            &["provider", "generate_audio", "callback_url"],
            &request.request_options,
            &mut warnings,
        )?;
        merge_extensions(&mut body, extensions)?;

        let key = self.key()?;
        let url = format!("{}/videos", self.api_base());
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        parse_video_operation(
            &value,
            &request.model,
            task_for_openrouter_video(request.mode),
            warnings,
        )
    }

    async fn poll_video(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<VideoGenerationResult>> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToVideo,
                MediaTask::ImageToVideo,
                MediaTask::ReferenceToVideo,
            ],
        )?;
        let key = self.key()?;
        let url = format!("{}/videos/{}", self.api_base(), handle.remote_id);
        let options = RequestOptions::default();
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .bearer_auth(&key)
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        parse_video_operation(&value, &handle.model, handle.task, handle.warnings.clone())
    }

    async fn cancel_video(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToVideo,
                MediaTask::ImageToVideo,
                MediaTask::ReferenceToVideo,
            ],
        )?;
        Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: "cancel".to_string(),
            reason: "OpenRouter does not document a video cancellation endpoint".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for OpenRouterMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.trim().is_empty()
    }

    fn capabilities(&self, model: &str) -> SpeechCapabilities {
        self.descriptor(model, MediaTask::TextToSpeech)
    }

    async fn submit_speech(
        &self,
        request: SpeechSynthesisRequest,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        let (body, mut warnings) = build_speech_body(&request)?;

        let key = self.key()?;
        let url = format!("{}/audio/speech", self.api_base());
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        shared::require_success(PROVIDER, &response)?;
        let content_type = response
            .headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or_else(|| speech_response_media_type(&body))
            .to_string();
        let generation_id = response
            .headers
            .get("x-generation-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let cost = match generation_id.as_deref() {
            Some(id) => {
                self.fetch_generation_cost(id, &key, &request.request_options)
                    .await
            }
            None => None,
        };
        if cost.is_none() {
            warnings.push(cost_unavailable_warning());
        }
        let usage = MediaUsage {
            line_items: vec![UsageLineItem {
                unit: UsageUnit::Characters,
                quantity: request.text.chars().count() as f64,
                cost,
                description: Some("speech input characters".to_string()),
            }],
            provider_reported_cost: cost,
            estimated_cost: None,
            currency: "USD".to_string(),
            metadata: generation_id
                .as_ref()
                .map_or(Value::Null, |id| json!({"generation_id":id})),
        };
        let bytes = response.body;
        Ok(Operation::completed(SpeechSynthesisResult {
            artifact: MediaArtifact {
                kind: MediaKind::Audio,
                media_type: content_type,
                size_bytes: Some(bytes.len() as u64),
                source: ArtifactSource::Inline(bytes),
                dimensions: None,
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: None,
                channels: None,
                expires_at: None,
                metadata: Value::Null,
            },
            usage: Some(usage),
            warnings,
            provider_metadata: generation_id.map_or(Value::Null, |id| json!({"generation_id":id})),
        }))
    }

    async fn poll_speech(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }

    async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }

    async fn stream_speech(&self, request: SpeechSynthesisRequest) -> MediaResult<SpeechStream> {
        let (body, _) = build_speech_body(&request)?;
        let key = self.key()?;
        let url = format!("{}/audio/speech", self.api_base());
        let response = shared::send_stream(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let media_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or_else(|| speech_response_media_type(&body))
            .to_string();
        let generation_id = response
            .headers()
            .get("x-generation-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        Ok(SpeechStream::new(media_type, generation_id, response))
    }
}

#[async_trait::async_trait]
impl TranscriptionProvider for OpenRouterMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }

    fn supports_model(&self, model: &str) -> bool {
        !model.trim().is_empty()
    }

    fn capabilities(&self, model: &str) -> TranscriptionCapabilities {
        self.descriptor(model, MediaTask::SpeechToText)
    }

    async fn submit_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        if request.model.trim().is_empty() {
            return Err(MediaError::InvalidRequest(
                "model must not be empty".to_string(),
            ));
        }
        let mut warnings = request_warnings(&request.request_options);
        if request.prompt.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "prompt",
                "OpenRouter's base64 transcription path has no portable prompt field",
            )?;
        }
        let (data, media_type) =
            shared::source_to_base64(&request.audio, request.request_options.max_source_bytes)?;
        let format = audio_format_from_media_type(&media_type)?;
        let mut body = Map::new();
        body.insert("model".to_string(), json!(request.model));
        body.insert(
            "input_audio".to_string(),
            json!({"data":data,"format":format}),
        );
        if let Some(language) = request.language.as_ref() {
            body.insert("language".to_string(), json!(language));
        }
        if !request.timestamp_granularities.is_empty() {
            body.insert("response_format".to_string(), json!("verbose_json"));
            insert_timestamp_granularities(&mut body, &request.timestamp_granularities);
        }
        let extensions = openrouter_extensions(
            &request.provider_options,
            &["provider"],
            &request.request_options,
            &mut warnings,
        )?;
        merge_extensions(&mut body, extensions)?;
        let key = self.key()?;
        let url = format!("{}/audio/transcriptions", self.api_base());
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body)
        })
        .await?;
        let generation_id = response
            .headers
            .get("x-generation-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let value = shared::parse_json(PROVIDER, &response)?;
        Ok(Operation::completed(parse_transcription_response(
            &value,
            warnings,
            generation_id,
        )?))
    }

    async fn poll_transcription(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }

    async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: handle.model.clone(),
            task: handle.task,
        })
    }
}

fn parse_image_response(
    value: &Value,
    expected_outputs: Option<u32>,
    mut warnings: Vec<ProviderWarning>,
) -> MediaResult<ImageGenerationResult> {
    let data =
        value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "missing image data array".to_string(),
            })?;
    let mut artifacts = Vec::with_capacity(data.len());
    for item in data {
        let encoded = item
            .get("b64_json")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "image item is missing b64_json".to_string(),
            })?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
        let media_type = item
            .get("media_type")
            .and_then(Value::as_str)
            .unwrap_or("image/png")
            .to_string();
        artifacts.push(MediaArtifact {
            kind: MediaKind::Image,
            media_type,
            size_bytes: Some(bytes.len() as u64),
            source: ArtifactSource::Inline(bytes),
            dimensions: None,
            duration_secs: None,
            frame_rate: None,
            sample_rate_hz: None,
            channels: None,
            expires_at: None,
            metadata: Value::Null,
        });
    }
    if artifacts.is_empty() {
        return Err(MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "image response contained no artifacts".to_string(),
        });
    }
    if let Some(expected) = expected_outputs.filter(|expected| artifacts.len() < *expected as usize)
    {
        warnings.push(ProviderWarning {
            code: WarningCode::PartialOutput,
            message: format!(
                "OpenRouter returned {} artifact(s), fewer than the requested {expected}",
                artifacts.len()
            ),
            parameter: Some("count".to_string()),
            provider_metadata: Value::Null,
        });
    }
    let usage = parse_usage(value.get("usage"));
    if usage
        .as_ref()
        .and_then(|usage| usage.provider_reported_cost)
        .is_none()
    {
        warnings.push(cost_unavailable_warning());
    }
    Ok(ImageGenerationResult {
        artifacts,
        usage,
        warnings,
        safety: SafetyReport::default(),
        provider_metadata: value
            .get("created")
            .map_or(Value::Null, |created| json!({"created":created})),
    })
}

fn parse_video_operation(
    value: &Value,
    model: &str,
    task: MediaTask,
    mut warnings: Vec<ProviderWarning>,
) -> MediaResult<Operation<VideoGenerationResult>> {
    let id =
        value
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "video response is missing id".to_string(),
            })?;
    let native_status = value
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("pending");
    let status = match native_status {
        "pending" | "queued" => OperationStatus::Queued,
        "in_progress" | "processing" => OperationStatus::Running,
        "completed" | "succeeded" => OperationStatus::Succeeded,
        "failed" => OperationStatus::Failed,
        "cancelled" | "canceled" => OperationStatus::Cancelled,
        "expired" => OperationStatus::Expired,
        other => {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: format!("unknown video status '{other}'"),
            })
        }
    };
    let handle = JobHandle {
        provider: PROVIDER.to_string(),
        model: model.to_string(),
        task,
        remote_id: id.to_string(),
        cost_estimate: None,
        warnings: warnings.clone(),
        expected_outputs: None,
    };
    let error = if matches!(
        status,
        OperationStatus::Failed | OperationStatus::Cancelled | OperationStatus::Expired
    ) {
        Some(GenerationFailure {
            category: match status {
                OperationStatus::Cancelled => FailureCategory::Cancelled,
                OperationStatus::Expired => FailureCategory::Expired,
                _ => FailureCategory::RemoteJob,
            },
            message: value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("video generation failed")
                .to_string(),
            provider_code: Some(native_status.to_string()),
            http_status: None,
            request_id: None,
            provider_metadata: Value::Null,
        })
    } else {
        None
    };
    let result = if status == OperationStatus::Succeeded {
        let urls = value
            .get("unsigned_urls")
            .and_then(Value::as_array)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "completed video response is missing unsigned_urls".to_string(),
            })?;
        let artifacts = urls
            .iter()
            .filter_map(Value::as_str)
            .map(|url| MediaArtifact {
                kind: MediaKind::Video,
                media_type: shared::media_type_from_url(url, "video/mp4"),
                source: ArtifactSource::Url(url.to_string()),
                size_bytes: None,
                dimensions: None,
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: None,
                channels: None,
                expires_at: None,
                metadata: json!({"provider":PROVIDER,"requires_auth":true}),
            })
            .collect::<Vec<_>>();
        if artifacts.is_empty() {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "completed video response contained no artifact URLs".to_string(),
            });
        }
        let usage = parse_usage(value.get("usage"));
        if usage
            .as_ref()
            .and_then(|usage| usage.provider_reported_cost)
            .is_none()
        {
            warnings.push(cost_unavailable_warning());
        }
        Some(VideoGenerationResult {
            artifacts,
            usage,
            warnings,
            safety: SafetyReport::default(),
            provider_metadata: value
                .get("generation_id")
                .map_or(Value::Null, |id| json!({"generation_id":id})),
        })
    } else {
        None
    };
    Ok(Operation {
        handle: Some(handle),
        status,
        progress: (status == OperationStatus::Succeeded).then_some(1.0),
        result,
        error,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: json!({
            "native_status":native_status,
            "polling_url":value.get("polling_url"),
            "generation_id":value.get("generation_id")
        }),
    })
}

fn task_for_openrouter_video(mode: VideoGenerationMode) -> MediaTask {
    match mode {
        VideoGenerationMode::TextToVideo => MediaTask::TextToVideo,
        VideoGenerationMode::ImageToVideo => MediaTask::ImageToVideo,
        VideoGenerationMode::ReferenceToVideo => MediaTask::ReferenceToVideo,
        VideoGenerationMode::Extend => MediaTask::VideoExtend,
        VideoGenerationMode::Edit => MediaTask::VideoEdit,
    }
}

fn parse_transcription_response(
    value: &Value,
    mut warnings: Vec<ProviderWarning>,
    generation_id: Option<String>,
) -> MediaResult<TranscriptionResult> {
    let text =
        value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "transcription response is missing text".to_string(),
            })?;
    let usage = parse_usage(value.get("usage"));
    if usage
        .as_ref()
        .and_then(|usage| usage.provider_reported_cost)
        .is_none()
    {
        warnings.push(cost_unavailable_warning());
    }
    Ok(TranscriptionResult {
        text: text.to_string(),
        language: value
            .get("language")
            .and_then(Value::as_str)
            .map(str::to_string),
        duration_secs: value
            .get("duration")
            .and_then(Value::as_f64)
            .or_else(|| value.pointer("/usage/seconds").and_then(Value::as_f64)),
        segments: parse_segments(value.get("segments")),
        words: parse_words(value.get("words")),
        usage,
        warnings,
        provider_metadata: generation_id.map_or(Value::Null, |id| json!({"generation_id":id})),
    })
}

fn parse_usage(value: Option<&Value>) -> Option<MediaUsage> {
    let value = value?;
    let mut line_items = Vec::new();
    for (field, unit) in [
        ("prompt_tokens", UsageUnit::InputTokens),
        ("input_tokens", UsageUnit::InputTokens),
        ("completion_tokens", UsageUnit::OutputTokens),
        ("output_tokens", UsageUnit::OutputTokens),
        ("seconds", UsageUnit::AudioSeconds),
    ] {
        if let Some(quantity) = value.get(field).and_then(Value::as_f64) {
            if !line_items
                .iter()
                .any(|item: &UsageLineItem| item.unit == unit)
            {
                line_items.push(UsageLineItem {
                    unit,
                    quantity,
                    cost: None,
                    description: Some(field.to_string()),
                });
            }
        }
    }
    Some(MediaUsage {
        line_items,
        provider_reported_cost: value.get("cost").and_then(Value::as_f64),
        estimated_cost: None,
        currency: "USD".to_string(),
        metadata: value.clone(),
    })
}

fn parse_segments(value: Option<&Value>) -> Vec<TranscriptSegment> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            Some(TranscriptSegment {
                start_secs: item.get("start")?.as_f64()?,
                end_secs: item.get("end")?.as_f64()?,
                text: item.get("text")?.as_str()?.to_string(),
                speaker: item
                    .get("speaker")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

fn parse_words(value: Option<&Value>) -> Vec<TranscriptWord> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            Some(TranscriptWord {
                start_secs: item.get("start")?.as_f64()?,
                end_secs: item.get("end")?.as_f64()?,
                word: item
                    .get("word")
                    .or_else(|| item.get("text"))?
                    .as_str()?
                    .to_string(),
                speaker: item
                    .get("speaker")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect()
}

fn video_image_reference(
    source: &MediaSource,
    frame_type: Option<&str>,
    maximum_bytes: usize,
) -> MediaResult<Value> {
    let url = shared::source_to_data_or_uri(source, maximum_bytes)?;
    let mut value = json!({"type":"image_url","image_url":{"url":url}});
    if let Some(frame_type) = frame_type {
        value["frame_type"] = json!(frame_type);
    }
    Ok(value)
}

fn merge_extensions(
    body: &mut Map<String, Value>,
    extensions: Map<String, Value>,
) -> MediaResult<()> {
    for (key, value) in extensions {
        if body.contains_key(&key) {
            return Err(MediaError::InvalidRequest(format!(
                "OpenRouter field '{key}' was supplied both portably and in provider options"
            )));
        }
        body.insert(key, value);
    }
    Ok(())
}

fn unsupported(
    options: &RequestOptions,
    warnings: &mut Vec<ProviderWarning>,
    parameter: &str,
    reason: &str,
) -> MediaResult<()> {
    match options.unsupported_parameter_policy {
        UnsupportedParameterPolicy::Error => Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: parameter.to_string(),
            reason: reason.to_string(),
        }),
        UnsupportedParameterPolicy::WarnAndDrop => {
            warnings.push(ProviderWarning {
                code: WarningCode::UnsupportedParameterDropped,
                message: reason.to_string(),
                parameter: Some(parameter.to_string()),
                provider_metadata: Value::Null,
            });
            Ok(())
        }
    }
}

fn validate_text(model: &str, text: &str) -> MediaResult<()> {
    if model.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "model must not be empty".to_string(),
        ));
    }
    if text.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "prompt/text must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn audio_format_from_media_type(media_type: &str) -> MediaResult<&'static str> {
    match media_type.split(';').next().unwrap_or(media_type) {
        "audio/wav" | "audio/x-wav" => Ok("wav"),
        "audio/mpeg" | "audio/mp3" => Ok("mp3"),
        "audio/flac" => Ok("flac"),
        "audio/mp4" | "audio/m4a" => Ok("m4a"),
        "audio/ogg" => Ok("ogg"),
        "audio/webm" => Ok("webm"),
        "audio/aac" => Ok("aac"),
        other => Err(MediaError::InvalidRequest(format!(
            "unsupported transcription media type '{other}'"
        ))),
    }
}

fn build_speech_body(
    request: &SpeechSynthesisRequest,
) -> MediaResult<(Map<String, Value>, Vec<ProviderWarning>)> {
    validate_text(&request.model, &request.text)?;
    if request.voice.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "speech voice must not be empty".to_string(),
        ));
    }
    let mut warnings = request_warnings(&request.request_options);
    let supported_format = matches!(request.output.format, AudioFormat::Mp3 | AudioFormat::Pcm);
    if !supported_format {
        unsupported(
            &request.request_options,
            &mut warnings,
            "output.format",
            "OpenRouter documents mp3 and pcm for the speech endpoint",
        )?;
    }
    if request.language.is_some() {
        unsupported(
            &request.request_options,
            &mut warnings,
            "language",
            "OpenRouter's speech endpoint does not expose a portable language field",
        )?;
    }
    if request.output.sample_rate_hz.is_some() || request.output.channels.is_some() {
        unsupported(
            &request.request_options,
            &mut warnings,
            "output sample rate/channels",
            "the endpoint controls these through the selected model and format",
        )?;
    }
    let mut body = Map::new();
    body.insert("model".to_string(), json!(request.model));
    body.insert("input".to_string(), json!(request.text));
    body.insert("voice".to_string(), json!(request.voice));
    if supported_format {
        body.insert(
            "response_format".to_string(),
            json!(request.output.format.as_str()),
        );
    }
    if let Some(speed) = request.speed {
        if speed <= 0.0 {
            return Err(MediaError::InvalidRequest(
                "speech speed must be greater than zero".to_string(),
            ));
        }
        body.insert("speed".to_string(), json!(speed));
    }
    let mut extensions = openrouter_extensions(
        &request.provider_options,
        &["provider"],
        &request.request_options,
        &mut warnings,
    )?;
    if let Some(instructions) = request.instructions.as_ref() {
        if !request.model.starts_with("openai/") {
            unsupported(
                &request.request_options,
                &mut warnings,
                "instructions",
                "portable speech instructions are verified only for OpenAI TTS routes; use provider.options for another upstream",
            )?;
        } else {
            let provider = extensions
                .entry("provider".to_string())
                .or_insert_with(|| json!({"options":{}}));
            let options = provider
                .pointer_mut("/options")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| {
                    MediaError::InvalidRequest(
                        "provider_options.openrouter.provider.options must be an object"
                            .to_string(),
                    )
                })?;
            let upstream_options = options
                .entry("openai".to_string())
                .or_insert_with(|| json!({}));
            let upstream_options = upstream_options.as_object_mut().ok_or_else(|| {
                MediaError::InvalidRequest("provider.options.openai must be an object".to_string())
            })?;
            if upstream_options.contains_key("instructions") {
                return Err(MediaError::InvalidRequest(
                    "instructions was supplied both portably and in provider options".to_string(),
                ));
            }
            upstream_options.insert("instructions".to_string(), json!(instructions));
        }
    }
    merge_extensions(&mut body, extensions)?;
    Ok((body, warnings))
}

fn speech_response_media_type(body: &Map<String, Value>) -> &'static str {
    if body.get("response_format").and_then(Value::as_str) == Some("mp3") {
        "audio/mpeg"
    } else {
        "audio/pcm"
    }
}

fn insert_timestamp_granularities(
    body: &mut Map<String, Value>,
    granularities: &[TimestampGranularity],
) {
    body.insert(
        "timestamp_granularities".to_string(),
        Value::Array(
            granularities
                .iter()
                .map(|granularity| {
                    json!(match granularity {
                        TimestampGranularity::Segment => "segment",
                        TimestampGranularity::Word => "word",
                    })
                })
                .collect(),
        ),
    );
}

fn openrouter_extensions(
    options: &ProviderOptions,
    allowed: &[&str],
    controls: &RequestOptions,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<Map<String, Value>> {
    let mut extensions = shared::provider_options(options, PROVIDER)?;
    let typed: OpenRouterMediaOptions = serde_json::from_value(Value::Object(extensions.clone()))
        .map_err(|error| {
        MediaError::InvalidRequest(format!("invalid OpenRouter provider options: {error}"))
    })?;
    validate_openrouter_options(&typed)?;
    let unsupported_keys = extensions
        .keys()
        .filter(|key| !allowed.contains(&key.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    for key in unsupported_keys {
        unsupported(
            controls,
            warnings,
            &format!("provider_options.openrouter.{key}"),
            "this OpenRouter media task does not document the option",
        )?;
        extensions.remove(&key);
    }
    Ok(extensions)
}

fn validate_openrouter_options(options: &OpenRouterMediaOptions) -> MediaResult<()> {
    if options.output_compression.is_some_and(|value| value > 100) {
        return Err(MediaError::InvalidRequest(
            "OpenRouter output_compression must be between 0 and 100".to_string(),
        ));
    }
    if options
        .quality
        .as_deref()
        .is_some_and(|value| !matches!(value, "auto" | "low" | "medium" | "high"))
    {
        return Err(MediaError::InvalidRequest(
            "OpenRouter quality must be auto, low, medium, or high".to_string(),
        ));
    }
    if options
        .background
        .as_deref()
        .is_some_and(|value| !matches!(value, "auto" | "transparent" | "opaque"))
    {
        return Err(MediaError::InvalidRequest(
            "OpenRouter background must be auto, transparent, or opaque".to_string(),
        ));
    }
    if let Some(callback_url) = options.callback_url.as_deref() {
        shared::validate_https_url(callback_url, "OpenRouter callback_url")?;
    }
    if options
        .provider
        .as_ref()
        .is_some_and(|value| !value.is_object())
    {
        return Err(MediaError::InvalidRequest(
            "OpenRouter provider routing options must be an object".to_string(),
        ));
    }
    Ok(())
}

fn openrouter_options_schema(task: MediaTask) -> Value {
    let mut properties = Map::new();
    properties.insert("provider".to_string(), json!({"type":"object"}));
    match task {
        MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::ImageVariation => {
            properties.insert("quality".to_string(), json!({"type":"string"}));
            properties.insert("background".to_string(), json!({"type":"string"}));
            properties.insert(
                "output_compression".to_string(),
                json!({"type":"integer","minimum":0,"maximum":100}),
            );
        }
        MediaTask::TextToVideo | MediaTask::ImageToVideo | MediaTask::ReferenceToVideo => {
            properties.insert("generate_audio".to_string(), json!({"type":"boolean"}));
            properties.insert(
                "callback_url".to_string(),
                json!({"type":"string","format":"uri"}),
            );
        }
        _ => {}
    }
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":properties
    })
}

fn cost_unavailable_warning() -> ProviderWarning {
    ProviderWarning {
        code: WarningCode::CostUnavailable,
        message:
            "OpenRouter returned media successfully but generation cost metadata was unavailable"
                .to_string(),
        parameter: None,
        provider_metadata: Value::Null,
    }
}

fn request_warnings(options: &RequestOptions) -> Vec<ProviderWarning> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy == UnsupportedParameterPolicy::WarnAndDrop
    {
        vec![ProviderWarning {
            code: WarningCode::UnsupportedParameterDropped,
            message: "OpenRouter media endpoints do not document request idempotency keys"
                .to_string(),
            parameter: Some("request_options.idempotency_key".to_string()),
            provider_metadata: Value::Null,
        }]
    } else {
        Vec::new()
    }
}

fn execution(
    immediate: CapabilitySupport,
    persistent_jobs: CapabilitySupport,
    polling: CapabilitySupport,
    webhooks: CapabilitySupport,
) -> ExecutionCapabilities {
    ExecutionCapabilities {
        immediate,
        persistent_jobs,
        polling,
        cancellation: CapabilitySupport::Unsupported,
        progress: CapabilitySupport::Unsupported,
        webhooks,
        binary_streaming: CapabilitySupport::Unknown,
        resumable: persistent_jobs,
    }
}

#[allow(clippy::too_many_arguments)]
fn parameters(
    count: CapabilitySupport,
    seed: CapabilitySupport,
    dimensions: CapabilitySupport,
    aspect_ratio: CapabilitySupport,
    duration: CapabilitySupport,
    mask: CapabilitySupport,
    negative_prompt: CapabilitySupport,
    output_format: CapabilitySupport,
) -> ParameterCapabilities {
    ParameterCapabilities {
        count,
        seed,
        dimensions,
        aspect_ratio,
        duration,
        mask,
        negative_prompt,
        output_format,
    }
}

fn unknown_parameters() -> ParameterCapabilities {
    parameters(
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
        CapabilitySupport::Unknown,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_image_fixture_and_actual_cost() {
        let fixture = json!({
            "created": 1748372400,
            "data": [{"b64_json": "aW1hZ2U=", "media_type": "image/png"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "cost": 0.04}
        });
        let result = parse_image_response(&fixture, Some(1), Vec::new()).unwrap();
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.artifacts[0].size_bytes, Some(5));
        assert_eq!(
            result.usage.and_then(|usage| usage.provider_reported_cost),
            Some(0.04)
        );
    }

    #[test]
    fn fewer_images_are_a_partial_result_not_a_failure() {
        let fixture = json!({
            "data": [{"b64_json": "aW1hZ2U=", "media_type": "image/png"}],
            "usage": {"cost": 0.04}
        });
        let result = parse_image_response(&fixture, Some(2), Vec::new()).unwrap();
        assert_eq!(result.artifacts.len(), 1);
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.code == WarningCode::PartialOutput));
    }

    #[test]
    fn parses_completed_video_without_downloading_it() {
        let fixture = json!({
            "id":"job-1",
            "status":"completed",
            "generation_id":"gen-1",
            "unsigned_urls":["https://example.test/out.mp4"],
            "usage":{"cost":0.5}
        });
        let operation = parse_video_operation(
            &fixture,
            "google/veo-3.1",
            MediaTask::TextToVideo,
            Vec::new(),
        )
        .unwrap();
        assert_eq!(operation.status, OperationStatus::Succeeded);
        let result = operation.result.unwrap();
        assert!(matches!(result.artifacts[0].source, ArtifactSource::Url(_)));
        assert_eq!(result.usage.unwrap().provider_reported_cost, Some(0.5));
    }

    #[test]
    fn parses_transcription_usage_dimensions() {
        let fixture = json!({
            "text":"hello",
            "usage":{"seconds":9.2,"input_tokens":83,"output_tokens":30,"cost":0.000508}
        });
        let result = parse_transcription_response(&fixture, Vec::new(), None).unwrap();
        assert_eq!(result.duration_secs, Some(9.2));
        assert_eq!(result.usage.unwrap().provider_reported_cost, Some(0.000508));
    }

    #[test]
    fn missing_provider_cost_is_explicitly_warned() {
        let image = json!({
            "data": [{"b64_json": "aW1hZ2U=", "media_type": "image/png"}]
        });
        let result = parse_image_response(&image, None, Vec::new()).unwrap();
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.code == WarningCode::CostUnavailable));

        let transcript = json!({"text":"hello"});
        let result = parse_transcription_response(&transcript, Vec::new(), None).unwrap();
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.code == WarningCode::CostUnavailable));
    }

    #[test]
    fn transcription_accepts_all_documented_audio_types() {
        for (media_type, format) in [
            ("audio/wav", "wav"),
            ("audio/mpeg", "mp3"),
            ("audio/flac", "flac"),
            ("audio/mp4", "m4a"),
            ("audio/ogg", "ogg"),
            ("audio/webm", "webm"),
            ("audio/aac", "aac"),
        ] {
            assert_eq!(audio_format_from_media_type(media_type).unwrap(), format);
        }
    }

    #[test]
    fn capability_schema_is_task_specific() {
        let provider = OpenRouterMediaProvider::new();
        let image = ImageGenerationProvider::capabilities(&provider, "image/model");
        let speech = SpeechSynthesisProvider::capabilities(&provider, "speech/model");
        assert!(image
            .provider_options_schema
            .as_ref()
            .and_then(|schema| schema.pointer("/properties/quality"))
            .is_some());
        assert!(speech
            .provider_options_schema
            .as_ref()
            .and_then(|schema| schema.pointer("/properties/quality"))
            .is_none());
    }

    #[test]
    fn provider_options_cannot_override_portable_fields() {
        let mut body = Map::new();
        body.insert("model".to_string(), json!("model"));
        let mut extension = Map::new();
        extension.insert("model".to_string(), json!("other"));
        assert!(merge_extensions(&mut body, extension).is_err());
    }

    #[test]
    fn typed_options_validate_compression_and_webhook() {
        let invalid_compression = OpenRouterMediaOptions {
            output_compression: Some(101),
            ..OpenRouterMediaOptions::default()
        };
        assert!(invalid_compression.into_provider_options().is_err());
        let invalid_webhook = OpenRouterMediaOptions {
            callback_url: Some("http://example.test/hook".to_string()),
            ..OpenRouterMediaOptions::default()
        };
        assert!(invalid_webhook.into_provider_options().is_err());
    }

    #[test]
    fn speech_format_obeys_unsupported_parameter_policy() {
        let mut request =
            SpeechSynthesisRequest::new("hello", "alloy").with_model("openai/gpt-4o-mini-tts");
        request.output.format = AudioFormat::Wav;
        assert!(matches!(
            build_speech_body(&request),
            Err(MediaError::UnsupportedParameter { .. })
        ));

        request.request_options.unsupported_parameter_policy =
            UnsupportedParameterPolicy::WarnAndDrop;
        let (body, warnings) = build_speech_body(&request).unwrap();
        assert!(!body.contains_key("response_format"));
        assert_eq!(speech_response_media_type(&body), "audio/pcm");
        assert!(warnings
            .iter()
            .any(|warning| warning.code == WarningCode::UnsupportedParameterDropped));
    }

    #[test]
    fn transcription_timestamp_fields_match_the_json_contract() {
        let mut body = Map::new();
        body.insert("response_format".to_string(), json!("verbose_json"));
        insert_timestamp_granularities(
            &mut body,
            &[TimestampGranularity::Segment, TimestampGranularity::Word],
        );
        assert_eq!(
            body.get("timestamp_granularities"),
            Some(&json!(["segment", "word"]))
        );
    }

    #[test]
    fn provider_extensions_are_strict_per_task() {
        let options = OpenRouterMediaOptions {
            quality: Some("high".to_string()),
            ..OpenRouterMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        assert!(openrouter_extensions(
            &options,
            &["provider"],
            &RequestOptions::default(),
            &mut Vec::new()
        )
        .is_err());

        let controls = RequestOptions {
            unsupported_parameter_policy: UnsupportedParameterPolicy::WarnAndDrop,
            ..RequestOptions::default()
        };
        let mut warnings = Vec::new();
        let extensions =
            openrouter_extensions(&options, &["provider"], &controls, &mut warnings).unwrap();
        assert!(extensions.is_empty());
        assert_eq!(warnings.len(), 1);
    }
}
