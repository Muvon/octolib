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

//! fal.ai queue adapter for model-specific image, video, speech, and
//! transcription endpoints.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};

const PROVIDER: &str = "fal";
const API_KEY_ENV: &str = "FAL_KEY";
const API_BASE_ENV: &str = "FAL_API_URL";
const API_BASE: &str = "https://queue.fal.run";

#[derive(Debug, Clone, Default)]
pub struct FalMediaProvider;

/// Stable fal adapter controls. Model-native fields belong in `input`;
/// `field_map` maps portable semantic names such as `prompt`, `source_image`,
/// or `duration` to the selected endpoint's field names.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FalMediaOptions {
    #[serde(default)]
    pub input: Map<String, Value>,
    #[serde(default)]
    pub field_map: HashMap<String, String>,
    pub webhook: Option<String>,
    pub cost_estimate: Option<CostEstimate>,
}

impl FalMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_fal_options(&self)?;
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
        Ok(options)
    }
}

impl FalMediaProvider {
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
        let (inputs, outputs) = match task {
            MediaTask::TextToImage
            | MediaTask::ImageEdit
            | MediaTask::Inpainting
            | MediaTask::ImageVariation => {
                (vec![Modality::Text, Modality::Image], vec![Modality::Image])
            }
            MediaTask::TextToVideo
            | MediaTask::ImageToVideo
            | MediaTask::ReferenceToVideo
            | MediaTask::VideoExtend
            | MediaTask::VideoEdit => (
                vec![Modality::Text, Modality::Image, Modality::Video],
                vec![Modality::Video, Modality::Audio],
            ),
            MediaTask::TextToSpeech => (vec![Modality::Text], vec![Modality::Audio]),
            MediaTask::SpeechToText => (vec![Modality::Audio], vec![Modality::Text]),
            _ => (Vec::new(), Vec::new()),
        };
        MediaModelDescriptor {
            provider: PROVIDER.to_string(),
            model: model.to_string(),
            tasks: vec![task],
            input_modalities: inputs,
            output_modalities: outputs,
            execution: ExecutionCapabilities {
                immediate: CapabilitySupport::Unsupported,
                persistent_jobs: CapabilitySupport::Supported,
                polling: CapabilitySupport::Supported,
                cancellation: CapabilitySupport::Supported,
                progress: CapabilitySupport::Unknown,
                webhooks: CapabilitySupport::Supported,
                binary_streaming: CapabilitySupport::Unsupported,
                resumable: CapabilitySupport::Supported,
            },
            parameters: ParameterCapabilities {
                count: CapabilitySupport::Unknown,
                seed: CapabilitySupport::Unknown,
                dimensions: CapabilitySupport::Unknown,
                aspect_ratio: CapabilitySupport::Unknown,
                duration: CapabilitySupport::Unknown,
                mask: CapabilitySupport::Unknown,
                negative_prompt: CapabilitySupport::Unknown,
                output_format: CapabilitySupport::Unknown,
            },
            limits: MediaLimits::default(),
            provider_options_schema: Some(fal_options_schema()),
        }
    }

    async fn submit_queue(
        &self,
        model: &str,
        input: Map<String, Value>,
        options: &FalMediaOptions,
        request_options: &RequestOptions,
    ) -> MediaResult<Value> {
        validate_model(model)?;
        let url = format!("{}/{model}", self.api_base());
        let key = self.key()?;
        let webhook = options.webhook.clone();
        let body = Value::Object(input);
        let response = shared::send(PROVIDER, request_options, || {
            let mut builder = crate::llm::providers::shared::http_client()
                .post(&url)
                .header("Authorization", format!("Key {key}"))
                .json(&body);
            if let Some(webhook) = webhook.as_ref() {
                builder = builder.query(&[("fal_webhook", webhook)]);
            }
            builder
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        if value.get("request_id").and_then(Value::as_str).is_none() {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "queue submission returned no request_id".to_string(),
            });
        }
        Ok(value)
    }

    async fn queue_status(&self, handle: &JobHandle) -> MediaResult<Value> {
        let key = self.key()?;
        let url = format!(
            "{}/{}/requests/{}/status",
            self.api_base(),
            app_alias(&handle.model)?,
            handle.remote_id
        );
        let options = RequestOptions::default();
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .header("Authorization", format!("Key {key}"))
        })
        .await?;
        shared::parse_json(PROVIDER, &response)
    }

    /// Fetch the stored outcome of a COMPLETED queue request. Account and
    /// infrastructure errors propagate; other error statuses are the stored
    /// result of the failed generation itself.
    async fn fetch_result(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Result<Value, GenerationFailure>> {
        let key = self.key()?;
        let url = format!(
            "{}/{}/requests/{}",
            self.api_base(),
            app_alias(&handle.model)?,
            handle.remote_id
        );
        let options = RequestOptions::default();
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .header("Authorization", format!("Key {key}"))
        })
        .await?;
        if response.status.is_success() {
            return Ok(Ok(shared::parse_json(PROVIDER, &response)?));
        }
        if matches!(response.status.as_u16(), 401 | 402 | 403 | 429)
            || response.status.is_server_error()
        {
            shared::require_success(PROVIDER, &response)?;
        }
        Ok(Err(GenerationFailure {
            category: if response.status.as_u16() == 422 {
                FailureCategory::InvalidInput
            } else {
                FailureCategory::RemoteJob
            },
            message: failure_message(&response.body),
            provider_code: None,
            http_status: Some(response.status.as_u16()),
            request_id: Some(handle.remote_id.clone()),
            provider_metadata: Value::Null,
        }))
    }

    async fn cancel_request(&self, handle: &JobHandle) -> MediaResult<()> {
        let key = self.key()?;
        let url = format!(
            "{}/{}/requests/{}/cancel",
            self.api_base(),
            app_alias(&handle.model)?,
            handle.remote_id
        );
        let options = RequestOptions::default();
        let response = shared::send(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .put(&url)
                .header("Authorization", format!("Key {key}"))
        })
        .await?;
        shared::require_success(PROVIDER, &response)
    }
}

#[async_trait::async_trait]
impl ImageGenerationProvider for FalMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        validate_model(model).is_ok()
    }
    fn capabilities(&self, model: &str) -> ImageCapabilities {
        self.descriptor(model, MediaTask::TextToImage)
    }

    async fn submit_image(
        &self,
        request: ImageGenerationRequest,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        validate_nonempty(&request.model, &request.prompt)?;
        let mut options = parse_options(&request.provider_options)?;
        let warnings = request_warnings(&request.request_options);
        insert_semantic(&mut options, "prompt", "prompt", json!(request.prompt))?;
        if !request.source_images.is_empty() {
            let sources = request
                .source_images
                .iter()
                .map(|source| {
                    shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)
                })
                .collect::<MediaResult<Vec<_>>>()?;
            if sources.len() == 1 {
                insert_semantic(&mut options, "source_image", "image_url", json!(sources[0]))?;
            } else {
                insert_semantic(&mut options, "source_images", "image_urls", json!(sources))?;
            }
        }
        if let Some(mask) = request.mask.as_ref() {
            let mask =
                shared::source_to_data_or_uri(mask, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "mask", "mask_url", json!(mask))?;
        }
        if let Some(count) = request.count {
            insert_semantic(&mut options, "count", "num_images", json!(count))?;
        }
        if let Some(seed) = request.seed {
            insert_semantic(&mut options, "seed", "seed", json!(seed))?;
        }
        if let Some(negative) = request.negative_prompt.as_ref() {
            insert_semantic(
                &mut options,
                "negative_prompt",
                "negative_prompt",
                json!(negative),
            )?;
        }
        if let Some(format) = request.output_format {
            insert_semantic(
                &mut options,
                "output_format",
                "output_format",
                json!(format.as_str()),
            )?;
        }
        if let Some(geometry) = request.geometry {
            insert_geometry(&mut options, geometry)?;
        }
        if request.mode == ImageGenerationMode::Inpaint && request.mask.is_none() {
            return Err(MediaError::InvalidRequest(
                "inpainting requires a mask".to_string(),
            ));
        }
        if matches!(
            request.mode,
            ImageGenerationMode::Edit | ImageGenerationMode::Variation
        ) && request.source_images.is_empty()
        {
            return Err(MediaError::InvalidRequest(
                "image edit/variation requires a source image".to_string(),
            ));
        }
        let task = task_for_image_mode(request.mode);
        let value = self
            .submit_queue(
                &request.model,
                options.input.clone(),
                &options,
                &request.request_options,
            )
            .await?;
        Ok(queued_operation(
            &request.model,
            task,
            &value,
            options.cost_estimate,
            warnings,
            request.count,
        )?)
    }

    async fn poll_image(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToImage,
                MediaTask::ImageEdit,
                MediaTask::Inpainting,
                MediaTask::ImageVariation,
            ],
        )?;
        let status_value = self.queue_status(handle).await?;
        let status = queue_state(&status_value)?;
        if status != OperationStatus::Succeeded {
            return Ok(non_terminal_operation(handle, status, &status_value));
        }
        let mut warnings = handle.warnings.clone();
        match self.fetch_result(handle).await? {
            Ok(payload) => {
                let result = parse_image_result(
                    &payload,
                    &status_value,
                    handle.cost_estimate.as_ref(),
                    handle.expected_outputs,
                    &mut warnings,
                )?;
                Ok(succeeded_operation(handle, result, &payload))
            }
            Err(failure) => Ok(failed_operation(handle, failure)),
        }
    }

    async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToImage,
                MediaTask::ImageEdit,
                MediaTask::Inpainting,
                MediaTask::ImageVariation,
            ],
        )?;
        self.cancel_request(handle).await
    }
}

#[async_trait::async_trait]
impl VideoGenerationProvider for FalMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        validate_model(model).is_ok()
    }
    fn capabilities(&self, model: &str) -> VideoCapabilities {
        self.descriptor(model, MediaTask::TextToVideo)
    }

    async fn submit_video(
        &self,
        request: VideoGenerationRequest,
    ) -> MediaResult<Operation<VideoGenerationResult>> {
        validate_nonempty(&request.model, &request.prompt)?;
        let mut options = parse_options(&request.provider_options)?;
        let warnings = request_warnings(&request.request_options);
        insert_semantic(&mut options, "prompt", "prompt", json!(request.prompt))?;
        if let Some(source) = request.first_frame.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "first_frame", "image_url", json!(source))?;
        }
        if let Some(source) = request.last_frame.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "last_frame", "end_image_url", json!(source))?;
        }
        if !request.reference_images.is_empty() {
            let sources = request
                .reference_images
                .iter()
                .map(|source| {
                    shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)
                })
                .collect::<MediaResult<Vec<_>>>()?;
            insert_semantic(
                &mut options,
                "reference_images",
                "reference_image_urls",
                json!(sources),
            )?;
        }
        if let Some(source) = request.source_video.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "source_video", "video_url", json!(source))?;
        }
        if let Some(count) = request.count {
            insert_semantic(&mut options, "count", "num_videos", json!(count))?;
        }
        if let Some(seed) = request.seed {
            insert_semantic(&mut options, "seed", "seed", json!(seed))?;
        }
        if let Some(duration) = request.duration_secs {
            if duration <= 0.0 {
                return Err(MediaError::InvalidRequest(
                    "video duration must be positive".to_string(),
                ));
            }
            insert_semantic(&mut options, "duration", "duration", json!(duration))?;
        }
        if let Some(negative) = request.negative_prompt.as_ref() {
            insert_semantic(
                &mut options,
                "negative_prompt",
                "negative_prompt",
                json!(negative),
            )?;
        }
        if let Some(format) = request.output_format {
            insert_semantic(
                &mut options,
                "output_format",
                "output_format",
                json!(format.as_str()),
            )?;
        }
        if let Some(geometry) = request.geometry {
            insert_geometry(&mut options, geometry)?;
        }
        let task = task_for_video_mode(request.mode);
        validate_video_mode(&request)?;
        let value = self
            .submit_queue(
                &request.model,
                options.input.clone(),
                &options,
                &request.request_options,
            )
            .await?;
        Ok(queued_operation(
            &request.model,
            task,
            &value,
            options.cost_estimate,
            warnings,
            request.count,
        )?)
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
                MediaTask::VideoExtend,
                MediaTask::VideoEdit,
            ],
        )?;
        let status_value = self.queue_status(handle).await?;
        let status = queue_state(&status_value)?;
        if status != OperationStatus::Succeeded {
            return Ok(non_terminal_operation(handle, status, &status_value));
        }
        let mut warnings = handle.warnings.clone();
        match self.fetch_result(handle).await? {
            Ok(payload) => {
                let result = parse_video_result(
                    &payload,
                    &status_value,
                    handle.cost_estimate.as_ref(),
                    handle.expected_outputs,
                    &mut warnings,
                )?;
                Ok(succeeded_operation(handle, result, &payload))
            }
            Err(failure) => Ok(failed_operation(handle, failure)),
        }
    }

    async fn cancel_video(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToVideo,
                MediaTask::ImageToVideo,
                MediaTask::ReferenceToVideo,
                MediaTask::VideoExtend,
                MediaTask::VideoEdit,
            ],
        )?;
        self.cancel_request(handle).await
    }
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for FalMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        validate_model(model).is_ok()
    }
    fn capabilities(&self, model: &str) -> SpeechCapabilities {
        self.descriptor(model, MediaTask::TextToSpeech)
    }

    async fn submit_speech(
        &self,
        request: SpeechSynthesisRequest,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        validate_nonempty(&request.model, &request.text)?;
        if request.voice.trim().is_empty() {
            return Err(MediaError::InvalidRequest(
                "speech voice must not be empty".to_string(),
            ));
        }
        let mut options = parse_options(&request.provider_options)?;
        let warnings = request_warnings(&request.request_options);
        insert_semantic(&mut options, "text", "text", json!(request.text))?;
        insert_semantic(&mut options, "voice", "voice", json!(request.voice))?;
        if let Some(language) = request.language.as_ref() {
            insert_semantic(&mut options, "language", "language", json!(language))?;
        }
        if let Some(instructions) = request.instructions.as_ref() {
            insert_semantic(
                &mut options,
                "instructions",
                "instructions",
                json!(instructions),
            )?;
        }
        if let Some(speed) = request.speed {
            insert_semantic(&mut options, "speed", "speed", json!(speed))?;
        }
        insert_semantic(
            &mut options,
            "output_format",
            "output_format",
            json!(request.output.format.as_str()),
        )?;
        let value = self
            .submit_queue(
                &request.model,
                options.input.clone(),
                &options,
                &request.request_options,
            )
            .await?;
        Ok(queued_operation(
            &request.model,
            MediaTask::TextToSpeech,
            &value,
            options.cost_estimate,
            warnings,
            None,
        )?)
    }

    async fn poll_speech(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToSpeech])?;
        let status_value = self.queue_status(handle).await?;
        let status = queue_state(&status_value)?;
        if status != OperationStatus::Succeeded {
            return Ok(non_terminal_operation(handle, status, &status_value));
        }
        let mut warnings = handle.warnings.clone();
        match self.fetch_result(handle).await? {
            Ok(payload) => {
                let result = parse_speech_result(
                    &payload,
                    &status_value,
                    handle.cost_estimate.as_ref(),
                    &mut warnings,
                )?;
                Ok(succeeded_operation(handle, result, &payload))
            }
            Err(failure) => Ok(failed_operation(handle, failure)),
        }
    }

    async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToSpeech])?;
        self.cancel_request(handle).await
    }
}

#[async_trait::async_trait]
impl TranscriptionProvider for FalMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        validate_model(model).is_ok()
    }
    fn capabilities(&self, model: &str) -> TranscriptionCapabilities {
        self.descriptor(model, MediaTask::SpeechToText)
    }

    async fn submit_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        validate_model(&request.model)?;
        let mut options = parse_options(&request.provider_options)?;
        let warnings = request_warnings(&request.request_options);
        let audio = shared::source_to_data_or_uri(
            &request.audio,
            request.request_options.max_source_bytes,
        )?;
        insert_semantic(&mut options, "audio", "audio_url", json!(audio))?;
        if let Some(language) = request.language.as_ref() {
            insert_semantic(&mut options, "language", "language", json!(language))?;
        }
        if let Some(prompt) = request.prompt.as_ref() {
            insert_semantic(&mut options, "prompt", "prompt", json!(prompt))?;
        }
        if !request.timestamp_granularities.is_empty() {
            insert_semantic(
                &mut options,
                "timestamp_granularities",
                "timestamp_granularities",
                json!(request.timestamp_granularities),
            )?;
        }
        let value = self
            .submit_queue(
                &request.model,
                options.input.clone(),
                &options,
                &request.request_options,
            )
            .await?;
        Ok(queued_operation(
            &request.model,
            MediaTask::SpeechToText,
            &value,
            options.cost_estimate,
            warnings,
            None,
        )?)
    }

    async fn poll_transcription(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::SpeechToText])?;
        let status_value = self.queue_status(handle).await?;
        let status = queue_state(&status_value)?;
        if status != OperationStatus::Succeeded {
            return Ok(non_terminal_operation(handle, status, &status_value));
        }
        let mut warnings = handle.warnings.clone();
        match self.fetch_result(handle).await? {
            Ok(payload) => {
                let result = parse_transcription_result(
                    &payload,
                    &status_value,
                    handle.cost_estimate.as_ref(),
                    &mut warnings,
                )?;
                Ok(succeeded_operation(handle, result, &payload))
            }
            Err(failure) => Ok(failed_operation(handle, failure)),
        }
    }

    async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::SpeechToText])?;
        self.cancel_request(handle).await
    }
}

fn parse_options(options: &ProviderOptions) -> MediaResult<FalMediaOptions> {
    let raw = shared::provider_options(options, PROVIDER)?;
    let parsed: FalMediaOptions = serde_json::from_value(Value::Object(raw)).map_err(|error| {
        MediaError::InvalidRequest(format!("invalid fal provider options: {error}"))
    })?;
    validate_fal_options(&parsed)?;
    Ok(parsed)
}

fn validate_fal_options(options: &FalMediaOptions) -> MediaResult<()> {
    if let Some(webhook) = options.webhook.as_deref() {
        shared::validate_https_url(webhook, "fal webhook")?;
    }
    if options.field_map.values().any(|field| field.is_empty()) {
        return Err(MediaError::InvalidRequest(
            "fal field_map target names must not be empty".to_string(),
        ));
    }
    if let Some(estimate) = options.cost_estimate.as_ref() {
        if !estimate.usd_per_unit.is_finite() || estimate.usd_per_unit < 0.0 {
            return Err(MediaError::InvalidRequest(
                "cost_estimate.usd_per_unit must be finite and non-negative".to_string(),
            ));
        }
        if estimate
            .quantity
            .is_some_and(|quantity| !quantity.is_finite() || quantity < 0.0)
        {
            return Err(MediaError::InvalidRequest(
                "cost_estimate.quantity must be finite and non-negative".to_string(),
            ));
        }
    }
    Ok(())
}

fn insert_semantic(
    options: &mut FalMediaOptions,
    semantic: &str,
    default_field: &str,
    value: Value,
) -> MediaResult<()> {
    let field = options
        .field_map
        .get(semantic)
        .cloned()
        .unwrap_or_else(|| default_field.to_string());
    if options.input.contains_key(&field) {
        return Err(MediaError::InvalidRequest(format!(
            "fal input field '{field}' was supplied both portably and in provider_options.fal.input"
        )));
    }
    options.input.insert(field, value);
    Ok(())
}

fn insert_geometry(options: &mut FalMediaOptions, geometry: OutputGeometry) -> MediaResult<()> {
    match geometry {
        OutputGeometry::Dimensions { width, height } => insert_semantic(
            options,
            "dimensions",
            "image_size",
            json!({"width": width, "height": height}),
        ),
        OutputGeometry::AspectRatio { .. } => insert_semantic(
            options,
            "aspect_ratio",
            "aspect_ratio",
            json!(geometry.as_api_string()),
        ),
    }
}

fn queue_state(value: &Value) -> MediaResult<OperationStatus> {
    let native =
        value
            .get("status")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "queue response is missing status".to_string(),
            })?;
    match native {
        "IN_QUEUE" => Ok(OperationStatus::Queued),
        "IN_PROGRESS" => Ok(OperationStatus::Running),
        "COMPLETED" => Ok(OperationStatus::Succeeded),
        "CANCELLATION_REQUESTED" => Ok(OperationStatus::CancellationRequested),
        "CANCELLED" => Ok(OperationStatus::Cancelled),
        other => Err(MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: format!("unknown queue status '{other}'"),
        }),
    }
}

fn make_handle(
    model: &str,
    task: MediaTask,
    remote_id: &str,
    estimate: Option<CostEstimate>,
    warnings: Vec<ProviderWarning>,
    expected_outputs: Option<u32>,
) -> JobHandle {
    JobHandle {
        provider: PROVIDER.to_string(),
        model: model.to_string(),
        task,
        remote_id: remote_id.to_string(),
        cost_estimate: estimate,
        warnings,
        expected_outputs,
    }
}

fn queued_operation<T>(
    model: &str,
    task: MediaTask,
    submit_response: &Value,
    estimate: Option<CostEstimate>,
    warnings: Vec<ProviderWarning>,
    expected_outputs: Option<u32>,
) -> MediaResult<Operation<T>> {
    let remote_id = submit_response
        .get("request_id")
        .and_then(Value::as_str)
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "queue submission returned no request_id".to_string(),
        })?;
    Ok(Operation {
        handle: Some(make_handle(
            model,
            task,
            remote_id,
            estimate,
            warnings,
            expected_outputs,
        )),
        status: OperationStatus::Queued,
        progress: None,
        result: None,
        error: None,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: json!({
            "queue_position": submit_response.get("queue_position")
        }),
    })
}

fn non_terminal_operation<T>(
    handle: &JobHandle,
    status: OperationStatus,
    status_value: &Value,
) -> Operation<T> {
    let error = (status == OperationStatus::Cancelled).then(|| GenerationFailure {
        category: FailureCategory::Cancelled,
        message: "fal queue request was cancelled".to_string(),
        provider_code: status_value
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_string),
        http_status: None,
        request_id: Some(handle.remote_id.clone()),
        provider_metadata: Value::Null,
    });
    Operation {
        handle: Some(handle.clone()),
        status,
        progress: None,
        result: None,
        error,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: json!({
            "native_status": status_value.get("status"),
            "queue_position": status_value.get("queue_position")
        }),
    }
}

fn succeeded_operation<T>(handle: &JobHandle, result: T, payload: &Value) -> Operation<T> {
    Operation {
        handle: Some(handle.clone()),
        status: OperationStatus::Succeeded,
        progress: Some(1.0),
        result: Some(result),
        error: None,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: json!({"raw_output": sanitize_media_value(payload)}),
    }
}

fn failed_operation<T>(handle: &JobHandle, failure: GenerationFailure) -> Operation<T> {
    Operation {
        handle: Some(handle.clone()),
        status: OperationStatus::Failed,
        progress: None,
        result: None,
        error: Some(failure),
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: Value::Null,
    }
}

fn parse_image_result(
    payload: &Value,
    status_value: &Value,
    estimate: Option<&CostEstimate>,
    expected_outputs: Option<u32>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<ImageGenerationResult> {
    let artifacts = collect_artifacts(payload, MediaKind::Image)?;
    if artifacts.is_empty() {
        return Err(no_artifacts("image"));
    }
    warn_if_partial(expected_outputs, artifacts.len(), warnings);
    let usage = queue_usage(
        status_value,
        estimate,
        Some(artifacts.len() as f64),
        warnings,
    );
    Ok(ImageGenerationResult {
        artifacts,
        usage: Some(usage),
        warnings: std::mem::take(warnings),
        safety: SafetyReport::default(),
        provider_metadata: json!({"raw_output": sanitize_media_value(payload)}),
    })
}

fn parse_video_result(
    payload: &Value,
    status_value: &Value,
    estimate: Option<&CostEstimate>,
    expected_outputs: Option<u32>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<VideoGenerationResult> {
    let artifacts = collect_artifacts(payload, MediaKind::Video)?;
    if artifacts.is_empty() {
        return Err(no_artifacts("video"));
    }
    warn_if_partial(expected_outputs, artifacts.len(), warnings);
    let usage = queue_usage(status_value, estimate, None, warnings);
    Ok(VideoGenerationResult {
        artifacts,
        usage: Some(usage),
        warnings: std::mem::take(warnings),
        safety: SafetyReport::default(),
        provider_metadata: json!({"raw_output": sanitize_media_value(payload)}),
    })
}

fn parse_speech_result(
    payload: &Value,
    status_value: &Value,
    estimate: Option<&CostEstimate>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<SpeechSynthesisResult> {
    let mut artifacts = collect_artifacts(payload, MediaKind::Audio)?;
    if artifacts.is_empty() {
        return Err(no_artifacts("audio"));
    }
    if artifacts.len() > 1 {
        warnings.push(ProviderWarning {
            code: WarningCode::PartialOutput,
            message:
                "speech result contained multiple media outputs; the first artifact was selected"
                    .to_string(),
            parameter: None,
            provider_metadata: json!({"artifact_count": artifacts.len()}),
        });
    }
    let usage = queue_usage(status_value, estimate, None, warnings);
    Ok(SpeechSynthesisResult {
        artifact: artifacts.remove(0),
        usage: Some(usage),
        warnings: std::mem::take(warnings),
        provider_metadata: json!({"raw_output": sanitize_media_value(payload)}),
    })
}

fn parse_transcription_result(
    payload: &Value,
    status_value: &Value,
    estimate: Option<&CostEstimate>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<TranscriptionResult> {
    let text = payload
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| payload.get("transcription").and_then(Value::as_str))
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "fal output does not contain transcription text; preserve the raw output from provider_metadata for model-specific handling".to_string(),
        })?;
    let usage = queue_usage(status_value, estimate, None, warnings);
    Ok(TranscriptionResult {
        text: text.to_string(),
        language: payload
            .get("language")
            .and_then(Value::as_str)
            .map(str::to_string),
        duration_secs: payload.get("duration").and_then(Value::as_f64),
        segments: transcript_segments(payload.get("segments")),
        words: transcript_words(payload.get("words")),
        usage: Some(usage),
        warnings: std::mem::take(warnings),
        provider_metadata: json!({"raw_output": sanitize_media_value(payload)}),
    })
}

fn collect_artifacts(payload: &Value, kind: MediaKind) -> MediaResult<Vec<MediaArtifact>> {
    let mut candidates = Vec::new();
    collect_strings(payload, &mut candidates);
    let mut seen = HashSet::new();
    let mut artifacts = Vec::new();
    for candidate in candidates {
        if !seen.insert(candidate.to_string()) {
            continue;
        }
        let (source, media_type, size) = if candidate.starts_with("data:") {
            let (bytes, media_type) = shared::decode_data_url(candidate)?;
            let size = Some(bytes.len() as u64);
            (ArtifactSource::Inline(bytes), media_type, size)
        } else if candidate.starts_with("http://") || candidate.starts_with("https://") {
            (
                ArtifactSource::Url(candidate.to_string()),
                shared::media_type_from_url(candidate, default_media_type(kind)),
                None,
            )
        } else {
            continue;
        };
        let artifact_kind = media_kind_for_type(&media_type).unwrap_or(kind);
        if !artifact_kind_compatible(kind, artifact_kind) {
            continue;
        }
        artifacts.push(MediaArtifact {
            kind: artifact_kind,
            media_type,
            source,
            size_bytes: size,
            dimensions: None,
            duration_secs: None,
            frame_rate: None,
            sample_rate_hz: None,
            channels: None,
            expires_at: None,
            metadata: json!({"provider": PROVIDER}),
        });
    }
    Ok(artifacts)
}

fn collect_strings<'a>(value: &'a Value, output: &mut Vec<&'a str>) {
    match value {
        Value::String(value) => output.push(value),
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_strings(value, output)),
        Value::Object(values) => values
            .values()
            .for_each(|value| collect_strings(value, output)),
        _ => {}
    }
}

fn media_kind_for_type(media_type: &str) -> Option<MediaKind> {
    if media_type.starts_with("image/") {
        Some(MediaKind::Image)
    } else if media_type.starts_with("video/") {
        Some(MediaKind::Video)
    } else if media_type.starts_with("audio/") {
        Some(MediaKind::Audio)
    } else {
        None
    }
}

fn artifact_kind_compatible(requested: MediaKind, actual: MediaKind) -> bool {
    requested == actual || (requested == MediaKind::Video && actual == MediaKind::Audio)
}

fn warn_if_partial(
    expected_outputs: Option<u32>,
    actual_outputs: usize,
    warnings: &mut Vec<ProviderWarning>,
) {
    if expected_outputs.is_some_and(|expected| actual_outputs < expected as usize) {
        warnings.push(ProviderWarning {
            code: WarningCode::PartialOutput,
            message: "provider returned fewer media artifacts than requested".to_string(),
            parameter: Some("count".to_string()),
            provider_metadata: json!({
                "expected": expected_outputs,
                "actual": actual_outputs
            }),
        });
    }
}

fn queue_usage(
    status_value: &Value,
    estimate: Option<&CostEstimate>,
    output_count: Option<f64>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaUsage {
    let metrics = status_value.get("metrics").cloned().unwrap_or(Value::Null);
    let inference_time = metrics.get("inference_time").and_then(Value::as_f64);
    let mut line_items = Vec::new();
    if let Some(seconds) = inference_time {
        line_items.push(UsageLineItem {
            unit: UsageUnit::ComputeSeconds,
            quantity: seconds,
            cost: None,
            description: Some("fal queue inference time".to_string()),
        });
    }
    if let Some(count) = output_count {
        line_items.push(UsageLineItem {
            unit: UsageUnit::Images,
            quantity: count,
            cost: None,
            description: Some("normalized output images".to_string()),
        });
    }
    let estimated_cost = estimate.and_then(|rate| {
        let quantity = rate.quantity.or(match rate.unit {
            UsageUnit::ComputeSeconds => inference_time,
            UsageUnit::Images => output_count,
            _ => None,
        });
        quantity.map(|quantity| quantity * rate.usd_per_unit)
    });
    if estimated_cost.is_none() {
        warnings.push(ProviderWarning {
            code: WarningCode::CostUnavailable,
            message: "fal queue metrics do not report dollar cost; supply a verified cost_estimate rate when the endpoint's billing unit is known".to_string(),
            parameter: None,
            provider_metadata: Value::Null,
        });
    }
    MediaUsage {
        line_items,
        provider_reported_cost: None,
        estimated_cost,
        currency: "USD".to_string(),
        metadata: metrics,
    }
}

fn transcript_segments(value: Option<&Value>) -> Vec<TranscriptSegment> {
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

fn transcript_words(value: Option<&Value>) -> Vec<TranscriptWord> {
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

fn sanitize_media_value(value: &Value) -> Value {
    match value {
        Value::String(value) if value.starts_with("data:") => {
            let header = value
                .split_once(',')
                .map_or(value.as_str(), |(header, _)| header);
            json!({"inline_media_omitted":true,"encoding":header,"encoded_bytes":value.len()})
        }
        Value::String(value) if value.len() > 16 * 1024 => {
            json!({"large_string_omitted":true,"bytes":value.len()})
        }
        Value::Array(values) => Value::Array(values.iter().map(sanitize_media_value).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), sanitize_media_value(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn failure_message(body: &[u8]) -> String {
    const MAX: usize = 16 * 1024;
    let body = &body[..body.len().min(MAX)];
    if let Ok(value) = serde_json::from_slice::<Value>(body) {
        if let Some(message) = value.get("detail").and_then(Value::as_str) {
            return message.to_string();
        }
        return value.to_string();
    }
    String::from_utf8_lossy(body).into_owned()
}

fn fal_options_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":{
            "input":{"type":"object"},
            "field_map":{"type":"object","additionalProperties":{"type":"string"}},
            "webhook":{"type":"string","format":"uri"},
            "cost_estimate":{"type":"object"}
        }
    })
}

fn request_warnings(options: &RequestOptions) -> Vec<ProviderWarning> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy == UnsupportedParameterPolicy::WarnAndDrop
    {
        vec![ProviderWarning {
            code: WarningCode::UnsupportedParameterDropped,
            message: "fal does not document queue idempotency keys".to_string(),
            parameter: Some("request_options.idempotency_key".to_string()),
            provider_metadata: Value::Null,
        }]
    } else {
        Vec::new()
    }
}

fn validate_model(model: &str) -> MediaResult<()> {
    let mut segments = 0;
    for segment in model.split('/') {
        if segment.is_empty() || segment == "." || segment == ".." || !valid_path_segment(segment) {
            return Err(MediaError::InvalidRequest(
                "fal model contains unsupported URL path characters".to_string(),
            ));
        }
        segments += 1;
    }
    if segments < 2 {
        return Err(MediaError::InvalidRequest(
            "fal model must be an owner/app endpoint path such as fal-ai/flux/dev".to_string(),
        ));
    }
    Ok(())
}

/// Queue request URLs address the owner/app pair even when the model id
/// carries additional endpoint path segments.
fn app_alias(model: &str) -> MediaResult<String> {
    validate_model(model)?;
    let mut segments = model.split('/');
    Ok(format!(
        "{}/{}",
        segments.next().unwrap_or_default(),
        segments.next().unwrap_or_default()
    ))
}

fn valid_path_segment(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn validate_nonempty(model: &str, prompt: &str) -> MediaResult<()> {
    validate_model(model)?;
    if prompt.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "prompt/text must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn validate_video_mode(request: &VideoGenerationRequest) -> MediaResult<()> {
    match request.mode {
        VideoGenerationMode::ImageToVideo if request.first_frame.is_none() => Err(
            MediaError::InvalidRequest("image-to-video requires first_frame".to_string()),
        ),
        VideoGenerationMode::ReferenceToVideo if request.reference_images.is_empty() => Err(
            MediaError::InvalidRequest("reference-to-video requires reference_images".to_string()),
        ),
        VideoGenerationMode::Extend | VideoGenerationMode::Edit
            if request.source_video.is_none() =>
        {
            Err(MediaError::InvalidRequest(
                "video extension/edit requires source_video".to_string(),
            ))
        }
        _ => Ok(()),
    }
}

fn task_for_image_mode(mode: ImageGenerationMode) -> MediaTask {
    match mode {
        ImageGenerationMode::Generate => MediaTask::TextToImage,
        ImageGenerationMode::Edit => MediaTask::ImageEdit,
        ImageGenerationMode::Inpaint => MediaTask::Inpainting,
        ImageGenerationMode::Variation => MediaTask::ImageVariation,
    }
}

fn task_for_video_mode(mode: VideoGenerationMode) -> MediaTask {
    match mode {
        VideoGenerationMode::TextToVideo => MediaTask::TextToVideo,
        VideoGenerationMode::ImageToVideo => MediaTask::ImageToVideo,
        VideoGenerationMode::ReferenceToVideo => MediaTask::ReferenceToVideo,
        VideoGenerationMode::Extend => MediaTask::VideoExtend,
        VideoGenerationMode::Edit => MediaTask::VideoEdit,
    }
}

fn default_media_type(kind: MediaKind) -> &'static str {
    match kind {
        MediaKind::Image => "image/*",
        MediaKind::Video => "video/*",
        MediaKind::Audio => "audio/*",
    }
}

fn no_artifacts(kind: &str) -> MediaError {
    MediaError::InvalidResponse {
        provider: PROVIDER.to_string(),
        message: format!(
            "completed fal request contained no recognizable {kind} artifact URL; raw output is endpoint-specific"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_paths_validate_and_reduce_to_app_alias() {
        assert!(validate_model("fal-ai/flux/dev").is_ok());
        assert_eq!(app_alias("fal-ai/flux/dev").unwrap(), "fal-ai/flux");
        assert_eq!(app_alias("fal-ai/whisper").unwrap(), "fal-ai/whisper");
        assert!(validate_model("flux").is_err());
        assert!(validate_model("fal-ai//dev").is_err());
        assert!(validate_model("fal-ai/../requests").is_err());
    }

    #[test]
    fn parses_all_queue_states() {
        for (native, expected) in [
            ("IN_QUEUE", OperationStatus::Queued),
            ("IN_PROGRESS", OperationStatus::Running),
            ("COMPLETED", OperationStatus::Succeeded),
            (
                "CANCELLATION_REQUESTED",
                OperationStatus::CancellationRequested,
            ),
            ("CANCELLED", OperationStatus::Cancelled),
        ] {
            assert_eq!(queue_state(&json!({"status": native})).unwrap(), expected);
        }
        assert!(queue_state(&json!({"status":"MYSTERY"})).is_err());
        assert!(queue_state(&json!({})).is_err());
    }

    #[test]
    fn typed_options_round_trip_and_reject_unknown_keys() {
        let options = FalMediaOptions {
            input: serde_json::from_value(json!({"guidance_scale":3.5})).unwrap(),
            field_map: HashMap::from([("prompt".to_string(), "caption".to_string())]),
            ..FalMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        let parsed = parse_options(&options).unwrap();
        assert_eq!(
            parsed.field_map.get("prompt").map(String::as_str),
            Some("caption")
        );
        assert_eq!(parsed.input.get("guidance_scale"), Some(&json!(3.5)));

        let mut unknown = ProviderOptions::new();
        unknown.insert(PROVIDER.to_string(), json!({"mystery":true}));
        assert!(parse_options(&unknown).is_err());

        let insecure = FalMediaOptions {
            webhook: Some("http://example.test/hook".to_string()),
            ..FalMediaOptions::default()
        };
        assert!(insecure.into_provider_options().is_err());
    }

    #[test]
    fn field_map_is_explicit_and_conflicts_fail_closed() {
        let mut options = FalMediaOptions {
            field_map: HashMap::from([("prompt".to_string(), "caption".to_string())]),
            ..FalMediaOptions::default()
        };
        insert_semantic(&mut options, "prompt", "prompt", json!("hello")).unwrap();
        assert_eq!(options.input.get("caption"), Some(&json!("hello")));
        assert!(insert_semantic(&mut options, "prompt", "prompt", json!("again")).is_err());
    }

    #[test]
    fn normalizes_typical_fal_payloads_without_download() {
        let payload = json!({
            "images":[{"url":"https://v3.fal.media/files/a.png","content_type":"image/png","width":1024}],
            "seed": 42
        });
        let artifacts = collect_artifacts(&payload, MediaKind::Image).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert!(matches!(artifacts[0].source, ArtifactSource::Url(_)));
        assert_eq!(artifacts[0].media_type, "image/png");

        let video = json!({"video":{"url":"https://v3.fal.media/files/clip.mp4"}});
        let artifacts = collect_artifacts(&video, MediaKind::Video).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, MediaKind::Video);
    }

    #[test]
    fn queued_and_failed_operations_keep_the_resumable_handle() {
        let submit = json!({"request_id":"a1b2-c3","queue_position":4});
        let operation: Operation<ImageGenerationResult> = queued_operation(
            "fal-ai/flux/dev",
            MediaTask::TextToImage,
            &submit,
            None,
            Vec::new(),
            Some(2),
        )
        .unwrap();
        assert_eq!(operation.status, OperationStatus::Queued);
        let handle = operation.handle.unwrap();
        assert_eq!(handle.remote_id, "a1b2-c3");
        assert_eq!(handle.expected_outputs, Some(2));

        let failed: Operation<ImageGenerationResult> = failed_operation(
            &handle,
            GenerationFailure {
                category: FailureCategory::InvalidInput,
                message: "bad input".to_string(),
                provider_code: None,
                http_status: Some(422),
                request_id: Some(handle.remote_id.clone()),
                provider_metadata: Value::Null,
            },
        );
        assert_eq!(failed.status, OperationStatus::Failed);
        assert_eq!(failed.error.unwrap().http_status, Some(422));
    }

    #[test]
    fn computes_only_labeled_estimated_cost_from_inference_time() {
        let status_value = json!({"status":"COMPLETED","metrics":{"inference_time":2.5}});
        let estimate = CostEstimate {
            unit: UsageUnit::ComputeSeconds,
            usd_per_unit: 0.001,
            quantity: None,
        };
        let usage = queue_usage(&status_value, Some(&estimate), None, &mut Vec::new());
        assert_eq!(usage.provider_reported_cost, None);
        assert_eq!(usage.estimated_cost, Some(0.0025));

        let mut warnings = Vec::new();
        let usage = queue_usage(&status_value, None, None, &mut warnings);
        assert_eq!(usage.estimated_cost, None);
        assert!(warnings
            .iter()
            .any(|warning| warning.code == WarningCode::CostUnavailable));
    }

    #[test]
    fn transcription_payload_normalizes_text_and_segments() {
        let payload = json!({
            "text":"hello world",
            "segments":[{"start":0.0,"end":1.0,"text":"hello world"}]
        });
        let status_value = json!({"status":"COMPLETED"});
        let mut warnings = Vec::new();
        let result =
            parse_transcription_result(&payload, &status_value, None, &mut warnings).unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.segments.len(), 1);
    }
}
