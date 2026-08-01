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

//! Replicate prediction adapter for model-specific image, video, speech, and
//! transcription models.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, RwLock};
use std::time::{Duration, Instant};

const PROVIDER: &str = "replicate";
const API_KEY_ENV: &str = "REPLICATE_API_TOKEN";
const API_BASE_ENV: &str = "REPLICATE_API_URL";
const API_BASE: &str = "https://api.replicate.com/v1";
const DEFAULT_ARTIFACT_TTL_SECS: u64 = 3600;
const SCHEMA_CACHE_TTL: Duration = Duration::from_secs(600);
const SCHEMA_CACHE_CAPACITY: usize = 128;

#[derive(Debug, Clone)]
struct CachedSchema {
    stored_at: Instant,
    schema: Value,
}

static SCHEMA_CACHE: LazyLock<RwLock<HashMap<String, CachedSchema>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

#[derive(Debug, Clone, Default)]
pub struct ReplicateMediaProvider;

/// Stable Replicate adapter controls. Model-native fields belong in `input`;
/// `field_map` maps portable semantic names such as `prompt`, `source_image`,
/// or `duration` to the selected model's OpenAPI field names.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplicateMediaOptions {
    pub version: Option<String>,
    #[serde(default)]
    pub input: Map<String, Value>,
    #[serde(default)]
    pub field_map: HashMap<String, String>,
    pub webhook: Option<String>,
    pub webhook_events_filter: Option<Vec<String>>,
    pub cancel_after: Option<String>,
    pub cost_estimate: Option<CostEstimate>,
}

impl ReplicateMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_replicate_options(&self)?;
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
        Ok(options)
    }
}

impl ReplicateMediaProvider {
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
                immediate: CapabilitySupport::Unknown,
                persistent_jobs: CapabilitySupport::Supported,
                polling: CapabilitySupport::Supported,
                cancellation: CapabilitySupport::Supported,
                progress: CapabilitySupport::Unknown,
                webhooks: CapabilitySupport::Supported,
                binary_streaming: CapabilitySupport::Unknown,
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
            provider_options_schema: Some(replicate_options_schema()),
        }
    }

    /// Fetch the authoritative per-version OpenAPI schema exposed by Replicate.
    /// Transient discovery failure is returned to the caller; sync capability
    /// methods deliberately remain `Unknown` rather than claiming unsupported.
    pub async fn discover_openapi_schema(
        &self,
        model: &str,
        version: Option<&str>,
    ) -> MediaResult<Value> {
        let (owner, name) = split_model(model)?;
        if version.is_some_and(|value| value.is_empty() || !valid_path_segment(value)) {
            return Err(MediaError::InvalidRequest(
                "Replicate version contains unsupported URL path characters".to_string(),
            ));
        }
        let cache_key = format!(
            "{}:{model}@{}",
            self.api_base(),
            version.unwrap_or("latest")
        );
        if let Some(schema) = cached_schema(&cache_key) {
            return Ok(schema);
        }
        let key = self.key()?;
        let url = match version {
            Some(version) => format!(
                "{}/models/{owner}/{name}/versions/{version}",
                self.api_base()
            ),
            None => format!("{}/models/{owner}/{name}", self.api_base()),
        };
        let options = RequestOptions {
            max_retries: 0,
            submit_timeout: Some(Duration::from_secs(10)),
            max_response_bytes: 10 * 1024 * 1024,
            ..RequestOptions::default()
        };
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .bearer_auth(&key)
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        let schema = if version.is_some() {
            value.get("openapi_schema")
        } else {
            value.pointer("/latest_version/openapi_schema")
        }
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "model response did not contain an OpenAPI schema".to_string(),
        })?;
        let schema = schema.clone();
        cache_schema(cache_key, schema.clone());
        Ok(schema)
    }

    async fn submit_prediction(
        &self,
        model: &str,
        task: MediaTask,
        input: Map<String, Value>,
        options: &ReplicateMediaOptions,
        request_options: &RequestOptions,
        warnings: &mut Vec<ProviderWarning>,
    ) -> MediaResult<Value> {
        let (owner, name) = split_model(model)?;
        // Successful discovery gives callers and adapters authoritative model
        // schema data. Its failure is deliberately non-fatal: arbitrary model
        // platforms must remain `Unknown`, not become falsely unsupported.
        match self
            .discover_openapi_schema(model, options.version.as_deref())
            .await
        {
            Ok(schema) => validate_known_input_fields(&schema, &input)?,
            Err(error) => warnings.push(ProviderWarning {
                code: WarningCode::SchemaUnavailable,
                message: format!(
                    "Replicate model schema discovery was unavailable; input validation is deferred to the prediction API: {error}"
                ),
                parameter: None,
                provider_metadata: Value::Null,
            }),
        }
        let mut body = Map::new();
        body.insert("input".to_string(), Value::Object(input));
        let url = if let Some(version) = options.version.as_ref() {
            body.insert("version".to_string(), json!(version));
            format!("{}/predictions", self.api_base())
        } else {
            format!("{}/models/{owner}/{name}/predictions", self.api_base())
        };
        if let Some(webhook) = options.webhook.as_ref() {
            shared::validate_https_url(webhook, "Replicate webhook")?;
            body.insert("webhook".to_string(), json!(webhook));
        }
        if let Some(events) = options.webhook_events_filter.as_ref() {
            body.insert("webhook_events_filter".to_string(), json!(events));
        }
        let key = self.key()?;
        let cancel_after = options.cancel_after.clone();
        let response = shared::send(PROVIDER, request_options, || {
            let mut builder = crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(&body);
            if let Some(deadline) = cancel_after.as_ref() {
                builder = builder.header("Cancel-After", deadline);
            }
            builder
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        if value.get("id").and_then(Value::as_str).is_none() {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: format!("prediction submission for {task:?} returned no id"),
            });
        }
        Ok(value)
    }

    async fn poll_prediction(&self, handle: &JobHandle) -> MediaResult<Value> {
        let key = self.key()?;
        let url = format!("{}/predictions/{}", self.api_base(), handle.remote_id);
        let options = RequestOptions::default();
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .bearer_auth(&key)
        })
        .await?;
        shared::parse_json(PROVIDER, &response)
    }

    async fn cancel_prediction(&self, handle: &JobHandle) -> MediaResult<()> {
        let key = self.key()?;
        let url = format!(
            "{}/predictions/{}/cancel",
            self.api_base(),
            handle.remote_id
        );
        let options = RequestOptions::default();
        let response = shared::send(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
        })
        .await?;
        shared::require_success(PROVIDER, &response)
    }
}

#[async_trait::async_trait]
impl ImageGenerationProvider for ReplicateMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        split_model(model).is_ok()
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
        let mut warnings = request_warnings(&request.request_options);
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
                insert_semantic(
                    &mut options,
                    "source_image",
                    "input_image",
                    json!(sources[0]),
                )?;
            } else {
                insert_semantic(
                    &mut options,
                    "source_images",
                    "input_images",
                    json!(sources),
                )?;
            }
        }
        if let Some(mask) = request.mask.as_ref() {
            let mask =
                shared::source_to_data_or_uri(mask, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "mask", "mask", json!(mask))?;
        }
        if let Some(count) = request.count {
            insert_semantic(&mut options, "count", "num_outputs", json!(count))?;
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
        let value = self
            .submit_prediction(
                &request.model,
                task_for_image_mode(request.mode),
                options.input.clone(),
                &options,
                &request.request_options,
                &mut warnings,
            )
            .await?;
        parse_image_operation(
            &value,
            &request.model,
            task_for_image_mode(request.mode),
            options.cost_estimate,
            request.count,
            &mut warnings,
        )
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
        let value = self.poll_prediction(handle).await?;
        let mut warnings = handle.warnings.clone();
        parse_image_operation(
            &value,
            &handle.model,
            handle.task,
            handle.cost_estimate.clone(),
            handle.expected_outputs,
            &mut warnings,
        )
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
        self.cancel_prediction(handle).await
    }
}

#[async_trait::async_trait]
impl VideoGenerationProvider for ReplicateMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        split_model(model).is_ok()
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
        insert_semantic(&mut options, "prompt", "prompt", json!(request.prompt))?;
        if let Some(source) = request.first_frame.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(
                &mut options,
                "first_frame",
                "first_frame_image",
                json!(source),
            )?;
        }
        if let Some(source) = request.last_frame.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(
                &mut options,
                "last_frame",
                "last_frame_image",
                json!(source),
            )?;
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
                "reference_images",
                json!(sources),
            )?;
        }
        if let Some(source) = request.source_video.as_ref() {
            let source =
                shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)?;
            insert_semantic(&mut options, "source_video", "video", json!(source))?;
        }
        if let Some(count) = request.count {
            insert_semantic(&mut options, "count", "num_outputs", json!(count))?;
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
        let mut warnings = request_warnings(&request.request_options);
        let value = self
            .submit_prediction(
                &request.model,
                task,
                options.input.clone(),
                &options,
                &request.request_options,
                &mut warnings,
            )
            .await?;
        parse_video_operation(
            &value,
            &request.model,
            task,
            options.cost_estimate,
            request.count,
            &mut warnings,
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
                MediaTask::VideoExtend,
                MediaTask::VideoEdit,
            ],
        )?;
        let value = self.poll_prediction(handle).await?;
        let mut warnings = handle.warnings.clone();
        parse_video_operation(
            &value,
            &handle.model,
            handle.task,
            handle.cost_estimate.clone(),
            handle.expected_outputs,
            &mut warnings,
        )
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
        self.cancel_prediction(handle).await
    }
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for ReplicateMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        split_model(model).is_ok()
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
        let mut warnings = request_warnings(&request.request_options);
        let value = self
            .submit_prediction(
                &request.model,
                MediaTask::TextToSpeech,
                options.input.clone(),
                &options,
                &request.request_options,
                &mut warnings,
            )
            .await?;
        parse_speech_operation(&value, &request.model, options.cost_estimate, &mut warnings)
    }

    async fn poll_speech(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToSpeech])?;
        let value = self.poll_prediction(handle).await?;
        let mut warnings = handle.warnings.clone();
        parse_speech_operation(
            &value,
            &handle.model,
            handle.cost_estimate.clone(),
            &mut warnings,
        )
    }

    async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToSpeech])?;
        self.cancel_prediction(handle).await
    }
}

#[async_trait::async_trait]
impl TranscriptionProvider for ReplicateMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        split_model(model).is_ok()
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
        let mut options = parse_options(&request.provider_options)?;
        let audio = shared::source_to_data_or_uri(
            &request.audio,
            request.request_options.max_source_bytes,
        )?;
        insert_semantic(&mut options, "audio", "audio", json!(audio))?;
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
        let mut warnings = request_warnings(&request.request_options);
        let value = self
            .submit_prediction(
                &request.model,
                MediaTask::SpeechToText,
                options.input.clone(),
                &options,
                &request.request_options,
                &mut warnings,
            )
            .await?;
        parse_transcription_operation(&value, &request.model, options.cost_estimate, &mut warnings)
    }

    async fn poll_transcription(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::SpeechToText])?;
        let value = self.poll_prediction(handle).await?;
        let mut warnings = handle.warnings.clone();
        parse_transcription_operation(
            &value,
            &handle.model,
            handle.cost_estimate.clone(),
            &mut warnings,
        )
    }

    async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(handle, PROVIDER, &[MediaTask::SpeechToText])?;
        self.cancel_prediction(handle).await
    }
}

fn parse_options(options: &ProviderOptions) -> MediaResult<ReplicateMediaOptions> {
    let mut raw = shared::provider_options(options, PROVIDER)?;
    let version = take_string(&mut raw, "version")?;
    let input = match raw.remove("input") {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(input)) => input,
        Some(_) => {
            return Err(MediaError::InvalidRequest(
                "provider_options.replicate.input must be an object".to_string(),
            ))
        }
    };
    let field_map = match raw.remove("field_map") {
        None | Some(Value::Null) => HashMap::new(),
        Some(Value::Object(fields)) => fields
            .into_iter()
            .map(|(key, value)| {
                value
                    .as_str()
                    .map(|value| (key.clone(), value.to_string()))
                    .ok_or_else(|| {
                        MediaError::InvalidRequest(format!("field_map.{key} must be a string"))
                    })
            })
            .collect::<MediaResult<HashMap<_, _>>>()?,
        Some(_) => {
            return Err(MediaError::InvalidRequest(
                "provider_options.replicate.field_map must be an object".to_string(),
            ))
        }
    };
    let webhook = take_string(&mut raw, "webhook")?;
    let cancel_after = take_string(&mut raw, "cancel_after")?;
    let webhook_events_filter = match raw.remove("webhook_events_filter") {
        None | Some(Value::Null) => None,
        Some(Value::Array(values)) => Some(
            values
                .into_iter()
                .map(|value| {
                    value.as_str().map(str::to_string).ok_or_else(|| {
                        MediaError::InvalidRequest(
                            "webhook_events_filter entries must be strings".to_string(),
                        )
                    })
                })
                .collect::<MediaResult<Vec<_>>>()?,
        ),
        Some(_) => {
            return Err(MediaError::InvalidRequest(
                "webhook_events_filter must be an array".to_string(),
            ))
        }
    };
    let cost_estimate: Option<CostEstimate> = match raw.remove("cost_estimate") {
        None | Some(Value::Null) => None,
        Some(value) => Some(serde_json::from_value(value).map_err(|error| {
            MediaError::InvalidRequest(format!("invalid cost_estimate: {error}"))
        })?),
    };
    if !raw.is_empty() {
        return Err(MediaError::InvalidRequest(format!(
            "unknown Replicate provider options: {}",
            raw.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    let parsed = ReplicateMediaOptions {
        version,
        input,
        field_map,
        webhook,
        webhook_events_filter,
        cancel_after,
        cost_estimate,
    };
    validate_replicate_options(&parsed)?;
    Ok(parsed)
}

fn validate_replicate_options(options: &ReplicateMediaOptions) -> MediaResult<()> {
    if options
        .version
        .as_deref()
        .is_some_and(|value| value.is_empty() || !valid_path_segment(value))
    {
        return Err(MediaError::InvalidRequest(
            "provider_options.replicate.version contains unsupported URL path characters"
                .to_string(),
        ));
    }
    if let Some(webhook) = options.webhook.as_deref() {
        shared::validate_https_url(webhook, "Replicate webhook")?;
    }
    if options
        .webhook_events_filter
        .as_ref()
        .is_some_and(|events| {
            events
                .iter()
                .any(|event| !matches!(event.as_str(), "start" | "output" | "logs" | "completed"))
        })
    {
        return Err(MediaError::InvalidRequest(
            "Replicate webhook events must be start, output, logs, or completed".to_string(),
        ));
    }
    if options.cancel_after.as_deref().is_some_and(|value| {
        value.is_empty()
            || value.len() > 64
            || !value.bytes().all(|byte| byte.is_ascii_alphanumeric())
    }) {
        return Err(MediaError::InvalidRequest(
            "Replicate cancel_after must be a compact duration such as 5m or 1h".to_string(),
        ));
    }
    if options.field_map.values().any(|field| field.is_empty()) {
        return Err(MediaError::InvalidRequest(
            "Replicate field_map target names must not be empty".to_string(),
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
    options: &mut ReplicateMediaOptions,
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
        return Err(MediaError::InvalidRequest(format!("Replicate input field '{field}' was supplied both portably and in provider_options.replicate.input")));
    }
    options.input.insert(field, value);
    Ok(())
}

fn insert_geometry(
    options: &mut ReplicateMediaOptions,
    geometry: OutputGeometry,
) -> MediaResult<()> {
    match geometry {
        OutputGeometry::Dimensions { width, height } => {
            insert_semantic(options, "width", "width", json!(width))?;
            insert_semantic(options, "height", "height", json!(height))
        }
        OutputGeometry::AspectRatio { .. } => insert_semantic(
            options,
            "aspect_ratio",
            "aspect_ratio",
            json!(geometry.as_api_string()),
        ),
    }
}

fn prediction_state(value: &Value) -> MediaResult<(OperationStatus, Option<GenerationFailure>)> {
    let native =
        value
            .get("status")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "prediction is missing status".to_string(),
            })?;
    let status = match native {
        "starting" => OperationStatus::Queued,
        "processing" => OperationStatus::Running,
        "succeeded" | "successful" => OperationStatus::Succeeded,
        "failed" => OperationStatus::Failed,
        "canceled" | "cancelled" => OperationStatus::Cancelled,
        "aborted" => OperationStatus::Expired,
        other => {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: format!("unknown prediction status '{other}'"),
            })
        }
    };
    let failure = if matches!(
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
                .unwrap_or(native)
                .to_string(),
            provider_code: Some(native.to_string()),
            http_status: None,
            request_id: value.get("id").and_then(Value::as_str).map(str::to_string),
            provider_metadata: Value::Null,
        })
    } else {
        None
    };
    Ok((status, failure))
}

fn base_operation<T>(
    value: &Value,
    model: &str,
    task: MediaTask,
    estimate: Option<CostEstimate>,
    handle_warnings: Vec<ProviderWarning>,
    expected_outputs: Option<u32>,
    result: Option<T>,
) -> MediaResult<Operation<T>> {
    let id =
        value
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "prediction is missing id".to_string(),
            })?;
    let (status, error) = prediction_state(value)?;
    if status == OperationStatus::Succeeded && result.is_none() {
        return Err(MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "succeeded prediction has no normalized result".to_string(),
        });
    }
    Ok(Operation {
        handle: Some(JobHandle {
            provider: PROVIDER.to_string(),
            model: model.to_string(),
            task,
            remote_id: id.to_string(),
            cost_estimate: estimate,
            warnings: handle_warnings,
            expected_outputs,
        }),
        status,
        progress: (status == OperationStatus::Succeeded).then_some(1.0),
        result,
        error,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: sanitized_prediction_metadata(value),
    })
}

fn parse_image_operation(
    value: &Value,
    model: &str,
    task: MediaTask,
    estimate: Option<CostEstimate>,
    expected_outputs: Option<u32>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<Operation<ImageGenerationResult>> {
    let handle_warnings = warnings.clone();
    let status = prediction_state(value)?.0;
    let result = if status == OperationStatus::Succeeded {
        let artifacts = collect_artifacts(value.get("output"), MediaKind::Image)?;
        if artifacts.is_empty() {
            return Err(no_artifacts("image"));
        }
        warn_if_partial(expected_outputs, artifacts.len(), warnings);
        let usage = prediction_usage(
            value,
            estimate.as_ref(),
            Some(artifacts.len() as f64),
            warnings,
        );
        Some(ImageGenerationResult {
            artifacts,
            usage: Some(usage),
            warnings: std::mem::take(warnings),
            safety: SafetyReport::default(),
            provider_metadata: output_metadata(value),
        })
    } else {
        None
    };
    base_operation(
        value,
        model,
        task,
        estimate,
        handle_warnings,
        expected_outputs,
        result,
    )
}

fn parse_video_operation(
    value: &Value,
    model: &str,
    task: MediaTask,
    estimate: Option<CostEstimate>,
    expected_outputs: Option<u32>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<Operation<VideoGenerationResult>> {
    let handle_warnings = warnings.clone();
    let status = prediction_state(value)?.0;
    let result = if status == OperationStatus::Succeeded {
        let artifacts = collect_artifacts(value.get("output"), MediaKind::Video)?;
        if artifacts.is_empty() {
            return Err(no_artifacts("video"));
        }
        warn_if_partial(expected_outputs, artifacts.len(), warnings);
        let usage = prediction_usage(value, estimate.as_ref(), None, warnings);
        Some(VideoGenerationResult {
            artifacts,
            usage: Some(usage),
            warnings: std::mem::take(warnings),
            safety: SafetyReport::default(),
            provider_metadata: output_metadata(value),
        })
    } else {
        None
    };
    base_operation(
        value,
        model,
        task,
        estimate,
        handle_warnings,
        expected_outputs,
        result,
    )
}

fn parse_speech_operation(
    value: &Value,
    model: &str,
    estimate: Option<CostEstimate>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<Operation<SpeechSynthesisResult>> {
    let handle_warnings = warnings.clone();
    let status = prediction_state(value)?.0;
    let result = if status == OperationStatus::Succeeded {
        let mut artifacts = collect_artifacts(value.get("output"), MediaKind::Audio)?;
        if artifacts.is_empty() {
            return Err(no_artifacts("audio"));
        }
        if artifacts.len() > 1 {
            warnings.push(ProviderWarning { code: WarningCode::PartialOutput, message: "speech result contained multiple media outputs; the first artifact was selected".to_string(), parameter: None, provider_metadata: json!({"artifact_count":artifacts.len()}) });
        }
        let usage = prediction_usage(value, estimate.as_ref(), None, warnings);
        Some(SpeechSynthesisResult {
            artifact: artifacts.remove(0),
            usage: Some(usage),
            warnings: std::mem::take(warnings),
            provider_metadata: output_metadata(value),
        })
    } else {
        None
    };
    base_operation(
        value,
        model,
        MediaTask::TextToSpeech,
        estimate,
        handle_warnings,
        None,
        result,
    )
}

fn parse_transcription_operation(
    value: &Value,
    model: &str,
    estimate: Option<CostEstimate>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<Operation<TranscriptionResult>> {
    let handle_warnings = warnings.clone();
    let status = prediction_state(value)?.0;
    let result = if status == OperationStatus::Succeeded {
        let output = value
            .get("output")
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "transcription prediction is missing output".to_string(),
            })?;
        let text = output.as_str().filter(|text| !text.starts_with("http://") && !text.starts_with("https://"))
            .or_else(|| output.get("text").and_then(Value::as_str))
            .or_else(|| output.get("transcription").and_then(Value::as_str))
            .ok_or_else(|| MediaError::InvalidResponse { provider: PROVIDER.to_string(), message: "Replicate output does not contain transcription text; preserve the raw output from provider_metadata for model-specific handling".to_string() })?;
        let usage = prediction_usage(value, estimate.as_ref(), None, warnings);
        Some(TranscriptionResult {
            text: text.to_string(),
            language: output
                .get("language")
                .and_then(Value::as_str)
                .map(str::to_string),
            duration_secs: output.get("duration").and_then(Value::as_f64),
            segments: transcript_segments(output.get("segments")),
            words: transcript_words(output.get("words")),
            usage: Some(usage),
            warnings: std::mem::take(warnings),
            provider_metadata: output_metadata(value),
        })
    } else {
        None
    };
    base_operation(
        value,
        model,
        MediaTask::SpeechToText,
        estimate,
        handle_warnings,
        None,
        result,
    )
}

fn collect_artifacts(output: Option<&Value>, kind: MediaKind) -> MediaResult<Vec<MediaArtifact>> {
    let Some(output) = output else {
        return Ok(Vec::new());
    };
    let mut candidates = Vec::new();
    collect_strings(output, &mut candidates);
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
        } else if candidate.starts_with("s3://") || candidate.starts_with("gs://") {
            (
                ArtifactSource::ObjectStorageUri(candidate.to_string()),
                default_media_type(kind).to_string(),
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
            metadata: json!({
                "provider": PROVIDER,
                "requires_auth": true,
                "retention_hint_secs": DEFAULT_ARTIFACT_TTL_SECS
            }),
        });
    }
    Ok(artifacts)
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

fn prediction_usage(
    value: &Value,
    estimate: Option<&CostEstimate>,
    output_count: Option<f64>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaUsage {
    let metrics = value.get("metrics").cloned().unwrap_or(Value::Null);
    let predict_time = metrics.get("predict_time").and_then(Value::as_f64);
    let provider_cost = metrics.get("cost").and_then(Value::as_f64);
    let mut line_items = Vec::new();
    if let Some(seconds) = predict_time {
        line_items.push(UsageLineItem {
            unit: UsageUnit::ComputeSeconds,
            quantity: seconds,
            cost: None,
            description: Some("Replicate prediction compute time".to_string()),
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
            UsageUnit::ComputeSeconds => predict_time,
            UsageUnit::Images => output_count,
            _ => None,
        });
        quantity.map(|quantity| quantity * rate.usd_per_unit)
    });
    if provider_cost.is_none() && estimated_cost.is_none() {
        warnings.push(ProviderWarning { code: WarningCode::CostUnavailable, message: "Replicate prediction metrics do not report dollar cost; supply a verified cost_estimate rate when the model's billing unit is known".to_string(), parameter: None, provider_metadata: Value::Null });
    }
    MediaUsage {
        line_items,
        provider_reported_cost: provider_cost,
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

fn sanitized_prediction_metadata(value: &Value) -> Value {
    json!({
        "native_status": value.get("status"),
        "model": value.get("model"),
        "version": value.get("version"),
        "created_at": value.get("created_at"),
        "started_at": value.get("started_at"),
        "completed_at": value.get("completed_at"),
        "deadline": value.get("deadline"),
        "data_removed": value.get("data_removed"),
        "metrics": value.get("metrics"),
        "urls": value.get("urls")
    })
}

fn output_metadata(value: &Value) -> Value {
    json!({
        "raw_output": value.get("output").map(sanitize_media_value),
        "prediction":sanitized_prediction_metadata(value)
    })
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

fn cached_schema(key: &str) -> Option<Value> {
    let cache = SCHEMA_CACHE.read().ok()?;
    let cached = cache.get(key)?;
    (cached.stored_at.elapsed() <= SCHEMA_CACHE_TTL).then(|| cached.schema.clone())
}

fn cache_schema(key: String, schema: Value) {
    let Some(mut cache) = SCHEMA_CACHE.write().ok() else {
        tracing::warn!("Replicate schema cache lock is poisoned; skipping cache write");
        return;
    };
    cache.retain(|_, value| value.stored_at.elapsed() <= SCHEMA_CACHE_TTL);
    if cache.len() >= SCHEMA_CACHE_CAPACITY {
        if let Some(oldest) = cache
            .iter()
            .min_by_key(|(_, value)| value.stored_at)
            .map(|(key, _)| key.clone())
        {
            cache.remove(&oldest);
        }
    }
    cache.insert(
        key,
        CachedSchema {
            stored_at: Instant::now(),
            schema,
        },
    );
}

fn validate_known_input_fields(schema: &Value, input: &Map<String, Value>) -> MediaResult<()> {
    let Some(input_schema) = schema.pointer("/components/schemas/Input") else {
        return Ok(());
    };
    let Some(properties) = input_schema.get("properties").and_then(Value::as_object) else {
        return Ok(());
    };
    if input_schema
        .get("additionalProperties")
        .and_then(Value::as_bool)
        != Some(false)
    {
        return Ok(());
    }
    let unknown = input
        .keys()
        .filter(|key| !properties.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        return Ok(());
    }
    Err(MediaError::InvalidRequest(format!(
        "Replicate model schema does not define input fields: {}. Use provider_options.replicate.field_map to map portable fields",
        unknown.join(", ")
    )))
}

fn replicate_options_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":{
            "version":{"type":"string"},
            "input":{"type":"object"},
            "field_map":{"type":"object","additionalProperties":{"type":"string"}},
            "webhook":{"type":"string","format":"uri"},
            "webhook_events_filter":{"type":"array","items":{"enum":["start","output","logs","completed"]}},
            "cancel_after":{"type":"string"},
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
            message: "Replicate does not document prediction idempotency keys".to_string(),
            parameter: Some("request_options.idempotency_key".to_string()),
            provider_metadata: Value::Null,
        }]
    } else {
        Vec::new()
    }
}

fn split_model(model: &str) -> MediaResult<(&str, &str)> {
    let (owner, name) = model.split_once('/').ok_or_else(|| MediaError::InvalidRequest("Replicate model must be owner/name; put a community version in provider_options.replicate.version".to_string()))?;
    if owner.trim().is_empty() || name.trim().is_empty() || name.contains('/') {
        return Err(MediaError::InvalidRequest(
            "Replicate model must contain exactly one owner/name pair".to_string(),
        ));
    }
    if !valid_path_segment(owner) || !valid_path_segment(name) {
        return Err(MediaError::InvalidRequest(
            "Replicate owner/name contains unsupported URL path characters".to_string(),
        ));
    }
    Ok((owner, name))
}

fn valid_path_segment(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn validate_nonempty(model: &str, prompt: &str) -> MediaResult<()> {
    split_model(model)?;
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

fn take_string(values: &mut Map<String, Value>, key: &str) -> MediaResult<Option<String>> {
    match values.remove(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(MediaError::InvalidRequest(format!(
            "provider_options.replicate.{key} must be a string"
        ))),
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
    MediaError::InvalidResponse { provider: PROVIDER.to_string(), message: format!("succeeded Replicate prediction contained no recognizable {kind} artifact URL; raw output is model-specific") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_prediction_states() {
        for (native, expected) in [
            ("starting", OperationStatus::Queued),
            ("processing", OperationStatus::Running),
            ("succeeded", OperationStatus::Succeeded),
            ("failed", OperationStatus::Failed),
            ("canceled", OperationStatus::Cancelled),
            ("aborted", OperationStatus::Expired),
        ] {
            let value = json!({"status":native});
            assert_eq!(prediction_state(&value).unwrap().0, expected);
        }
    }

    #[test]
    fn normalizes_nested_and_multiple_outputs_without_download() {
        let output = json!({"files":["https://replicate.delivery/a.png", {"url":"https://replicate.delivery/b.webp"}]});
        let artifacts = collect_artifacts(Some(&output), MediaKind::Image).unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(artifacts
            .iter()
            .all(|artifact| matches!(artifact.source, ArtifactSource::Url(_))));
        assert!(artifacts.iter().all(|artifact| {
            artifact.metadata.get("retention_hint_secs") == Some(&json!(3600))
        }));
    }

    #[test]
    fn filters_cross_modality_urls_but_keeps_video_audio_companions() {
        let output = json!([
            "https://replicate.delivery/image.png",
            "https://replicate.delivery/video.mp4",
            "https://replicate.delivery/audio.mp3"
        ]);
        let images = collect_artifacts(Some(&output), MediaKind::Image).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].kind, MediaKind::Image);

        let video = collect_artifacts(Some(&output), MediaKind::Video).unwrap();
        assert_eq!(video.len(), 2);
        assert_eq!(video[0].kind, MediaKind::Video);
        assert_eq!(video[1].kind, MediaKind::Audio);
    }

    #[test]
    fn resumed_prediction_preserves_expected_count_and_reports_partial_output() {
        let value = json!({
            "id":"prediction-id",
            "status":"succeeded",
            "output":["https://replicate.delivery/image.png"]
        });
        let mut warnings = Vec::new();
        let operation = parse_image_operation(
            &value,
            "owner/model",
            MediaTask::TextToImage,
            None,
            Some(2),
            &mut warnings,
        )
        .unwrap();
        assert_eq!(operation.handle.unwrap().expected_outputs, Some(2));
        assert!(operation
            .result
            .unwrap()
            .warnings
            .iter()
            .any(|warning| warning.code == WarningCode::PartialOutput));
    }

    #[test]
    fn computes_only_labeled_estimated_cost() {
        let value = json!({"metrics":{"predict_time":2.5}});
        let estimate = CostEstimate {
            unit: UsageUnit::ComputeSeconds,
            usd_per_unit: 0.001,
            quantity: None,
        };
        let usage = prediction_usage(&value, Some(&estimate), None, &mut Vec::new());
        assert_eq!(usage.provider_reported_cost, None);
        assert_eq!(usage.estimated_cost, Some(0.0025));
    }

    #[test]
    fn failed_prediction_keeps_native_id_and_error() {
        let value = json!({
            "id":"prediction-id",
            "status":"failed",
            "error":"model ran out of memory"
        });
        let mut warnings = Vec::new();
        let operation = parse_video_operation(
            &value,
            "owner/model",
            MediaTask::TextToVideo,
            None,
            None,
            &mut warnings,
        )
        .unwrap();
        assert_eq!(operation.status, OperationStatus::Failed);
        let error = operation.error.unwrap();
        assert_eq!(error.request_id.as_deref(), Some("prediction-id"));
        assert_eq!(error.category, FailureCategory::RemoteJob);
        assert!(error.message.contains("out of memory"));
    }

    #[test]
    fn object_storage_outputs_remain_object_storage_artifacts() {
        let output = json!(["s3://bucket/video.mp4", "gs://bucket/audio.wav"]);
        let artifacts = collect_artifacts(Some(&output), MediaKind::Video).unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(artifacts
            .iter()
            .all(|artifact| matches!(artifact.source, ArtifactSource::ObjectStorageUri(_))));
    }

    #[test]
    fn raw_model_output_is_retained_without_inline_media() {
        let value = json!({
            "id":"prediction-id",
            "status":"succeeded",
            "output":{"image":"data:image/png;base64,aW1hZ2U=","score":0.9}
        });
        let metadata = output_metadata(&value);
        assert_eq!(
            metadata.pointer("/raw_output/image/inline_media_omitted"),
            Some(&json!(true))
        );
        assert_eq!(metadata.pointer("/raw_output/score"), Some(&json!(0.9)));
        assert!(!metadata.to_string().contains("aW1hZ2U="));
    }

    #[test]
    fn rejects_unknown_provider_options() {
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), json!({"mystery":true}));
        assert!(parse_options(&options).is_err());
    }

    #[test]
    fn serialized_handle_has_no_credentials() {
        let handle = JobHandle {
            provider: PROVIDER.to_string(),
            model: "owner/model".to_string(),
            task: MediaTask::TextToImage,
            remote_id: "prediction-id".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        let encoded = serde_json::to_string(&handle).unwrap();
        assert!(!encoded.to_ascii_lowercase().contains("token"));
        assert!(!encoded.to_ascii_lowercase().contains("authorization"));
        let decoded: JobHandle = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, handle);
    }

    #[test]
    fn rejects_unsafe_prediction_version() {
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), json!({"version":"../secret"}));
        assert!(parse_options(&options).is_err());
    }

    #[test]
    fn validates_webhook_events_and_cost_quantities() {
        let invalid_event = ReplicateMediaOptions {
            webhook_events_filter: Some(vec!["mystery".to_string()]),
            ..ReplicateMediaOptions::default()
        };
        assert!(invalid_event.into_provider_options().is_err());

        let invalid_cost = ReplicateMediaOptions {
            cost_estimate: Some(CostEstimate {
                unit: UsageUnit::Images,
                usd_per_unit: 0.1,
                quantity: Some(-1.0),
            }),
            ..ReplicateMediaOptions::default()
        };
        assert!(invalid_cost.into_provider_options().is_err());
    }

    #[test]
    fn field_map_is_explicit_and_conflicts_fail_closed() {
        let mut options = ReplicateMediaOptions {
            field_map: HashMap::from([("prompt".to_string(), "text_prompt".to_string())]),
            ..ReplicateMediaOptions::default()
        };
        insert_semantic(&mut options, "prompt", "prompt", json!("hello")).unwrap();
        assert_eq!(options.input.get("text_prompt"), Some(&json!("hello")));
        assert!(insert_semantic(&mut options, "prompt", "prompt", json!("again")).is_err());
    }

    #[test]
    fn transcript_segments_and_words_are_normalized() {
        let output = json!({
            "segments":[{"start":0.0,"end":1.0,"text":"hello","speaker":"A"}],
            "words":[{"start":0.0,"end":0.5,"word":"hello"}]
        });
        let segments = transcript_segments(output.get("segments"));
        let words = transcript_words(output.get("words"));
        assert_eq!(segments[0].speaker.as_deref(), Some("A"));
        assert_eq!(words[0].word, "hello");
    }

    #[tokio::test]
    async fn prediction_adapter_does_not_claim_binary_speech_streaming() {
        let provider = ReplicateMediaProvider::new();
        let request = SpeechSynthesisRequest::new("hello", "voice").with_model("owner/model");
        assert!(matches!(
            provider.stream_speech(request).await,
            Err(MediaError::UnsupportedTask { .. })
        ));
    }

    #[test]
    fn schema_cache_reuses_and_sanitizes_entries() {
        let key = "test:model@latest".to_string();
        let schema = json!({"components":{"schemas":{"Input":{"type":"object"}}}});
        cache_schema(key.clone(), schema.clone());
        assert_eq!(cached_schema(&key), Some(schema));
    }

    #[test]
    fn strict_model_schema_rejects_unknown_input_fields() {
        let schema = json!({"components":{"schemas":{"Input":{
            "type":"object",
            "additionalProperties":false,
            "properties":{"prompt":{"type":"string"}}
        }}}});
        let input = serde_json::from_value(json!({"wrong":"field"})).unwrap();
        assert!(validate_known_input_fields(&schema, &input).is_err());
    }

    #[test]
    fn typed_options_round_trip_through_strict_parser() {
        let options = ReplicateMediaOptions {
            version: Some("version-id".to_string()),
            input: serde_json::from_value(json!({"guidance":3.5})).unwrap(),
            field_map: HashMap::from([("prompt".to_string(), "caption".to_string())]),
            ..ReplicateMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        let parsed = parse_options(&options).unwrap();
        assert_eq!(parsed.version.as_deref(), Some("version-id"));
        assert_eq!(
            parsed.field_map.get("prompt").map(String::as_str),
            Some("caption")
        );
    }
}
