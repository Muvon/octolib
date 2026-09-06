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

//! Runway task adapter for the Gen-4 family of image and video models.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use serde_json::{json, Map, Value};

const PROVIDER: &str = "runway";
const API_KEY_ENV: &str = "RUNWAYML_API_SECRET";
const API_BASE_ENV: &str = "RUNWAY_API_URL";
const API_BASE: &str = "https://api.dev.runwayml.com/v1";
/// Runway pins request and response shapes to a dated contract.
const API_VERSION: &str = "2024-11-06";

#[derive(Debug, Clone, Default)]
pub struct RunwayMediaProvider;

/// Stable Runway adapter controls. `body` carries verified endpoint fields that
/// have no portable equivalent, such as `contentModeration`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunwayMediaOptions {
    #[serde(default)]
    pub body: Map<String, Value>,
    pub cost_estimate: Option<CostEstimate>,
}

impl RunwayMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_runway_options(&self)?;
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
        Ok(options)
    }
}

impl RunwayMediaProvider {
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
            MediaTask::TextToImage | MediaTask::ImageEdit | MediaTask::ImageVariation => {
                (vec![Modality::Text, Modality::Image], vec![Modality::Image])
            }
            MediaTask::TextToVideo | MediaTask::ImageToVideo => (
                vec![Modality::Text, Modality::Image],
                vec![Modality::Video, Modality::Audio],
            ),
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
                progress: CapabilitySupport::Supported,
                webhooks: CapabilitySupport::Unsupported,
                binary_streaming: CapabilitySupport::Unsupported,
                resumable: CapabilitySupport::Supported,
            },
            parameters: ParameterCapabilities {
                count: CapabilitySupport::Unsupported,
                seed: CapabilitySupport::Supported,
                dimensions: CapabilitySupport::Supported,
                aspect_ratio: CapabilitySupport::Supported,
                duration: match task {
                    MediaTask::TextToVideo | MediaTask::ImageToVideo => {
                        CapabilitySupport::Supported
                    }
                    _ => CapabilitySupport::Unsupported,
                },
                mask: CapabilitySupport::Unsupported,
                negative_prompt: CapabilitySupport::Unsupported,
                output_format: CapabilitySupport::Unsupported,
            },
            limits: MediaLimits::default(),
            provider_options_schema: Some(runway_options_schema()),
        }
    }

    async fn submit_task(
        &self,
        endpoint: &str,
        body: Map<String, Value>,
        request_options: &RequestOptions,
    ) -> MediaResult<String> {
        let url = format!("{}/{endpoint}", self.api_base());
        let key = self.key()?;
        let body = Value::Object(body);
        let response = shared::send(PROVIDER, request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .bearer_auth(&key)
                .header("X-Runway-Version", API_VERSION)
                .json(&body)
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "task submission returned no id".to_string(),
            })
    }

    async fn poll_task(&self, handle: &JobHandle) -> MediaResult<Value> {
        let key = self.key()?;
        let url = format!("{}/tasks/{}", self.api_base(), handle.remote_id);
        let options = RequestOptions::default();
        let response = shared::send_idempotent(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .get(&url)
                .bearer_auth(&key)
                .header("X-Runway-Version", API_VERSION)
        })
        .await?;
        shared::parse_json(PROVIDER, &response)
    }

    async fn cancel_task(&self, handle: &JobHandle) -> MediaResult<()> {
        let key = self.key()?;
        let url = format!("{}/tasks/{}", self.api_base(), handle.remote_id);
        let options = RequestOptions::default();
        let response = shared::send(PROVIDER, &options, || {
            crate::llm::providers::shared::http_client()
                .delete(&url)
                .bearer_auth(&key)
                .header("X-Runway-Version", API_VERSION)
        })
        .await?;
        shared::require_success(PROVIDER, &response)
    }
}

#[async_trait::async_trait]
impl ImageGenerationProvider for RunwayMediaProvider {
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
        let options = parse_options(&request.provider_options)?;
        let mut warnings = request_warnings(&request.request_options);
        if request.mode == ImageGenerationMode::Inpaint || request.mask.is_some() {
            return Err(MediaError::UnsupportedParameter {
                provider: PROVIDER.to_string(),
                parameter: "mask".to_string(),
                reason: "Runway image generation has no documented mask input".to_string(),
            });
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
        let mut body = options.body.clone();
        insert_field(&mut body, "model", json!(request.model))?;
        insert_field(&mut body, "promptText", json!(request.prompt))?;
        insert_field(
            &mut body,
            "ratio",
            json!(ratio_string(request.geometry).ok_or_else(|| {
                MediaError::InvalidRequest(
                    "Runway requires an output ratio; set request.geometry".to_string(),
                )
            })?),
        )?;
        if !request.source_images.is_empty() {
            let references = request
                .source_images
                .iter()
                .map(|source| {
                    shared::source_to_data_or_uri(source, request.request_options.max_source_bytes)
                        .map(|uri| json!({"uri": uri}))
                })
                .collect::<MediaResult<Vec<_>>>()?;
            insert_field(&mut body, "referenceImages", json!(references))?;
        }
        if let Some(seed) = request.seed {
            insert_field(&mut body, "seed", json!(seed))?;
        }
        drop_unsupported(
            &request.request_options,
            &mut warnings,
            &[
                ("count", request.count.is_some()),
                ("negative_prompt", request.negative_prompt.is_some()),
                ("output_format", request.output_format.is_some()),
            ],
        )?;
        let task = task_for_image_mode(request.mode);
        let id = self
            .submit_task("text_to_image", body, &request.request_options)
            .await?;
        Ok(queued_operation(
            &request.model,
            task,
            &id,
            options.cost_estimate,
            warnings,
            None,
        ))
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
                MediaTask::ImageVariation,
            ],
        )?;
        let value = self.poll_task(handle).await?;
        let mut warnings = handle.warnings.clone();
        let (status, error) = task_state(&value)?;
        let result = if status == OperationStatus::Succeeded {
            let artifacts = collect_artifacts(&value, MediaKind::Image)?;
            if artifacts.is_empty() {
                return Err(no_artifacts("image"));
            }
            let usage = task_usage(
                handle.cost_estimate.as_ref(),
                Some(artifacts.len() as f64),
                &mut warnings,
            );
            Some(ImageGenerationResult {
                artifacts,
                usage: Some(usage),
                warnings: std::mem::take(&mut warnings),
                safety: SafetyReport::default(),
                provider_metadata: task_metadata(&value),
            })
        } else {
            None
        };
        Ok(operation_from_task(handle, status, error, result, &value))
    }

    async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[
                MediaTask::TextToImage,
                MediaTask::ImageEdit,
                MediaTask::ImageVariation,
            ],
        )?;
        self.cancel_task(handle).await
    }
}

#[async_trait::async_trait]
impl VideoGenerationProvider for RunwayMediaProvider {
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
        let options = parse_options(&request.provider_options)?;
        let mut warnings = request_warnings(&request.request_options);
        let task = match request.mode {
            VideoGenerationMode::TextToVideo => MediaTask::TextToVideo,
            VideoGenerationMode::ImageToVideo => MediaTask::ImageToVideo,
            mode => {
                return Err(MediaError::InvalidRequest(format!(
                    "Runway's dated API contract documents only text-to-video and image-to-video; {mode:?} has no verified endpoint"
                )))
            }
        };
        if task == MediaTask::ImageToVideo && request.first_frame.is_none() {
            return Err(MediaError::InvalidRequest(
                "image-to-video requires first_frame".to_string(),
            ));
        }
        if task == MediaTask::TextToVideo && request.first_frame.is_some() {
            return Err(MediaError::InvalidRequest(
                "text-to-video does not accept a first frame; use ImageToVideo".to_string(),
            ));
        }
        let mut body = options.body.clone();
        insert_field(&mut body, "model", json!(request.model))?;
        insert_field(&mut body, "promptText", json!(request.prompt))?;
        insert_field(
            &mut body,
            "ratio",
            json!(ratio_string(request.geometry).ok_or_else(|| {
                MediaError::InvalidRequest(
                    "Runway requires an output ratio; set request.geometry".to_string(),
                )
            })?),
        )?;
        if let Some(duration) = request.duration_secs {
            insert_field(&mut body, "duration", json!(whole_seconds(duration)?))?;
        }
        if let Some(seed) = request.seed {
            insert_field(&mut body, "seed", json!(seed))?;
        }
        let endpoint = if task == MediaTask::ImageToVideo {
            let first = shared::source_to_data_or_uri(
                request.first_frame.as_ref().expect("checked above"),
                request.request_options.max_source_bytes,
            )?;
            let prompt_image = match request.last_frame.as_ref() {
                Some(last) => {
                    let last = shared::source_to_data_or_uri(
                        last,
                        request.request_options.max_source_bytes,
                    )?;
                    json!([
                        {"uri": first, "position": "first"},
                        {"uri": last, "position": "last"}
                    ])
                }
                None => json!(first),
            };
            insert_field(&mut body, "promptImage", prompt_image)?;
            "image_to_video"
        } else {
            "text_to_video"
        };
        drop_unsupported(
            &request.request_options,
            &mut warnings,
            &[
                ("count", request.count.is_some()),
                ("negative_prompt", request.negative_prompt.is_some()),
                ("output_format", request.output_format.is_some()),
                ("reference_images", !request.reference_images.is_empty()),
                ("source_video", request.source_video.is_some()),
            ],
        )?;
        let id = self
            .submit_task(endpoint, body, &request.request_options)
            .await?;
        Ok(queued_operation(
            &request.model,
            task,
            &id,
            options.cost_estimate,
            warnings,
            None,
        ))
    }

    async fn poll_video(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<VideoGenerationResult>> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[MediaTask::TextToVideo, MediaTask::ImageToVideo],
        )?;
        let value = self.poll_task(handle).await?;
        let mut warnings = handle.warnings.clone();
        let (status, error) = task_state(&value)?;
        let result = if status == OperationStatus::Succeeded {
            let artifacts = collect_artifacts(&value, MediaKind::Video)?;
            if artifacts.is_empty() {
                return Err(no_artifacts("video"));
            }
            let usage = task_usage(handle.cost_estimate.as_ref(), None, &mut warnings);
            Some(VideoGenerationResult {
                artifacts,
                usage: Some(usage),
                warnings: std::mem::take(&mut warnings),
                safety: SafetyReport::default(),
                provider_metadata: task_metadata(&value),
            })
        } else {
            None
        };
        Ok(operation_from_task(handle, status, error, result, &value))
    }

    async fn cancel_video(&self, handle: &JobHandle) -> MediaResult<()> {
        shared::validate_handle(
            handle,
            PROVIDER,
            &[MediaTask::TextToVideo, MediaTask::ImageToVideo],
        )?;
        self.cancel_task(handle).await
    }
}

fn parse_options(options: &ProviderOptions) -> MediaResult<RunwayMediaOptions> {
    let raw = shared::provider_options(options, PROVIDER)?;
    let parsed: RunwayMediaOptions =
        serde_json::from_value(Value::Object(raw)).map_err(|error| {
            MediaError::InvalidRequest(format!("invalid Runway provider options: {error}"))
        })?;
    validate_runway_options(&parsed)?;
    Ok(parsed)
}

fn validate_runway_options(options: &RunwayMediaOptions) -> MediaResult<()> {
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

fn insert_field(body: &mut Map<String, Value>, field: &str, value: Value) -> MediaResult<()> {
    if body.contains_key(field) {
        return Err(MediaError::InvalidRequest(format!(
            "Runway field '{field}' was supplied both portably and in provider_options.runway.body"
        )));
    }
    body.insert(field.to_string(), value);
    Ok(())
}

/// Runway's `ratio` is the literal output resolution, written `width:height`.
fn ratio_string(geometry: Option<OutputGeometry>) -> Option<String> {
    match geometry? {
        OutputGeometry::Dimensions { width, height }
        | OutputGeometry::AspectRatio { width, height } => Some(format!("{width}:{height}")),
    }
}

fn whole_seconds(duration: f64) -> MediaResult<u32> {
    if !duration.is_finite() || duration <= 0.0 || duration.fract() != 0.0 {
        return Err(MediaError::InvalidRequest(
            "Runway duration must be a whole number of seconds".to_string(),
        ));
    }
    Ok(duration as u32)
}

fn task_state(value: &Value) -> MediaResult<(OperationStatus, Option<GenerationFailure>)> {
    let native =
        value
            .get("status")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "task is missing status".to_string(),
            })?;
    let status = match native {
        "PENDING" | "THROTTLED" => OperationStatus::Queued,
        "RUNNING" => OperationStatus::Running,
        "SUCCEEDED" => OperationStatus::Succeeded,
        "FAILED" => OperationStatus::Failed,
        "CANCELLED" | "CANCELED" => OperationStatus::Cancelled,
        other => {
            return Err(MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: format!("unknown task status '{other}'"),
            })
        }
    };
    let failure =
        matches!(status, OperationStatus::Failed | OperationStatus::Cancelled).then(|| {
            GenerationFailure {
                category: if status == OperationStatus::Cancelled {
                    FailureCategory::Cancelled
                } else {
                    FailureCategory::RemoteJob
                },
                message: value
                    .get("failure")
                    .and_then(Value::as_str)
                    .unwrap_or(native)
                    .to_string(),
                provider_code: value
                    .get("failureCode")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                http_status: None,
                request_id: value.get("id").and_then(Value::as_str).map(str::to_string),
                provider_metadata: Value::Null,
            }
        });
    Ok((status, failure))
}

fn queued_operation<T>(
    model: &str,
    task: MediaTask,
    remote_id: &str,
    estimate: Option<CostEstimate>,
    warnings: Vec<ProviderWarning>,
    expected_outputs: Option<u32>,
) -> Operation<T> {
    Operation {
        handle: Some(JobHandle {
            provider: PROVIDER.to_string(),
            model: model.to_string(),
            task,
            remote_id: remote_id.to_string(),
            cost_estimate: estimate,
            warnings,
            expected_outputs,
        }),
        status: OperationStatus::Queued,
        progress: None,
        result: None,
        error: None,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: Value::Null,
    }
}

fn operation_from_task<T>(
    handle: &JobHandle,
    status: OperationStatus,
    error: Option<GenerationFailure>,
    result: Option<T>,
    value: &Value,
) -> Operation<T> {
    Operation {
        handle: Some(handle.clone()),
        status,
        progress: value
            .get("progress")
            .and_then(Value::as_f64)
            .map(|progress| progress as f32),
        result,
        error,
        created_at: None,
        started_at: None,
        completed_at: None,
        provider_metadata: task_metadata(value),
    }
}

fn task_metadata(value: &Value) -> Value {
    json!({
        "native_status": value.get("status"),
        "id": value.get("id"),
        "createdAt": value.get("createdAt"),
        "progress": value.get("progress"),
        "failureCode": value.get("failureCode")
    })
}

fn collect_artifacts(value: &Value, kind: MediaKind) -> MediaResult<Vec<MediaArtifact>> {
    let outputs = value
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "succeeded task has no output array".to_string(),
        })?;
    let mut artifacts = Vec::with_capacity(outputs.len());
    for output in outputs {
        let Some(url) = output.as_str() else {
            continue;
        };
        if !url.starts_with("https://") {
            continue;
        }
        let media_type = shared::media_type_from_url(url, default_media_type(kind));
        artifacts.push(MediaArtifact {
            kind,
            media_type,
            source: ArtifactSource::Url(url.to_string()),
            size_bytes: None,
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

fn task_usage(
    estimate: Option<&CostEstimate>,
    output_count: Option<f64>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaUsage {
    let mut line_items = Vec::new();
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
            UsageUnit::Images => output_count,
            _ => None,
        });
        quantity.map(|quantity| quantity * rate.usd_per_unit)
    });
    if estimated_cost.is_none() {
        warnings.push(ProviderWarning {
            code: WarningCode::CostUnavailable,
            message: "Runway tasks do not report dollar cost; supply a verified cost_estimate rate when the model's billing unit is known".to_string(),
            parameter: None,
            provider_metadata: Value::Null,
        });
    }
    MediaUsage {
        line_items,
        provider_reported_cost: None,
        estimated_cost,
        currency: "USD".to_string(),
        metadata: Value::Null,
    }
}

fn drop_unsupported(
    options: &RequestOptions,
    warnings: &mut Vec<ProviderWarning>,
    parameters: &[(&str, bool)],
) -> MediaResult<()> {
    for (parameter, present) in parameters {
        if !present {
            continue;
        }
        if options.unsupported_parameter_policy == UnsupportedParameterPolicy::Error {
            return Err(MediaError::UnsupportedParameter {
                provider: PROVIDER.to_string(),
                parameter: (*parameter).to_string(),
                reason: "Runway's dated API contract has no equivalent field".to_string(),
            });
        }
        warnings.push(ProviderWarning {
            code: WarningCode::UnsupportedParameterDropped,
            message: format!("Runway has no equivalent for '{parameter}'"),
            parameter: Some((*parameter).to_string()),
            provider_metadata: Value::Null,
        });
    }
    Ok(())
}

fn request_warnings(options: &RequestOptions) -> Vec<ProviderWarning> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy == UnsupportedParameterPolicy::WarnAndDrop
    {
        vec![ProviderWarning {
            code: WarningCode::UnsupportedParameterDropped,
            message: "Runway does not document task idempotency keys".to_string(),
            parameter: Some("request_options.idempotency_key".to_string()),
            provider_metadata: Value::Null,
        }]
    } else {
        Vec::new()
    }
}

fn runway_options_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":{
            "body":{"type":"object"},
            "cost_estimate":{"type":"object"}
        }
    })
}

fn validate_model(model: &str) -> MediaResult<()> {
    let model = model.trim();
    if model.is_empty() || model.len() > 128 || model.contains(['\r', '\n']) {
        return Err(MediaError::InvalidRequest(
            "Runway model must be a non-empty model name such as gen4_turbo".to_string(),
        ));
    }
    Ok(())
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

fn task_for_image_mode(mode: ImageGenerationMode) -> MediaTask {
    match mode {
        ImageGenerationMode::Generate => MediaTask::TextToImage,
        ImageGenerationMode::Edit => MediaTask::ImageEdit,
        ImageGenerationMode::Variation => MediaTask::ImageVariation,
        ImageGenerationMode::Inpaint => MediaTask::Inpainting,
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
        message: format!("succeeded Runway task contained no usable {kind} output URL"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_task_states() {
        for (native, expected) in [
            ("PENDING", OperationStatus::Queued),
            ("THROTTLED", OperationStatus::Queued),
            ("RUNNING", OperationStatus::Running),
            ("SUCCEEDED", OperationStatus::Succeeded),
            ("FAILED", OperationStatus::Failed),
            ("CANCELLED", OperationStatus::Cancelled),
        ] {
            let value = json!({"status": native});
            assert_eq!(task_state(&value).unwrap().0, expected);
        }
        assert!(task_state(&json!({"status":"MYSTERY"})).is_err());
        assert!(task_state(&json!({})).is_err());
    }

    #[test]
    fn failed_task_keeps_failure_code_and_id() {
        let value = json!({
            "id":"task-id",
            "status":"FAILED",
            "failure":"content moderation blocked the prompt",
            "failureCode":"SAFETY.INPUT.TEXT"
        });
        let (status, failure) = task_state(&value).unwrap();
        assert_eq!(status, OperationStatus::Failed);
        let failure = failure.unwrap();
        assert_eq!(failure.provider_code.as_deref(), Some("SAFETY.INPUT.TEXT"));
        assert_eq!(failure.request_id.as_deref(), Some("task-id"));
        assert_eq!(failure.category, FailureCategory::RemoteJob);
    }

    #[test]
    fn ratio_is_a_colon_separated_resolution() {
        assert_eq!(
            ratio_string(Some(OutputGeometry::Dimensions {
                width: 1280,
                height: 768
            })),
            Some("1280:768".to_string())
        );
        assert_eq!(
            ratio_string(Some(OutputGeometry::AspectRatio {
                width: 16,
                height: 9
            })),
            Some("16:9".to_string())
        );
        assert_eq!(ratio_string(None), None);
    }

    #[test]
    fn duration_must_be_whole_seconds() {
        assert_eq!(whole_seconds(8.0).unwrap(), 8);
        assert!(whole_seconds(7.5).is_err());
        assert!(whole_seconds(0.0).is_err());
        assert!(whole_seconds(f64::NAN).is_err());
    }

    #[test]
    fn output_urls_become_artifacts_without_download() {
        let value = json!({
            "status":"SUCCEEDED",
            "output":["https://cdn.runwayml.com/generated.mp4"]
        });
        let artifacts = collect_artifacts(&value, MediaKind::Video).unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].media_type, "video/mp4");
        assert!(matches!(artifacts[0].source, ArtifactSource::Url(_)));
        assert!(collect_artifacts(&json!({"status":"SUCCEEDED"}), MediaKind::Video).is_err());
    }

    #[test]
    fn progress_is_reported_while_running() {
        let handle = JobHandle {
            provider: PROVIDER.to_string(),
            model: "gen4_turbo".to_string(),
            task: MediaTask::TextToVideo,
            remote_id: "task-id".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        let value = json!({"id":"task-id","status":"RUNNING","progress":0.42});
        let operation: Operation<VideoGenerationResult> =
            operation_from_task(&handle, OperationStatus::Running, None, None, &value);
        assert_eq!(operation.status, OperationStatus::Running);
        assert_eq!(operation.progress, Some(0.42));
        assert_eq!(operation.handle.unwrap().remote_id, "task-id");
    }

    #[test]
    fn body_conflicts_fail_closed_and_options_round_trip() {
        let options = RunwayMediaOptions {
            body: serde_json::from_value(
                json!({"contentModeration":{"publicFigureThreshold":"low"}}),
            )
            .unwrap(),
            ..RunwayMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        let parsed = parse_options(&options).unwrap();
        assert!(parsed.body.contains_key("contentModeration"));

        let mut body = parsed.body.clone();
        insert_field(&mut body, "model", json!("gen4_turbo")).unwrap();
        assert!(insert_field(&mut body, "model", json!("other")).is_err());

        let mut unknown = ProviderOptions::new();
        unknown.insert(PROVIDER.to_string(), json!({"mystery":true}));
        assert!(parse_options(&unknown).is_err());
    }

    #[test]
    fn unsupported_parameters_error_or_warn_by_policy() {
        let strict = RequestOptions::default();
        let mut warnings = Vec::new();
        assert!(drop_unsupported(&strict, &mut warnings, &[("count", true)]).is_err());

        let lenient = RequestOptions {
            unsupported_parameter_policy: UnsupportedParameterPolicy::WarnAndDrop,
            ..RequestOptions::default()
        };
        drop_unsupported(&lenient, &mut warnings, &[("count", true)]).unwrap();
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].code, WarningCode::UnsupportedParameterDropped);
    }

    #[tokio::test]
    async fn unverified_video_modes_are_rejected_before_any_request() {
        let provider = RunwayMediaProvider::new();
        let request = VideoGenerationRequest {
            mode: VideoGenerationMode::Extend,
            source_video: Some(MediaSource::Url {
                url: "https://example.test/clip.mp4".to_string(),
                media_type: None,
            }),
            ..VideoGenerationRequest::new("extend this").with_model("gen4_turbo")
        };
        assert!(matches!(
            provider.submit_video(request).await,
            Err(MediaError::InvalidRequest(_))
        ));
    }

    #[test]
    fn serialized_handle_has_no_credentials() {
        let handle = JobHandle {
            provider: PROVIDER.to_string(),
            model: "gen4_turbo".to_string(),
            task: MediaTask::TextToVideo,
            remote_id: "task-id".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        let encoded = serde_json::to_string(&handle).unwrap();
        assert!(!encoded.to_ascii_lowercase().contains("secret"));
        assert!(!encoded.to_ascii_lowercase().contains("authorization"));
        assert_eq!(serde_json::from_str::<JobHandle>(&encoded).unwrap(), handle);
    }
}
