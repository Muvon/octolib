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

//! Higgsfield adapter for model-specific image and video generation endpoints.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

const PROVIDER: &str = "higgsfield";
/// Holds `<key_id>:<key_secret>`, the pair the official SDKs read from one variable.
const API_KEY_ENV: &str = "HIGGSFIELD_API_KEY";
const API_BASE_ENV: &str = "HIGGSFIELD_API_URL";
const API_BASE: &str = "https://api.higgsfield.ai";
const MAX_IDEMPOTENCY_KEY_LEN: usize = 255;
/// API routes that share the URL namespace with model endpoint ids.
const RESERVED_PATH_ROOTS: &[&str] = &["requests", "files", "estimate"];
const IMAGE_TASKS: &[MediaTask] = &[MediaTask::TextToImage, MediaTask::ImageEdit, MediaTask::ImageVariation];
const VIDEO_TASKS: &[MediaTask] = &[
	MediaTask::TextToVideo,
	MediaTask::ImageToVideo,
	MediaTask::ReferenceToVideo,
	MediaTask::VideoExtend,
	MediaTask::VideoEdit,
];

#[derive(Debug, Clone, Default)]
pub struct HiggsfieldMediaProvider;

/// Stable Higgsfield adapter controls. Model-native fields belong in `input`;
/// `field_map` maps portable semantic names such as `prompt`, `first_frame`,
/// or `duration` to the selected endpoint's field names.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HiggsfieldMediaOptions {
	#[serde(default)]
	pub input: Map<String, Value>,
	#[serde(default)]
	pub field_map: HashMap<String, String>,
	pub webhook: Option<String>,
	pub cost_estimate: Option<CostEstimate>,
}

impl HiggsfieldMediaOptions {
	pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
		validate_options(&self)?;
		let mut options = ProviderOptions::new();
		options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
		Ok(options)
	}
}

impl HiggsfieldMediaProvider {
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
		let key = shared::api_key(API_KEY_ENV)?;
		validate_credentials(&key)?;
		Ok(key)
	}

	fn descriptor(&self, model: &str, task: MediaTask) -> MediaModelDescriptor {
		let video = VIDEO_TASKS.contains(&task);
		let (inputs, outputs) = if video {
			(
				vec![Modality::Text, Modality::Image, Modality::Video],
				vec![Modality::Video, Modality::Audio],
			)
		} else {
			(vec![Modality::Text, Modality::Image], vec![Modality::Image])
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
				progress: CapabilitySupport::Unsupported,
				webhooks: CapabilitySupport::Supported,
				binary_streaming: CapabilitySupport::Unsupported,
				resumable: CapabilitySupport::Supported,
			},
			parameters: ParameterCapabilities {
				count: if video {
					CapabilitySupport::Unsupported
				} else {
					CapabilitySupport::Unknown
				},
				seed: CapabilitySupport::Unknown,
				dimensions: CapabilitySupport::Unsupported,
				aspect_ratio: CapabilitySupport::Unknown,
				duration: if video {
					CapabilitySupport::Unknown
				} else {
					CapabilitySupport::Unsupported
				},
				mask: CapabilitySupport::Unsupported,
				negative_prompt: CapabilitySupport::Unknown,
				output_format: CapabilitySupport::Unsupported,
			},
			limits: MediaLimits::default(),
			provider_options_schema: Some(higgsfield_options_schema()),
		}
	}

	async fn submit_request(
		&self,
		model: &str,
		input: Map<String, Value>,
		webhook: Option<&str>,
		request_options: &RequestOptions,
	) -> MediaResult<String> {
		validate_model(model)?;
		let idempotency_key = request_options.idempotency_key.as_deref();
		if let Some(idempotency_key) = idempotency_key {
			validate_idempotency_key(idempotency_key)?;
		}
		let url = format!("{}/{model}", self.api_base());
		let key = self.key()?;
		let body = Value::Object(input);
		let transport = without_idempotency_key(request_options);
		let build = || {
			let mut builder = crate::http::http_client()
				.post(&url)
				.header("Authorization", format!("Key {key}"))
				.json(&body);
			if let Some(webhook) = webhook {
				builder = builder.query(&[("hf_webhook", webhook)]);
			}
			if let Some(idempotency_key) = idempotency_key {
				builder = builder.header("Idempotency-Key", idempotency_key);
			}
			builder
		};
		// A replayed idempotency key returns the original request instead of
		// creating a second paid job, so only keyed submissions are retried.
		let response = if idempotency_key.is_some() {
			shared::send_idempotent(PROVIDER, &transport, build).await?
		} else {
			shared::send(PROVIDER, &transport, build).await?
		};
		let value = parse_json(&response)?;
		value
			.get("request_id")
			.and_then(Value::as_str)
			.map(str::to_string)
			.ok_or_else(|| MediaError::InvalidResponse {
				provider: PROVIDER.to_string(),
				message: "submission returned no request_id".to_string(),
			})
	}

	async fn request_status(&self, handle: &JobHandle) -> MediaResult<Value> {
		let key = self.key()?;
		let url = format!("{}/requests/{}/status", self.api_base(), handle.remote_id);
		let options = RequestOptions::default();
		let response = shared::send_idempotent(PROVIDER, &options, || {
			crate::http::http_client()
				.get(&url)
				.header("Authorization", format!("Key {key}"))
		})
		.await?;
		parse_json(&response)
	}

	async fn cancel_request(&self, handle: &JobHandle) -> MediaResult<()> {
		let key = self.key()?;
		let url = format!("{}/requests/{}/cancel", self.api_base(), handle.remote_id);
		let options = RequestOptions::default();
		let response = shared::send(PROVIDER, &options, || {
			crate::http::http_client()
				.post(&url)
				.header("Authorization", format!("Key {key}"))
		})
		.await?;
		require_success(&response)
	}

	/// Higgsfield endpoints take public HTTPS URLs, so inline and local media
	/// go through its presigned upload flow first.
	async fn resolve_source(&self, source: &MediaSource, request_options: &RequestOptions) -> MediaResult<String> {
		match source {
			MediaSource::Url { url, .. } => {
				shared::validate_https_url(url, "Higgsfield media source URL")?;
				Ok(url.clone())
			}
			MediaSource::Bytes { .. } | MediaSource::File { .. } => {
				let (data, media_type) = shared::source_to_bytes(source, request_options.max_source_bytes)?;
				self.upload(data, &media_type, request_options).await
			}
			_ => Err(MediaError::InvalidRequest(
				"Higgsfield media sources must be HTTPS URLs, inline bytes, or local files".to_string(),
			)),
		}
	}

	async fn upload(&self, data: Vec<u8>, media_type: &str, request_options: &RequestOptions) -> MediaResult<String> {
		let key = self.key()?;
		let url = format!("{}/files/generate-upload-url", self.api_base());
		let mut transport = without_idempotency_key(request_options);
		let body = json!({"content_type": media_type});
		let response = shared::send_idempotent(PROVIDER, &transport, || {
			crate::http::http_client()
				.post(&url)
				.header("Authorization", format!("Key {key}"))
				.json(&body)
		})
		.await?;
		let ticket = parse_json(&response)?;
		let field = |name: &str| {
			ticket
				.get(name)
				.and_then(Value::as_str)
				.ok_or_else(|| MediaError::InvalidResponse {
					provider: PROVIDER.to_string(),
					message: format!("upload URL response is missing {name}"),
				})
		};
		let upload_url = field("upload_url")?;
		let public_url = field("public_url")?;
		shared::validate_https_url(upload_url, "Higgsfield upload_url")?;
		shared::validate_https_url(public_url, "Higgsfield public_url")?;
		let upload_headers = upload_headers(&ticket)?;
		// Caller headers belong to the Higgsfield API, not the storage bucket.
		transport.extra_headers = None;
		let response = shared::send_idempotent(PROVIDER, &transport, || {
			let mut builder = crate::http::http_client().put(upload_url).body(data.clone());
			for (name, value) in &upload_headers {
				builder = builder.header(name.as_str(), value.as_str());
			}
			builder
		})
		.await?;
		shared::require_success(PROVIDER, &response)?;
		Ok(public_url.to_string())
	}
}

#[async_trait::async_trait]
impl ImageGenerationProvider for HiggsfieldMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		validate_model(model).is_ok()
	}
	fn capabilities(&self, model: &str) -> ImageCapabilities {
		self.descriptor(model, MediaTask::TextToImage)
	}

	async fn submit_image(&self, request: ImageGenerationRequest) -> MediaResult<Operation<ImageGenerationResult>> {
		validate_nonempty(&request.model, &request.prompt)?;
		let mut options = parse_options(&request.provider_options)?;
		let mut warnings = Vec::new();
		if request.mode == ImageGenerationMode::Inpaint || request.mask.is_some() {
			return Err(MediaError::UnsupportedParameter {
				provider: PROVIDER.to_string(),
				parameter: "mask".to_string(),
				reason: "Higgsfield image endpoints document no mask input".to_string(),
			});
		}
		if matches!(request.mode, ImageGenerationMode::Edit | ImageGenerationMode::Variation)
			&& request.source_images.is_empty()
		{
			return Err(MediaError::InvalidRequest(
				"image edit/variation requires a source image".to_string(),
			));
		}
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				(
					"dimensions",
					matches!(request.geometry, Some(OutputGeometry::Dimensions { .. })),
				),
				("output_format", request.output_format.is_some()),
			],
		)?;
		insert_semantic(&mut options, "prompt", "prompt", json!(request.prompt))?;
		if let Some(count) = request.count {
			insert_semantic(&mut options, "count", "batch_size", json!(count))?;
		}
		if let Some(seed) = request.seed {
			insert_semantic(&mut options, "seed", "seed", json!(seed))?;
		}
		if let Some(negative) = request.negative_prompt.as_ref() {
			insert_semantic(&mut options, "negative_prompt", "negative_prompt", json!(negative))?;
		}
		if let Some(geometry @ OutputGeometry::AspectRatio { .. }) = request.geometry {
			insert_semantic(
				&mut options,
				"aspect_ratio",
				"aspect_ratio",
				json!(geometry.as_api_string()),
			)?;
		}
		if !request.source_images.is_empty() {
			let mut urls = Vec::with_capacity(request.source_images.len());
			for source in &request.source_images {
				urls.push(self.resolve_source(source, &request.request_options).await?);
			}
			insert_semantic(&mut options, "source_images", "image_urls", json!(urls))?;
		}
		let task = task_for_image_mode(request.mode);
		let id = self
			.submit_request(
				&request.model,
				std::mem::take(&mut options.input),
				options.webhook.as_deref(),
				&request.request_options,
			)
			.await?;
		Ok(queued_operation(
			&request.model,
			task,
			&id,
			shared::resolved_cost_estimate(PROVIDER, &request.model, options.cost_estimate, None),
			warnings,
			request.count,
		))
	}

	async fn poll_image(&self, handle: &JobHandle) -> MediaResult<Operation<ImageGenerationResult>> {
		shared::validate_handle(handle, PROVIDER, IMAGE_TASKS)?;
		let value = self.request_status(handle).await?;
		let (status, error) = request_state(&value)?;
		if status != OperationStatus::Succeeded {
			return Ok(operation_from_request(handle, status, error, None, &value));
		}
		Ok(
			match shared::readable_output(image_result(handle, &value), &handle.remote_id, value.clone())? {
				Ok(result) => operation_from_request(handle, status, None, Some(result), &value),
				Err(failure) => operation_from_request(handle, OperationStatus::Failed, Some(failure), None, &value),
			},
		)
	}

	async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
		shared::validate_handle(handle, PROVIDER, IMAGE_TASKS)?;
		self.cancel_request(handle).await
	}
}

#[async_trait::async_trait]
impl VideoGenerationProvider for HiggsfieldMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		validate_model(model).is_ok()
	}
	fn capabilities(&self, model: &str) -> VideoCapabilities {
		self.descriptor(model, MediaTask::TextToVideo)
	}

	async fn submit_video(&self, request: VideoGenerationRequest) -> MediaResult<Operation<VideoGenerationResult>> {
		validate_nonempty(&request.model, &request.prompt)?;
		validate_video_mode(&request)?;
		let mut options = parse_options(&request.provider_options)?;
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("count", request.count.is_some()),
				(
					"dimensions",
					matches!(request.geometry, Some(OutputGeometry::Dimensions { .. })),
				),
				("output_format", request.output_format.is_some()),
			],
		)?;
		insert_semantic(&mut options, "prompt", "prompt", json!(request.prompt))?;
		if let Some(duration) = request.duration_secs {
			insert_semantic(&mut options, "duration", "duration", json!(whole_seconds(duration)?))?;
		}
		if let Some(seed) = request.seed {
			insert_semantic(&mut options, "seed", "seed", json!(seed))?;
		}
		if let Some(negative) = request.negative_prompt.as_ref() {
			insert_semantic(&mut options, "negative_prompt", "negative_prompt", json!(negative))?;
		}
		if let Some(geometry @ OutputGeometry::AspectRatio { .. }) = request.geometry {
			insert_semantic(
				&mut options,
				"aspect_ratio",
				"aspect_ratio",
				json!(geometry.as_api_string()),
			)?;
		}
		if let Some(source) = request.first_frame.as_ref() {
			let url = self.resolve_source(source, &request.request_options).await?;
			insert_semantic(&mut options, "first_frame", "image_url", json!(url))?;
		}
		if let Some(source) = request.last_frame.as_ref() {
			let url = self.resolve_source(source, &request.request_options).await?;
			insert_semantic(&mut options, "last_frame", "end_image_url", json!(url))?;
		}
		if !request.reference_images.is_empty() {
			let mut urls = Vec::with_capacity(request.reference_images.len());
			for source in &request.reference_images {
				urls.push(self.resolve_source(source, &request.request_options).await?);
			}
			insert_semantic(&mut options, "reference_images", "image_urls", json!(urls))?;
		}
		if let Some(source) = request.source_video.as_ref() {
			let url = self.resolve_source(source, &request.request_options).await?;
			insert_semantic(&mut options, "source_video", "video_url", json!(url))?;
		}
		let task = task_for_video_mode(request.mode);
		let id = self
			.submit_request(
				&request.model,
				std::mem::take(&mut options.input),
				options.webhook.as_deref(),
				&request.request_options,
			)
			.await?;
		Ok(queued_operation(
			&request.model,
			task,
			&id,
			shared::resolved_cost_estimate(
				PROVIDER,
				&request.model,
				options.cost_estimate,
				request.duration_secs.map(|seconds| (UsageUnit::VideoSeconds, seconds)),
			),
			warnings,
			None,
		))
	}

	async fn poll_video(&self, handle: &JobHandle) -> MediaResult<Operation<VideoGenerationResult>> {
		shared::validate_handle(handle, PROVIDER, VIDEO_TASKS)?;
		let value = self.request_status(handle).await?;
		let (status, error) = request_state(&value)?;
		if status != OperationStatus::Succeeded {
			return Ok(operation_from_request(handle, status, error, None, &value));
		}
		Ok(
			match shared::readable_output(video_result(handle, &value), &handle.remote_id, value.clone())? {
				Ok(result) => operation_from_request(handle, status, None, Some(result), &value),
				Err(failure) => operation_from_request(handle, OperationStatus::Failed, Some(failure), None, &value),
			},
		)
	}

	async fn cancel_video(&self, handle: &JobHandle) -> MediaResult<()> {
		shared::validate_handle(handle, PROVIDER, VIDEO_TASKS)?;
		self.cancel_request(handle).await
	}
}

/// `shared::send` rejects idempotency keys for providers without a documented
/// contract; Higgsfield documents one, so the adapter sends the header itself.
fn without_idempotency_key(request_options: &RequestOptions) -> RequestOptions {
	let mut options = request_options.clone();
	options.idempotency_key = None;
	options
}

/// Higgsfield answers 403 for an unfunded balance, which the shared
/// classifier would report as a permission failure.
fn require_success(response: &shared::CapturedResponse) -> MediaResult<()> {
	if response.status == reqwest::StatusCode::FORBIDDEN {
		return Err(MediaError::InsufficientCredits {
			provider: PROVIDER.to_string(),
			message: shared::error_message(&response.body),
		});
	}
	shared::require_success(PROVIDER, response)
}

fn parse_json(response: &shared::CapturedResponse) -> MediaResult<Value> {
	require_success(response)?;
	shared::parse_json(PROVIDER, response)
}

fn upload_headers(ticket: &Value) -> MediaResult<Vec<(String, String)>> {
	let headers =
		ticket
			.get("upload_headers")
			.and_then(Value::as_object)
			.ok_or_else(|| MediaError::InvalidResponse {
				provider: PROVIDER.to_string(),
				message: "upload URL response is missing upload_headers".to_string(),
			})?;
	headers
		.iter()
		.map(|(name, value)| {
			value
				.as_str()
				.map(|value| (name.clone(), value.to_string()))
				.ok_or_else(|| MediaError::InvalidResponse {
					provider: PROVIDER.to_string(),
					message: format!("upload header '{name}' is not a string"),
				})
		})
		.collect()
}

fn parse_options(options: &ProviderOptions) -> MediaResult<HiggsfieldMediaOptions> {
	let raw = shared::provider_options(options, PROVIDER)?;
	let parsed: HiggsfieldMediaOptions = serde_json::from_value(Value::Object(raw))
		.map_err(|error| MediaError::InvalidRequest(format!("invalid Higgsfield provider options: {error}")))?;
	validate_options(&parsed)?;
	Ok(parsed)
}

fn validate_options(options: &HiggsfieldMediaOptions) -> MediaResult<()> {
	if let Some(webhook) = options.webhook.as_deref() {
		shared::validate_https_url(webhook, "Higgsfield webhook")?;
	}
	if options.field_map.values().any(|field| field.is_empty()) {
		return Err(MediaError::InvalidRequest(
			"Higgsfield field_map target names must not be empty".to_string(),
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

fn validate_credentials(key: &str) -> MediaResult<()> {
	match key.split_once(':') {
		Some((id, secret)) if !id.trim().is_empty() && !secret.trim().is_empty() => Ok(()),
		_ => Err(MediaError::Authentication {
			provider: PROVIDER.to_string(),
			message: format!("{API_KEY_ENV} must be <key_id>:<key_secret>"),
		}),
	}
}

fn validate_idempotency_key(key: &str) -> MediaResult<()> {
	if key.is_empty() || key.len() > MAX_IDEMPOTENCY_KEY_LEN || !key.bytes().all(|byte| byte.is_ascii_graphic()) {
		return Err(MediaError::InvalidRequest(format!(
            "Higgsfield idempotency keys must be 1-{MAX_IDEMPOTENCY_KEY_LEN} visible ASCII characters without whitespace"
        )));
	}
	Ok(())
}

fn insert_semantic(
	options: &mut HiggsfieldMediaOptions,
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
			"Higgsfield input field '{field}' was supplied both portably and in provider_options.higgsfield.input"
		)));
	}
	options.input.insert(field, value);
	Ok(())
}

fn whole_seconds(duration: f64) -> MediaResult<u32> {
	if !duration.is_finite() || duration <= 0.0 || duration.fract() != 0.0 {
		return Err(MediaError::InvalidRequest(
			"Higgsfield duration must be a whole number of seconds".to_string(),
		));
	}
	Ok(duration as u32)
}

fn request_state(value: &Value) -> MediaResult<(OperationStatus, Option<GenerationFailure>)> {
	let native = value
		.get("status")
		.and_then(Value::as_str)
		.ok_or_else(|| MediaError::InvalidResponse {
			provider: PROVIDER.to_string(),
			message: "request is missing status".to_string(),
		})?;
	let (status, failure) = match native {
		"queued" => (OperationStatus::Queued, None),
		"in_progress" => (OperationStatus::Running, None),
		"completed" => (OperationStatus::Succeeded, None),
		"failed" => (
			OperationStatus::Failed,
			Some((FailureCategory::RemoteJob, "generation failed")),
		),
		"nsfw" => (
			OperationStatus::Failed,
			Some((
				FailureCategory::ContentPolicy,
				"content moderation rejected the input or output",
			)),
		),
		"canceled" => (
			OperationStatus::Cancelled,
			Some((FailureCategory::Cancelled, "request was canceled")),
		),
		other => {
			return Err(MediaError::InvalidResponse {
				provider: PROVIDER.to_string(),
				message: format!("unknown request status '{other}'"),
			})
		}
	};
	let failure = failure.map(|(category, default_message)| GenerationFailure {
		category,
		message: value
			.get("error")
			.and_then(Value::as_str)
			.unwrap_or(default_message)
			.to_string(),
		provider_code: Some(native.to_string()),
		http_status: None,
		request_id: value.get("request_id").and_then(Value::as_str).map(str::to_string),
		provider_metadata: Value::Null,
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

fn operation_from_request<T>(
	handle: &JobHandle,
	status: OperationStatus,
	error: Option<GenerationFailure>,
	result: Option<T>,
	value: &Value,
) -> Operation<T> {
	Operation {
		handle: Some(handle.clone()),
		status,
		progress: (status == OperationStatus::Succeeded).then_some(1.0),
		result,
		error,
		created_at: None,
		started_at: None,
		completed_at: None,
		provider_metadata: json!({
			"native_status": value.get("status"),
			"request_id": value.get("request_id"),
			"error": value.get("error")
		}),
	}
}

fn image_result(handle: &JobHandle, value: &Value) -> MediaResult<ImageGenerationResult> {
	let artifacts = image_artifacts(value)?;
	let mut warnings = handle.warnings.clone();
	warn_if_partial(handle.expected_outputs, artifacts.len(), &mut warnings);
	let usage = request_usage(
		handle.cost_estimate.as_ref(),
		Some(artifacts.len() as f64),
		&mut warnings,
	);
	Ok(ImageGenerationResult {
		artifacts,
		usage: Some(usage),
		warnings,
		safety: SafetyReport::default(),
		provider_metadata: value.clone(),
	})
}

fn video_result(handle: &JobHandle, value: &Value) -> MediaResult<VideoGenerationResult> {
	let artifacts = video_artifacts(value)?;
	let mut warnings = handle.warnings.clone();
	let usage = request_usage(handle.cost_estimate.as_ref(), None, &mut warnings);
	Ok(VideoGenerationResult {
		artifacts,
		usage: Some(usage),
		warnings,
		safety: SafetyReport::default(),
		provider_metadata: value.clone(),
	})
}

fn image_artifacts(value: &Value) -> MediaResult<Vec<MediaArtifact>> {
	let artifacts = value
		.get("images")
		.and_then(Value::as_array)
		.into_iter()
		.flatten()
		.map(|output| artifact(output, MediaKind::Image))
		.collect::<MediaResult<Vec<_>>>()?;
	if artifacts.is_empty() {
		return Err(no_artifacts("image"));
	}
	Ok(artifacts)
}

fn video_artifacts(value: &Value) -> MediaResult<Vec<MediaArtifact>> {
	let output = value.get("video").ok_or_else(|| no_artifacts("video"))?;
	Ok(vec![artifact(output, MediaKind::Video)?])
}

fn artifact(output: &Value, kind: MediaKind) -> MediaResult<MediaArtifact> {
	let url = output
		.get("url")
		.and_then(Value::as_str)
		.filter(|url| url.starts_with("https://"))
		.ok_or_else(|| MediaError::InvalidResponse {
			provider: PROVIDER.to_string(),
			message: "output is missing an HTTPS url".to_string(),
		})?;
	let media_type = output
		.get("content_type")
		.and_then(Value::as_str)
		.map(str::to_string)
		.unwrap_or_else(|| shared::media_type_from_url(url, default_media_type(kind)));
	Ok(MediaArtifact {
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
	})
}

fn warn_if_partial(expected_outputs: Option<u32>, actual_outputs: usize, warnings: &mut Vec<ProviderWarning>) {
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

fn request_usage(
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
            message: "Higgsfield request status does not report dollar cost; supply a verified cost_estimate rate when the endpoint's billing unit is known".to_string(),
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
                reason: "Higgsfield endpoints document no portable field for it; set the endpoint's own field in provider_options.higgsfield.input".to_string(),
            });
		}
		warnings.push(ProviderWarning {
			code: WarningCode::UnsupportedParameterDropped,
			message: format!("Higgsfield has no portable equivalent for '{parameter}'"),
			parameter: Some((*parameter).to_string()),
			provider_metadata: Value::Null,
		});
	}
	Ok(())
}

fn higgsfield_options_schema() -> Value {
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

fn validate_model(model: &str) -> MediaResult<()> {
	if model
		.split('/')
		.next()
		.is_some_and(|root| RESERVED_PATH_ROOTS.contains(&root))
	{
		return Err(MediaError::InvalidRequest(format!(
			"Higgsfield model '{model}' collides with a reserved API route"
		)));
	}
	let mut count = 0;
	for segment in model.split('/') {
		if segment.is_empty() || segment == "." || segment == ".." || !valid_path_segment(segment) {
			return Err(MediaError::InvalidRequest(
				"Higgsfield model contains unsupported URL path characters".to_string(),
			));
		}
		count += 1;
	}
	if count < 2 {
		return Err(MediaError::InvalidRequest(
			"Higgsfield model must be an endpoint id such as kling-video/v3.0/std/text-to-video".to_string(),
		));
	}
	Ok(())
}

fn valid_path_segment(value: &str) -> bool {
	value
		.bytes()
		.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn validate_nonempty(model: &str, prompt: &str) -> MediaResult<()> {
	validate_model(model)?;
	if prompt.trim().is_empty() {
		return Err(MediaError::InvalidRequest("prompt/text must not be empty".to_string()));
	}
	Ok(())
}

fn validate_video_mode(request: &VideoGenerationRequest) -> MediaResult<()> {
	match request.mode {
		VideoGenerationMode::ImageToVideo if request.first_frame.is_none() => Err(MediaError::InvalidRequest(
			"image-to-video requires first_frame".to_string(),
		)),
		VideoGenerationMode::ReferenceToVideo if request.reference_images.is_empty() => Err(
			MediaError::InvalidRequest("reference-to-video requires reference_images".to_string()),
		),
		VideoGenerationMode::Extend | VideoGenerationMode::Edit if request.source_video.is_none() => Err(
			MediaError::InvalidRequest("video extension/edit requires source_video".to_string()),
		),
		_ => Ok(()),
	}
}

fn task_for_image_mode(mode: ImageGenerationMode) -> MediaTask {
	match mode {
		ImageGenerationMode::Generate => MediaTask::TextToImage,
		ImageGenerationMode::Edit => MediaTask::ImageEdit,
		ImageGenerationMode::Variation => MediaTask::ImageVariation,
		ImageGenerationMode::Inpaint => MediaTask::Inpainting,
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
		message: format!("completed Higgsfield request contained no {kind} output"),
	}
}

#[cfg(test)]
#[path = "higgsfield_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "higgsfield_http_tests.rs"]
mod http_tests;
