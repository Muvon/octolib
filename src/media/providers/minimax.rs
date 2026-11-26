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

//! Native MiniMax image, H3 video, synchronous speech, and speech-to-text generation.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use base64::Engine;
use serde_json::{json, Value};

const PROVIDER: &str = "minimax";
const API_KEY_ENV: &str = "MINIMAX_API_KEY";
const API_BASE_ENV: &str = "MINIMAX_MEDIA_API_URL";
const API_BASE: &str = "https://api.minimax.io";
const IMAGE_ENDPOINT: &str = "/v1/image_generation";
const H3_IMAGE_ENDPOINT: &str = "/v1/images/generations";
const H3_IMAGE_EDIT_ENDPOINT: &str = "/v1/images/edits";
const TRANSCRIPTION_ENDPOINT: &str = "/v1/speech_to_text";
const VIDEO_ENDPOINT: &str = "/v2/video_generation";
const VIDEO_QUERY_ENDPOINT: &str = "/v2/query/video_generation";
const SPEECH_ENDPOINT: &str = "/v1/t2a_v2";
const FILE_UPLOAD_ENDPOINT: &str = "/v1/files/upload";
const FILE_DELETE_ENDPOINT: &str = "/v1/files/delete";
/// The file purpose the V2 video endpoint reads `mm_file://` inputs from.
const VIDEO_INPUT_PURPOSE: &str = "video_generation_input";
/// The file purpose the H3-Image edit endpoint reads `mm_file://` inputs from.
const IMAGE_INPUT_PURPOSE: &str = "image_generation_input";
const DEFAULT_VOICE: &str = "English_expressive_narrator";
const DEFAULT_VOICE_SENTINEL: &str = "default";
const DEFAULT_VIDEO_DURATION_SECS: u32 = 5;
const DEFAULT_RESOLUTION: &str = "768P";
const IMAGE_PROMPT_LIMIT: usize = 1500;
const H3_IMAGE_PROMPT_LIMIT: usize = 32000;
const H3_MIN_IMAGE_PIXELS: u64 = 655_360;
const H3_MAX_IMAGE_PIXELS: u64 = 20_000_000;
const MAX_H3_RATIO_SIDE: u32 = 300;
const MAX_TRANSCRIPTION_BYTES: usize = 50 * 1024 * 1024;
const SPEECH_TEXT_LIMIT: usize = 10000;
const MAX_IMAGES: u32 = 9;
const MAX_H3_IMAGES: u32 = 10;
const MAX_H3_REFERENCE_IMAGES: usize = 16;
const MAX_REFERENCE_IMAGES: usize = 9;
const MIN_IMAGE_DIMENSION: u32 = 512;
const MAX_IMAGE_DIMENSION: u32 = 2048;
const IMAGE_DIMENSION_MULTIPLE: u32 = 8;
const MIN_VIDEO_SECONDS: f64 = 4.0;
const H3_MAX_MIN_SECONDS: f64 = 5.0;
const MAX_VIDEO_SECONDS: f64 = 15.0;
const MIN_SPEECH_SPEED: f64 = 0.5;
const MAX_SPEECH_SPEED: f64 = 2.0;
const IMAGE_RATIOS: &[&str] = &["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"];
const VIDEO_RATIOS: &[&str] = &["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const VIDEO_MODELS: &[&str] = &[
	"MiniMax-H3",
	"MiniMax-H3-Highspeed",
	"MiniMax-H3-Lite",
	"MiniMax-H3-Max",
	"MiniMax-H3-Max-Turbo",
	"MiniMax-H3-MG",
	"MiniMax-H3-NC",
];
const SPEECH_MODELS: &[&str] = &[
	"speech-2.8-hd",
	"speech-2.8-turbo",
	"speech-2.6-hd",
	"speech-2.6-turbo",
	"speech-02-hd",
	"speech-02-turbo",
];
const IMAGE_MODELS: &[&str] = &[
	"image-01",
	"image-01-live",
	"MiniMax-H3-Image",
	"MiniMax-H3-Image-Highspeed",
];
const TRANSCRIPTION_MODEL: &str = "asr-1.0";

fn is_h3_image(model: &str) -> bool {
	model.starts_with("MiniMax-H3-Image")
}

#[derive(Debug, Clone, Default)]
pub struct MinimaxMediaProvider;

impl MinimaxMediaProvider {
	pub fn new() -> Self {
		Self
	}

	fn api_base(&self) -> MediaResult<String> {
		match std::env::var(API_BASE_ENV) {
			Ok(base) => Ok(base.trim_end_matches('/').to_string()),
			Err(std::env::VarError::NotPresent) => Ok(API_BASE.to_string()),
			Err(error) => Err(MediaError::InvalidRequest(format!("invalid {API_BASE_ENV}: {error}"))),
		}
	}

	/// Upload one input to MiniMax's file store under `purpose` and return its file id.
	async fn upload_input(&self, source: &MediaSource, purpose: &str, options: &RequestOptions) -> MediaResult<u64> {
		let (data, media_type) = shared::source_to_bytes(source, options.max_source_bytes)?;
		// MiniMax types the file by its name; the part carries no Content-Type.
		let file_name = format!("input.{}", file_extension(&media_type));
		let url = format!("{}{FILE_UPLOAD_ENDPOINT}", self.api_base()?);
		let key = shared::api_key(API_KEY_ENV)?;
		let response = shared::send(PROVIDER, options, || {
			let file = reqwest::multipart::Part::bytes(data.clone()).file_name(file_name.clone());
			let form = reqwest::multipart::Form::new()
				.text("purpose", purpose.to_string())
				.part("file", file);
			crate::http::http_client().post(&url).bearer_auth(&key).multipart(form)
		})
		.await?;
		parse_response(&response, true)?
			.pointer("/file/file_id")
			.and_then(Value::as_u64)
			.ok_or_else(|| invalid_response("missing file.file_id"))
	}

	/// Delete an uploaded input. A failure does not fail the generation it fed:
	/// the task already exists, and MiniMax drops the file after 7 days anyway.
	async fn delete_input(&self, file_id: u64, purpose: &str) {
		let body = json!({"file_id": file_id, "purpose": purpose});
		if let Err(error) = self
			.post(FILE_DELETE_ENDPOINT, &body, &RequestOptions::default(), true)
			.await
		{
			tracing::warn!(file_id, %error, "MiniMax input was not deleted; it expires after 7 days");
		}
	}

	/// Reference one model input. Local bytes are uploaded to MiniMax's private
	/// file store and travel as `mm_file://`, never inlined: MiniMax caps the
	/// whole body at 64 MB and its file store exposes no download link.
	async fn input_url(
		&self,
		source: &MediaSource,
		purpose: &str,
		options: &RequestOptions,
		uploaded: &mut Vec<u64>,
	) -> MediaResult<String> {
		match source {
			MediaSource::Bytes { .. } | MediaSource::File { .. } => {
				let file_id = self.upload_input(source, purpose, options).await?;
				uploaded.push(file_id);
				Ok(format!("mm_file://{file_id}"))
			}
			_ => shared::source_to_data_or_uri(source, options.max_source_bytes),
		}
	}

	/// H3-Image answers with URLs and bills by token, so the cost is computed
	/// from the usage MiniMax reports instead of a flat per-image rate.
	/// https://platform.minimax.io/docs/api-reference/image-generation-h3-generations
	async fn submit_h3_image(
		&self,
		request: &ImageGenerationRequest,
		edit: bool,
		count: u32,
		mut warnings: Vec<ProviderWarning>,
	) -> MediaResult<Operation<ImageGenerationResult>> {
		let mut body = json!({"model":request.model, "prompt":request.prompt, "n":count, "response_format":"url"});
		match request.geometry {
			None => {}
			Some(OutputGeometry::Dimensions { width, height }) => {
				if !h3_image_shape(
					u64::from(width),
					u64::from(height),
					H3_MIN_IMAGE_PIXELS,
					H3_MAX_IMAGE_PIXELS,
				) {
					return Err(MediaError::InvalidRequest(format!(
						"MiniMax H3-Image size must cover {H3_MIN_IMAGE_PIXELS}..={H3_MAX_IMAGE_PIXELS} pixels with a side ratio in 1:3..=3:1"
					)));
				}
				body["size"] = json!(format!("{width}x{height}"));
			}
			Some(OutputGeometry::AspectRatio { width, height }) => {
				if width > MAX_H3_RATIO_SIDE
					|| height > MAX_H3_RATIO_SIDE
					|| !h3_image_shape(u64::from(width), u64::from(height), 1, u64::MAX)
				{
					return Err(MediaError::InvalidRequest(format!(
						"MiniMax H3-Image aspect ratio sides must be 1..={MAX_H3_RATIO_SIDE} with a ratio in 1:3..=3:1"
					)));
				}
				body["aspect_ratio"] = json!(format!("{width}:{height}"));
			}
		}
		let mut uploaded = Vec::new();
		let endpoint = if edit {
			let mut images = Vec::with_capacity(request.source_images.len());
			for source in &request.source_images {
				let image_url = self
					.input_url(source, IMAGE_INPUT_PURPOSE, &request.request_options, &mut uploaded)
					.await?;
				images.push(json!({"image_url": image_url}));
			}
			body["images"] = Value::Array(images);
			H3_IMAGE_EDIT_ENDPOINT
		} else {
			H3_IMAGE_ENDPOINT
		};
		let response = self.post(endpoint, &body, &request.request_options, false).await;
		// The endpoint answers only once every image is done, so each input has been read.
		for file_id in uploaded {
			self.delete_input(file_id, IMAGE_INPUT_PURPOSE).await;
		}
		let value = response?;
		let data = value
			.get("data")
			.and_then(Value::as_array)
			.ok_or_else(|| invalid_response("missing data array"))?;
		let mut artifacts = Vec::with_capacity(data.len());
		let mut reasons = Vec::new();
		let mut first_code = None;
		for (index, output) in data.iter().enumerate() {
			if let Some(error) = output.get("error") {
				let code = required_str(error, "/code")?;
				reasons.push(format!("image {index}: {} ({code})", required_str(error, "/message")?));
				first_code.get_or_insert(code);
				continue;
			}
			let url = required_str(output, "/url")?;
			shared::validate_https_url(url, "data.url").map_err(|error| invalid_response(error.to_string()))?;
			artifacts.push(MediaArtifact {
				kind: MediaKind::Image,
				media_type: h3_image_media_type(required_str(output, "/output_format")?)?.to_string(),
				size_bytes: None,
				dimensions: Some(image_size(required_str(output, "/size")?)?),
				duration_secs: None,
				frame_rate: None,
				sample_rate_hz: None,
				channels: None,
				expires_at: None,
				source: ArtifactSource::Url(url.to_string()),
				metadata: output
					.get("revised_prompt")
					.map(|prompt| json!({"revised_prompt": prompt}))
					.unwrap_or(Value::Null),
			});
		}
		if artifacts.is_empty() {
			if reasons.is_empty() {
				return Err(invalid_response("data contains no images"));
			}
			return Err(MediaError::RemoteFailure(GenerationFailure {
				// 1026 and 1027 are the content-safety codes, 1008 insufficient balance.
				category: match first_code {
					Some("1026" | "1027") => FailureCategory::ContentPolicy,
					Some("1008") => FailureCategory::InsufficientCredits,
					_ => FailureCategory::RemoteJob,
				},
				message: format!("MiniMax generated no images: {}", reasons.join("; ")),
				provider_code: first_code.map(str::to_string),
				http_status: None,
				request_id: None,
				provider_metadata: Value::Null,
			}));
		}
		let safety = if reasons.is_empty() {
			SafetyReport::default()
		} else {
			warnings.push(ProviderWarning {
				code: WarningCode::PartialOutput,
				message: format!(
					"MiniMax returned {} of {count} requested images; {}",
					artifacts.len(),
					reasons.join("; ")
				),
				parameter: Some("count".to_string()),
				provider_metadata: Value::Null,
			});
			SafetyReport {
				status: SafetyStatus::PartiallyFiltered,
				filtered_count: reasons.len() as u32,
				reasons,
				provider_metadata: Value::Null,
			}
		};
		let usage = match value.get("usage") {
			Some(reported) => h3_image_usage(&request.model, reported)?,
			None => {
				warnings.push(ProviderWarning {
					code: WarningCode::CostUnavailable,
					message: "MiniMax reported no usage; H3-Image cost is unavailable".to_string(),
					parameter: None,
					provider_metadata: Value::Null,
				});
				MediaUsage::default()
			}
		};
		Ok(Operation::completed(ImageGenerationResult {
			artifacts,
			usage: Some(usage),
			warnings,
			safety,
			provider_metadata: json!({"created":value.get("created"), "request_id":value.get("request_id")}),
		}))
	}

	async fn post(&self, endpoint: &str, body: &Value, options: &RequestOptions, v1: bool) -> MediaResult<Value> {
		let url = format!("{}{endpoint}", self.api_base()?);
		let key = shared::api_key(API_KEY_ENV)?;
		let response = shared::send(PROVIDER, options, || {
			crate::http::http_client().post(&url).bearer_auth(&key).json(body)
		})
		.await?;
		parse_response(&response, v1)
	}

	fn descriptor(&self, model: &str, task: MediaTask) -> MediaModelDescriptor {
		let video = task == MediaTask::TextToVideo;
		let image = task == MediaTask::TextToImage;
		let transcription = task == MediaTask::SpeechToText;
		let support = |yes| {
			if yes {
				CapabilitySupport::Supported
			} else {
				CapabilitySupport::Unsupported
			}
		};
		MediaModelDescriptor {
			provider: PROVIDER.to_string(),
			model: model.to_string(),
			tasks: if video {
				vec![task, MediaTask::ImageToVideo, MediaTask::ReferenceToVideo]
			} else {
				vec![task]
			},
			input_modalities: if video {
				vec![Modality::Text, Modality::Image, Modality::Video]
			} else if transcription {
				vec![Modality::Audio]
			} else {
				vec![Modality::Text]
			},
			output_modalities: vec![if image {
				Modality::Image
			} else if video {
				Modality::Video
			} else if transcription {
				Modality::Text
			} else {
				Modality::Audio
			}],
			execution: ExecutionCapabilities {
				immediate: support(!video),
				persistent_jobs: support(video),
				polling: support(video),
				cancellation: CapabilitySupport::Unsupported,
				progress: CapabilitySupport::Unsupported,
				webhooks: CapabilitySupport::Unsupported,
				binary_streaming: CapabilitySupport::Unsupported,
				resumable: support(video),
			},
			parameters: ParameterCapabilities {
				count: support(image),
				seed: CapabilitySupport::Unsupported,
				dimensions: support(image),
				aspect_ratio: support(image || video),
				duration: support(video),
				mask: CapabilitySupport::Unsupported,
				negative_prompt: CapabilitySupport::Unsupported,
				output_format: support(!image && !video && !transcription),
			},
			limits: MediaLimits {
				max_outputs: Some(if !image {
					1
				} else if is_h3_image(model) {
					MAX_H3_IMAGES
				} else {
					MAX_IMAGES
				}),
				max_duration_secs: video.then_some(MAX_VIDEO_SECONDS),
				..MediaLimits::default()
			},
			provider_options_schema: Some(if video {
				json!({"type":"object", "additionalProperties":false, "properties":{
					"resolution":{"type":"string", "enum":["480P", "768P", "2K"]}
				}})
			} else {
				json!({"type":"object", "additionalProperties":false})
			}),
		}
	}
}

#[async_trait::async_trait]
impl ImageGenerationProvider for MinimaxMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		IMAGE_MODELS.contains(&model)
	}
	fn capabilities(&self, model: &str) -> ImageCapabilities {
		self.descriptor(model, MediaTask::TextToImage)
	}

	async fn submit_image(&self, request: ImageGenerationRequest) -> MediaResult<Operation<ImageGenerationResult>> {
		let h3 = is_h3_image(&request.model);
		validate_request(&request.model, &request.prompt, MediaTask::TextToImage)?;
		let prompt_limit = if h3 { H3_IMAGE_PROMPT_LIMIT } else { IMAGE_PROMPT_LIMIT };
		if request.prompt.chars().count() > prompt_limit {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax prompt exceeds {prompt_limit} characters"
			)));
		}
		// H3-Image also edits from reference images; image-01 only generates.
		let edit = matches!(request.mode, ImageGenerationMode::Edit);
		match request.mode {
			ImageGenerationMode::Generate => {}
			ImageGenerationMode::Edit if h3 => {}
			ImageGenerationMode::Edit => return Err(unsupported_task(&request.model, MediaTask::ImageEdit)),
			ImageGenerationMode::Variation => return Err(unsupported_task(&request.model, MediaTask::ImageVariation)),
			ImageGenerationMode::Inpaint => return Err(unsupported_task(&request.model, MediaTask::Inpainting)),
		}
		validate_options(&request.provider_options, false)?;
		if edit && (request.source_images.is_empty() || request.source_images.len() > MAX_H3_REFERENCE_IMAGES) {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax image editing requires 1..={MAX_H3_REFERENCE_IMAGES} source_images"
			)));
		}
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("source_images", !edit && !request.source_images.is_empty()),
				("mask", request.mask.is_some()),
				("seed", request.seed.is_some()),
				("negative_prompt", request.negative_prompt.is_some()),
				("output_format", request.output_format.is_some()),
				// image-01-live picks its own width and height.
				(
					"geometry",
					&request.model == "image-01-live"
						&& matches!(request.geometry, Some(OutputGeometry::Dimensions { .. })),
				),
			],
		)?;
		let count = request.count.unwrap_or(1);
		let max_images = if h3 { MAX_H3_IMAGES } else { MAX_IMAGES };
		if !(1..=max_images).contains(&count) {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax count must be 1..={max_images}"
			)));
		}
		if h3 {
			return self.submit_h3_image(&request, edit, count, warnings).await;
		}
		// image-01-live picks its own size, so dropped width and height are not sent either.
		let geometry = match request.geometry {
			Some(OutputGeometry::Dimensions { .. }) if &request.model == "image-01-live" => None,
			geometry => geometry,
		};
		let mut body = json!({"model":request.model, "prompt":request.prompt, "n":count, "response_format":"base64"});
		match geometry {
			Some(OutputGeometry::Dimensions { width, height }) => {
				if [width, height].iter().any(|value| {
					!(MIN_IMAGE_DIMENSION..=MAX_IMAGE_DIMENSION).contains(value)
						|| value % IMAGE_DIMENSION_MULTIPLE != 0
				}) {
					return Err(MediaError::InvalidRequest(format!("MiniMax image dimensions must be {MIN_IMAGE_DIMENSION}..={MAX_IMAGE_DIMENSION} and divisible by {IMAGE_DIMENSION_MULTIPLE}")));
				}
				body["width"] = json!(width);
				body["height"] = json!(height);
			}
			geometry => body["aspect_ratio"] = json!(aspect_ratio(geometry, "1:1", IMAGE_RATIOS)?),
		}
		let value = self.post(IMAGE_ENDPOINT, &body, &request.request_options, true).await?;
		let metadata = value
			.get("metadata")
			.ok_or_else(|| invalid_response("missing metadata"))?;
		let outputs = value
			.pointer("/data/image_base64")
			.and_then(Value::as_array)
			.ok_or_else(|| invalid_response("missing data.image_base64 array"))?;
		if outputs.is_empty() {
			return Err(invalid_response("data.image_base64 contains no images"));
		}
		let mut artifacts = Vec::with_capacity(outputs.len());
		for output in outputs {
			let encoded = output
				.as_str()
				.ok_or_else(|| invalid_response("data.image_base64 contains a non-string image"))?;
			let bytes = base64::engine::general_purpose::STANDARD
				.decode(encoded)
				.map_err(|error| invalid_response(format!("invalid data.image_base64: {error}")))?;
			let media_type = image_media_type(&bytes)?;
			artifacts.push(inline_artifact(MediaKind::Image, media_type, bytes, None));
		}
		if artifacts.len() < count as usize {
			warnings.push(ProviderWarning {
				code: WarningCode::PartialOutput,
				message: format!("MiniMax returned {} of {count} requested images", artifacts.len()),
				parameter: Some("count".to_string()),
				provider_metadata: Value::Null,
			});
		}
		let estimate = shared::resolved_cost_estimate(PROVIDER, &request.model, None, None, None);
		let usage = usage(
			&request.model,
			UsageUnit::Images,
			Some(artifacts.len() as f64),
			estimate.as_ref(),
			&mut warnings,
		);
		Ok(Operation::completed(ImageGenerationResult {
			artifacts,
			usage: Some(usage),
			warnings,
			safety: SafetyReport::default(),
			provider_metadata: metadata.clone(),
		}))
	}

	async fn poll_image(&self, handle: &JobHandle) -> MediaResult<Operation<ImageGenerationResult>> {
		Err(unsupported_task(&handle.model, handle.task))
	}
	async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
		Err(unsupported_task(&handle.model, handle.task))
	}
}

#[async_trait::async_trait]
impl VideoGenerationProvider for MinimaxMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		VIDEO_MODELS.contains(&model)
	}
	fn capabilities(&self, model: &str) -> VideoCapabilities {
		self.descriptor(model, MediaTask::TextToVideo)
	}

	async fn submit_video(&self, request: VideoGenerationRequest) -> MediaResult<Operation<VideoGenerationResult>> {
		validate_request(&request.model, &request.prompt, MediaTask::TextToVideo)?;
		// MiniMax forbids mixing start/end frames with reference inputs.
		let framed = request.first_frame.is_some() || request.last_frame.is_some();
		let referenced = !request.reference_images.is_empty() || request.source_video.is_some();
		let task = match (request.mode, framed, referenced) {
			(VideoGenerationMode::TextToVideo, false, false) => MediaTask::TextToVideo,
			(VideoGenerationMode::ImageToVideo, true, false) => MediaTask::ImageToVideo,
			(VideoGenerationMode::ImageToVideo, false, false) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax image-to-video requires first_frame or last_frame".to_string(),
				))
			}
			(VideoGenerationMode::ReferenceToVideo, false, true) => MediaTask::ReferenceToVideo,
			(VideoGenerationMode::ReferenceToVideo, false, false) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax reference-to-video requires reference_images or source_video".to_string(),
				))
			}
			(VideoGenerationMode::TextToVideo, true, false) | (VideoGenerationMode::ReferenceToVideo, true, false) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax first_frame or last_frame requires ImageToVideo mode".to_string(),
				))
			}
			(VideoGenerationMode::TextToVideo, false, true) | (VideoGenerationMode::ImageToVideo, false, true) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax reference inputs require ReferenceToVideo mode".to_string(),
				))
			}
			(VideoGenerationMode::Extend, _, _) => {
				return Err(unsupported_task(&request.model, MediaTask::VideoExtend))
			}
			(VideoGenerationMode::Edit, _, _) => return Err(unsupported_task(&request.model, MediaTask::VideoEdit)),
			_ => {
				return Err(MediaError::InvalidRequest(
					"MiniMax first/last frames and reference inputs are mutually exclusive".to_string(),
				))
			}
		};
		let resolution =
			validate_options(&request.provider_options, true)?.unwrap_or_else(|| DEFAULT_RESOLUTION.to_string());
		let rates = video_rates(&request.model, &resolution)?;
		// The extended models bill a reference task's output at a higher rate.
		let estimate = CostEstimate {
			usd_per_unit: if task == MediaTask::ReferenceToVideo {
				rates.reference_output
			} else {
				rates.output
			},
			..rates.estimate()
		};
		let duration = request.duration_secs.unwrap_or(f64::from(DEFAULT_VIDEO_DURATION_SECS));
		validate_duration(&request.model, duration)?;
		if request.reference_images.len() > MAX_REFERENCE_IMAGES {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax accepts at most {MAX_REFERENCE_IMAGES} reference images"
			)));
		}
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("count", request.count.is_some_and(|count| count != 1)),
				("seed", request.seed.is_some()),
				("negative_prompt", request.negative_prompt.is_some()),
				(
					"geometry",
					task == MediaTask::ImageToVideo && request.geometry.is_some(),
				),
				(
					"output_format",
					request.output_format.is_some_and(|format| format != VideoFormat::Mp4),
				),
			],
		)?;
		// H3 derives image-to-video geometry from the frame. Lite and NC do not
		// accept `adaptive` for reference tasks, so those name a fixed ratio.
		let ratio = match task {
			MediaTask::ImageToVideo => "adaptive".to_string(),
			MediaTask::ReferenceToVideo if adaptive_reference(&request.model) => {
				aspect_ratio(request.geometry, "adaptive", VIDEO_RATIOS)?
			}
			_ => aspect_ratio(request.geometry, "16:9", VIDEO_RATIOS)?,
		};
		let mut content = vec![json!({"type":"text", "text":request.prompt})];
		let mut uploaded = Vec::new();
		for (role, source) in [
			("first_frame", request.first_frame.as_ref()),
			("last_frame", request.last_frame.as_ref()),
		] {
			if let Some(source) = source {
				let url = self
					.input_url(source, VIDEO_INPUT_PURPOSE, &request.request_options, &mut uploaded)
					.await?;
				content.push(json!({"type":"image_url", "image_url":{"url":url}, "role":role}));
			}
		}
		for source in &request.reference_images {
			let url = self
				.input_url(source, VIDEO_INPUT_PURPOSE, &request.request_options, &mut uploaded)
				.await?;
			content.push(json!({"type":"image_url", "image_url":{"url":url}, "role":"reference_image"}));
		}
		if let Some(source) = request.source_video.as_ref() {
			let url = self
				.input_url(source, VIDEO_INPUT_PURPOSE, &request.request_options, &mut uploaded)
				.await?;
			content.push(json!({"type":"video_url", "video_url":{"url":url}, "role":"reference_video"}));
		}
		let body = json!({"model":request.model, "content":content, "resolution":resolution, "duration":duration as u32, "ratio":ratio});
		let submitted = self.post(VIDEO_ENDPOINT, &body, &request.request_options, false).await;
		// MiniMax reads the inputs when the task is created (verified live
		// 2026-10-31: deleted right after submit, the task still succeeded), so the
		// user's media is removed now rather than kept for the 7-day file retention.
		for file_id in uploaded {
			self.delete_input(file_id, VIDEO_INPUT_PURPOSE).await;
		}
		let value = submitted?;
		let id = required_str(&value, "/task_id")?;
		let handle = JobHandle {
			provider: PROVIDER.to_string(),
			model: request.model,
			task,
			remote_id: id.to_string(),
			// The requested length stands in when the finished task reports none.
			cost_estimate: Some(CostEstimate {
				quantity: Some(duration),
				..estimate
			}),
			warnings,
			expected_outputs: Some(1),
		};
		Ok(operation(&handle, OperationStatus::Queued, None, None, Value::Null))
	}

	async fn poll_video(&self, handle: &JobHandle) -> MediaResult<Operation<VideoGenerationResult>> {
		shared::validate_handle(
			handle,
			PROVIDER,
			&[
				MediaTask::TextToVideo,
				MediaTask::ImageToVideo,
				MediaTask::ReferenceToVideo,
			],
		)?;
		let url = format!("{}{VIDEO_QUERY_ENDPOINT}/{}", self.api_base()?, handle.remote_id);
		let key = shared::api_key(API_KEY_ENV)?;
		let response = shared::send_idempotent(PROVIDER, &RequestOptions::default(), || {
			crate::http::http_client().get(&url).bearer_auth(&key)
		})
		.await?;
		let value = parse_response(&response, false)?;
		let task = value.get("task").ok_or_else(|| invalid_response("missing task"))?;
		let (status, failure) = task_state(task, handle)?;
		if status != OperationStatus::Succeeded {
			return Ok(operation(handle, status, failure, None, task.clone()));
		}
		Ok(
			match shared::readable_output(video_result(handle, task), &handle.remote_id, task.clone())? {
				Ok(result) => operation(handle, status, None, Some(result), task.clone()),
				Err(failure) => operation(handle, OperationStatus::Failed, Some(failure), None, task.clone()),
			},
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
			],
		)?;
		Err(unsupported_task(&handle.model, handle.task))
	}
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for MinimaxMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		SPEECH_MODELS.contains(&model)
	}
	fn capabilities(&self, model: &str) -> SpeechCapabilities {
		self.descriptor(model, MediaTask::TextToSpeech)
	}

	async fn submit_speech(&self, request: SpeechSynthesisRequest) -> MediaResult<Operation<SpeechSynthesisResult>> {
		validate_request(&request.model, &request.text, MediaTask::TextToSpeech)?;
		let characters = request.text.chars().count();
		if characters >= SPEECH_TEXT_LIMIT {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax text must contain fewer than {SPEECH_TEXT_LIMIT} characters"
			)));
		}
		validate_options(&request.provider_options, false)?;
		let format = match request.output.format {
			AudioFormat::Mp3 => "mp3",
			AudioFormat::Pcm => "pcm",
			AudioFormat::Flac => "flac",
			AudioFormat::Wav => "wav",
			AudioFormat::Ogg => "opus",
			_ => {
				return Err(MediaError::UnsupportedParameter {
					provider: PROVIDER.to_string(),
					parameter: "output.format".to_string(),
					reason: "MiniMax supports mp3, pcm, flac, wav, and opus".to_string(),
				})
			}
		};
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("language", request.language.is_some()),
				("instructions", request.instructions.is_some()),
				("output.sample_rate_hz", request.output.sample_rate_hz.is_some()),
				("output.channels", request.output.channels.is_some()),
			],
		)?;
		let voice = if request.voice == DEFAULT_VOICE_SENTINEL {
			DEFAULT_VOICE
		} else {
			&request.voice
		};
		let mut body = json!({"model":request.model, "text":request.text, "stream":false, "voice_setting":{"voice_id":voice}, "audio_setting":{"format":format}, "output_format":"hex"});
		if let Some(speed) = request.speed {
			if !speed.is_finite() || !(MIN_SPEECH_SPEED..=MAX_SPEECH_SPEED).contains(&speed) {
				return Err(MediaError::InvalidRequest(format!(
					"MiniMax speed must be between {MIN_SPEECH_SPEED} and {MAX_SPEECH_SPEED}"
				)));
			}
			body["voice_setting"]["speed"] = json!(speed);
		}
		let value = self
			.post(SPEECH_ENDPOINT, &body, &request.request_options, true)
			.await?;
		if value.pointer("/data/status").and_then(Value::as_u64) != Some(2) {
			return Err(invalid_response("data.status must be 2 for completed speech"));
		}
		let bytes = hex::decode(required_str(&value, "/data/audio")?)
			.map_err(|error| invalid_response(format!("invalid data.audio hex: {error}")))?;
		if bytes.is_empty() {
			return Err(invalid_response("data.audio contains no audio bytes"));
		}
		let duration_secs = value
			.pointer("/extra_info/audio_length")
			.map(|length| {
				length
					.as_f64()
					.filter(|milliseconds| milliseconds.is_finite() && *milliseconds >= 0.0)
					.map(|milliseconds| milliseconds / 1000.0)
					.ok_or_else(|| invalid_response("invalid extra_info.audio_length"))
			})
			.transpose()?;
		let trace_id = required_str(&value, "/trace_id")?;
		let billable_characters = value
			.pointer("/extra_info/usage_characters")
			.map(|characters| {
				characters
					.as_u64()
					.map(|characters| characters as f64)
					.ok_or_else(|| invalid_response("extra_info.usage_characters must be a non-negative integer"))
			})
			.transpose()?;
		if billable_characters.is_none() {
			warnings.push(ProviderWarning {
				code: WarningCode::UsageUnavailable,
				message: "MiniMax did not report extra_info.usage_characters; speech usage is unavailable".to_string(),
				parameter: None,
				provider_metadata: Value::Null,
			});
		}
		// The text's length stands in when MiniMax reports no billable characters.
		let estimate = shared::resolved_cost_estimate(
			PROVIDER,
			&request.model,
			None,
			None,
			Some((UsageUnit::Characters, characters as f64)),
		);
		let usage = usage(
			&request.model,
			UsageUnit::Characters,
			billable_characters,
			estimate.as_ref(),
			&mut warnings,
		);
		Ok(Operation::completed(SpeechSynthesisResult {
			artifact: inline_artifact(
				MediaKind::Audio,
				request.output.format.media_type(),
				bytes,
				duration_secs,
			),
			usage: Some(usage),
			warnings,
			provider_metadata: json!({"extra_info":value.get("extra_info"), "trace_id":trace_id}),
		}))
	}

	async fn poll_speech(&self, handle: &JobHandle) -> MediaResult<Operation<SpeechSynthesisResult>> {
		Err(unsupported_task(&handle.model, handle.task))
	}
	async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()> {
		Err(unsupported_task(&handle.model, handle.task))
	}
}

#[async_trait::async_trait]
impl TranscriptionProvider for MinimaxMediaProvider {
	fn name(&self) -> &str {
		PROVIDER
	}
	fn supports_model(&self, model: &str) -> bool {
		model == TRANSCRIPTION_MODEL
	}
	fn capabilities(&self, model: &str) -> TranscriptionCapabilities {
		self.descriptor(model, MediaTask::SpeechToText)
	}

	/// https://platform.minimax.io/docs/api-reference/speech-to-text-create
	async fn submit_transcription(&self, request: TranscriptionRequest) -> MediaResult<Operation<TranscriptionResult>> {
		validate_request(&request.model, TRANSCRIPTION_MODEL, MediaTask::SpeechToText)?;
		validate_options(&request.provider_options, false)?;
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[("prompt", request.prompt.is_some())],
		)?;
		// MiniMax takes the recording as a multipart upload and caps it at 50 MB.
		let (data, media_type) = shared::source_to_bytes(&request.audio, request.request_options.max_source_bytes)?;
		if data.len() > MAX_TRANSCRIPTION_BYTES {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax audio must not exceed {MAX_TRANSCRIPTION_BYTES} bytes"
			)));
		}
		let word_level = request.timestamp_granularities.contains(&TimestampGranularity::Word);
		// Timestamps come back only in `verbose_json`, at word level when asked for.
		let response_format = if request.timestamp_granularities.is_empty() {
			"json"
		} else {
			"verbose_json"
		};
		// MiniMax types the file by its name; the part carries no Content-Type.
		let file_name = format!("audio.{}", file_extension(&media_type));
		let url = format!("{}{TRANSCRIPTION_ENDPOINT}", self.api_base()?);
		let key = shared::api_key(API_KEY_ENV)?;
		let language = request.language.clone();
		let response = shared::send(PROVIDER, &request.request_options, || {
			let file = reqwest::multipart::Part::bytes(data.clone()).file_name(file_name.clone());
			let mut form = reqwest::multipart::Form::new()
				.text("model", TRANSCRIPTION_MODEL)
				.text("response_format", response_format)
				.part("file", file);
			if word_level {
				form = form.text("timestamp_level", "word");
			}
			let mut builder = crate::http::http_client().post(&url).bearer_auth(&key).multipart(form);
			if let Some(language) = language.as_deref() {
				builder = builder.header("language", language);
			}
			builder
		})
		.await?;
		let value = parse_response(&response, false)?;
		// Silence transcribes to an empty string, so only the field has to exist.
		let text = value
			.get("text")
			.and_then(Value::as_str)
			.ok_or_else(|| invalid_response("missing text"))?
			.to_string();
		// MiniMax bills the duration it reports for the input audio.
		let duration_secs = reported_seconds(&value, "/duration")?;
		let (segments, words) = if request.timestamp_granularities.is_empty() {
			(Vec::new(), Vec::new())
		} else {
			transcript_units(
				value
					.get("segments")
					.ok_or_else(|| invalid_response("missing segments"))?,
				word_level,
			)?
		};
		let estimate = shared::resolved_cost_estimate(
			PROVIDER,
			TRANSCRIPTION_MODEL,
			None,
			None,
			Some((UsageUnit::AudioSeconds, duration_secs)),
		);
		let usage = usage(
			TRANSCRIPTION_MODEL,
			UsageUnit::AudioSeconds,
			Some(duration_secs),
			estimate.as_ref(),
			&mut warnings,
		);
		Ok(Operation::completed(TranscriptionResult {
			text,
			language: None,
			duration_secs: Some(duration_secs),
			segments,
			words,
			usage: Some(usage),
			warnings,
			provider_metadata: json!({"trace_id":value.get("trace_id"), "n_speakers":value.get("n_speakers")}),
		}))
	}

	async fn poll_transcription(&self, handle: &JobHandle) -> MediaResult<Operation<TranscriptionResult>> {
		Err(unsupported_task(&handle.model, handle.task))
	}
	async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()> {
		Err(unsupported_task(&handle.model, handle.task))
	}
}

fn parse_response(response: &shared::CapturedResponse, v1: bool) -> MediaResult<Value> {
	match response.status.as_u16() {
		400 => return Err(MediaError::InvalidRequest(shared::error_message(&response.body))),
		422 => {
			return Err(MediaError::RemoteFailure(GenerationFailure {
				category: FailureCategory::ContentPolicy,
				message: shared::error_message(&response.body),
				provider_code: None,
				http_status: Some(422),
				request_id: None,
				provider_metadata: Value::Null,
			}))
		}
		_ => shared::require_success(PROVIDER, response)?,
	}
	let value = shared::parse_json(PROVIDER, response)?;
	if v1 {
		let code = value
			.pointer("/base_resp/status_code")
			.and_then(Value::as_u64)
			.ok_or_else(|| invalid_response("missing base_resp.status_code"))?;
		if code != 0 {
			let message = required_str(&value, "/base_resp/status_msg")?.to_string();
			return Err(match code {
				1002 | 1039 | 1041 | 1045 | 2045 | 2046 | 2047 | 2062 => MediaError::RateLimit {
					provider: PROVIDER.to_string(),
					message,
					retry_after_secs: None,
				},
				1008 | 2052 | 2053 | 2056 | 2067 => MediaError::InsufficientCredits {
					provider: PROVIDER.to_string(),
					message,
				},
				1004 | 2049 | 2057 => MediaError::Authentication {
					provider: PROVIDER.to_string(),
					message,
				},
				1042 | 1043 | 1044 | 2013 | 20132 | 2014 | 2022 | 2037 | 2039 | 2043 | 2048 | 2051 | 2054 | 2069 => {
					MediaError::InvalidRequest(format!("MiniMax: {message} ({code})"))
				}
				1026 | 1027 | 1040 | 1046 => MediaError::RemoteFailure(GenerationFailure {
					category: FailureCategory::ContentPolicy,
					message,
					provider_code: Some(code.to_string()),
					http_status: Some(response.status.as_u16()),
					request_id: None,
					provider_metadata: json!({"base_resp":value.get("base_resp")}),
				}),
				1000 | 1001 | 1024 | 1033 | 2064 | 2066 | 2160 => MediaError::Api {
					provider: PROVIDER.to_string(),
					status: 500,
					message: format!("{message} ({code})"),
				},
				2038 | 2042 | 2061 | 2068 => MediaError::Permission {
					provider: PROVIDER.to_string(),
					message,
				},
				_ => MediaError::Api {
					provider: PROVIDER.to_string(),
					status: response.status.as_u16(),
					message: format!("{message} ({code})"),
				},
			});
		}
	}
	Ok(value)
}

fn validate_options(options: &ProviderOptions, video: bool) -> MediaResult<Option<String>> {
	#[derive(serde::Deserialize)]
	#[serde(deny_unknown_fields)]
	struct VideoOptions {
		resolution: Option<String>,
	}
	let raw = shared::provider_options(options, PROVIDER)?;
	if !video && !raw.is_empty() {
		return Err(MediaError::InvalidRequest(
			"MiniMax image, speech and transcription have no provider options".to_string(),
		));
	}
	let options: VideoOptions = serde_json::from_value(Value::Object(raw))
		.map_err(|error| MediaError::InvalidRequest(format!("invalid MiniMax provider options: {error}")))?;
	Ok(options.resolution)
}

fn validate_request(model: &str, text: &str, task: MediaTask) -> MediaResult<()> {
	let supported = match task {
		MediaTask::TextToImage => IMAGE_MODELS.contains(&model),
		MediaTask::TextToVideo => VIDEO_MODELS.contains(&model),
		MediaTask::TextToSpeech => SPEECH_MODELS.contains(&model),
		MediaTask::SpeechToText => model == TRANSCRIPTION_MODEL,
		_ => false,
	};
	if !supported {
		return Err(unsupported_task(model, task));
	}
	if text.trim().is_empty() {
		return Err(MediaError::InvalidRequest(
			"MiniMax prompt/text must not be empty".to_string(),
		));
	}
	Ok(())
}

/// Pay-as-you-go USD per second of output, per second of output on a reference
/// task (`reference_output`), per second of input video and per reference image
/// beyond `free_images`. The flagship models bill every task type alike; the
/// extended ones host H3 with their own rate card.
/// https://platform.minimax.io/docs/guides/pricing-paygo (read 2026-11-26)
#[derive(Clone, Copy)]
struct VideoRates {
	output: f64,
	reference_output: f64,
	input_video: f64,
	input_image: f64,
	free_images: u32,
}

impl VideoRates {
	fn estimate(self) -> CostEstimate {
		CostEstimate {
			unit: UsageUnit::VideoSeconds,
			usd_per_unit: self.output,
			quantity: None,
		}
	}
}

fn video_rates(model: &str, resolution: &str) -> MediaResult<VideoRates> {
	// (output, reference output, input video, input image beyond the free quota)
	let (output, reference_output, input_video, input_image) = match (model, resolution) {
		// 480P is served by RunningHub under the same model name.
		("MiniMax-H3", "480P") => (0.038, 0.038, 0.038, 0.04),
		("MiniMax-H3", "768P") => (0.08, 0.08, 0.08, 0.04),
		("MiniMax-H3", "2K") => (0.13, 0.13, 0.13, 0.04),
		("MiniMax-H3-Highspeed", "480P") => (0.04, 0.04, 0.04, 0.03),
		("MiniMax-H3-Highspeed", "768P") => (0.06, 0.06, 0.06, 0.03),
		("MiniMax-H3-Lite", "480P") => (0.025, 0.025, 0.025, 0.02),
		("MiniMax-H3-Lite", "768P") => (0.04, 0.04, 0.04, 0.02),
		("MiniMax-H3-Max", "480P") => (0.05, 0.05, 0.0553, 0.074),
		("MiniMax-H3-Max", "768P") => (0.08, 0.08, 0.143, 0.074),
		("MiniMax-H3-Max-Turbo", "480P") => (0.025, 0.05, 0.04, 0.06),
		("MiniMax-H3-Max-Turbo", "768P") => (0.04, 0.08, 0.12, 0.06),
		// MG takes no image-input fee.
		("MiniMax-H3-MG", "768P") => (0.04, 0.06, 0.06, 0.0),
		("MiniMax-H3-MG", "2K") => (0.10, 0.12, 0.12, 0.0),
		("MiniMax-H3-NC", "480P") => (0.022, 0.022, 0.022, 0.011),
		("MiniMax-H3-NC", "768P") => (0.03, 0.03, 0.03, 0.015),
		_ => {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax resolution '{resolution}' is not supported by {model}"
			)))
		}
	};
	let free_images = if matches!(model, "MiniMax-H3-Max" | "MiniMax-H3-Max-Turbo") {
		2
	} else {
		5
	};
	Ok(VideoRates {
		output,
		reference_output,
		input_video,
		input_image,
		free_images,
	})
}

/// MiniMax-H3-Lite and MiniMax-H3-NC do not accept `adaptive` yet.
fn adaptive_reference(model: &str) -> bool {
	!matches!(model, "MiniMax-H3-Lite" | "MiniMax-H3-NC")
}

/// Reference tasks also bill each second of input video and each reference
/// image beyond a free quota; the finished task reports both counts.
fn reference_input_usage(
	handle: &JobHandle,
	task: &Value,
	usage: &mut MediaUsage,
	warnings: &mut Vec<ProviderWarning>,
) -> MediaResult<()> {
	if handle.task != MediaTask::ReferenceToVideo {
		return Ok(());
	}
	let seconds = optional_count(task, "/usage/input_seconds")?;
	let images = optional_count(task, "/usage/input_image_count")?;
	if seconds == 0.0 && images == 0.0 {
		return Ok(());
	}
	// The input rates and the free image quota come from the model and resolution.
	let rates = match required_str(task, "/resolution").and_then(|resolution| video_rates(&handle.model, resolution)) {
		Ok(rates) => rates,
		Err(_) => {
			warnings.push(ProviderWarning {
				code: WarningCode::CostUnavailable,
				message: "MiniMax reported reference input usage without a priceable task.resolution; the task cost is unavailable".to_string(),
				parameter: None,
				provider_metadata: Value::Null,
			});
			for (unit, quantity, description) in [
				(UsageUnit::VideoSeconds, seconds, "input video seconds"),
				(UsageUnit::Images, images, "reference images"),
			] {
				if quantity > 0.0 {
					usage.line_items.push(UsageLineItem {
						unit,
						quantity,
						cost: None,
						description: Some(description.to_string()),
					});
				}
			}
			// A total that omits the input share would read as the full price.
			usage.estimated_cost = None;
			return Ok(());
		}
	};
	let extra = seconds * rates.input_video + (images - f64::from(rates.free_images)).max(0.0) * rates.input_image;
	for (unit, quantity, rate, description) in [
		(
			UsageUnit::VideoSeconds,
			seconds,
			rates.input_video,
			"input video seconds",
		),
		(
			UsageUnit::Images,
			(images - f64::from(rates.free_images)).max(0.0),
			rates.input_image,
			"reference images beyond the free quota",
		),
	] {
		if quantity > 0.0 {
			usage.line_items.push(UsageLineItem {
				unit,
				quantity,
				cost: Some(quantity * rate),
				description: Some(description.to_string()),
			});
		}
	}
	// An unknown output cost stays unknown rather than shrinking to the input share.
	usage.estimated_cost = usage.estimated_cost.map(|cost| cost + extra);
	Ok(())
}

fn validate_duration(model: &str, duration: f64) -> MediaResult<()> {
	let minimum = if matches!(model, "MiniMax-H3-Max" | "MiniMax-H3-Max-Turbo") {
		H3_MAX_MIN_SECONDS
	} else {
		MIN_VIDEO_SECONDS
	};
	if !duration.is_finite() || duration.fract() != 0.0 || !(minimum..=MAX_VIDEO_SECONDS).contains(&duration) {
		return Err(MediaError::InvalidRequest(format!(
			"{model} duration must be a whole number of seconds in {minimum}..={MAX_VIDEO_SECONDS}"
		)));
	}
	Ok(())
}

fn aspect_ratio(geometry: Option<OutputGeometry>, default: &str, allowed: &[&str]) -> MediaResult<String> {
	let ratio =
		match geometry {
			None => return Ok(default.to_string()),
			Some(OutputGeometry::AspectRatio { width, height }) => format!("{width}:{height}"),
			Some(OutputGeometry::Dimensions { .. }) => return Err(MediaError::UnsupportedParameter {
				provider: PROVIDER.to_string(),
				parameter: "geometry".to_string(),
				reason:
					"MiniMax video requires an aspect ratio; select resolution with provider_options.minimax.resolution"
						.to_string(),
			}),
		};
	if !allowed.contains(&ratio.as_str()) {
		return Err(MediaError::InvalidRequest(format!(
			"unsupported MiniMax aspect ratio '{ratio}'"
		)));
	}
	Ok(ratio)
}

fn task_state(task: &Value, handle: &JobHandle) -> MediaResult<(OperationStatus, Option<GenerationFailure>)> {
	let status = match required_str(task, "/status")? {
		"queued" => OperationStatus::Queued,
		"running" => OperationStatus::Running,
		"succeeded" => OperationStatus::Succeeded,
		"failed" => OperationStatus::Failed,
		"cancelled" => OperationStatus::Cancelled,
		other => return Err(invalid_response(format!("unknown task.status '{other}'"))),
	};
	let failure = if status == OperationStatus::Failed {
		let code = required_str(task, "/error/code")?;
		Some(GenerationFailure {
			category: match code {
				"1026" => FailureCategory::ContentPolicy,
				"1002" | "1039" => FailureCategory::RateLimit,
				"1008" => FailureCategory::InsufficientCredits,
				"1004" | "2049" => FailureCategory::Authentication,
				"2013" => FailureCategory::InvalidInput,
				_ => FailureCategory::RemoteJob,
			},
			message: required_str(task, "/error/message")?.to_string(),
			provider_code: Some(code.to_string()),
			http_status: None,
			request_id: Some(handle.remote_id.clone()),
			provider_metadata: task.clone(),
		})
	} else {
		None
	};
	Ok((status, failure))
}

fn video_result(handle: &JobHandle, task: &Value) -> MediaResult<VideoGenerationResult> {
	let url = required_str(task, "/content/url")?;
	shared::validate_https_url(url, "task.content.url").map_err(|error| invalid_response(error.to_string()))?;
	let seconds = task
		.get("duration")
		.map(|duration| {
			duration
				.as_f64()
				.filter(|seconds| seconds.is_finite() && *seconds > 0.0)
				.ok_or_else(|| invalid_response("invalid task.duration"))
		})
		.transpose()?;
	let mut warnings = handle.warnings.clone();
	let mut usage = usage(
		&handle.model,
		UsageUnit::VideoSeconds,
		seconds,
		handle.cost_estimate.as_ref(),
		&mut warnings,
	);
	reference_input_usage(handle, task, &mut usage, &mut warnings)?;
	Ok(VideoGenerationResult {
		artifacts: vec![MediaArtifact {
			kind: MediaKind::Video,
			media_type: "video/mp4".to_string(),
			source: ArtifactSource::Url(url.to_string()),
			size_bytes: None,
			dimensions: None,
			duration_secs: seconds,
			frame_rate: None,
			sample_rate_hz: None,
			channels: None,
			expires_at: None,
			metadata: json!({"provider":PROVIDER}),
		}],
		usage: Some(usage),
		warnings,
		safety: SafetyReport::default(),
		provider_metadata: task.clone(),
	})
}

fn operation<T>(
	handle: &JobHandle,
	status: OperationStatus,
	error: Option<GenerationFailure>,
	result: Option<T>,
	metadata: Value,
) -> Operation<T> {
	Operation {
		handle: Some(handle.clone()),
		status,
		progress: None,
		result,
		error,
		created_at: None,
		started_at: None,
		completed_at: None,
		provider_metadata: metadata,
	}
}

fn inline_artifact(kind: MediaKind, media_type: &str, bytes: Vec<u8>, duration_secs: Option<f64>) -> MediaArtifact {
	MediaArtifact {
		kind,
		media_type: media_type.to_string(),
		size_bytes: Some(bytes.len() as u64),
		source: ArtifactSource::Inline(bytes),
		dimensions: None,
		duration_secs,
		frame_rate: None,
		sample_rate_hz: None,
		channels: None,
		expires_at: None,
		metadata: json!({"provider":PROVIDER}),
	}
}

fn file_extension(media_type: &str) -> &str {
	media_type.split_once('/').map_or("bin", |(_, subtype)| subtype)
}

/// Width and height pass MiniMax's pixel budget (inclusive) and stay within 1:3..=3:1.
fn h3_image_shape(width: u64, height: u64, min_pixels: u64, max_pixels: u64) -> bool {
	let (short, long) = (width.min(height), width.max(height));
	(min_pixels..=max_pixels).contains(&(width * height)) && long <= 3 * short
}

/// H3-Image token prices per 1M tokens: text input, image input, image output.
fn h3_image_rates(model: &str) -> (f64, f64, f64) {
	if model == "MiniMax-H3-Image-Highspeed" {
		(2.30, 3.00, 7.60)
	} else {
		(2.30, 1.50, 3.80)
	}
}

fn h3_image_usage(model: &str, reported: &Value) -> MediaResult<MediaUsage> {
	let (text_rate, image_rate, output_rate) = h3_image_rates(model);
	let text = reported_tokens(reported, "/input_tokens_details/text_tokens")?;
	let image = reported_tokens(reported, "/input_tokens_details/image_tokens")?;
	let output = reported_tokens(reported, "/output_tokens_details/image_tokens")?;
	let per_million = |rate: f64| rate / 1_000_000.0;
	Ok(MediaUsage {
		line_items: [
			(UsageUnit::InputTokens, text, per_million(text_rate)),
			(UsageUnit::InputTokens, image, per_million(image_rate)),
			(UsageUnit::OutputTokens, output, per_million(output_rate)),
		]
		.into_iter()
		.map(|(unit, quantity, usd_per_token)| UsageLineItem {
			unit,
			quantity,
			cost: Some(quantity * usd_per_token),
			description: None,
		})
		.collect(),
		provider_reported_cost: None,
		estimated_cost: Some((text * text_rate + image * image_rate + output * output_rate) / 1_000_000.0),
		currency: "USD".to_string(),
		metadata: reported.clone(),
	})
}

/// A token count MiniMax reports; a missing or malformed one is never defaulted.
fn reported_tokens(value: &Value, path: &str) -> MediaResult<f64> {
	value
		.pointer(path)
		.and_then(Value::as_u64)
		.map(|tokens| tokens as f64)
		.ok_or_else(|| invalid_response(format!("invalid {path}")))
}

/// A non-negative count or duration MiniMax reports.
fn reported_seconds(value: &Value, path: &str) -> MediaResult<f64> {
	value
		.pointer(path)
		.and_then(Value::as_f64)
		.filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
		.ok_or_else(|| invalid_response(format!("invalid {path}")))
}

/// A count that is absent when the task had no such input.
fn optional_count(value: &Value, path: &str) -> MediaResult<f64> {
	value
		.pointer(path)
		.map(|_| reported_seconds(value, path))
		.transpose()
		.map(|count| count.unwrap_or(0.0))
}

/// MiniMax reports one `segments` list whose granularity follows the requested
/// `timestamp_level`: sentences, or single words with word timestamps.
fn transcript_units(segments: &Value, word_level: bool) -> MediaResult<(Vec<TranscriptSegment>, Vec<TranscriptWord>)> {
	let units = segments
		.as_array()
		.ok_or_else(|| invalid_response("segments must be an array"))?
		.iter()
		.map(|unit| {
			Ok((
				reported_seconds(unit, "/start")?,
				reported_seconds(unit, "/end")?,
				unit.get("text")
					.and_then(Value::as_str)
					.ok_or_else(|| invalid_response("invalid segment text"))?
					.to_string(),
				unit.get("speaker").and_then(Value::as_str).map(str::to_string),
			))
		})
		.collect::<MediaResult<Vec<_>>>()?;
	Ok(if word_level {
		(
			Vec::new(),
			units
				.into_iter()
				.map(|(start_secs, end_secs, word, speaker)| TranscriptWord {
					start_secs,
					end_secs,
					word,
					speaker,
				})
				.collect(),
		)
	} else {
		(
			units
				.into_iter()
				.map(|(start_secs, end_secs, text, speaker)| TranscriptSegment {
					start_secs,
					end_secs,
					text,
					speaker,
				})
				.collect(),
			Vec::new(),
		)
	})
}

fn h3_image_media_type(format: &str) -> MediaResult<&'static str> {
	match format {
		"png" => Ok("image/png"),
		"jpeg" | "jpg" => Ok("image/jpeg"),
		"webp" => Ok("image/webp"),
		other => Err(invalid_response(format!("unknown data.output_format '{other}'"))),
	}
}

/// `WIDTHxHEIGHT`, the size MiniMax reports for each generated image.
fn image_size(size: &str) -> MediaResult<Dimensions> {
	let parse = |side: &str| {
		side.parse::<u32>()
			.ok()
			.filter(|side| *side > 0)
			.ok_or_else(|| invalid_response(format!("invalid data.size '{size}'")))
	};
	let (width, height) = size
		.split_once('x')
		.ok_or_else(|| invalid_response(format!("invalid data.size '{size}'")))?;
	Ok(Dimensions {
		width: parse(width)?,
		height: parse(height)?,
	})
}

fn image_media_type(bytes: &[u8]) -> MediaResult<&'static str> {
	if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
		Ok("image/png")
	} else if bytes.starts_with(b"\xff\xd8\xff") {
		Ok("image/jpeg")
	} else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
		Ok("image/webp")
	} else {
		Err(invalid_response("data.image_base64 contains no recognizable image"))
	}
}

fn usage(
	model: &str,
	unit: UsageUnit,
	quantity: Option<f64>,
	rate: Option<&CostEstimate>,
	warnings: &mut Vec<ProviderWarning>,
) -> MediaUsage {
	let estimated_cost = rate
		.filter(|rate| rate.unit == unit)
		.and_then(|rate| quantity.or(rate.quantity).map(|quantity| rate.usd_per_unit * quantity));
	if estimated_cost.is_none() {
		warnings.push(ProviderWarning {
			code: WarningCode::CostUnavailable,
			message: format!("MiniMax cost estimate for {model} requires a matching rate and reported quantity"),
			parameter: None,
			provider_metadata: Value::Null,
		});
	}
	MediaUsage {
		line_items: quantity
			.into_iter()
			.map(|quantity| UsageLineItem {
				unit,
				quantity,
				cost: None,
				description: None,
			})
			.collect(),
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
	for (parameter, present) in parameters.iter().copied().chain(std::iter::once((
		"request_options.idempotency_key",
		options.idempotency_key.is_some(),
	))) {
		if !present {
			continue;
		}
		let reason = format!("MiniMax has no supported mapping for '{parameter}'");
		if options.unsupported_parameter_policy == UnsupportedParameterPolicy::Error {
			return Err(MediaError::UnsupportedParameter {
				provider: PROVIDER.to_string(),
				parameter: parameter.to_string(),
				reason,
			});
		}
		warnings.push(ProviderWarning {
			code: WarningCode::UnsupportedParameterDropped,
			message: reason,
			parameter: Some(parameter.to_string()),
			provider_metadata: Value::Null,
		});
	}
	Ok(())
}

fn required_str<'a>(value: &'a Value, path: &str) -> MediaResult<&'a str> {
	value
		.pointer(path)
		.and_then(Value::as_str)
		.filter(|value| !value.is_empty())
		.ok_or_else(|| invalid_response(format!("missing or empty {path}")))
}

fn invalid_response(message: impl Into<String>) -> MediaError {
	MediaError::InvalidResponse {
		provider: PROVIDER.to_string(),
		message: message.into(),
	}
}
fn unsupported_task(model: &str, task: MediaTask) -> MediaError {
	MediaError::UnsupportedTask {
		provider: PROVIDER.to_string(),
		model: model.to_string(),
		task,
	}
}

#[cfg(test)]
#[path = "minimax_http_tests.rs"]
mod http_tests;
