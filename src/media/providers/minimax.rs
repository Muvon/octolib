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

//! Native MiniMax image, H3 video, and synchronous speech generation.

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
const VIDEO_ENDPOINT: &str = "/v2/video_generation";
const VIDEO_QUERY_ENDPOINT: &str = "/v2/query/video_generation";
const SPEECH_ENDPOINT: &str = "/v1/t2a_v2";
const DEFAULT_VOICE: &str = "English_expressive_narrator";
const DEFAULT_VOICE_SENTINEL: &str = "default";
const DEFAULT_VIDEO_DURATION_SECS: u32 = 5;
const DEFAULT_RESOLUTION: &str = "768P";
const IMAGE_PROMPT_LIMIT: usize = 1500;
const SPEECH_TEXT_LIMIT: usize = 10000;
const MAX_IMAGES: u32 = 9;
const MIN_IMAGE_DIMENSION: u32 = 512;
const MAX_IMAGE_DIMENSION: u32 = 2048;
const IMAGE_DIMENSION_MULTIPLE: u32 = 8;
const MIN_VIDEO_SECONDS: f64 = 4.0;
const H3_MAX_MIN_SECONDS: f64 = 5.0;
const MAX_VIDEO_SECONDS: f64 = 15.0;
const MIN_SPEECH_SPEED: f32 = 0.5;
const MAX_SPEECH_SPEED: f32 = 2.0;
const IMAGE_RATIOS: &[&str] = &["1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"];
const VIDEO_RATIOS: &[&str] = &["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const VIDEO_MODELS: &[&str] = &[
	"MiniMax-H3",
	"MiniMax-H3-Highspeed",
	"MiniMax-H3-Lite",
	"MiniMax-H3-Max",
];
const SPEECH_MODELS: &[&str] = &["speech-2.8-hd", "speech-2.8-turbo", "speech-02-hd", "speech-02-turbo"];

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
				vec![task, MediaTask::ImageToVideo]
			} else {
				vec![task]
			},
			input_modalities: if video {
				vec![Modality::Text, Modality::Image]
			} else {
				vec![Modality::Text]
			},
			output_modalities: vec![if image {
				Modality::Image
			} else if video {
				Modality::Video
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
				output_format: support(!image && !video),
			},
			limits: MediaLimits {
				max_outputs: Some(if image { MAX_IMAGES } else { 1 }),
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
		model == "image-01"
	}
	fn capabilities(&self, model: &str) -> ImageCapabilities {
		self.descriptor(model, MediaTask::TextToImage)
	}

	async fn submit_image(&self, request: ImageGenerationRequest) -> MediaResult<Operation<ImageGenerationResult>> {
		validate_request(&request.model, &request.prompt, MediaTask::TextToImage)?;
		if request.prompt.chars().count() > IMAGE_PROMPT_LIMIT {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax prompt exceeds {IMAGE_PROMPT_LIMIT} characters"
			)));
		}
		match request.mode {
			ImageGenerationMode::Generate => {}
			ImageGenerationMode::Edit => return Err(unsupported_task(&request.model, MediaTask::ImageEdit)),
			ImageGenerationMode::Variation => return Err(unsupported_task(&request.model, MediaTask::ImageVariation)),
			ImageGenerationMode::Inpaint => return Err(unsupported_task(&request.model, MediaTask::Inpainting)),
		}
		validate_options(&request.provider_options, false)?;
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("source_images", !request.source_images.is_empty()),
				("mask", request.mask.is_some()),
				("seed", request.seed.is_some()),
				("negative_prompt", request.negative_prompt.is_some()),
				("output_format", request.output_format.is_some()),
			],
		)?;
		let count = request.count.unwrap_or(1);
		if !(1..=MAX_IMAGES).contains(&count) {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax count must be 1..={MAX_IMAGES}"
			)));
		}
		let mut body = json!({"model":request.model, "prompt":request.prompt, "n":count, "response_format":"base64"});
		match request.geometry {
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
		let estimate = shared::resolved_cost_estimate(PROVIDER, &request.model, None, None);
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
		let task = match (request.mode, request.first_frame.as_ref()) {
			(VideoGenerationMode::TextToVideo, None) => MediaTask::TextToVideo,
			(VideoGenerationMode::ImageToVideo, Some(_)) => MediaTask::ImageToVideo,
			(VideoGenerationMode::ImageToVideo, None) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax image-to-video requires first_frame".to_string(),
				))
			}
			(VideoGenerationMode::TextToVideo, Some(_)) => {
				return Err(MediaError::InvalidRequest(
					"MiniMax first_frame requires ImageToVideo mode".to_string(),
				))
			}
			(VideoGenerationMode::ReferenceToVideo, _) => {
				return Err(unsupported_task(&request.model, MediaTask::ReferenceToVideo))
			}
			(VideoGenerationMode::Extend, _) => return Err(unsupported_task(&request.model, MediaTask::VideoExtend)),
			(VideoGenerationMode::Edit, _) => return Err(unsupported_task(&request.model, MediaTask::VideoEdit)),
		};
		let resolution =
			validate_options(&request.provider_options, true)?.unwrap_or_else(|| DEFAULT_RESOLUTION.to_string());
		let estimate = video_cost_estimate(&request.model, &resolution)?;
		let duration = request.duration_secs.unwrap_or(f64::from(DEFAULT_VIDEO_DURATION_SECS));
		validate_duration(&request.model, duration)?;
		let mut warnings = Vec::new();
		drop_unsupported(
			&request.request_options,
			&mut warnings,
			&[
				("count", request.count.is_some_and(|count| count != 1)),
				("seed", request.seed.is_some()),
				("last_frame", request.last_frame.is_some()),
				("reference_images", !request.reference_images.is_empty()),
				("source_video", request.source_video.is_some()),
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
		let ratio = if task == MediaTask::ImageToVideo {
			// H3 derives image-to-video geometry from the first frame.
			"adaptive".to_string()
		} else {
			aspect_ratio(request.geometry, "16:9", VIDEO_RATIOS)?
		};
		let mut content = vec![json!({"type":"text", "text":request.prompt})];
		if let Some(first) = request.first_frame.as_ref() {
			let url = shared::source_to_data_or_uri(first, request.request_options.max_source_bytes)?;
			content.push(json!({"type":"image_url", "image_url":{"url":url}, "role":"first_frame"}));
		}
		let body = json!({"model":request.model, "content":content, "resolution":resolution, "duration":duration as u32, "ratio":ratio});
		let value = self
			.post(VIDEO_ENDPOINT, &body, &request.request_options, false)
			.await?;
		let id = required_str(&value, "/task_id")?;
		let handle = JobHandle {
			provider: PROVIDER.to_string(),
			model: request.model,
			task,
			remote_id: id.to_string(),
			cost_estimate: Some(estimate),
			warnings,
			expected_outputs: Some(1),
		};
		Ok(operation(&handle, OperationStatus::Queued, None, None, Value::Null))
	}

	async fn poll_video(&self, handle: &JobHandle) -> MediaResult<Operation<VideoGenerationResult>> {
		shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToVideo, MediaTask::ImageToVideo])?;
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
		shared::validate_handle(handle, PROVIDER, &[MediaTask::TextToVideo, MediaTask::ImageToVideo])?;
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
		let estimate = shared::resolved_cost_estimate(PROVIDER, &request.model, None, None);
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
			"MiniMax image and speech have no provider options".to_string(),
		));
	}
	let options: VideoOptions = serde_json::from_value(Value::Object(raw))
		.map_err(|error| MediaError::InvalidRequest(format!("invalid MiniMax provider options: {error}")))?;
	Ok(options.resolution)
}

fn validate_request(model: &str, text: &str, task: MediaTask) -> MediaResult<()> {
	let supported = match task {
		MediaTask::TextToImage => model == "image-01",
		MediaTask::TextToVideo => VIDEO_MODELS.contains(&model),
		MediaTask::TextToSpeech => SPEECH_MODELS.contains(&model),
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

fn video_cost_estimate(model: &str, resolution: &str) -> MediaResult<CostEstimate> {
	let usd_per_unit = match (model, resolution) {
		("MiniMax-H3", "2K") => 0.13,
		("MiniMax-H3", "768P") => 0.08,
		("MiniMax-H3-Highspeed", "768P") => 0.06,
		("MiniMax-H3-Lite", "768P") => 0.04,
		("MiniMax-H3-Lite", "480P") => 0.025,
		("MiniMax-H3-Max", "768P") => 0.08,
		("MiniMax-H3-Max", "480P") => 0.05,
		_ => {
			return Err(MediaError::InvalidRequest(format!(
				"MiniMax resolution '{resolution}' is not supported by {model}"
			)))
		}
	};
	Ok(CostEstimate {
		unit: UsageUnit::VideoSeconds,
		usd_per_unit,
		quantity: None,
	})
}

fn validate_duration(model: &str, duration: f64) -> MediaResult<()> {
	let minimum = if model == "MiniMax-H3-Max" {
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
	let usage = usage(
		&handle.model,
		UsageUnit::VideoSeconds,
		seconds,
		handle.cost_estimate.as_ref(),
		&mut warnings,
	);
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
		.zip(quantity)
		.map(|(rate, quantity)| rate.usd_per_unit * quantity);
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
