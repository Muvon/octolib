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

use super::*;
use crate::test_support::env::EnvGuard;
use crate::test_support::mock_http::MockHttp;

fn start_mock(replies: Vec<(u16, String)>) -> (MockHttp, EnvGuard) {
	let mock = MockHttp::start(replies);
	let env = EnvGuard::new()
		.set(API_BASE_ENV, mock.base_url())
		.set(API_KEY_ENV, "mock-key");
	(mock, env)
}

fn image_reply() -> String {
	json!({"data":{"image_base64":[base64::engine::general_purpose::STANDARD.encode(b"\x89PNG\r\n\x1a\nimage")]}, "metadata":{"success_count":"1", "failed_count":"0"}, "base_resp":{"status_code":0,"status_msg":"success"}}).to_string()
}

fn speech_reply() -> String {
	json!({"data":{"audio":"49443300ff", "status":2}, "extra_info":{"audio_length":1250, "usage_characters":999}, "trace_id":"trace-1", "base_resp":{"status_code":0,"status_msg":"success"}}).to_string()
}

fn video_request(model: &str, duration: f64) -> VideoGenerationRequest {
	let mut request = VideoGenerationRequest::new("a castle").with_model(model);
	request.duration_secs = Some(duration);
	request
}

fn handle() -> JobHandle {
	JobHandle {
		provider: PROVIDER.to_string(),
		model: "MiniMax-H3".to_string(),
		task: MediaTask::TextToVideo,
		remote_id: "task-1".to_string(),
		cost_estimate: shared::resolved_cost_estimate(PROVIDER, "MiniMax-H3", None, None),
		warnings: Vec::new(),
		expected_outputs: Some(1),
	}
}

#[tokio::test]
#[serial_test::serial]
async fn image_base64_returns_inline_bytes_and_prices_returned_count() {
	let (mock, _env) = start_mock(vec![(200, image_reply())]);
	let mut request = ImageGenerationRequest::new("a castle").with_model("image-01");
	request.count = Some(2);
	request.geometry = Some(OutputGeometry::AspectRatio { width: 16, height: 9 });
	let result = MinimaxMediaProvider::new()
		.submit_image(request)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.artifacts[0].media_type, "image/png");
	assert_eq!(
		result.artifacts[0].source,
		ArtifactSource::Inline(b"\x89PNG\r\n\x1a\nimage".to_vec())
	);
	let usage = result.usage.unwrap();
	assert_eq!(usage.line_items[0].unit, UsageUnit::Images);
	assert_eq!(usage.line_items[0].quantity, 1.0);
	assert_eq!(usage.estimated_cost, Some(0.0035));
	assert_eq!(usage.provider_reported_cost, None);
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::PartialOutput));
	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].path, IMAGE_ENDPOINT);
	let body: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(
		body,
		json!({"model":"image-01", "prompt":"a castle", "n":2,"aspect_ratio":"16:9", "response_format":"base64"})
	);
}

#[tokio::test]
#[serial_test::serial]
async fn image_maps_default_ratio_and_explicit_size() {
	let (mock, _env) = start_mock(vec![(200, image_reply())]);
	let provider = MinimaxMediaProvider::new();
	provider
		.submit_image(ImageGenerationRequest::new("castle").with_model("image-01"))
		.await
		.unwrap();
	let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
	request.geometry = Some(OutputGeometry::Dimensions {
		width: 1024,
		height: 768,
	});
	provider.submit_image(request).await.unwrap();
	let requests = mock.requests();
	let default: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(default["aspect_ratio"], "1:1");
	assert_eq!(default["n"], 1);
	let sized: Value = serde_json::from_str(&requests[1].body).unwrap();
	assert_eq!(sized["width"], 1024);
	assert_eq!(sized["height"], 768);
	assert!(sized.get("aspect_ratio").is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn video_submit_and_poll_queued_running_succeeded() {
	let (mock, _env) = start_mock(vec![
		(200, json!({"task_id":"task-1"}).to_string()),
		(200, json!({"task":{"id":"task-1", "model":"MiniMax-H3", "status":"queued"}}).to_string()),
		(200, json!({"task":{"id":"task-1", "model":"MiniMax-H3", "status":"running"}}).to_string()),
		(200, json!({"task":{"id":"task-1", "model":"MiniMax-H3", "status":"succeeded", "duration":7, "resolution":"2K", "content":{"url":"https://cdn.example.test/out.mp4"}, "usage":{"output_seconds":7}}}).to_string()),
	]);
	let provider = MinimaxMediaProvider::new();
	let mut request = video_request("MiniMax-H3", 5.0);
	request
		.provider_options
		.insert(PROVIDER.to_string(), json!({"resolution":"2K"}));
	request.geometry = Some(OutputGeometry::AspectRatio { width: 9, height: 16 });
	let submitted = provider.submit_video(request).await.unwrap();
	assert_eq!(submitted.status, OperationStatus::Queued);
	let handle: JobHandle = serde_json::from_str(&serde_json::to_string(&submitted.handle.unwrap()).unwrap()).unwrap();
	assert_eq!(handle.cost_estimate.as_ref().unwrap().quantity, None);
	assert_eq!(
		provider.poll_video(&handle).await.unwrap().status,
		OperationStatus::Queued
	);
	assert_eq!(
		provider.poll_video(&handle).await.unwrap().status,
		OperationStatus::Running
	);
	let completed = provider.poll_video(&handle).await.unwrap();
	assert_eq!(completed.status, OperationStatus::Succeeded);
	let result = completed.result.unwrap();
	assert_eq!(
		result.artifacts[0].source,
		ArtifactSource::Url("https://cdn.example.test/out.mp4".to_string())
	);
	assert_eq!(result.artifacts[0].duration_secs, Some(7.0));
	let usage = result.usage.unwrap();
	assert_eq!(usage.line_items[0].unit, UsageUnit::VideoSeconds);
	assert_eq!(usage.line_items[0].quantity, 7.0);
	assert_eq!(usage.estimated_cost, Some(7.0 * 0.13));
	assert_eq!(usage.provider_reported_cost, None);
	let requests = mock.requests();
	assert_eq!(requests[0].path, VIDEO_ENDPOINT);
	let body: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(
		body,
		json!({"model":"MiniMax-H3", "content":[{"type":"text","text":"a castle"}], "resolution":"2K", "duration":5,"ratio":"9:16"})
	);
	assert!(body["duration"].is_u64());
	for request in &requests[1..] {
		assert_eq!(request.method, "GET");
		assert_eq!(request.path, "/v2/query/video_generation/task-1");
	}
}

#[tokio::test]
#[serial_test::serial]
async fn video_start_image_uses_v2_content_and_default_resolution() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"task-1"}).to_string())]);
	let mut request = video_request("MiniMax-H3-Lite", 4.0);
	request.mode = VideoGenerationMode::ImageToVideo;
	request.first_frame = Some(MediaSource::Bytes {
		data: b"image".to_vec(),
		media_type: "image/png".to_string(),
	});
	let submitted = MinimaxMediaProvider::new().submit_video(request).await.unwrap();
	assert_eq!(submitted.handle.unwrap().task, MediaTask::ImageToVideo);
	let body: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
	assert_eq!(body["resolution"], "768P");
	assert_eq!(body["ratio"], "adaptive");
	assert_eq!(
		body["content"][1],
		json!({"type":"image_url", "image_url":{"url":"data:image/png;base64,aW1hZ2U="}, "role":"first_frame"})
	);
}

#[tokio::test]
#[serial_test::serial]
async fn video_failed_preserves_error_message_and_category() {
	let (_mock, _env) = start_mock(vec![
		(
			200,
			json!({"task":{"status":"failed", "error":{"code":"1000", "message":"render crashed"}}}).to_string(),
		),
		(
			200,
			json!({"task":{"status":"failed", "error":{"code":"1026", "message":"sensitive content"}}}).to_string(),
		),
		(200, json!({"task":{"status":"cancelled"}}).to_string()),
	]);
	let provider = MinimaxMediaProvider::new();
	let failed = provider.poll_video(&handle()).await.unwrap();
	assert_eq!(failed.status, OperationStatus::Failed);
	let error = failed.error.unwrap();
	assert_eq!(error.category, FailureCategory::RemoteJob);
	assert_eq!(error.message, "render crashed");
	assert_eq!(error.provider_code.as_deref(), Some("1000"));
	assert_eq!(
		provider.poll_video(&handle()).await.unwrap().error.unwrap().category,
		FailureCategory::ContentPolicy
	);
	assert_eq!(
		provider.poll_video(&handle()).await.unwrap().status,
		OperationStatus::Cancelled
	);
}

#[tokio::test]
#[serial_test::serial]
async fn succeeded_video_with_unreadable_output_resolves_as_failed() {
	let (_mock, _env) = start_mock(vec![
		(
			200,
			json!({"task":{"status":"succeeded", "duration":5, "content":{}}}).to_string(),
		),
		(
			200,
			json!({"task":{"status":"succeeded", "duration":"invalid", "content":{"url":"https://cdn.example.test/out.mp4"}}}).to_string(),
		),
	]);
	for _ in 0..2 {
		let operation = MinimaxMediaProvider::new().poll_video(&handle()).await.unwrap();
		assert_eq!(operation.status, OperationStatus::Failed);
		assert_eq!(operation.error.unwrap().category, FailureCategory::InvalidResponse);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn speech_hex_decodes_and_prices_reported_characters() {
	let (mock, _env) = start_mock(vec![(200, speech_reply())]);
	let mut request = SpeechSynthesisRequest::new("hello 世", "default").with_model("speech-2.8-hd");
	request.speed = Some(1.25);
	let completed = MinimaxMediaProvider::new().submit_speech(request).await.unwrap();
	assert_eq!(completed.status, OperationStatus::Succeeded);
	let result = completed.result.unwrap();
	assert_eq!(
		result.artifact.source,
		ArtifactSource::Inline(vec![0x49, 0x44, 0x33, 0, 0xff])
	);
	assert_eq!(result.artifact.media_type, "audio/mpeg");
	assert_eq!(result.artifact.duration_secs, Some(1.25));
	let usage = result.usage.unwrap();
	assert_eq!(usage.line_items[0].unit, UsageUnit::Characters);
	assert_eq!(usage.line_items[0].quantity, 999.0);
	assert_eq!(usage.estimated_cost, Some(999.0 * 0.0001));
	assert_eq!(usage.provider_reported_cost, None);
	let requests = mock.requests();
	assert_eq!(requests[0].path, SPEECH_ENDPOINT);
	let body: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(
		body,
		json!({"model":"speech-2.8-hd", "text":"hello 世", "stream":false,"voice_setting":{"voice_id":DEFAULT_VOICE, "speed":1.25}, "audio_setting":{"format":"mp3"},"output_format":"hex"})
	);
}

#[tokio::test]
#[serial_test::serial]
async fn speech_maps_each_supported_audio_format_and_explicit_voice() {
	let (mock, _env) = start_mock(vec![(200, speech_reply())]);
	for (format, native) in [
		(AudioFormat::Mp3, "mp3"),
		(AudioFormat::Pcm, "pcm"),
		(AudioFormat::Flac, "flac"),
		(AudioFormat::Wav, "wav"),
		(AudioFormat::Ogg, "opus"),
	] {
		let mut request = SpeechSynthesisRequest::new("hello", "custom_voice").with_model("speech-02-turbo");
		request.output.format = format;
		let result = MinimaxMediaProvider::new()
			.submit_speech(request)
			.await
			.unwrap()
			.result
			.unwrap();
		assert_eq!(result.artifact.media_type, format.media_type());
		let requests = mock.requests();
		let body: Value = serde_json::from_str(&requests.last().unwrap().body).unwrap();
		assert_eq!(body["audio_setting"]["format"], native);
		assert_eq!(body["voice_setting"]["voice_id"], "custom_voice");
	}
}

fn assert_error_kind(error: &MediaError, kind: &str) {
	let actual = match error {
		MediaError::RateLimit { .. } => "rate",
		MediaError::InsufficientCredits { .. } => "balance",
		MediaError::Authentication { .. } => "auth",
		MediaError::InvalidRequest(_) => "params",
		MediaError::RemoteFailure(failure) if failure.category == FailureCategory::ContentPolicy => "policy",
		MediaError::Api { status, .. } if *status >= 500 => "server",
		_ => panic!("unexpected error: {error}"),
	};
	assert_eq!(actual, kind, "{error}");
}

#[tokio::test]
#[serial_test::serial]
async fn v1_nonzero_base_resp_on_http_200_is_an_error_for_image_and_speech() {
	let cases = [
		(1002, "rate"),
		(1039, "rate"),
		(1045, "rate"),
		(1008, "balance"),
		(2053, "balance"),
		(1004, "auth"),
		(2049, "auth"),
		(1026, "policy"),
		(1027, "policy"),
		(2013, "params"),
		(2014, "params"),
		(2022, "params"),
		(20132, "params"),
		(1000, "server"),
		(1001, "server"),
		(1024, "server"),
	];
	let replies = cases
		.iter()
		.flat_map(|(code, _)| {
			let body = json!({"base_resp":{"status_code":code,"status_msg":"rejected"}}).to_string();
			[(200, body.clone()), (200, body)]
		})
		.collect();
	let (mock, _env) = start_mock(replies);
	let provider = MinimaxMediaProvider::new();
	for (_, kind) in cases {
		let error = provider
			.submit_image(ImageGenerationRequest::new("castle").with_model("image-01"))
			.await
			.unwrap_err();
		assert_error_kind(&error, kind);
		let error = provider
			.submit_speech(SpeechSynthesisRequest::new("hello", "voice").with_model("speech-2.8-hd"))
			.await
			.unwrap_err();
		assert_error_kind(&error, kind);
	}
	assert_eq!(mock.requests().len(), cases.len() * 2);
}

#[tokio::test]
#[serial_test::serial]
async fn v2_http_errors_use_existing_failure_kinds_without_retrying_creation() {
	let cases = [
		(429, "rate"),
		(402, "balance"),
		(401, "auth"),
		(422, "policy"),
		(400, "params"),
		(500, "server"),
		(529, "server"),
	];
	let replies = cases.iter().map(|(status,_)| (*status,json!({"type":"error", "error":{"type":"provider_error", "message":"rejected", "http_code":status.to_string()}, "request_id":"trace-1"}).to_string())).collect();
	let (mock, _env) = start_mock(replies);
	for (_, kind) in cases {
		let error = MinimaxMediaProvider::new()
			.submit_video(video_request("MiniMax-H3", 5.0))
			.await
			.unwrap_err();
		assert_error_kind(&error, kind);
	}
	assert_eq!(mock.requests().len(), cases.len());
}

#[tokio::test]
#[serial_test::serial]
async fn duration_and_resolution_limits_fail_before_http() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"unexpected"}).to_string())]);
	let provider = MinimaxMediaProvider::new();
	for (model, duration) in [
		("MiniMax-H3", 3.0),
		("MiniMax-H3", 16.0),
		("MiniMax-H3", 5.5),
		("MiniMax-H3", f64::NAN),
		("MiniMax-H3", f64::INFINITY),
		("MiniMax-H3-Max", 4.0),
		("MiniMax-H3-Lite", 3.0),
		("MiniMax-H3-Highspeed", 16.0),
	] {
		assert!(matches!(
			provider.submit_video(video_request(model, duration)).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	for (model, resolution) in [
		("MiniMax-H3", "480P"),
		("MiniMax-H3-Highspeed", "2K"),
		("MiniMax-H3-Lite", "2K"),
		("MiniMax-H3-Max", "2K"),
		("MiniMax-H3", "1080P"),
	] {
		let mut request = video_request(model, 5.0);
		request
			.provider_options
			.insert(PROVIDER.to_string(), json!({"resolution":resolution}));
		assert!(matches!(
			provider.submit_video(request).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	assert!(mock.requests().is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn each_video_model_accepts_its_resolution_and_duration_boundaries() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"task-1"}).to_string())]);
	for (model, resolutions, minimum) in [
		("MiniMax-H3", vec!["768P", "2K"], 4.0),
		("MiniMax-H3-Highspeed", vec!["768P"], 4.0),
		("MiniMax-H3-Lite", vec!["480P", "768P"], 4.0),
		("MiniMax-H3-Max", vec!["480P", "768P"], 5.0),
	] {
		for resolution in resolutions {
			for seconds in [minimum, 15.0] {
				let mut request = video_request(model, seconds);
				request
					.provider_options
					.insert(PROVIDER.to_string(), json!({"resolution":resolution}));
				MinimaxMediaProvider::new().submit_video(request).await.unwrap();
			}
		}
	}
	assert_eq!(mock.requests().len(), 14);
}

#[tokio::test]
#[serial_test::serial]
async fn image_and_speech_limits_reject_before_http() {
	let (mock, _env) = start_mock(vec![(200, image_reply())]);
	let provider = MinimaxMediaProvider::new();
	for count in [0, 10] {
		let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
		request.count = Some(count);
		assert!(matches!(
			provider.submit_image(request).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	let request = ImageGenerationRequest::new("世".repeat(1501)).with_model("image-01");
	assert!(matches!(
		provider.submit_image(request).await,
		Err(MediaError::InvalidRequest(_))
	));
	for (width, height) in [(504, 768), (2056, 768), (1025, 768)] {
		let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
		request.geometry = Some(OutputGeometry::Dimensions { width, height });
		assert!(matches!(
			provider.submit_image(request).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	let request = SpeechSynthesisRequest::new("世".repeat(10000), "voice").with_model("speech-2.8-hd");
	assert!(matches!(
		provider.submit_speech(request).await,
		Err(MediaError::InvalidRequest(_))
	));
	assert!(mock.requests().is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn malformed_response_fields_are_never_defaulted() {
	let provider = MinimaxMediaProvider::new();
	for body in [
		json!({}),
		json!({"base_resp":{"status_code":0},"metadata":{},"data":{}}),
		json!({"base_resp":{"status_code":0},"metadata":{},"data":{"image_base64":[]}}),
		json!({"base_resp":{"status_code":0},"metadata":{},"data":{"image_base64":["not base64"]}}),
		json!({"base_resp":{"status_code":0},"metadata":{},"data":{"image_base64":["aW1hZ2U="]}}),
	] {
		let (_mock, _env) = start_mock(vec![(200, body.to_string())]);
		assert!(matches!(
			provider
				.submit_image(ImageGenerationRequest::new("castle").with_model("image-01"))
				.await,
			Err(MediaError::InvalidResponse { .. })
		));
	}
	for body in [
		json!({"base_resp":{"status_code":0},"data":{"status":2,"audio":"zz"}}),
		json!({"base_resp":{"status_code":0},"data":{"status":1,"audio":"49"}}),
		json!({"base_resp":{"status_code":0},"data":{"status":2,"audio":"49"},"extra_info":{"audio_length":"invalid"},"trace_id":"trace-1"}),
	] {
		let (_mock, _env) = start_mock(vec![(200, body.to_string())]);
		assert!(matches!(
			provider
				.submit_speech(SpeechSynthesisRequest::new("hello", "voice").with_model("speech-2.8-hd"))
				.await,
			Err(MediaError::InvalidResponse { .. })
		));
	}
	let (_mock, _env) = start_mock(vec![(200, "{}".to_string())]);
	assert!(matches!(
		provider.submit_video(video_request("MiniMax-H3", 5.0)).await,
		Err(MediaError::InvalidResponse { .. })
	));
	assert!(matches!(
		provider.poll_video(&handle()).await,
		Err(MediaError::InvalidResponse { .. })
	));
}

#[tokio::test]
#[serial_test::serial]
async fn video_prices_returned_duration_with_the_stored_rate() {
	let (_mock, _env) = start_mock(vec![(
		200,
		json!({"task":{"status":"succeeded", "duration":7, "content":{"url":"https://cdn.example.test/out.mp4"}}})
			.to_string(),
	)]);
	let mut handle = handle();
	handle.cost_estimate = Some(CostEstimate {
		unit: UsageUnit::VideoSeconds,
		usd_per_unit: 0.5,
		quantity: Some(99.0),
	});
	let result = MinimaxMediaProvider::new()
		.poll_video(&handle)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.usage.unwrap().estimated_cost, Some(3.5));
	handle.cost_estimate = None;
	let result = MinimaxMediaProvider::new()
		.poll_video(&handle)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.usage.unwrap().estimated_cost, None);
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::CostUnavailable));
}

#[tokio::test]
#[serial_test::serial]
async fn text_video_defaults_to_a_fixed_ratio_and_image_video_rejects_geometry() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"task-1"}).to_string())]);
	let provider = MinimaxMediaProvider::new();
	provider.submit_video(video_request("MiniMax-H3", 5.0)).await.unwrap();
	let body: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
	assert_eq!(body["ratio"], "16:9");
	let mut request = video_request("MiniMax-H3", 5.0);
	request.mode = VideoGenerationMode::ImageToVideo;
	request.first_frame = Some(MediaSource::Bytes {
		data: b"image".to_vec(),
		media_type: "image/png".to_string(),
	});
	request.geometry = Some(OutputGeometry::AspectRatio { width: 16, height: 9 });
	assert!(matches!(
		provider.submit_video(request).await,
		Err(MediaError::UnsupportedParameter { .. })
	));
	assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
#[serial_test::serial]
async fn unsupported_options_follow_the_request_policy() {
	let (mock, _env) = start_mock(vec![(200, image_reply())]);
	let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
	request.negative_prompt = Some("rain".to_string());
	assert!(matches!(
		MinimaxMediaProvider::new().submit_image(request.clone()).await,
		Err(MediaError::UnsupportedParameter { .. })
	));
	assert!(mock.requests().is_empty());
	request.request_options.unsupported_parameter_policy = UnsupportedParameterPolicy::WarnAndDrop;
	let result = MinimaxMediaProvider::new()
		.submit_image(request)
		.await
		.unwrap()
		.result
		.unwrap();
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::UnsupportedParameterDropped
			&& warning.parameter.as_deref() == Some("negative_prompt")));
}

#[tokio::test]
#[serial_test::serial]
async fn cancellation_and_immediate_polling_make_no_http_calls() {
	let (mock, _env) = start_mock(vec![(200, "{}".to_string())]);
	let provider = MinimaxMediaProvider::new();
	assert!(matches!(
		provider.cancel_video(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(matches!(
		provider.poll_image(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(matches!(
		provider.cancel_image(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(matches!(
		provider.poll_speech(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(matches!(
		provider.cancel_speech(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(mock.requests().is_empty());
}

#[test]
fn minimax_is_registered_for_image_video_speech_only() {
	use crate::media::factory::MediaProviderFactory;
	let (image, model) = MediaProviderFactory::get_image_provider_for_model("minimax:image-01").unwrap();
	assert_eq!(image.name(), PROVIDER);
	assert_eq!(model, "image-01");
	assert!(MediaProviderFactory::get_video_provider_for_model("minimax:MiniMax-H3-Max").is_ok());
	assert!(MediaProviderFactory::get_speech_provider_for_model("minimax:speech-2.8-turbo").is_ok());
	assert!(MediaProviderFactory::create_transcription_provider("minimax").is_err());
	assert!(MediaProviderFactory::supported_providers().contains(&PROVIDER));
	assert!(MediaProviderFactory::get_video_provider_for_model("minimax:image-01").is_err());
}

#[tokio::test]
#[serial_test::serial]
async fn speech_voice_sentinel_matches_exactly_and_other_values_pass_through() {
	let (mock, _env) = start_mock(vec![(200, speech_reply())]);
	for (voice, expected) in [
		("default", DEFAULT_VOICE),
		("custom_voice", "custom_voice"),
		("Default", "Default"),
		(" default ", " default "),
		("", ""),
	] {
		MinimaxMediaProvider::new()
			.submit_speech(SpeechSynthesisRequest::new("hello", voice).with_model("speech-2.8-hd"))
			.await
			.unwrap();
		let requests = mock.requests();
		let body: Value = serde_json::from_str(&requests.last().unwrap().body).unwrap();
		assert_eq!(body["voice_setting"]["voice_id"], expected);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn speech_duration_is_unknown_when_extra_info_or_audio_length_is_absent() {
	let mut no_extra_info: Value = serde_json::from_str(&speech_reply()).unwrap();
	no_extra_info.as_object_mut().unwrap().remove("extra_info");
	let mut no_audio_length: Value = serde_json::from_str(&speech_reply()).unwrap();
	no_audio_length["extra_info"]
		.as_object_mut()
		.unwrap()
		.remove("audio_length");
	let (_mock, _env) = start_mock(vec![
		(200, no_extra_info.to_string()),
		(200, no_audio_length.to_string()),
	]);
	for _ in 0..2 {
		let operation = MinimaxMediaProvider::new()
			.submit_speech(SpeechSynthesisRequest::new("hello", "default").with_model("speech-2.8-hd"))
			.await
			.unwrap();
		assert_eq!(operation.status, OperationStatus::Succeeded);
		assert_eq!(operation.result.unwrap().artifact.duration_secs, None);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn video_without_duration_sends_five_seconds_for_every_h3_model() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"task-1"}).to_string())]);
	for model in [
		"MiniMax-H3",
		"MiniMax-H3-Highspeed",
		"MiniMax-H3-Lite",
		"MiniMax-H3-Max",
	] {
		MinimaxMediaProvider::new()
			.submit_video(VideoGenerationRequest::new("castle").with_model(model))
			.await
			.unwrap();
		let requests = mock.requests();
		let body: Value = serde_json::from_str(&requests.last().unwrap().body).unwrap();
		assert_eq!(body["duration"], json!(5));
		assert!(body["duration"].is_u64());
	}
}

#[tokio::test]
#[serial_test::serial]
async fn video_without_reported_duration_keeps_the_artifact_and_usage_unknown() {
	let (_mock, _env) = start_mock(vec![(
		200,
		json!({"task":{"status":"succeeded", "content":{"url":"https://cdn.example.test/out.mp4"}}}).to_string(),
	)]);
	let operation = MinimaxMediaProvider::new().poll_video(&handle()).await.unwrap();
	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.unwrap();
	assert_eq!(result.artifacts[0].duration_secs, None);
	let usage = result.usage.unwrap();
	assert!(usage.line_items.is_empty());
	assert_eq!(usage.estimated_cost, None);
}

#[tokio::test]
#[serial_test::serial]
async fn video_estimates_snapshot_the_selected_model_and_resolution_rate() {
	let cases = [
		("MiniMax-H3", None, "768P", 0.08),
		("MiniMax-H3", Some("2K"), "2K", 0.13),
		("MiniMax-H3-Highspeed", None, "768P", 0.06),
		("MiniMax-H3-Lite", None, "768P", 0.04),
		("MiniMax-H3-Lite", Some("480P"), "480P", 0.025),
		("MiniMax-H3-Max", None, "768P", 0.08),
		("MiniMax-H3-Max", Some("480P"), "480P", 0.05),
	];
	let replies = cases.iter().flat_map(|(model, _, resolution, _)| [
		(200, json!({"task_id":"task-1"}).to_string()),
		(200, json!({"task":{"id":"task-1", "model":model, "status":"succeeded", "resolution":resolution, "duration":7, "content":{"url":"https://cdn.example.test/out.mp4"}}}).to_string()),
	]).collect();
	let (mock, _env) = start_mock(replies);
	let provider = MinimaxMediaProvider::new();
	for (index, (model, selected_resolution, sent_resolution, rate)) in cases.into_iter().enumerate() {
		let mut request = video_request(model, 5.0);
		if let Some(resolution) = selected_resolution {
			request
				.provider_options
				.insert(PROVIDER.to_string(), json!({"resolution":resolution}));
		}
		let submitted = provider.submit_video(request).await.unwrap();
		let handle: JobHandle =
			serde_json::from_str(&serde_json::to_string(&submitted.handle.unwrap()).unwrap()).unwrap();
		let estimate = handle.cost_estimate.as_ref().unwrap();
		assert_eq!(estimate.unit, UsageUnit::VideoSeconds);
		assert_eq!(estimate.usd_per_unit, rate);
		assert_eq!(estimate.quantity, None);
		let result = provider.poll_video(&handle).await.unwrap().result.unwrap();
		let usage = result.usage.unwrap();
		assert_eq!(usage.line_items[0].quantity, 7.0);
		assert_eq!(usage.estimated_cost, Some(rate * 7.0));
		assert_eq!(usage.provider_reported_cost, None);
		let body: Value = serde_json::from_str(&mock.requests()[index * 2].body).unwrap();
		assert_eq!(body["resolution"], sent_resolution);
	}
}

#[tokio::test]
#[serial_test::serial]
async fn speech_without_reported_characters_marks_usage_unavailable() {
	let mut missing_characters: Value = serde_json::from_str(&speech_reply()).unwrap();
	missing_characters["extra_info"]
		.as_object_mut()
		.unwrap()
		.remove("usage_characters");
	let mut missing_extra_info: Value = serde_json::from_str(&speech_reply()).unwrap();
	missing_extra_info.as_object_mut().unwrap().remove("extra_info");
	let (_mock, _env) = start_mock(vec![
		(200, missing_characters.to_string()),
		(200, missing_extra_info.to_string()),
	]);
	for _ in 0..2 {
		let result = MinimaxMediaProvider::new()
			.submit_speech(SpeechSynthesisRequest::new("hello", "default").with_model("speech-2.8-hd"))
			.await
			.unwrap()
			.result
			.unwrap();
		let usage = result.usage.unwrap();
		assert!(usage.line_items.is_empty());
		assert_eq!(usage.estimated_cost, None);
		assert_eq!(usage.provider_reported_cost, None);
		assert!(result
			.warnings
			.iter()
			.any(|warning| warning.code == WarningCode::UsageUnavailable));
	}
}

#[tokio::test]
#[serial_test::serial]
async fn speech_rejects_malformed_reported_characters_and_accepts_zero() {
	let invalid = [
		json!(-1),
		json!(1.5),
		json!(1.0),
		json!("999"),
		Value::Null,
		json!(true),
	];
	let mut replies: Vec<_> = invalid
		.iter()
		.map(|characters| {
			let mut value: Value = serde_json::from_str(&speech_reply()).unwrap();
			value["extra_info"]["usage_characters"] = characters.clone();
			(200, value.to_string())
		})
		.collect();
	let mut zero: Value = serde_json::from_str(&speech_reply()).unwrap();
	zero["extra_info"]["usage_characters"] = json!(0);
	replies.push((200, zero.to_string()));
	let (_mock, _env) = start_mock(replies);
	let provider = MinimaxMediaProvider::new();
	for _ in invalid {
		let error = provider
			.submit_speech(SpeechSynthesisRequest::new("hello", "default").with_model("speech-2.8-hd"))
			.await
			.unwrap_err();
		assert!(
			matches!(&error, MediaError::InvalidResponse {message, ..} if message.contains("extra_info.usage_characters")),
			"{error}"
		);
	}
	let result = provider
		.submit_speech(SpeechSynthesisRequest::new("hello", "default").with_model("speech-2.8-hd"))
		.await
		.unwrap()
		.result
		.unwrap();
	let usage = result.usage.unwrap();
	assert_eq!(usage.line_items[0].quantity, 0.0);
	assert_eq!(usage.estimated_cost, Some(0.0));
	assert!(!result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::UsageUnavailable));
}
