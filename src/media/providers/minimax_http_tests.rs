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

fn image_video_request(first_frame: MediaSource) -> VideoGenerationRequest {
	let mut request = video_request("MiniMax-H3-Lite", 4.0);
	request.mode = VideoGenerationMode::ImageToVideo;
	request.first_frame = Some(first_frame);
	request
}

fn png_frame() -> MediaSource {
	MediaSource::Bytes {
		data: b"image".to_vec(),
		media_type: "image/png".to_string(),
	}
}

fn upload_reply(file_id: u64) -> String {
	json!({"file":{"file_id":file_id, "bytes":5, "filename":"input.png", "purpose":"video_generation_input"}, "base_resp":{"status_code":0,"status_msg":"success"}}).to_string()
}

fn delete_reply(file_id: u64) -> String {
	json!({"file_id":file_id, "base_resp":{"status_code":0,"status_msg":"success"}}).to_string()
}

fn handle() -> JobHandle {
	JobHandle {
		provider: PROVIDER.to_string(),
		model: "MiniMax-H3".to_string(),
		task: MediaTask::TextToVideo,
		remote_id: "task-1".to_string(),
		cost_estimate: shared::resolved_cost_estimate(PROVIDER, "MiniMax-H3", None, None, None),
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
	assert_eq!(handle.cost_estimate.as_ref().unwrap().quantity, Some(5.0));
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
async fn video_start_image_is_uploaded_referenced_and_deleted() {
	let (mock, _env) = start_mock(vec![
		(200, upload_reply(42)),
		(200, json!({"task_id":"task-1"}).to_string()),
		(200, delete_reply(42)),
	]);
	let submitted = MinimaxMediaProvider::new()
		.submit_video(image_video_request(png_frame()))
		.await
		.unwrap();
	assert_eq!(submitted.handle.unwrap().task, MediaTask::ImageToVideo);

	let requests = mock.requests();
	assert_eq!(requests.len(), 3, "upload, create, delete");
	assert_eq!(requests[0].path, "/v1/files/upload");
	assert!(requests[0]
		.body
		.contains("name=\"purpose\"\r\n\r\nvideo_generation_input"));
	assert!(requests[0].body.contains("filename=\"input.png\""));
	assert!(
		!requests[0].body.contains("base64"),
		"the frame travels as a file, not inline"
	);

	assert_eq!(requests[1].path, "/v2/video_generation");
	let body: Value = serde_json::from_str(&requests[1].body).unwrap();
	assert_eq!(body["resolution"], "768P");
	assert_eq!(body["ratio"], "adaptive");
	assert_eq!(
		body["content"][1],
		json!({"type":"image_url", "image_url":{"url":"mm_file://42"}, "role":"first_frame"})
	);

	assert_eq!(requests[2].path, "/v1/files/delete");
	let body: Value = serde_json::from_str(&requests[2].body).unwrap();
	assert_eq!(body, json!({"file_id":42, "purpose":"video_generation_input"}));
}

#[tokio::test]
#[serial_test::serial]
async fn a_refused_create_still_deletes_the_uploaded_frame() {
	let (mock, _env) = start_mock(vec![
		(200, upload_reply(7)),
		(
			400,
			json!({"base_resp":{"status_code":2013,"status_msg":"invalid params"}}).to_string(),
		),
		(200, delete_reply(7)),
	]);
	assert!(MinimaxMediaProvider::new()
		.submit_video(image_video_request(png_frame()))
		.await
		.is_err());
	let requests = mock.requests();
	assert_eq!(requests.len(), 3);
	assert_eq!(requests[2].path, "/v1/files/delete");
}

#[tokio::test]
#[serial_test::serial]
async fn a_failed_delete_does_not_fail_the_created_task() {
	let (_mock, _env) = start_mock(vec![
		(200, upload_reply(7)),
		(200, json!({"task_id":"task-1"}).to_string()),
		(
			200,
			json!({"file_id":0,"base_resp":{"status_code":1000,"status_msg":"record not found"}}).to_string(),
		),
	]);
	let submitted = MinimaxMediaProvider::new()
		.submit_video(image_video_request(png_frame()))
		.await
		.expect("the task exists; a leftover file expires on its own");
	assert_eq!(submitted.handle.unwrap().remote_id, "task-1");
}

#[tokio::test]
#[serial_test::serial]
async fn a_url_start_image_is_passed_through_without_upload() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"task-1"}).to_string())]);
	let url = "https://example.test/frame.png";
	MinimaxMediaProvider::new()
		.submit_video(image_video_request(MediaSource::Url {
			url: url.to_string(),
			media_type: None,
		}))
		.await
		.unwrap();
	let requests = mock.requests();
	assert_eq!(requests.len(), 1, "no upload, no delete");
	let body: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(body["content"][1]["image_url"]["url"], url);
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
		("MiniMax-H3-Max-Turbo", 4.0),
		("MiniMax-H3-MG", 3.0),
		("MiniMax-H3-NC", 16.0),
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
		("MiniMax-H3-Highspeed", "2K"),
		("MiniMax-H3-Max-Turbo", "2K"),
		("MiniMax-H3-MG", "480P"),
		("MiniMax-H3-NC", "2K"),
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
		("MiniMax-H3", vec!["480P", "768P", "2K"], 4.0),
		("MiniMax-H3-Highspeed", vec!["480P", "768P"], 4.0),
		("MiniMax-H3-Lite", vec!["480P", "768P"], 4.0),
		("MiniMax-H3-Max", vec!["480P", "768P"], 5.0),
		("MiniMax-H3-Max-Turbo", vec!["480P", "768P"], 5.0),
		("MiniMax-H3-MG", vec!["768P", "2K"], 4.0),
		("MiniMax-H3-NC", vec!["480P", "768P"], 4.0),
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
	assert_eq!(mock.requests().len(), 30);
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
			provider.submit_image(request.clone()).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	let request = ImageGenerationRequest::new("世".repeat(1501)).with_model("image-01");
	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::InvalidRequest(_))
	));
	for (width, height) in [(504, 768), (2056, 768), (1025, 768)] {
		let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
		request.geometry = Some(OutputGeometry::Dimensions { width, height });
		assert!(matches!(
			provider.submit_image(request.clone()).await,
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
fn minimax_is_registered_for_image_video_speech_and_transcription() {
	use crate::media::factory::MediaProviderFactory;
	let (image, model) = MediaProviderFactory::get_image_provider_for_model("minimax:image-01").unwrap();
	assert_eq!(image.name(), PROVIDER);
	assert_eq!(model, "image-01");
	assert!(MediaProviderFactory::get_video_provider_for_model("minimax:MiniMax-H3-Max").is_ok());
	assert!(MediaProviderFactory::get_speech_provider_for_model("minimax:speech-2.8-turbo").is_ok());
	let (transcription, model) = MediaProviderFactory::get_transcription_provider_for_model("minimax:asr-1.0").unwrap();
	assert_eq!(transcription.name(), PROVIDER);
	assert_eq!(model, "asr-1.0");
	assert!(MediaProviderFactory::get_transcription_provider_for_model("minimax:speech-2.8-turbo").is_err());
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
		"MiniMax-H3-Max-Turbo",
		"MiniMax-H3-MG",
		"MiniMax-H3-NC",
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
		assert_eq!(estimate.quantity, Some(5.0));
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
async fn speech_without_reported_characters_bills_the_text_length() {
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
		// "hello" at speech-2.8-hd's $0.0001 a character.
		assert!((usage.estimated_cost.unwrap() - 5.0 * 0.0001).abs() < 1e-12);
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

/// Costs are sums of floats, so they are compared within a fraction of a cent.
fn assert_usd(actual: Option<f64>, expected: f64) {
	let actual = actual.expect("a priced MiniMax reply");
	assert!((actual - expected).abs() < 1e-9, "cost {actual} is not {expected}");
}

fn h3_image_reply() -> String {
	json!({
		"created":1765823,
		"request_id":"req-1",
		"data":[{"url":"https://cdn.example.test/a.jpg","size":"1024x1024","output_format":"jpeg","revised_prompt":"a castle at dusk"}],
		"usage":{"input_tokens":40,"output_tokens":4080,"total_tokens":4120,
			"input_tokens_details":{"text_tokens":40,"image_tokens":0},
			"output_tokens_details":{"image_tokens":4080}}
	})
	.to_string()
}

#[tokio::test]
#[serial_test::serial]
async fn h3_image_sends_geometry_and_prices_the_reported_tokens() {
	let (mock, _env) = start_mock(vec![(200, h3_image_reply())]);
	let mut request = ImageGenerationRequest::new("castle").with_model("MiniMax-H3-Image");
	request.count = Some(10);
	request.geometry = Some(OutputGeometry::Dimensions {
		width: 1024,
		height: 768,
	});
	let result = MinimaxMediaProvider::new()
		.submit_image(request)
		.await
		.unwrap()
		.result
		.unwrap();
	let requests = mock.requests();
	assert_eq!(requests[0].path, "/v1/images/generations");
	let body: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert_eq!(
		body,
		json!({"model":"MiniMax-H3-Image", "prompt":"castle", "n":10, "response_format":"url", "size":"1024x768"})
	);
	let artifact = &result.artifacts[0];
	assert_eq!(
		artifact.source,
		ArtifactSource::Url("https://cdn.example.test/a.jpg".to_string())
	);
	assert_eq!(artifact.media_type, "image/jpeg");
	assert_eq!(
		artifact.dimensions,
		Some(Dimensions {
			width: 1024,
			height: 1024
		})
	);
	let usage = result.usage.unwrap();
	assert_eq!(usage.line_items.len(), 3);
	// 40 text input tokens at $2.30 and 4080 output tokens at $3.80 per 1M.
	assert_eq!(usage.estimated_cost, Some(0.015596));
	assert_usd(usage.line_items[2].cost, 0.015504);
}

#[tokio::test]
#[serial_test::serial]
async fn h3_image_highspeed_takes_an_aspect_ratio_and_rejects_bad_geometry() {
	let (mock, _env) = start_mock(vec![(200, h3_image_reply())]);
	let mut request = ImageGenerationRequest::new("castle").with_model("MiniMax-H3-Image-Highspeed");
	request.geometry = Some(OutputGeometry::AspectRatio { width: 21, height: 9 });
	MinimaxMediaProvider::new().submit_image(request).await.unwrap();
	let body: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
	assert_eq!(body["aspect_ratio"], "21:9");
	let provider = MinimaxMediaProvider::new();
	for geometry in [
		OutputGeometry::Dimensions {
			width: 512,
			height: 512,
		},
		OutputGeometry::Dimensions {
			width: 4096,
			height: 1024,
		},
		OutputGeometry::AspectRatio { width: 4, height: 1 },
		OutputGeometry::AspectRatio { width: 0, height: 9 },
	] {
		let mut request = ImageGenerationRequest::new("castle").with_model("MiniMax-H3-Image");
		request.geometry = Some(geometry);
		assert!(matches!(
			provider.submit_image(request.clone()).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	let request = ImageGenerationRequest::new("世".repeat(32001)).with_model("MiniMax-H3-Image");
	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::InvalidRequest(_))
	));
	let mut request = ImageGenerationRequest::new("世".repeat(1501)).with_model("image-01");
	request.count = Some(10);
	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::InvalidRequest(_))
	));
	assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
#[serial_test::serial]
async fn h3_image_edit_uploads_references_and_deletes_them() {
	let (mock, _env) = start_mock(vec![
		(200, upload_reply(9)),
		(200, h3_image_reply()),
		(200, delete_reply(9)),
	]);
	let mut request = ImageGenerationRequest::new("castle at dusk").with_model("MiniMax-H3-Image");
	request.mode = ImageGenerationMode::Edit;
	request.source_images = vec![
		png_frame(),
		MediaSource::Url {
			url: "https://example.test/ref.jpg".to_string(),
			media_type: None,
		},
	];
	MinimaxMediaProvider::new().submit_image(request).await.unwrap();
	let requests = mock.requests();
	assert_eq!(requests.len(), 3, "upload, edit, delete");
	assert!(requests[0]
		.body
		.contains("name=\"purpose\"\r\n\r\nimage_generation_input"));
	assert_eq!(requests[1].path, "/v1/images/edits");
	let body: Value = serde_json::from_str(&requests[1].body).unwrap();
	assert_eq!(
		body["images"],
		json!([{"image_url":"mm_file://9"}, {"image_url":"https://example.test/ref.jpg"}])
	);
	assert_eq!(requests[2].path, "/v1/files/delete");
	assert!(requests[2].body.contains("image_generation_input"));

	// image-01 still cannot edit, and an edit needs at least one image.
	let mut request = ImageGenerationRequest::new("castle").with_model("image-01");
	request.mode = ImageGenerationMode::Edit;
	request.source_images = vec![png_frame()];
	assert!(matches!(
		MinimaxMediaProvider::new().submit_image(request).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	let mut request = ImageGenerationRequest::new("castle").with_model("MiniMax-H3-Image");
	request.mode = ImageGenerationMode::Edit;
	assert!(matches!(
		MinimaxMediaProvider::new().submit_image(request).await,
		Err(MediaError::InvalidRequest(_))
	));
	assert_eq!(mock.requests().len(), 3);
}

#[tokio::test]
#[serial_test::serial]
async fn h3_image_failed_items_are_filtered_and_all_failed_is_content_policy() {
	let partial = json!({"request_id":"req-1","data":[
		{"error":{"code":"1027","message":"generated image failed the safety check"}},
		{"url":"https://cdn.example.test/a.jpg","size":"1024x1024","output_format":"png"}
	]})
	.to_string();
	let (_mock, _env) = start_mock(vec![
		(200, partial),
		(
			200,
			json!({"request_id":"req-1","data":[
				{"error":{"code":"1027","message":"blocked"}},
				{"error":{"code":"1026","message":"prompt blocked"}}
			]})
			.to_string(),
		),
		(200, json!({"request_id":"req-1","data":[]}).to_string()),
		(
			200,
			json!({"request_id":"req-1","data":[{"url":"https://cdn.example.test/a.jpg"}]}).to_string(),
		),
	]);
	let provider = MinimaxMediaProvider::new();
	let request = ImageGenerationRequest::new("castle").with_model("MiniMax-H3-Image");
	let operation = provider.submit_image(request.clone()).await.unwrap();
	let result = operation.result.unwrap();
	assert_eq!(result.artifacts.len(), 1);
	assert_eq!(result.safety.status, SafetyStatus::PartiallyFiltered);
	assert_eq!(result.safety.filtered_count, 1);
	assert!(result.safety.reasons[0].contains("1027"));
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::PartialOutput));
	// No usage and no media type: the cost and the image are not fabricated.
	assert_eq!(result.usage.unwrap().estimated_cost, None);

	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::RemoteFailure(GenerationFailure {
			category: FailureCategory::ContentPolicy,
			..
		}))
	));
	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::InvalidResponse { .. })
	));
	assert!(matches!(
		provider.submit_image(request.clone()).await,
		Err(MediaError::InvalidResponse { .. })
	));
}

#[tokio::test]
#[serial_test::serial]
async fn image_01_live_accepts_a_ratio_and_drops_dimensions_only_under_warn_policy() {
	let (mock, _env) = start_mock(vec![(200, image_reply())]);
	let mut request = ImageGenerationRequest::new("castle").with_model("image-01-live");
	request.geometry = Some(OutputGeometry::Dimensions {
		width: 1024,
		height: 1024,
	});
	assert!(matches!(
		MinimaxMediaProvider::new().submit_image(request.clone()).await,
		Err(MediaError::UnsupportedParameter { .. })
	));
	request.request_options.unsupported_parameter_policy = UnsupportedParameterPolicy::WarnAndDrop;
	let operation = MinimaxMediaProvider::new().submit_image(request).await.unwrap();
	assert!(operation
		.result
		.unwrap()
		.warnings
		.iter()
		.any(|warning| warning.parameter.as_deref() == Some("geometry")));
	let mut request = ImageGenerationRequest::new("castle").with_model("image-01-live");
	request.geometry = Some(OutputGeometry::AspectRatio { width: 9, height: 16 });
	MinimaxMediaProvider::new().submit_image(request).await.unwrap();
	let requests = mock.requests();
	assert_eq!(requests.len(), 2);
	let first: Value = serde_json::from_str(&requests[0].body).unwrap();
	assert!(first.get("width").is_none() && first.get("height").is_none());
	let second: Value = serde_json::from_str(&requests[1].body).unwrap();
	assert_eq!(second["aspect_ratio"], "9:16");
	assert_eq!(
		shared::resolved_cost_estimate(PROVIDER, "image-01-live", None, None, None)
			.unwrap()
			.usd_per_unit,
		0.0035
	);
}

fn reference_video_request() -> VideoGenerationRequest {
	let mut request = video_request("MiniMax-H3-Max-Turbo", 6.0);
	request.mode = VideoGenerationMode::ReferenceToVideo;
	request.reference_images = vec![
		png_frame(),
		MediaSource::Url {
			url: "https://example.test/style.jpg".to_string(),
			media_type: None,
		},
	];
	request.source_video = Some(MediaSource::Bytes {
		data: b"clip".to_vec(),
		media_type: "video/mp4".to_string(),
	});
	request
}

#[tokio::test]
#[serial_test::serial]
async fn reference_inputs_are_uploaded_tagged_with_roles_and_deleted() {
	let (mock, _env) = start_mock(vec![
		(200, upload_reply(1)),
		(200, upload_reply(2)),
		(200, json!({"task_id":"task-1"}).to_string()),
		(200, delete_reply(1)),
		(200, delete_reply(2)),
	]);
	let operation = MinimaxMediaProvider::new()
		.submit_video(reference_video_request())
		.await
		.unwrap();
	let handle = operation.handle.unwrap();
	assert_eq!(handle.task, MediaTask::ReferenceToVideo);
	// A reference task on Max-Turbo 768P bills its output at $0.08 per second.
	assert_eq!(handle.cost_estimate.as_ref().unwrap().usd_per_unit, 0.08);
	let requests = mock.requests();
	assert_eq!(requests.len(), 5, "two uploads, create, two deletes");
	assert!(requests[1].body.contains("filename=\"input.mp4\""));
	let body: Value = serde_json::from_str(&requests[2].body).unwrap();
	assert_eq!(body["ratio"], "adaptive");
	assert_eq!(
		body["content"],
		json!([
			{"type":"text", "text":"a castle"},
			{"type":"image_url", "image_url":{"url":"mm_file://1"}, "role":"reference_image"},
			{"type":"image_url", "image_url":{"url":"https://example.test/style.jpg"}, "role":"reference_image"},
			{"type":"video_url", "video_url":{"url":"mm_file://2"}, "role":"reference_video"}
		])
	);
	assert!(requests[3..].iter().all(|request| request.path == "/v1/files/delete"));
}

#[tokio::test]
#[serial_test::serial]
async fn last_frame_and_reference_ratios_follow_each_models_contract() {
	let (mock, _env) = start_mock(vec![
		(200, json!({"task_id":"task-1"}).to_string()),
		(200, json!({"task_id":"task-2"}).to_string()),
		(200, json!({"task_id":"task-3"}).to_string()),
	]);
	let provider = MinimaxMediaProvider::new();
	let mut request = video_request("MiniMax-H3", 5.0);
	request.mode = VideoGenerationMode::ImageToVideo;
	request.last_frame = Some(MediaSource::Url {
		url: "https://example.test/last.png".to_string(),
		media_type: None,
	});
	let operation = provider.submit_video(request).await.unwrap();
	assert_eq!(operation.handle.unwrap().task, MediaTask::ImageToVideo);
	let body: Value = serde_json::from_str(&mock.requests()[0].body).unwrap();
	assert_eq!(body["ratio"], "adaptive");
	assert_eq!(body["content"][1]["role"], "last_frame");

	// Lite has no `adaptive` for reference tasks, so the default is a fixed ratio.
	// URL inputs need no upload, so each request here is one create call.
	let url_reference = || {
		let mut request = reference_video_request();
		request.reference_images = vec![MediaSource::Url {
			url: "https://example.test/style.jpg".to_string(),
			media_type: None,
		}];
		request.source_video = None;
		request
	};
	let mut request = url_reference();
	request.model = "MiniMax-H3-Lite".to_string();
	provider.submit_video(request).await.unwrap();
	let body: Value = serde_json::from_str(&mock.requests()[1].body).unwrap();
	assert_eq!(body["ratio"], "16:9");

	// A named ratio still overrides the default.
	let mut request = url_reference();
	request.geometry = Some(OutputGeometry::AspectRatio { width: 1, height: 1 });
	provider.submit_video(request).await.unwrap();
	let body: Value = serde_json::from_str(&mock.requests()[2].body).unwrap();
	assert_eq!(body["ratio"], "1:1");
	assert_eq!(mock.requests().len(), 3);
}

#[tokio::test]
#[serial_test::serial]
async fn mismatched_video_modes_and_inputs_are_refused_before_http() {
	let (mock, _env) = start_mock(vec![(200, json!({"task_id":"unexpected"}).to_string())]);
	let provider = MinimaxMediaProvider::new();
	let mut frames_and_references = reference_video_request();
	frames_and_references.first_frame = Some(MediaSource::Url {
		url: "https://example.test/first.png".to_string(),
		media_type: None,
	});
	let mut references_in_image_mode = reference_video_request();
	references_in_image_mode.mode = VideoGenerationMode::ImageToVideo;
	references_in_image_mode.first_frame = Some(png_frame());
	let mut empty_reference = video_request("MiniMax-H3", 5.0);
	empty_reference.mode = VideoGenerationMode::ReferenceToVideo;
	let mut too_many_references = reference_video_request();
	too_many_references.reference_images = vec![png_frame(); 10];
	for request in [
		frames_and_references,
		references_in_image_mode,
		empty_reference,
		too_many_references,
	] {
		assert!(matches!(
			provider.submit_video(request).await,
			Err(MediaError::InvalidRequest(_))
		));
	}
	let mut reference_in_text_mode = reference_video_request();
	reference_in_text_mode.mode = VideoGenerationMode::TextToVideo;
	assert!(matches!(
		provider.submit_video(reference_in_text_mode).await,
		Err(MediaError::InvalidRequest(_))
	));
	assert!(mock.requests().is_empty());
}

#[tokio::test]
#[serial_test::serial]
async fn reference_input_video_and_extra_images_are_billed_from_reported_usage() {
	let finished = |usage: Value| {
		json!({"task":{
			"id":"task-1", "model":"MiniMax-H3-Max", "status":"succeeded", "duration":6,
			"resolution":"768P", "content":{"url":"https://cdn.example.test/out.mp4"}, "usage":usage
		}})
		.to_string()
	};
	let (mock, _env) = start_mock(vec![
		(
			200,
			finished(json!({"output_seconds":6, "input_seconds":4, "input_image_count":7})),
		),
		(
			200,
			finished(json!({"output_seconds":6, "input_seconds":4, "input_image_count":2})),
		),
		(
			200,
			json!({"task":{"id":"task-1", "status":"succeeded", "duration":6, "content":{"url":"https://cdn.example.test/out.mp4"}, "usage":{"input_seconds":4}}}).to_string(),
		),
	]);
	let mut handle = handle();
	handle.model = "MiniMax-H3-Max".to_string();
	handle.task = MediaTask::ReferenceToVideo;
	handle.cost_estimate = Some(CostEstimate {
		unit: UsageUnit::VideoSeconds,
		usd_per_unit: 0.08,
		quantity: Some(6.0),
	});
	let provider = MinimaxMediaProvider::new();
	let usage = provider
		.poll_video(&handle)
		.await
		.unwrap()
		.result
		.unwrap()
		.usage
		.unwrap();
	// 6 s output at $0.08, 4 s input video at $0.143, 5 of 7 images at $0.074.
	assert_usd(usage.estimated_cost, 0.48 + 0.572 + 5.0 * 0.074);
	assert_eq!(usage.line_items.len(), 3);
	assert_eq!(usage.line_items[0].quantity, 6.0);
	assert_eq!(usage.line_items[1].unit, UsageUnit::VideoSeconds);
	assert_eq!(usage.line_items[1].quantity, 4.0);
	assert_eq!(usage.line_items[2].unit, UsageUnit::Images);
	assert_eq!(usage.line_items[2].quantity, 5.0);
	// Two images fall inside the free quota, so only the video is added.
	let usage = provider
		.poll_video(&handle)
		.await
		.unwrap()
		.result
		.unwrap()
		.usage
		.unwrap();
	assert_eq!(usage.line_items.len(), 2);
	assert_usd(usage.estimated_cost, 0.48 + 0.572);
	// Without the resolution the input rate is unknown: the task still
	// resolves, with the input quantities unpriced and no total.
	let operation = provider.poll_video(&handle).await.unwrap();
	assert_eq!(operation.status, OperationStatus::Succeeded);
	let result = operation.result.unwrap();
	let usage = result.usage.unwrap();
	assert_eq!(usage.estimated_cost, None);
	assert_eq!(usage.line_items.len(), 2);
	assert!(result
		.warnings
		.iter()
		.any(|warning| warning.code == WarningCode::CostUnavailable));
	assert_eq!(mock.requests().len(), 3);
}

#[tokio::test]
#[serial_test::serial]
async fn speech_2_6_models_share_the_speech_endpoint() {
	let (mock, _env) = start_mock(vec![(200, speech_reply()), (200, speech_reply())]);
	for model in ["speech-2.6-hd", "speech-2.6-turbo"] {
		let operation = MinimaxMediaProvider::new()
			.submit_speech(SpeechSynthesisRequest::new("hello", "voice").with_model(model))
			.await
			.unwrap();
		let result = operation.result.unwrap();
		// The list page publishes no pay-as-you-go price for speech-2.6.
		assert_eq!(result.usage.unwrap().estimated_cost, None);
	}
	for request in mock.requests() {
		let body: Value = serde_json::from_str(&request.body).unwrap();
		assert!(body["model"].as_str().unwrap().starts_with("speech-2.6"));
	}
}

#[tokio::test]
#[serial_test::serial]
async fn transcription_uploads_the_recording_and_prices_the_reported_duration() {
	let (mock, _env) = start_mock(vec![
		(200, json!({"text":"hello there","duration":26.325,"trace_id":"trace-1"}).to_string()),
		(
			200,
			json!({"text":"hi","duration":1.5,"n_speakers":1,"trace_id":"trace-2",
				"segments":[{"start":0.5,"end":1.25,"text":"hi","speaker":"S1"}]})
			.to_string(),
		),
		(
			200,
			json!({"text":"hi","duration":1.5,
				"segments":[{"start":0.5,"end":0.75,"text":"h","speaker":"S1"},{"start":0.75,"end":1.25,"text":"i","speaker":"S1"}]})
			.to_string(),
		),
	]);
	let audio = || MediaSource::Bytes {
		data: b"RIFFwave".to_vec(),
		media_type: "audio/wav".to_string(),
	};
	let mut request = TranscriptionRequest::new(audio()).with_model("asr-1.0");
	request.language = Some("en".to_string());
	let result = MinimaxMediaProvider::new()
		.submit_transcription(request)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.text, "hello there");
	assert!(result.segments.is_empty() && result.words.is_empty());
	// $0.38 per hour of reported audio.
	assert_usd(result.usage.unwrap().estimated_cost, 26.325 * 0.38 / 3_600.0);
	let requests = mock.requests();
	assert_eq!(requests[0].path, "/v1/speech_to_text");
	assert_eq!(requests[0].method, "POST");
	assert_eq!(requests[0].header("language"), Some("en"));
	assert!(requests[0].body.contains("name=\"response_format\"\r\n\r\njson"));
	assert!(requests[0].body.contains("filename=\"audio.wav\""));
	assert!(!requests[0].body.contains("timestamp_level"));

	let request = TranscriptionRequest::new(audio()).with_model("asr-1.0");
	let mut request = request;
	request.timestamp_granularities = vec![TimestampGranularity::Segment];
	let result = MinimaxMediaProvider::new()
		.submit_transcription(request)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.segments[0].text, "hi");
	assert_eq!(result.segments[0].speaker.as_deref(), Some("S1"));
	assert!(mock.requests()[1].body.contains("verbose_json"));

	let mut request = TranscriptionRequest::new(audio()).with_model("asr-1.0");
	request.timestamp_granularities = vec![TimestampGranularity::Word];
	let result = MinimaxMediaProvider::new()
		.submit_transcription(request)
		.await
		.unwrap()
		.result
		.unwrap();
	assert_eq!(result.words.len(), 2);
	assert!(result.words[0].word == "h");
	assert!(mock.requests()[2].body.contains("timestamp_level"));

	assert!(matches!(
		MinimaxMediaProvider::new().poll_transcription(&handle()).await,
		Err(MediaError::UnsupportedTask { .. })
	));
}

#[tokio::test]
#[serial_test::serial]
async fn transcription_refuses_other_models_urls_and_malformed_replies() {
	let (mock, _env) = start_mock(vec![
		(200, json!({"text":"hello"}).to_string()),
		(200, json!({"duration":1.5}).to_string()),
	]);
	let provider = MinimaxMediaProvider::new();
	let audio = MediaSource::Bytes {
		data: b"RIFFwave".to_vec(),
		media_type: "audio/wav".to_string(),
	};
	let mut request = TranscriptionRequest::new(audio.clone()).with_model("speech-2.8-hd");
	request.prompt = Some("names".to_string());
	// An unknown model is refused before its options are looked at.
	assert!(matches!(
		provider.submit_transcription(request).await,
		Err(MediaError::UnsupportedTask { .. })
	));
	assert!(matches!(
		provider
			.submit_transcription(
				TranscriptionRequest::new(MediaSource::Url {
					url: "https://example.test/a.wav".to_string(),
					media_type: None,
				})
				.with_model("asr-1.0")
			)
			.await,
		Err(MediaError::InvalidRequest(_))
	));
	let mut request = TranscriptionRequest::new(audio).with_model("asr-1.0");
	request.prompt = Some("names".to_string());
	assert!(matches!(
		provider.submit_transcription(request).await,
		Err(MediaError::UnsupportedParameter { .. })
	));
	assert!(mock.requests().is_empty());

	let request = TranscriptionRequest::new(MediaSource::Bytes {
		data: b"RIFFwave".to_vec(),
		media_type: "audio/wav".to_string(),
	})
	.with_model("asr-1.0");
	// The duration is what MiniMax bills, so a reply without it is malformed.
	assert!(matches!(
		provider.submit_transcription(request).await,
		Err(MediaError::InvalidResponse { .. })
	));
	assert!(matches!(
		provider
			.submit_transcription(
				TranscriptionRequest::new(MediaSource::Bytes {
					data: b"RIFFwave".to_vec(),
					media_type: "audio/wav".to_string()
				})
				.with_model("asr-1.0")
			)
			.await,
		Err(MediaError::InvalidResponse { .. })
	));
}
