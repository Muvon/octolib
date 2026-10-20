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

fn handle(task: MediaTask) -> JobHandle {
	JobHandle {
		provider: PROVIDER.to_string(),
		model: "kling-video/v3.0/std/text-to-video".to_string(),
		task,
		remote_id: "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff".to_string(),
		cost_estimate: None,
		warnings: Vec::new(),
		expected_outputs: None,
	}
}

#[test]
fn endpoint_ids_validate_and_reserved_routes_are_rejected() {
	assert!(validate_model("kling-video/v3.0/std/text-to-video").is_ok());
	assert!(validate_model("xai/grok-imagine-image-2.0").is_ok());
	assert!(validate_model("soul").is_err());
	assert!(validate_model("higgsfield-ai/../requests").is_err());
	assert!(validate_model("higgsfield-ai/soul?x=1").is_err());
	assert!(validate_model("requests/d7e6c0f3/cancel").is_err());
	assert!(validate_model("files/generate-upload-url").is_err());
	assert!(validate_model("estimate/higgsfield-ai/soul/standard").is_err());
}

#[test]
fn credentials_require_key_id_and_secret() {
	assert!(validate_credentials("key-id:key-secret").is_ok());
	assert!(validate_credentials("key-id").is_err());
	assert!(validate_credentials(":key-secret").is_err());
	assert!(validate_credentials("key-id: ").is_err());
}

#[test]
fn idempotency_keys_follow_the_documented_shape() {
	assert!(validate_idempotency_key("2f1c7d0e-5c1b-4a2e-9a57-5b8c1e1f0a11").is_ok());
	assert!(validate_idempotency_key("").is_err());
	assert!(validate_idempotency_key("has space").is_err());
	assert!(validate_idempotency_key(&"a".repeat(MAX_IDEMPOTENCY_KEY_LEN + 1)).is_err());
}

#[test]
fn parses_all_request_states() {
	for (native, expected) in [
		("queued", OperationStatus::Queued),
		("in_progress", OperationStatus::Running),
		("completed", OperationStatus::Succeeded),
		("failed", OperationStatus::Failed),
		("nsfw", OperationStatus::Failed),
		("canceled", OperationStatus::Cancelled),
	] {
		let value = json!({"status": native});
		assert_eq!(request_state(&value).unwrap().0, expected);
	}
	assert!(request_state(&json!({"status":"mystery"})).is_err());
	assert!(request_state(&json!({})).is_err());
}

#[test]
fn terminal_failures_keep_category_message_and_id() {
	let failed = json!({
		"status":"failed",
		"request_id":"d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff",
		"error":"Generation failed"
	});
	let failure = request_state(&failed).unwrap().1.unwrap();
	assert_eq!(failure.category, FailureCategory::RemoteJob);
	assert_eq!(failure.message, "Generation failed");
	assert_eq!(
		failure.request_id.as_deref(),
		Some("d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff")
	);

	let moderated = request_state(&json!({"status":"nsfw","error":null}))
		.unwrap()
		.1
		.unwrap();
	assert_eq!(moderated.category, FailureCategory::ContentPolicy);
	assert_eq!(moderated.provider_code.as_deref(), Some("nsfw"));

	assert!(request_state(&json!({"status":"completed"})).unwrap().1.is_none());
}

#[test]
fn documented_outputs_become_artifacts_without_download() {
	let images = json!({
		"status":"completed",
		"images":[
			{"url":"https://cdn.example.com/image-1.jpg"},
			{"url":"https://cdn.example.com/image-2","content_type":"image/png"}
		]
	});
	let artifacts = image_artifacts(&images).unwrap();
	assert_eq!(artifacts.len(), 2);
	assert_eq!(artifacts[0].media_type, "image/jpeg");
	assert_eq!(artifacts[1].media_type, "image/png");
	assert!(matches!(artifacts[0].source, ArtifactSource::Url(_)));

	let video = json!({
		"status":"completed",
		"video":{"url":"https://cdn.example.com/video.mp4"}
	});
	let artifacts = video_artifacts(&video).unwrap();
	assert_eq!(artifacts.len(), 1);
	assert_eq!(artifacts[0].kind, MediaKind::Video);
	assert_eq!(artifacts[0].media_type, "video/mp4");

	assert!(image_artifacts(&json!({"status":"completed","images":[]})).is_err());
	assert!(video_artifacts(&json!({"status":"completed"})).is_err());
	assert!(video_artifacts(&json!({"video":{"url":"http://cdn.example.com/v.mp4"}})).is_err());
}

#[test]
fn completed_operation_reports_progress_and_native_status() {
	let value = json!({"status":"completed","request_id":"d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"});
	let operation: Operation<VideoGenerationResult> = operation_from_request(
		&handle(MediaTask::TextToVideo),
		OperationStatus::Succeeded,
		None,
		None,
		&value,
	);
	assert_eq!(operation.progress, Some(1.0));
	assert_eq!(operation.provider_metadata["native_status"], "completed");
}

#[test]
fn estimated_cost_uses_the_frozen_rate_or_warns() {
	let rate = CostEstimate {
		unit: UsageUnit::Images,
		usd_per_unit: 0.03,
		quantity: None,
	};
	let mut warnings = Vec::new();
	let usage = request_usage(Some(&rate), Some(4.0), &mut warnings);
	assert!((usage.estimated_cost.unwrap() - 0.12).abs() < 1e-9);
	assert!(usage.provider_reported_cost.is_none());
	assert!(warnings.is_empty());

	let usage = request_usage(None, None, &mut warnings);
	assert!(usage.estimated_cost.is_none());
	assert_eq!(warnings[0].code, WarningCode::CostUnavailable);
}

#[test]
fn partial_image_batches_warn() {
	let mut warnings = Vec::new();
	warn_if_partial(Some(4), 1, &mut warnings);
	assert_eq!(warnings[0].code, WarningCode::PartialOutput);
	warn_if_partial(Some(1), 1, &mut warnings);
	assert_eq!(warnings.len(), 1);
}

#[test]
fn options_round_trip_and_reject_unknown_keys() {
	let options = HiggsfieldMediaOptions {
		input: serde_json::from_value(json!({"resolution":"1080p"})).unwrap(),
		webhook: Some("https://example.com/webhooks/higgsfield".to_string()),
		..HiggsfieldMediaOptions::default()
	}
	.into_provider_options()
	.unwrap();
	let parsed = parse_options(&options).unwrap();
	assert_eq!(parsed.input["resolution"], "1080p");

	let mut unknown = ProviderOptions::new();
	unknown.insert(PROVIDER.to_string(), json!({"mystery":true}));
	assert!(parse_options(&unknown).is_err());

	let insecure = HiggsfieldMediaOptions {
		webhook: Some("http://example.com/hook".to_string()),
		..HiggsfieldMediaOptions::default()
	};
	assert!(insecure.into_provider_options().is_err());
}

#[test]
fn field_map_overrides_defaults_and_conflicts_fail_closed() {
	let mut options = HiggsfieldMediaOptions {
		field_map: HashMap::from([("last_frame".to_string(), "last_image_url".to_string())]),
		..HiggsfieldMediaOptions::default()
	};
	insert_semantic(
		&mut options,
		"last_frame",
		"end_image_url",
		json!("https://example.com/end.jpg"),
	)
	.unwrap();
	assert!(options.input.contains_key("last_image_url"));
	assert!(!options.input.contains_key("end_image_url"));

	insert_semantic(&mut options, "prompt", "prompt", json!("first")).unwrap();
	assert!(insert_semantic(&mut options, "prompt", "prompt", json!("again")).is_err());
}

#[test]
fn unsupported_parameters_error_or_warn_by_policy() {
	let strict = RequestOptions::default();
	let mut warnings = Vec::new();
	assert!(drop_unsupported(&strict, &mut warnings, &[("dimensions", true)]).is_err());

	let lenient = RequestOptions {
		unsupported_parameter_policy: UnsupportedParameterPolicy::WarnAndDrop,
		..RequestOptions::default()
	};
	drop_unsupported(&lenient, &mut warnings, &[("dimensions", true)]).unwrap();
	assert_eq!(warnings.len(), 1);
	assert_eq!(warnings[0].code, WarningCode::UnsupportedParameterDropped);
}

#[test]
fn duration_must_be_whole_seconds() {
	assert_eq!(whole_seconds(5.0).unwrap(), 5);
	assert!(whole_seconds(5.5).is_err());
	assert!(whole_seconds(0.0).is_err());
	assert!(whole_seconds(f64::NAN).is_err());
}

#[test]
fn upload_headers_must_be_strings() {
	let ticket = json!({"upload_headers":{"Content-Type":"image/jpeg","x-amz-tagging":"retention=temporary"}});
	assert_eq!(upload_headers(&ticket).unwrap().len(), 2);
	assert!(upload_headers(&json!({"upload_headers":{"Content-Type":1}})).is_err());
	assert!(upload_headers(&json!({})).is_err());
}

#[tokio::test]
async fn masks_are_rejected_before_any_request() {
	let provider = HiggsfieldMediaProvider::new();
	let request = ImageGenerationRequest {
		mode: ImageGenerationMode::Inpaint,
		mask: Some(MediaSource::Url {
			url: "https://example.com/mask.png".to_string(),
			media_type: None,
		}),
		..ImageGenerationRequest::new("fill the gap").with_model("higgsfield-ai/soul/standard")
	};
	assert!(matches!(
		provider.submit_image(request).await,
		Err(MediaError::UnsupportedParameter { .. })
	));
}

#[test]
fn serialized_handle_has_no_credentials() {
	let handle = handle(MediaTask::TextToImage);
	let encoded = serde_json::to_string(&handle).unwrap();
	assert!(!encoded.to_ascii_lowercase().contains("secret"));
	assert!(!encoded.to_ascii_lowercase().contains("authorization"));
	assert_eq!(serde_json::from_str::<JobHandle>(&encoded).unwrap(), handle);
}
