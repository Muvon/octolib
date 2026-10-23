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

//! MockHTTP-driven coverage for the ElevenLabs speech and transcription
//! endpoints, complementing the pure-parser tests in `elevenlabs.rs`.
#![cfg(feature = "llm")]

use super::*;

fn start_mock(replies: Vec<(u16, String)>) -> crate::llm::providers::mock_http::MockHttp {
	let mock = crate::llm::providers::mock_http::MockHttp::start(replies);
	std::env::set_var(API_BASE_ENV, mock.base_url());
	std::env::set_var(API_KEY_ENV, "mock-key");
	mock
}

fn finish_mock() {
	std::env::remove_var(API_BASE_ENV);
	std::env::remove_var(API_KEY_ENV);
}

#[tokio::test]
#[serial_test::serial]
async fn submit_speech_returns_inline_audio_bytes() {
	let mock = start_mock(vec![(200, "raw-audio-bytes".to_string())]);

	let provider = ElevenLabsMediaProvider::new();
	let request =
		SpeechSynthesisRequest::new("hello there", "JBFqnCBsd6RMkjVDRZzb").with_model("eleven_multilingual_v2");
	let operation = provider.submit_speech(request).await.expect("speech completed");
	assert_eq!(operation.status, OperationStatus::Succeeded);

	let result = operation.result.expect("result present");
	assert_eq!(result.artifact.size_bytes, Some("raw-audio-bytes".len() as u64));
	assert!(matches!(result.artifact.source, ArtifactSource::Inline(_)));

	let requests = mock.requests();
	assert_eq!(requests[0].method, "POST");
	assert_eq!(
		requests[0].path,
		"/text-to-speech/JBFqnCBsd6RMkjVDRZzb?output_format=mp3_44100_128"
	);
	let body: Value = serde_json::from_str(&requests[0].body).expect("request json");
	assert_eq!(body["text"], "hello there");
	assert_eq!(body["model_id"], "eleven_multilingual_v2");

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_speech_rejects_unsafe_voice_ids() {
	let _mock = start_mock(vec![(200, "raw-audio-bytes".to_string())]);

	let provider = ElevenLabsMediaProvider::new();
	let request = SpeechSynthesisRequest::new("hello there", "../admin").with_model("eleven_multilingual_v2");
	let error = provider.submit_speech(request).await.expect_err("voice rejected");
	assert!(
		matches!(error, MediaError::InvalidRequest(_)),
		"unexpected error: {error}"
	);

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_speech_surfaces_authentication_errors() {
	let _mock = start_mock(vec![(401, r#"{"detail":{"message":"bad key"}}"#.to_string())]);

	let provider = ElevenLabsMediaProvider::new();
	let request =
		SpeechSynthesisRequest::new("hello there", "JBFqnCBsd6RMkjVDRZzb").with_model("eleven_multilingual_v2");
	let error = provider.submit_speech(request).await.expect_err("401 surfaces");
	assert!(
		matches!(error, MediaError::Authentication { .. }),
		"unexpected error: {error}"
	);

	finish_mock();
}

#[tokio::test]
#[serial_test::serial]
async fn submit_transcription_parses_the_scribe_response() {
	let mock = start_mock(vec![(
		200,
		r#"{"language_code":"en","text":"hello world"}"#.to_string(),
	)]);

	let provider = ElevenLabsMediaProvider::new();
	let audio = MediaSource::Bytes {
		data: b"fake-mp3".to_vec(),
		media_type: "audio/mpeg".to_string(),
	};
	let request = TranscriptionRequest::new(audio).with_model("scribe_v1");
	let operation = provider
		.submit_transcription(request)
		.await
		.expect("transcription completed");

	let result = operation.result.expect("result present");
	assert_eq!(result.text, "hello world");
	assert_eq!(result.language.as_deref(), Some("en"));

	let requests = mock.requests();
	assert_eq!(requests[0].path, "/speech-to-text");
	// multipart body carries the model id field
	assert!(requests[0].body.contains("scribe_v1"), "body: {}", requests[0].body);

	finish_mock();
}
