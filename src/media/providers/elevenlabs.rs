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

//! ElevenLabs adapter for speech synthesis and transcription. Both endpoints
//! answer synchronously, so results are complete on submission and there is no
//! persistent job to poll or cancel.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::*;
use crate::media::types::*;
use base64::Engine;
use serde_json::{json, Map, Value};

const PROVIDER: &str = "elevenlabs";
const API_KEY_ENV: &str = "ELEVENLABS_API_KEY";
const API_BASE_ENV: &str = "ELEVENLABS_API_URL";
const API_BASE: &str = "https://api.elevenlabs.io/v1";
const AUTH_HEADER: &str = "xi-api-key";

#[derive(Debug, Clone, Default)]
pub struct ElevenLabsMediaProvider;

/// Stable ElevenLabs adapter controls. `voice_settings` carries the native
/// stability, similarity and style knobs; portable `speed` is merged into it.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ElevenLabsMediaOptions {
    #[serde(default)]
    pub voice_settings: Map<String, Value>,
    pub seed: Option<u64>,
    pub previous_text: Option<String>,
    pub next_text: Option<String>,
    pub diarize: Option<bool>,
    pub cost_estimate: Option<CostEstimate>,
}

impl ElevenLabsMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_elevenlabs_options(&self)?;
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
        Ok(options)
    }
}

impl ElevenLabsMediaProvider {
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
                immediate: CapabilitySupport::Supported,
                persistent_jobs: CapabilitySupport::Unsupported,
                polling: CapabilitySupport::Unsupported,
                cancellation: CapabilitySupport::Unsupported,
                progress: CapabilitySupport::Unsupported,
                webhooks: CapabilitySupport::Unsupported,
                binary_streaming: match task {
                    MediaTask::TextToSpeech => CapabilitySupport::Supported,
                    _ => CapabilitySupport::Unsupported,
                },
                resumable: CapabilitySupport::Unsupported,
            },
            parameters: ParameterCapabilities {
                count: CapabilitySupport::Unsupported,
                seed: CapabilitySupport::Supported,
                dimensions: CapabilitySupport::Unsupported,
                aspect_ratio: CapabilitySupport::Unsupported,
                duration: CapabilitySupport::Unsupported,
                mask: CapabilitySupport::Unsupported,
                negative_prompt: CapabilitySupport::Unsupported,
                output_format: CapabilitySupport::Supported,
            },
            limits: MediaLimits {
                supported_formats: vec!["mp3".to_string(), "pcm".to_string(), "ogg".to_string()],
                ..MediaLimits::default()
            },
            provider_options_schema: Some(elevenlabs_options_schema()),
        }
    }

    fn speech_request(
        &self,
        request: &SpeechSynthesisRequest,
        stream: bool,
    ) -> MediaResult<(String, Value, ResolvedAudio, Vec<ProviderWarning>)> {
        validate_model(&request.model)?;
        if request.text.trim().is_empty() {
            return Err(MediaError::InvalidRequest(
                "prompt/text must not be empty".to_string(),
            ));
        }
        let voice = validate_voice(&request.voice)?;
        let mut warnings = request_warnings(&request.request_options);
        let options = parse_options(&request.provider_options)?;
        let audio = resolve_audio(&request.output)?;

        let mut voice_settings = options.voice_settings.clone();
        if let Some(speed) = request.speed {
            if voice_settings.contains_key("speed") {
                return Err(MediaError::InvalidRequest(
                    "speed was supplied both portably and in provider_options.elevenlabs.voice_settings".to_string(),
                ));
            }
            voice_settings.insert("speed".to_string(), json!(speed));
        }
        if request.instructions.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "instructions",
                "ElevenLabs steers delivery through voice_settings and audio tags, not a free-form instruction field",
            )?;
        }

        let mut body = Map::new();
        body.insert("text".to_string(), json!(request.text));
        body.insert("model_id".to_string(), json!(request.model));
        if let Some(language) = request.language.as_ref() {
            body.insert("language_code".to_string(), json!(language));
        }
        if !voice_settings.is_empty() {
            body.insert("voice_settings".to_string(), Value::Object(voice_settings));
        }
        if let Some(seed) = options.seed {
            body.insert("seed".to_string(), json!(seed));
        }
        if let Some(previous) = options.previous_text.as_ref() {
            body.insert("previous_text".to_string(), json!(previous));
        }
        if let Some(next) = options.next_text.as_ref() {
            body.insert("next_text".to_string(), json!(next));
        }

        let suffix = if stream { "/stream" } else { "" };
        let url = format!("{}/text-to-speech/{voice}{suffix}", self.api_base());
        Ok((url, Value::Object(body), audio, warnings))
    }
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for ElevenLabsMediaProvider {
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
        let (url, body, audio, mut warnings) = self.speech_request(&request, false)?;
        let options = parse_options(&request.provider_options)?;
        let key = self.key()?;
        let format = audio.api_format.clone();
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .header(AUTH_HEADER, &key)
                .query(&[("output_format", format.as_str())])
                .json(&body)
        })
        .await?;
        shared::require_success(PROVIDER, &response)?;
        let media_type = response
            .headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or(audio.media_type)
            .to_string();
        let characters = request.text.chars().count() as f64;
        let estimate = shared::resolved_cost_estimate(
            PROVIDER,
            &request.model,
            options.cost_estimate.clone(),
            Some((UsageUnit::Characters, characters)),
        );
        let usage = character_usage(characters, estimate.as_ref(), &mut warnings);
        let bytes = response.body;
        Ok(Operation::completed(SpeechSynthesisResult {
            artifact: MediaArtifact {
                kind: MediaKind::Audio,
                media_type,
                size_bytes: Some(bytes.len() as u64),
                source: ArtifactSource::Inline(bytes),
                dimensions: None,
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: Some(audio.sample_rate_hz),
                channels: Some(1),
                expires_at: None,
                metadata: json!({"provider": PROVIDER, "output_format": audio.api_format}),
            },
            usage: Some(usage),
            warnings,
            provider_metadata: json!({"output_format": audio.api_format}),
        }))
    }

    async fn poll_speech(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        Err(immediate_only(handle))
    }

    async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(immediate_only(handle))
    }

    async fn stream_speech(&self, request: SpeechSynthesisRequest) -> MediaResult<SpeechStream> {
        let (url, body, audio, _) = self.speech_request(&request, true)?;
        let key = self.key()?;
        let format = audio.api_format.clone();
        let response = shared::send_stream(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .header(AUTH_HEADER, &key)
                .query(&[("output_format", format.as_str())])
                .json(&body)
        })
        .await?;
        let media_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split(';').next())
            .unwrap_or(audio.media_type)
            .to_string();
        Ok(SpeechStream::new(media_type, None, response))
    }
}

#[async_trait::async_trait]
impl TranscriptionProvider for ElevenLabsMediaProvider {
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
        let options = parse_options(&request.provider_options)?;
        let mut warnings = request_warnings(&request.request_options);
        if request.prompt.is_some() {
            unsupported(
                &request.request_options,
                &mut warnings,
                "prompt",
                "ElevenLabs transcription has no prompt or biasing field",
            )?;
        }
        if request
            .timestamp_granularities
            .contains(&TimestampGranularity::Segment)
        {
            unsupported(
                &request.request_options,
                &mut warnings,
                "timestamp_granularities",
                "ElevenLabs reports word and character timestamps, never segments",
            )?;
        }

        // Reuse the shared trust-boundary checks (size limit, media type, no
        // implicit URL download) and recover the raw bytes for the file part.
        let (encoded, media_type) =
            shared::source_to_base64(&request.audio, request.request_options.max_source_bytes)?;
        let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;

        let mut fields: Vec<(&str, String)> = vec![("model_id", request.model.clone())];
        if let Some(language) = request.language.as_ref() {
            fields.push(("language_code", language.clone()));
        }
        if request
            .timestamp_granularities
            .contains(&TimestampGranularity::Word)
        {
            fields.push(("timestamps_granularity", "word".to_string()));
        }
        if let Some(diarize) = options.diarize {
            fields.push(("diarize", diarize.to_string()));
        }
        let form = multipart_form(&fields, &media_type, bytes)?;

        let key = self.key()?;
        let url = format!("{}/speech-to-text", self.api_base());
        let response = shared::send(PROVIDER, &request.request_options, || {
            crate::llm::providers::shared::http_client()
                .post(&url)
                .header(AUTH_HEADER, &key)
                .header(reqwest::header::CONTENT_TYPE, &form.content_type)
                .body(form.body.clone())
        })
        .await?;
        let value = shared::parse_json(PROVIDER, &response)?;
        // Scribe bills the input audio's duration, which ElevenLabs only
        // reports after the upload is processed, so no quantity is known here.
        let estimate =
            shared::resolved_cost_estimate(PROVIDER, &request.model, options.cost_estimate, None);
        Ok(Operation::completed(parse_transcription(
            &value,
            estimate.as_ref(),
            warnings,
        )?))
    }

    async fn poll_transcription(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        Err(immediate_only(handle))
    }

    async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(immediate_only(handle))
    }
}

/// The ElevenLabs `output_format` token and the audio properties it implies.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedAudio {
    api_format: String,
    media_type: &'static str,
    sample_rate_hz: u32,
}

fn resolve_audio(output: &AudioOutputSpec) -> MediaResult<ResolvedAudio> {
    if output.channels.is_some_and(|channels| channels != 1) {
        return Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: "output.channels".to_string(),
            reason: "ElevenLabs renders mono audio only".to_string(),
        });
    }
    let unsupported_rate = |rate: u32| MediaError::UnsupportedParameter {
        provider: PROVIDER.to_string(),
        parameter: "output.sample_rate_hz".to_string(),
        reason: format!("{rate} Hz is not offered for this container"),
    };
    match output.format {
        AudioFormat::Mp3 => {
            let (api_format, sample_rate_hz) = match output.sample_rate_hz {
                None | Some(44100) => ("mp3_44100_128", 44100),
                Some(22050) => ("mp3_22050_32", 22050),
                Some(24000) => ("mp3_24000_48", 24000),
                Some(rate) => return Err(unsupported_rate(rate)),
            };
            Ok(ResolvedAudio {
                api_format: api_format.to_string(),
                media_type: "audio/mpeg",
                sample_rate_hz,
            })
        }
        AudioFormat::Pcm => {
            let rate = output.sample_rate_hz.ok_or_else(|| MediaError::InvalidRequest(
                "PCM output requires an explicit output.sample_rate_hz because raw samples carry no header".to_string(),
            ))?;
            if !matches!(rate, 8000 | 16000 | 22050 | 24000 | 32000 | 44100 | 48000) {
                return Err(unsupported_rate(rate));
            }
            Ok(ResolvedAudio {
                api_format: format!("pcm_{rate}"),
                media_type: "audio/pcm",
                sample_rate_hz: rate,
            })
        }
        AudioFormat::Ogg => {
            if output.sample_rate_hz.is_some_and(|rate| rate != 48000) {
                return Err(unsupported_rate(output.sample_rate_hz.unwrap_or_default()));
            }
            Ok(ResolvedAudio {
                api_format: "opus_48000_128".to_string(),
                media_type: "audio/ogg",
                sample_rate_hz: 48000,
            })
        }
        other => Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: "output.format".to_string(),
            reason: format!(
                "ElevenLabs returns mp3, pcm, or opus-in-ogg; {} is not offered",
                other.as_str()
            ),
        }),
    }
}

struct MultipartForm {
    content_type: String,
    body: Vec<u8>,
}

fn multipart_form(
    fields: &[(&str, String)],
    media_type: &str,
    file: Vec<u8>,
) -> MediaResult<MultipartForm> {
    for (name, value) in fields {
        if value.contains(['\r', '\n', '"']) {
            return Err(MediaError::InvalidRequest(format!(
                "multipart field '{name}' contains unsupported characters"
            )));
        }
    }
    let boundary = format!("octolib{}", uuid::Uuid::new_v4().simple());
    let mut body = Vec::with_capacity(file.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{}\"\r\n",
            file_name(media_type)
        )
        .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {media_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(&file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    Ok(MultipartForm {
        content_type: format!("multipart/form-data; boundary={boundary}"),
        body,
    })
}

fn file_name(media_type: &str) -> &'static str {
    match media_type {
        "audio/mpeg" => "audio.mp3",
        "audio/wav" | "audio/x-wav" => "audio.wav",
        "audio/flac" => "audio.flac",
        "audio/ogg" => "audio.ogg",
        "audio/aac" => "audio.aac",
        "audio/mp4" | "audio/m4a" => "audio.m4a",
        "audio/webm" => "audio.webm",
        "video/mp4" => "video.mp4",
        _ => "audio.bin",
    }
}

fn parse_transcription(
    value: &Value,
    estimate: Option<&CostEstimate>,
    mut warnings: Vec<ProviderWarning>,
) -> MediaResult<TranscriptionResult> {
    let text =
        value
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| MediaError::InvalidResponse {
                provider: PROVIDER.to_string(),
                message: "transcription response is missing text".to_string(),
            })?;
    let words = value
        .get("words")
        .and_then(Value::as_array)
        .map(|words| {
            words
                .iter()
                .filter(|word| {
                    word.get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| kind == "word")
                })
                .filter_map(|word| {
                    Some(TranscriptWord {
                        start_secs: word.get("start")?.as_f64()?,
                        end_secs: word.get("end")?.as_f64()?,
                        word: word.get("text")?.as_str()?.to_string(),
                        speaker: word
                            .get("speaker_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    if estimate.is_none() {
        warnings.push(cost_unavailable());
    }
    Ok(TranscriptionResult {
        text: text.to_string(),
        language: value
            .get("language_code")
            .and_then(Value::as_str)
            .map(str::to_string),
        duration_secs: None,
        segments: Vec::new(),
        words,
        usage: Some(MediaUsage {
            line_items: Vec::new(),
            provider_reported_cost: None,
            estimated_cost: estimate
                .and_then(|rate| rate.quantity.map(|quantity| quantity * rate.usd_per_unit)),
            currency: "USD".to_string(),
            metadata: Value::Null,
        }),
        warnings,
        provider_metadata: json!({
            "language_probability": value.get("language_probability"),
            "language_code": value.get("language_code")
        }),
    })
}

fn character_usage(
    characters: f64,
    estimate: Option<&CostEstimate>,
    warnings: &mut Vec<ProviderWarning>,
) -> MediaUsage {
    let estimated_cost = estimate.and_then(|rate| {
        let quantity = rate.quantity.or(match rate.unit {
            UsageUnit::Characters => Some(characters),
            _ => None,
        });
        quantity.map(|quantity| quantity * rate.usd_per_unit)
    });
    if estimated_cost.is_none() {
        warnings.push(cost_unavailable());
    }
    MediaUsage {
        line_items: vec![UsageLineItem {
            unit: UsageUnit::Characters,
            quantity: characters,
            cost: None,
            description: Some("speech input characters".to_string()),
        }],
        provider_reported_cost: None,
        estimated_cost,
        currency: "USD".to_string(),
        metadata: Value::Null,
    }
}

fn cost_unavailable() -> ProviderWarning {
    ProviderWarning {
        code: WarningCode::CostUnavailable,
        message: "ElevenLabs bills against a subscription character quota and reports no per-request dollar amount; supply a verified cost_estimate rate to price it locally".to_string(),
        parameter: None,
        provider_metadata: Value::Null,
    }
}

fn parse_options(options: &ProviderOptions) -> MediaResult<ElevenLabsMediaOptions> {
    let raw = shared::provider_options(options, PROVIDER)?;
    let parsed: ElevenLabsMediaOptions =
        serde_json::from_value(Value::Object(raw)).map_err(|error| {
            MediaError::InvalidRequest(format!("invalid ElevenLabs provider options: {error}"))
        })?;
    validate_elevenlabs_options(&parsed)?;
    Ok(parsed)
}

fn validate_elevenlabs_options(options: &ElevenLabsMediaOptions) -> MediaResult<()> {
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

fn unsupported(
    options: &RequestOptions,
    warnings: &mut Vec<ProviderWarning>,
    parameter: &str,
    reason: &str,
) -> MediaResult<()> {
    if options.unsupported_parameter_policy == UnsupportedParameterPolicy::Error {
        return Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: parameter.to_string(),
            reason: reason.to_string(),
        });
    }
    warnings.push(ProviderWarning {
        code: WarningCode::UnsupportedParameterDropped,
        message: reason.to_string(),
        parameter: Some(parameter.to_string()),
        provider_metadata: Value::Null,
    });
    Ok(())
}

fn request_warnings(options: &RequestOptions) -> Vec<ProviderWarning> {
    if options.idempotency_key.is_some()
        && options.unsupported_parameter_policy == UnsupportedParameterPolicy::WarnAndDrop
    {
        vec![ProviderWarning {
            code: WarningCode::UnsupportedParameterDropped,
            message: "ElevenLabs does not document request idempotency keys".to_string(),
            parameter: Some("request_options.idempotency_key".to_string()),
            provider_metadata: Value::Null,
        }]
    } else {
        Vec::new()
    }
}

fn immediate_only(handle: &JobHandle) -> MediaError {
    MediaError::UnsupportedTask {
        provider: PROVIDER.to_string(),
        model: handle.model.clone(),
        task: handle.task,
    }
}

fn elevenlabs_options_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":{
            "voice_settings":{"type":"object"},
            "seed":{"type":"integer"},
            "previous_text":{"type":"string"},
            "next_text":{"type":"string"},
            "diarize":{"type":"boolean"},
            "cost_estimate":{"type":"object"}
        }
    })
}

fn validate_model(model: &str) -> MediaResult<()> {
    let model = model.trim();
    if model.is_empty() || model.len() > 128 || model.contains(['\r', '\n']) {
        return Err(MediaError::InvalidRequest(
            "ElevenLabs model must be a model id such as eleven_multilingual_v2".to_string(),
        ));
    }
    Ok(())
}

/// The voice id is a URL path segment, so it must not carry traversal or
/// query characters.
fn validate_voice(voice: &str) -> MediaResult<&str> {
    let voice = voice.trim();
    if voice.is_empty()
        || voice.len() > 128
        || !voice
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(MediaError::InvalidRequest(
            "ElevenLabs voice must be a voice id containing only letters, digits, '-' or '_'"
                .to_string(),
        ));
    }
    Ok(voice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_every_offered_container_and_rejects_the_rest() {
        let mp3 = resolve_audio(&AudioOutputSpec::default()).unwrap();
        assert_eq!(mp3.api_format, "mp3_44100_128");
        assert_eq!(mp3.media_type, "audio/mpeg");
        assert_eq!(mp3.sample_rate_hz, 44100);

        let mp3_low = resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Mp3,
            sample_rate_hz: Some(22050),
            channels: None,
        })
        .unwrap();
        assert_eq!(mp3_low.api_format, "mp3_22050_32");

        let pcm = resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Pcm,
            sample_rate_hz: Some(16000),
            channels: Some(1),
        })
        .unwrap();
        assert_eq!(pcm.api_format, "pcm_16000");

        let ogg = resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Ogg,
            sample_rate_hz: None,
            channels: None,
        })
        .unwrap();
        assert_eq!(ogg.api_format, "opus_48000_128");

        // PCM carries no header, so the rate cannot be guessed.
        assert!(resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Pcm,
            sample_rate_hz: None,
            channels: None,
        })
        .is_err());
        assert!(resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Mp3,
            sample_rate_hz: Some(48000),
            channels: None,
        })
        .is_err());
        assert!(resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Flac,
            sample_rate_hz: None,
            channels: None,
        })
        .is_err());
        assert!(resolve_audio(&AudioOutputSpec {
            format: AudioFormat::Mp3,
            sample_rate_hz: None,
            channels: Some(2),
        })
        .is_err());
    }

    #[test]
    fn voice_ids_are_validated_as_path_segments() {
        assert_eq!(
            validate_voice(" 21m00Tcm4TlvDq8ikWAM ").unwrap(),
            "21m00Tcm4TlvDq8ikWAM"
        );
        assert!(validate_voice("../../v1/history").is_err());
        assert!(validate_voice("voice?query=1").is_err());
        assert!(validate_voice("").is_err());
    }

    #[test]
    fn multipart_body_is_well_formed_and_rejects_header_injection() {
        let form = multipart_form(
            &[("model_id", "scribe_v1".to_string())],
            "audio/mpeg",
            b"ID3audio".to_vec(),
        )
        .unwrap();
        let boundary = form
            .content_type
            .strip_prefix("multipart/form-data; boundary=")
            .unwrap()
            .to_string();
        let body = String::from_utf8_lossy(&form.body);
        assert!(body.starts_with(&format!("--{boundary}\r\n")));
        assert!(body.contains("Content-Disposition: form-data; name=\"model_id\"\r\n\r\nscribe_v1"));
        assert!(body.contains("filename=\"audio.mp3\""));
        assert!(body.contains("Content-Type: audio/mpeg"));
        assert!(body.contains("ID3audio"));
        assert!(body.ends_with(&format!("--{boundary}--\r\n")));

        assert!(multipart_form(
            &[("model_id", "a\r\nX-Injected: 1".to_string())],
            "audio/mpeg",
            Vec::new()
        )
        .is_err());
    }

    #[test]
    fn transcription_keeps_words_and_drops_spacing_tokens() {
        let value = json!({
            "language_code":"eng",
            "language_probability":0.99,
            "text":"hello world",
            "words":[
                {"text":"hello","start":0.0,"end":0.5,"type":"word","speaker_id":"speaker_0"},
                {"text":" ","start":0.5,"end":0.5,"type":"spacing"},
                {"text":"world","start":0.5,"end":1.0,"type":"word","speaker_id":"speaker_1"}
            ]
        });
        let result = parse_transcription(&value, None, Vec::new()).unwrap();
        assert_eq!(result.text, "hello world");
        assert_eq!(result.language.as_deref(), Some("eng"));
        assert_eq!(result.words.len(), 2);
        assert_eq!(result.words[1].word, "world");
        assert_eq!(result.words[0].speaker.as_deref(), Some("speaker_0"));
        assert!(result.segments.is_empty());
        assert!(result
            .warnings
            .iter()
            .any(|warning| warning.code == WarningCode::CostUnavailable));

        assert!(parse_transcription(&json!({"words":[]}), None, Vec::new()).is_err());
    }

    #[test]
    fn character_usage_prices_only_with_a_supplied_rate() {
        let mut warnings = Vec::new();
        let usage = character_usage(
            100.0,
            Some(&CostEstimate {
                unit: UsageUnit::Characters,
                usd_per_unit: 0.0001,
                quantity: None,
            }),
            &mut warnings,
        );
        assert_eq!(usage.provider_reported_cost, None);
        assert_eq!(usage.estimated_cost, Some(0.01));
        assert!(warnings.is_empty());

        let usage = character_usage(100.0, None, &mut warnings);
        assert_eq!(usage.estimated_cost, None);
        assert_eq!(usage.line_items[0].quantity, 100.0);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn typed_options_round_trip_and_reject_unknown_keys() {
        let options = ElevenLabsMediaOptions {
            voice_settings: serde_json::from_value(json!({"stability":0.4})).unwrap(),
            diarize: Some(true),
            ..ElevenLabsMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        let parsed = parse_options(&options).unwrap();
        assert_eq!(parsed.voice_settings.get("stability"), Some(&json!(0.4)));
        assert_eq!(parsed.diarize, Some(true));

        let mut unknown = ProviderOptions::new();
        unknown.insert(PROVIDER.to_string(), json!({"mystery":true}));
        assert!(parse_options(&unknown).is_err());
    }

    #[tokio::test]
    async fn speech_body_merges_speed_and_fails_on_conflict() {
        let provider = ElevenLabsMediaProvider::new();
        let mut request = SpeechSynthesisRequest::new("hello", "21m00Tcm4TlvDq8ikWAM")
            .with_model("eleven_multilingual_v2");
        request.speed = Some(1.1);
        request.language = Some("en".to_string());
        let (url, body, audio, _) = provider.speech_request(&request, false).unwrap();
        assert!(url.ends_with("/text-to-speech/21m00Tcm4TlvDq8ikWAM"));
        // `speed` is an f32 on the request, so compare with tolerance.
        let speed = body
            .pointer("/voice_settings/speed")
            .and_then(Value::as_f64)
            .unwrap();
        assert!((speed - 1.1).abs() < 1e-6);
        assert_eq!(
            body.pointer("/model_id"),
            Some(&json!("eleven_multilingual_v2"))
        );
        assert_eq!(body.pointer("/language_code"), Some(&json!("en")));
        assert_eq!(audio.api_format, "mp3_44100_128");

        let (stream_url, _, _, _) = provider.speech_request(&request, true).unwrap();
        assert!(stream_url.ends_with("/stream"));

        request.provider_options = ElevenLabsMediaOptions {
            voice_settings: serde_json::from_value(json!({"speed":0.9})).unwrap(),
            ..ElevenLabsMediaOptions::default()
        }
        .into_provider_options()
        .unwrap();
        assert!(provider.speech_request(&request, false).is_err());
    }

    #[tokio::test]
    async fn immediate_provider_refuses_polling_and_cancellation() {
        let provider = ElevenLabsMediaProvider::new();
        let handle = JobHandle {
            provider: PROVIDER.to_string(),
            model: "eleven_multilingual_v2".to_string(),
            task: MediaTask::TextToSpeech,
            remote_id: "none".to_string(),
            cost_estimate: None,
            warnings: Vec::new(),
            expected_outputs: None,
        };
        assert!(matches!(
            provider.poll_speech(&handle).await,
            Err(MediaError::UnsupportedTask { .. })
        ));
        assert!(matches!(
            provider.cancel_speech(&handle).await,
            Err(MediaError::UnsupportedTask { .. })
        ));
        assert!(matches!(
            provider.poll_transcription(&handle).await,
            Err(MediaError::UnsupportedTask { .. })
        ));
    }
}
