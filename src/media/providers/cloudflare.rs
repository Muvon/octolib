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

//! Cloudflare Workers AI media provider.
//!
//! Every call is a synchronous `POST /accounts/{account_id}/ai/run/{model}`.
//! JSON inputs return `{"result": ...}`; binary inputs (Whisper, Nova-3) go as
//! the raw request body under their media type; binary outputs (Aura audio,
//! Phoenix images) come back as raw bytes. Nothing is queued, so a submit is
//! complete on return and there is no job to poll or cancel.
//!
//! Configuration:
//! - `CLOUDFLARE_API_KEY`: API token with Workers AI permissions
//! - `CLOUDFLARE_ACCOUNT_ID`: account the models run under
//! - `CLOUDFLARE_MEDIA_API_URL`: optional override of the `.../ai/run` base
//!
//! Models, schemas and rates verified Sep 18, 2026 against the model pages.
//! FLUX.2 (dev/klein) is deliberately absent: its input is an undocumented
//! multipart body.

use super::shared;
use crate::media::errors::{MediaError, MediaResult};
use crate::media::traits::{
    ImageGenerationProvider, SpeechSynthesisProvider, TranscriptionProvider,
};
use crate::media::types::*;
use base64::Engine;
use serde_json::{json, Map, Value};

const PROVIDER: &str = "cloudflare";
const API_KEY_ENV: &str = "CLOUDFLARE_API_KEY";
const ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
const API_BASE_ENV: &str = "CLOUDFLARE_MEDIA_API_URL";
const API_BASE: &str = "https://api.cloudflare.com/client/v4/accounts";
const TILE_PIXELS: f64 = 512.0 * 512.0;
/// Workers AI bills $0.011 per 1,000 neurons; some JSON results report the
/// neurons consumed, which is the authoritative charge for that request.
const NEURON_USD: f64 = 0.011 / 1000.0;

#[derive(Debug, Clone, Default)]
pub struct CloudflareMediaProvider;

/// Stable Cloudflare adapter controls. Image knobs map onto the model's own
/// `num_steps`/`steps` and `guidance` fields; the Nova-3 flags go out as query
/// parameters on the binary upload.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloudflareMediaOptions {
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub detect_language: Option<bool>,
    pub diarize: Option<bool>,
    pub punctuate: Option<bool>,
    pub smart_format: Option<bool>,
    pub cost_estimate: Option<CostEstimate>,
}

impl CloudflareMediaOptions {
    pub fn into_provider_options(self) -> MediaResult<ProviderOptions> {
        validate_options(&self)?;
        let mut options = ProviderOptions::new();
        options.insert(PROVIDER.to_string(), serde_json::to_value(self)?);
        Ok(options)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImageOutput {
    /// `{"result": {"image": "<base64>"}}`
    Base64Json,
    /// Raw image bytes.
    Binary,
}

/// Text-to-image models. Cloudflare bills a 512x512 tile rate on the output
/// size plus a flat per-step rate, so the estimate needs both; a model with no
/// documented default for either stays unpriced until the caller supplies it.
struct ImageModel {
    id: &'static str,
    default_size: Option<(u32, u32)>,
    default_steps: Option<u32>,
    usd_per_tile: f64,
    usd_per_step: f64,
    output: ImageOutput,
    steps_field: &'static str,
    seed: bool,
    dimensions: bool,
    negative_prompt: bool,
}

const IMAGE_MODELS: &[ImageModel] = &[
    // Output size is not documented, so only the per-step part is known. The
    // schema is closed (`additionalProperties: false`): a `seed` is rejected
    // live even though the docs' curl example sends one.
    ImageModel {
        id: "@cf/black-forest-labs/flux-1-schnell",
        default_size: None,
        default_steps: Some(4),
        usd_per_tile: 0.000_052_8,
        usd_per_step: 0.000_105_6,
        output: ImageOutput::Base64Json,
        steps_field: "steps",
        seed: false,
        dimensions: false,
        negative_prompt: false,
    },
    ImageModel {
        id: "@cf/leonardo/lucid-origin",
        default_size: Some((1120, 1120)),
        default_steps: None,
        usd_per_tile: 0.007,
        usd_per_step: 0.000_132,
        output: ImageOutput::Base64Json,
        steps_field: "num_steps",
        seed: true,
        dimensions: true,
        negative_prompt: false,
    },
    ImageModel {
        id: "@cf/leonardo/phoenix-1.0",
        default_size: Some((1024, 1024)),
        default_steps: Some(25),
        usd_per_tile: 0.005_83,
        usd_per_step: 0.000_11,
        output: ImageOutput::Binary,
        steps_field: "num_steps",
        seed: true,
        dimensions: true,
        negative_prompt: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpeechApi {
    /// Deepgram Aura: `{text, speaker, encoding, container, sample_rate}` → raw audio.
    Aura,
    /// MeloTTS: `{prompt, lang}` → `{"result": {"audio": "<base64 mp3>"}}` or raw MP3.
    Melo,
}

const SPEECH_MODELS: &[(&str, SpeechApi)] = &[
    ("@cf/deepgram/aura-1", SpeechApi::Aura),
    ("@cf/deepgram/aura-2-en", SpeechApi::Aura),
    ("@cf/deepgram/aura-2-es", SpeechApi::Aura),
    ("@cf/myshell-ai/melotts", SpeechApi::Melo),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranscriptionApi {
    /// Raw audio body → `{text, words[{word,start,end}], vtt}`; no duration reported.
    Whisper,
    /// `{audio: <base64>, language?, initial_prompt?}` → `{text, transcription_info, segments}`.
    WhisperTurbo,
    /// Raw audio body plus query flags → Deepgram `{results.channels[].alternatives[]}`.
    Nova3,
}

const TRANSCRIPTION_MODELS: &[(&str, TranscriptionApi)] = &[
    (
        "@cf/openai/whisper-large-v3-turbo",
        TranscriptionApi::WhisperTurbo,
    ),
    ("@cf/openai/whisper", TranscriptionApi::Whisper),
    ("@cf/deepgram/nova-3", TranscriptionApi::Nova3),
];

fn image_model(model: &str) -> Option<&'static ImageModel> {
    IMAGE_MODELS
        .iter()
        .find(|m| m.id.eq_ignore_ascii_case(model.trim()))
}

fn speech_api(model: &str) -> Option<SpeechApi> {
    SPEECH_MODELS
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(model.trim()))
        .map(|(_, api)| *api)
}

fn transcription_api(model: &str) -> Option<TranscriptionApi> {
    TRANSCRIPTION_MODELS
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(model.trim()))
        .map(|(_, api)| *api)
}

impl CloudflareMediaProvider {
    pub fn new() -> Self {
        Self
    }

    fn key(&self) -> MediaResult<String> {
        shared::api_key(API_KEY_ENV)
    }

    fn run_url(&self, model: &str) -> MediaResult<String> {
        let base = match std::env::var(API_BASE_ENV) {
            Ok(value) => value.trim_end_matches('/').to_string(),
            Err(_) => {
                let account = shared::api_key(ACCOUNT_ID_ENV)?;
                if !account
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                {
                    return Err(MediaError::InvalidRequest(
                        "CLOUDFLARE_ACCOUNT_ID must be an account identifier".to_string(),
                    ));
                }
                format!("{API_BASE}/{account}/ai/run")
            }
        };
        Ok(format!("{base}/{}", model.trim()))
    }

    fn descriptor(&self, model: &str, task: MediaTask) -> MediaModelDescriptor {
        let (inputs, outputs) = match task {
            MediaTask::TextToImage => (vec![Modality::Text], vec![Modality::Image]),
            MediaTask::TextToSpeech => (vec![Modality::Text], vec![Modality::Audio]),
            MediaTask::SpeechToText => (vec![Modality::Audio], vec![Modality::Text]),
            _ => (Vec::new(), Vec::new()),
        };
        let image = image_model(model);
        let supported = |flag: bool| {
            if flag {
                CapabilitySupport::Supported
            } else {
                CapabilitySupport::Unsupported
            }
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
                binary_streaming: CapabilitySupport::Unsupported,
                resumable: CapabilitySupport::Unsupported,
            },
            parameters: ParameterCapabilities {
                count: CapabilitySupport::Unsupported,
                seed: supported(image.is_some_and(|m| m.seed)),
                dimensions: supported(image.is_some_and(|m| m.dimensions)),
                aspect_ratio: CapabilitySupport::Unsupported,
                duration: CapabilitySupport::Unsupported,
                mask: CapabilitySupport::Unsupported,
                negative_prompt: supported(image.is_some_and(|m| m.negative_prompt)),
                output_format: supported(matches!(speech_api(model), Some(SpeechApi::Aura))),
            },
            limits: MediaLimits {
                supported_formats: match speech_api(model) {
                    Some(SpeechApi::Aura) => ["mp3", "wav", "flac", "ogg", "aac", "pcm"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    Some(SpeechApi::Melo) => vec!["mp3".to_string()],
                    None => Vec::new(),
                },
                ..MediaLimits::default()
            },
            provider_options_schema: Some(options_schema()),
        }
    }

    async fn post_json(
        &self,
        model: &str,
        body: &Value,
        request_options: &RequestOptions,
    ) -> MediaResult<shared::CapturedResponse> {
        let url = self.run_url(model)?;
        let key = self.key()?;
        let response = shared::send(PROVIDER, request_options, || {
            crate::http::http_client()
                .post(&url)
                .bearer_auth(&key)
                .json(body)
        })
        .await?;
        shared::require_success(PROVIDER, &response)?;
        Ok(response)
    }

    async fn post_binary(
        &self,
        model: &str,
        query: &[(&str, String)],
        media_type: &str,
        bytes: Vec<u8>,
        request_options: &RequestOptions,
    ) -> MediaResult<shared::CapturedResponse> {
        let url = self.run_url(model)?;
        let key = self.key()?;
        let response = shared::send(PROVIDER, request_options, || {
            crate::http::http_client()
                .post(&url)
                .bearer_auth(&key)
                .query(query)
                .header(reqwest::header::CONTENT_TYPE, media_type)
                .body(bytes.clone())
        })
        .await?;
        shared::require_success(PROVIDER, &response)?;
        Ok(response)
    }
}

#[async_trait::async_trait]
impl ImageGenerationProvider for CloudflareMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        image_model(model).is_some()
    }
    fn capabilities(&self, model: &str) -> ImageCapabilities {
        self.descriptor(model, MediaTask::TextToImage)
    }

    async fn submit_image(
        &self,
        request: ImageGenerationRequest,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        let model = image_model(&request.model).ok_or_else(|| MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: request.model.clone(),
            task: MediaTask::TextToImage,
        })?;
        let mut plan = plan_image(model, &request)?;
        let response = self
            .post_json(model.id, &plan.body, &request.request_options)
            .await?;
        let bytes = match model.output {
            ImageOutput::Binary => response.body,
            ImageOutput::Base64Json => {
                let result = unwrap_result(&response)?;
                let encoded = result.get("image").and_then(Value::as_str).ok_or_else(|| {
                    MediaError::InvalidResponse {
                        provider: PROVIDER.to_string(),
                        message: "image response is missing result.image".to_string(),
                    }
                })?;
                base64::engine::general_purpose::STANDARD.decode(encoded)?
            }
        };
        let media_type = sniff_image_type(&bytes)
            .or_else(|| content_type(&response.headers))
            .unwrap_or_else(|| "image/png".to_string());
        let mut warnings = std::mem::take(&mut plan.warnings);
        let usage = image_usage(&plan, &mut warnings);
        Ok(Operation::completed(ImageGenerationResult {
            artifacts: vec![MediaArtifact {
                kind: MediaKind::Image,
                media_type,
                size_bytes: Some(bytes.len() as u64),
                source: ArtifactSource::Inline(bytes),
                dimensions: plan
                    .size
                    .map(|(width, height)| Dimensions { width, height }),
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: None,
                channels: None,
                expires_at: None,
                metadata: json!({"provider": PROVIDER}),
            }],
            usage: Some(usage),
            warnings,
            safety: SafetyReport::default(),
            provider_metadata: json!({"steps": plan.steps}),
        }))
    }

    async fn poll_image(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<ImageGenerationResult>> {
        Err(immediate_only(handle))
    }

    async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()> {
        Err(immediate_only(handle))
    }
}

#[async_trait::async_trait]
impl SpeechSynthesisProvider for CloudflareMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        speech_api(model).is_some()
    }
    fn capabilities(&self, model: &str) -> SpeechCapabilities {
        self.descriptor(model, MediaTask::TextToSpeech)
    }

    async fn submit_speech(
        &self,
        request: SpeechSynthesisRequest,
    ) -> MediaResult<Operation<SpeechSynthesisResult>> {
        let api = speech_api(&request.model).ok_or_else(|| MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: request.model.clone(),
            task: MediaTask::TextToSpeech,
        })?;
        let plan = plan_speech(api, &request)?;
        let options = parse_options(&request.provider_options)?;
        let response = self
            .post_json(request.model.trim(), &plan.body, &request.request_options)
            .await?;
        let header_type = content_type(&response.headers);
        let bytes = if header_type
            .as_deref()
            .is_some_and(|value| value.starts_with("application/json"))
        {
            // MeloTTS answers with base64 JSON over REST.
            let result = unwrap_result(&response)?;
            let encoded = result.get("audio").and_then(Value::as_str).ok_or_else(|| {
                MediaError::InvalidResponse {
                    provider: PROVIDER.to_string(),
                    message: "speech response is missing result.audio".to_string(),
                }
            })?;
            base64::engine::general_purpose::STANDARD.decode(encoded)?
        } else {
            response.body
        };
        // The requested encoding decides the container; Cloudflare's header
        // says audio/mpeg even when the bytes are a RIFF/WAV (verified live).
        let media_type = plan.media_type.to_string();
        let characters = request.text.chars().count() as f64;
        let estimate = shared::resolved_cost_estimate(
            PROVIDER,
            &request.model,
            options.cost_estimate,
            Some((UsageUnit::Characters, characters)),
        );
        let mut warnings = plan.warnings;
        let usage = character_usage(characters, estimate.as_ref(), &mut warnings);
        Ok(Operation::completed(SpeechSynthesisResult {
            artifact: MediaArtifact {
                kind: MediaKind::Audio,
                media_type,
                size_bytes: Some(bytes.len() as u64),
                source: ArtifactSource::Inline(bytes),
                dimensions: None,
                duration_secs: None,
                frame_rate: None,
                sample_rate_hz: plan.sample_rate_hz,
                channels: Some(1),
                expires_at: None,
                metadata: json!({"provider": PROVIDER}),
            },
            usage: Some(usage),
            warnings,
            provider_metadata: Value::Null,
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
}

#[async_trait::async_trait]
impl TranscriptionProvider for CloudflareMediaProvider {
    fn name(&self) -> &str {
        PROVIDER
    }
    fn supports_model(&self, model: &str) -> bool {
        transcription_api(model).is_some()
    }
    fn capabilities(&self, model: &str) -> TranscriptionCapabilities {
        self.descriptor(model, MediaTask::SpeechToText)
    }

    async fn submit_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> MediaResult<Operation<TranscriptionResult>> {
        let api = transcription_api(&request.model).ok_or_else(|| MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: request.model.clone(),
            task: MediaTask::SpeechToText,
        })?;
        let options = parse_options(&request.provider_options)?;
        let mut warnings = request_warnings(&request.request_options);
        if request.prompt.is_some() && api != TranscriptionApi::WhisperTurbo {
            unsupported(
                &request.request_options,
                &mut warnings,
                "prompt",
                "only whisper-large-v3-turbo accepts an initial_prompt",
            )?;
        }
        if request.language.is_some() && api == TranscriptionApi::Whisper {
            unsupported(
                &request.request_options,
                &mut warnings,
                "language",
                "@cf/openai/whisper has no language field",
            )?;
        }
        if request
            .timestamp_granularities
            .contains(&TimestampGranularity::Segment)
            && api != TranscriptionApi::WhisperTurbo
        {
            unsupported(
                &request.request_options,
                &mut warnings,
                "timestamp_granularities",
                "only whisper-large-v3-turbo reports segments; the others report words",
            )?;
        }

        // Shared trust-boundary checks (size limit, media type, no implicit
        // URL download), then the raw bytes for the binary uploads.
        let (encoded, media_type) =
            shared::source_to_base64(&request.audio, request.request_options.max_source_bytes)?;
        let model = request.model.trim();
        let response = match api {
            TranscriptionApi::Whisper => {
                let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
                self.post_binary(model, &[], &media_type, bytes, &request.request_options)
                    .await?
            }
            TranscriptionApi::WhisperTurbo => {
                let mut body = Map::new();
                body.insert("audio".to_string(), json!(encoded));
                body.insert("task".to_string(), json!("transcribe"));
                if let Some(language) = request.language.as_ref() {
                    body.insert("language".to_string(), json!(language));
                }
                if let Some(prompt) = request.prompt.as_ref() {
                    body.insert("initial_prompt".to_string(), json!(prompt));
                }
                self.post_json(model, &Value::Object(body), &request.request_options)
                    .await?
            }
            TranscriptionApi::Nova3 => {
                let bytes = base64::engine::general_purpose::STANDARD.decode(encoded)?;
                let query = nova3_query(&request, &options);
                self.post_binary(model, &query, &media_type, bytes, &request.request_options)
                    .await?
            }
        };
        let result = unwrap_result(&response)?;
        let estimate =
            shared::resolved_cost_estimate(PROVIDER, &request.model, options.cost_estimate, None);
        Ok(Operation::completed(parse_transcription(
            api,
            &result,
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

struct ImagePlan {
    body: Value,
    size: Option<(u32, u32)>,
    steps: Option<u32>,
    usd_per_tile: f64,
    usd_per_step: f64,
    supplied_estimate: Option<CostEstimate>,
    warnings: Vec<ProviderWarning>,
}

fn plan_image(model: &ImageModel, request: &ImageGenerationRequest) -> MediaResult<ImagePlan> {
    if request.prompt.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "prompt must not be empty".to_string(),
        ));
    }
    if request.mode != ImageGenerationMode::Generate
        || !request.source_images.is_empty()
        || request.mask.is_some()
    {
        return Err(MediaError::UnsupportedTask {
            provider: PROVIDER.to_string(),
            model: request.model.clone(),
            task: match request.mode {
                ImageGenerationMode::Inpaint => MediaTask::Inpainting,
                ImageGenerationMode::Variation => MediaTask::ImageVariation,
                _ => MediaTask::ImageEdit,
            },
        });
    }
    let options = parse_options(&request.provider_options)?;
    let mut warnings = request_warnings(&request.request_options);
    if request.count.is_some_and(|count| count != 1) {
        unsupported(
            &request.request_options,
            &mut warnings,
            "count",
            "Workers AI renders one image per request",
        )?;
    }
    if request.output_format.is_some() {
        unsupported(
            &request.request_options,
            &mut warnings,
            "output_format",
            "Workers AI decides the image container; the artifact reports what came back",
        )?;
    }
    if request.negative_prompt.is_some() && !model.negative_prompt {
        unsupported(
            &request.request_options,
            &mut warnings,
            "negative_prompt",
            "only phoenix-1.0 accepts a negative prompt",
        )?;
    }
    if request.seed.is_some() && !model.seed {
        unsupported(
            &request.request_options,
            &mut warnings,
            "seed",
            "flux-1-schnell's input schema is closed and rejects a seed",
        )?;
    }

    let mut size = model.default_size;
    match request.geometry {
        Some(OutputGeometry::Dimensions { width, height }) if model.dimensions => {
            size = Some((width, height));
        }
        Some(OutputGeometry::Dimensions { .. }) => {
            unsupported(
                &request.request_options,
                &mut warnings,
                "geometry",
                "this model has no width/height fields",
            )?;
        }
        Some(OutputGeometry::AspectRatio { .. }) => {
            unsupported(
                &request.request_options,
                &mut warnings,
                "geometry",
                "Workers AI takes explicit width/height, not an aspect ratio",
            )?;
        }
        None => {}
    }
    let steps = options.steps.or(model.default_steps);

    let mut body = Map::new();
    body.insert("prompt".to_string(), json!(request.prompt));
    if let Some(steps) = options.steps {
        body.insert(model.steps_field.to_string(), json!(steps));
    }
    if let (Some(seed), true) = (request.seed, model.seed) {
        body.insert("seed".to_string(), json!(seed));
    }
    if let Some(guidance) = options.guidance {
        body.insert("guidance".to_string(), json!(guidance));
    }
    if model.dimensions {
        if let Some(OutputGeometry::Dimensions { width, height }) = request.geometry {
            body.insert("width".to_string(), json!(width));
            body.insert("height".to_string(), json!(height));
        }
    }
    if let (Some(negative), true) = (request.negative_prompt.as_ref(), model.negative_prompt) {
        body.insert("negative_prompt".to_string(), json!(negative));
    }
    Ok(ImagePlan {
        body: Value::Object(body),
        size,
        steps,
        usd_per_tile: model.usd_per_tile,
        usd_per_step: model.usd_per_step,
        supplied_estimate: options.cost_estimate,
        warnings,
    })
}

/// Tile count is rounded up per axis, the way a 512-pixel grid covers the
/// output; the per-step charge is flat.
fn image_cost(plan: &ImagePlan) -> Option<f64> {
    if let Some(estimate) = plan.supplied_estimate.as_ref() {
        return match estimate.unit {
            UsageUnit::Images => Some(estimate.quantity.unwrap_or(1.0) * estimate.usd_per_unit),
            _ => None,
        };
    }
    let (width, height) = plan.size?;
    let steps = plan.steps?;
    let tiles = (f64::from(width) / 512.0).ceil() * (f64::from(height) / 512.0).ceil();
    Some(tiles * plan.usd_per_tile + f64::from(steps) * plan.usd_per_step)
}

fn image_usage(plan: &ImagePlan, warnings: &mut Vec<ProviderWarning>) -> MediaUsage {
    let estimated_cost = image_cost(plan);
    if estimated_cost.is_none() {
        warnings.push(ProviderWarning {
            code: WarningCode::CostUnavailable,
            message: "Workers AI bills per 512x512 output tile plus per step; the output size or step count is not known for this request, so supply steps and geometry or a cost_estimate per image".to_string(),
            parameter: None,
            provider_metadata: Value::Null,
        });
    }
    MediaUsage {
        line_items: vec![UsageLineItem {
            unit: UsageUnit::Images,
            quantity: 1.0,
            cost: estimated_cost,
            description: Some("generated image".to_string()),
        }],
        provider_reported_cost: None,
        estimated_cost,
        currency: "USD".to_string(),
        metadata: json!({
            "tiles": plan.size.map(|(w, h)| ((f64::from(w) / 512.0).ceil() * (f64::from(h) / 512.0).ceil()) as u32),
            "steps": plan.steps,
            "tile_pixels": TILE_PIXELS as u32,
        }),
    }
}

struct SpeechPlan {
    body: Value,
    media_type: &'static str,
    sample_rate_hz: Option<u32>,
    warnings: Vec<ProviderWarning>,
}

fn plan_speech(api: SpeechApi, request: &SpeechSynthesisRequest) -> MediaResult<SpeechPlan> {
    if request.text.trim().is_empty() {
        return Err(MediaError::InvalidRequest(
            "prompt/text must not be empty".to_string(),
        ));
    }
    let mut warnings = request_warnings(&request.request_options);
    if request.speed.is_some() {
        unsupported(
            &request.request_options,
            &mut warnings,
            "speed",
            "Workers AI speech models have no speed control",
        )?;
    }
    if request.instructions.is_some() {
        unsupported(
            &request.request_options,
            &mut warnings,
            "instructions",
            "Workers AI speech models have no instruction field",
        )?;
    }
    if request
        .output
        .channels
        .is_some_and(|channels| channels != 1)
    {
        return Err(MediaError::UnsupportedParameter {
            provider: PROVIDER.to_string(),
            parameter: "output.channels".to_string(),
            reason: "Workers AI renders mono audio only".to_string(),
        });
    }

    let mut body = Map::new();
    let (media_type, sample_rate_hz) = match api {
        SpeechApi::Aura => {
            let speaker = request.voice.trim();
            if speaker.is_empty()
                || !speaker
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
            {
                return Err(MediaError::InvalidRequest(
                    "Aura voice must be a speaker name such as asteria or luna".to_string(),
                ));
            }
            if request.language.is_some() {
                unsupported(
                    &request.request_options,
                    &mut warnings,
                    "language",
                    "Aura models are single-language; pick aura-2-en or aura-2-es instead",
                )?;
            }
            let (encoding, container, media_type) = match request.output.format {
                AudioFormat::Mp3 => ("mp3", None, "audio/mpeg"),
                AudioFormat::Wav => ("linear16", Some("wav"), "audio/wav"),
                AudioFormat::Flac => ("flac", None, "audio/flac"),
                AudioFormat::Ogg => ("opus", Some("ogg"), "audio/ogg"),
                AudioFormat::Aac => ("aac", None, "audio/aac"),
                AudioFormat::Pcm => ("linear16", Some("none"), "audio/pcm"),
                other => {
                    return Err(MediaError::UnsupportedParameter {
                        provider: PROVIDER.to_string(),
                        parameter: "output.format".to_string(),
                        reason: format!("{} is not an Aura encoding", other.as_str()),
                    })
                }
            };
            body.insert("text".to_string(), json!(request.text));
            body.insert("speaker".to_string(), json!(speaker));
            body.insert("encoding".to_string(), json!(encoding));
            if let Some(container) = container {
                body.insert("container".to_string(), json!(container));
            }
            if let Some(rate) = request.output.sample_rate_hz {
                body.insert("sample_rate".to_string(), json!(rate));
            }
            (media_type, request.output.sample_rate_hz)
        }
        SpeechApi::Melo => {
            if request.output.format != AudioFormat::Mp3 {
                return Err(MediaError::UnsupportedParameter {
                    provider: PROVIDER.to_string(),
                    parameter: "output.format".to_string(),
                    reason: "MeloTTS renders MP3 only".to_string(),
                });
            }
            if !request.voice.trim().is_empty() {
                unsupported(
                    &request.request_options,
                    &mut warnings,
                    "voice",
                    "MeloTTS has a single voice per language",
                )?;
            }
            body.insert("prompt".to_string(), json!(request.text));
            if let Some(language) = request.language.as_ref() {
                body.insert("lang".to_string(), json!(language));
            }
            ("audio/mpeg", None)
        }
    };
    Ok(SpeechPlan {
        body: Value::Object(body),
        media_type,
        sample_rate_hz,
        warnings,
    })
}

fn nova3_query(
    request: &TranscriptionRequest,
    options: &CloudflareMediaOptions,
) -> Vec<(&'static str, String)> {
    let mut query = Vec::new();
    if let Some(language) = request.language.as_ref() {
        query.push(("language", language.clone()));
    }
    for (name, value) in [
        ("detect_language", options.detect_language),
        ("diarize", options.diarize),
        ("punctuate", options.punctuate),
        ("smart_format", options.smart_format),
    ] {
        if let Some(value) = value {
            query.push((name, value.to_string()));
        }
    }
    query
}

fn parse_transcription(
    api: TranscriptionApi,
    result: &Value,
    estimate: Option<&CostEstimate>,
    mut warnings: Vec<ProviderWarning>,
) -> MediaResult<TranscriptionResult> {
    let missing = |field: &str| MediaError::InvalidResponse {
        provider: PROVIDER.to_string(),
        message: format!("transcription response is missing {field}"),
    };
    let words_from = |items: Option<&Vec<Value>>, speaker: bool| -> Vec<TranscriptWord> {
        items
            .map(|words| {
                words
                    .iter()
                    .filter_map(|word| {
                        Some(TranscriptWord {
                            start_secs: word.get("start")?.as_f64()?,
                            end_secs: word.get("end")?.as_f64()?,
                            word: word.get("word")?.as_str()?.trim().to_string(),
                            speaker: if speaker {
                                word.get("speaker").map(|value| value.to_string())
                            } else {
                                None
                            },
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let (text, language, duration_secs, segments, words, metadata) = match api {
        TranscriptionApi::Whisper => {
            let text = result
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("text"))?;
            (
                text.to_string(),
                None,
                None,
                Vec::new(),
                words_from(result.get("words").and_then(Value::as_array), false),
                json!({"vtt": result.get("vtt"), "word_count": result.get("word_count")}),
            )
        }
        TranscriptionApi::WhisperTurbo => {
            let text = result
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("text"))?;
            let info = result.get("transcription_info");
            let segments = result
                .get("segments")
                .and_then(Value::as_array)
                .map(|segments| {
                    segments
                        .iter()
                        .filter_map(|segment| {
                            Some(TranscriptSegment {
                                start_secs: segment.get("start")?.as_f64()?,
                                end_secs: segment.get("end")?.as_f64()?,
                                text: segment.get("text")?.as_str()?.trim().to_string(),
                                speaker: None,
                            })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let words = result
                .get("segments")
                .and_then(Value::as_array)
                .map(|segments| {
                    segments
                        .iter()
                        .flat_map(|segment| {
                            words_from(segment.get("words").and_then(Value::as_array), false)
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            (
                text.to_string(),
                info.and_then(|i| i.get("language"))
                    .and_then(Value::as_str)
                    .map(str::to_string),
                info.and_then(|i| i.get("duration")).and_then(Value::as_f64),
                segments,
                words,
                json!({"transcription_info": info, "word_count": result.get("word_count")}),
            )
        }
        TranscriptionApi::Nova3 => {
            let channel = result
                .pointer("/results/channels/0")
                .ok_or_else(|| missing("results.channels[0]"))?;
            let alternative = channel
                .pointer("/alternatives/0")
                .ok_or_else(|| missing("results.channels[0].alternatives[0]"))?;
            let text = alternative
                .get("transcript")
                .and_then(Value::as_str)
                .ok_or_else(|| missing("transcript"))?;
            (
                text.to_string(),
                alternative
                    .pointer("/languages/0")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                result.pointer("/metadata/duration").and_then(Value::as_f64),
                Vec::new(),
                words_from(alternative.get("words").and_then(Value::as_array), true),
                json!({"confidence": alternative.get("confidence"), "metadata": result.get("metadata")}),
            )
        }
    };

    let provider_reported_cost = result
        .pointer("/usage/neurons")
        .and_then(Value::as_f64)
        .map(|neurons| neurons * NEURON_USD);
    let estimated_cost = estimate.and_then(|rate| {
        let quantity = rate.quantity.or(match rate.unit {
            UsageUnit::AudioSeconds => duration_secs,
            _ => None,
        })?;
        Some(quantity * rate.usd_per_unit)
    });
    if provider_reported_cost.is_none() && estimated_cost.is_none() {
        warnings.push(ProviderWarning {
            code: WarningCode::CostUnavailable,
            message: "Workers AI bills transcription per audio minute and this response carries neither neurons nor a duration; supply a cost_estimate with the quantity in seconds to price it".to_string(),
            parameter: None,
            provider_metadata: Value::Null,
        });
    }
    Ok(TranscriptionResult {
        text,
        language,
        duration_secs,
        segments,
        words,
        usage: Some(MediaUsage {
            line_items: duration_secs
                .map(|seconds| {
                    vec![UsageLineItem {
                        unit: UsageUnit::AudioSeconds,
                        quantity: seconds,
                        cost: provider_reported_cost.or(estimated_cost),
                        description: Some("input audio".to_string()),
                    }]
                })
                .unwrap_or_default(),
            provider_reported_cost,
            estimated_cost,
            currency: "USD".to_string(),
            metadata: json!({"neurons": result.pointer("/usage/neurons")}),
        }),
        warnings,
        provider_metadata: metadata,
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
        warnings.push(ProviderWarning {
            code: WarningCode::CostUnavailable,
            message: "this Workers AI speech model bills per audio minute, which is unknown until playback; supply a cost_estimate to price it locally".to_string(),
            parameter: None,
            provider_metadata: Value::Null,
        });
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

/// REST JSON answers are `{"result": ..., "success": bool, "errors": [...]}`.
fn unwrap_result(response: &shared::CapturedResponse) -> MediaResult<Value> {
    let value = shared::parse_json(PROVIDER, response)?;
    if value.get("success").and_then(Value::as_bool) == Some(false) {
        let message = value
            .pointer("/errors/0/message")
            .and_then(Value::as_str)
            .unwrap_or("request failed")
            .to_string();
        return Err(MediaError::Api {
            provider: PROVIDER.to_string(),
            status: response.status.as_u16(),
            message,
        });
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| MediaError::InvalidResponse {
            provider: PROVIDER.to_string(),
            message: "response has no result field".to_string(),
        })
}

fn content_type(headers: &reqwest::header::HeaderMap) -> Option<String> {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(|value| value.trim().to_ascii_lowercase())
}

/// The image container is not in the schema, so read it off the bytes.
fn sniff_image_type(bytes: &[u8]) -> Option<String> {
    let media_type = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        "image/webp"
    } else {
        return None;
    };
    Some(media_type.to_string())
}

fn parse_options(options: &ProviderOptions) -> MediaResult<CloudflareMediaOptions> {
    let raw = shared::provider_options(options, PROVIDER)?;
    let parsed: CloudflareMediaOptions =
        serde_json::from_value(Value::Object(raw)).map_err(|error| {
            MediaError::InvalidRequest(format!("invalid Cloudflare provider options: {error}"))
        })?;
    validate_options(&parsed)?;
    Ok(parsed)
}

fn validate_options(options: &CloudflareMediaOptions) -> MediaResult<()> {
    if options.steps.is_some_and(|steps| steps == 0) {
        return Err(MediaError::InvalidRequest(
            "steps must be at least 1".to_string(),
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
            message: "Workers AI does not document request idempotency keys".to_string(),
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

fn options_schema() -> Value {
    json!({
        "type":"object",
        "additionalProperties":false,
        "properties":{
            "steps":{"type":"integer","minimum":1},
            "guidance":{"type":"number"},
            "detect_language":{"type":"boolean"},
            "diarize":{"type":"boolean"},
            "punctuate":{"type":"boolean"},
            "smart_format":{"type":"boolean"},
            "cost_estimate":{"type":"object"}
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_request(model: &str) -> ImageGenerationRequest {
        ImageGenerationRequest::new("a lighthouse at dusk").with_model(model)
    }

    #[test]
    fn only_documented_models_are_accepted_per_task() {
        let provider = CloudflareMediaProvider::new();
        assert!(ImageGenerationProvider::supports_model(
            &provider,
            "@cf/leonardo/phoenix-1.0"
        ));
        assert!(ImageGenerationProvider::supports_model(
            &provider,
            "@cf/black-forest-labs/flux-1-schnell"
        ));
        // Undocumented multipart input, deliberately not wired.
        assert!(!ImageGenerationProvider::supports_model(
            &provider,
            "@cf/black-forest-labs/flux-2-dev"
        ));
        assert!(SpeechSynthesisProvider::supports_model(
            &provider,
            "@cf/deepgram/aura-2-en"
        ));
        assert!(!SpeechSynthesisProvider::supports_model(
            &provider,
            "@cf/openai/whisper"
        ));
        assert!(TranscriptionProvider::supports_model(
            &provider,
            "@cf/openai/whisper-large-v3-turbo"
        ));
        assert!(TranscriptionProvider::supports_model(
            &provider,
            "@cf/deepgram/nova-3"
        ));
        assert!(!TranscriptionProvider::supports_model(
            &provider,
            "@cf/deepgram/aura-1"
        ));
    }

    #[test]
    fn phoenix_body_carries_dimensions_steps_guidance_and_negative_prompt() {
        let mut request = image_request("@cf/leonardo/phoenix-1.0");
        request.geometry = Some(OutputGeometry::Dimensions {
            width: 1536,
            height: 1024,
        });
        request.seed = Some(7);
        request.negative_prompt = Some("text, watermark".to_string());
        request.provider_options = CloudflareMediaOptions {
            steps: Some(30),
            guidance: Some(3.5),
            ..Default::default()
        }
        .into_provider_options()
        .unwrap();
        let plan = plan_image(image_model(&request.model).unwrap(), &request).unwrap();
        assert_eq!(
            plan.body,
            json!({
                "prompt": "a lighthouse at dusk",
                "num_steps": 30,
                "seed": 7,
                "guidance": 3.5,
                "width": 1536,
                "height": 1024,
                "negative_prompt": "text, watermark"
            })
        );
        assert_eq!(plan.size, Some((1536, 1024)));
        assert_eq!(plan.steps, Some(30));
        assert!(plan.warnings.is_empty());
    }

    #[test]
    fn flux_schnell_uses_steps_and_drops_unsupported_fields_by_policy() {
        let mut request = image_request("@cf/black-forest-labs/flux-1-schnell");
        request.geometry = Some(OutputGeometry::Dimensions {
            width: 1024,
            height: 1024,
        });
        request.negative_prompt = Some("blurry".to_string());
        request.count = Some(2);
        request.seed = Some(3);
        request.provider_options = CloudflareMediaOptions {
            steps: Some(6),
            ..Default::default()
        }
        .into_provider_options()
        .unwrap();
        // The default policy refuses silently dropped parameters.
        assert!(plan_image(image_model(&request.model).unwrap(), &request).is_err());

        request.request_options.unsupported_parameter_policy =
            UnsupportedParameterPolicy::WarnAndDrop;
        let plan = plan_image(image_model(&request.model).unwrap(), &request).unwrap();
        assert_eq!(
            plan.body,
            json!({"prompt": "a lighthouse at dusk", "steps": 6})
        );
        let dropped: Vec<_> = plan
            .warnings
            .iter()
            .filter_map(|w| w.parameter.as_deref())
            .collect();
        assert_eq!(dropped, ["count", "negative_prompt", "seed", "geometry"]);
    }

    #[test]
    fn image_cost_is_tiles_plus_steps_or_unavailable() {
        // Phoenix default 1024x1024 at 25 steps: 4 tiles * 0.00583 + 25 * 0.00011.
        let request = image_request("@cf/leonardo/phoenix-1.0");
        let plan = plan_image(image_model(&request.model).unwrap(), &request).unwrap();
        let cost = image_cost(&plan).unwrap();
        assert!((cost - 0.026_07).abs() < 1e-9, "{cost}");

        // A 1536x1024 output covers 3x2 tiles.
        let mut wide = image_request("@cf/leonardo/phoenix-1.0");
        wide.geometry = Some(OutputGeometry::Dimensions {
            width: 1536,
            height: 1024,
        });
        let plan = plan_image(image_model(&wide.model).unwrap(), &wide).unwrap();
        let cost = image_cost(&plan).unwrap();
        assert!((cost - (6.0 * 0.005_83 + 25.0 * 0.000_11)).abs() < 1e-9);

        // Lucid Origin has no documented default step count.
        let lucid = image_request("@cf/leonardo/lucid-origin");
        let plan = plan_image(image_model(&lucid.model).unwrap(), &lucid).unwrap();
        assert!(image_cost(&plan).is_none());
        let mut warnings = Vec::new();
        let usage = image_usage(&plan, &mut warnings);
        assert!(usage.estimated_cost.is_none());
        assert_eq!(warnings[0].code, WarningCode::CostUnavailable);

        // FLUX.1 schnell has no documented output size.
        let flux = image_request("@cf/black-forest-labs/flux-1-schnell");
        let plan = plan_image(image_model(&flux.model).unwrap(), &flux).unwrap();
        assert!(image_cost(&plan).is_none());

        // A caller-supplied per-image rate wins.
        let mut priced = image_request("@cf/black-forest-labs/flux-1-schnell");
        priced.provider_options = CloudflareMediaOptions {
            cost_estimate: Some(CostEstimate {
                unit: UsageUnit::Images,
                usd_per_unit: 0.01,
                quantity: None,
            }),
            ..Default::default()
        }
        .into_provider_options()
        .unwrap();
        let plan = plan_image(image_model(&priced.model).unwrap(), &priced).unwrap();
        assert_eq!(image_cost(&plan), Some(0.01));
    }

    #[test]
    fn edits_and_extra_namespaces_are_rejected() {
        let mut request = image_request("@cf/leonardo/phoenix-1.0");
        request.mode = ImageGenerationMode::Edit;
        assert!(matches!(
            plan_image(image_model(&request.model).unwrap(), &request),
            Err(MediaError::UnsupportedTask { .. })
        ));

        let mut request = image_request("@cf/leonardo/phoenix-1.0");
        request
            .provider_options
            .insert("replicate".to_string(), json!({}));
        assert!(plan_image(image_model(&request.model).unwrap(), &request).is_err());

        let mut request = image_request("@cf/leonardo/phoenix-1.0");
        request
            .provider_options
            .insert(PROVIDER.to_string(), json!({"steps": 0}));
        assert!(plan_image(image_model(&request.model).unwrap(), &request).is_err());
    }

    #[test]
    fn aura_body_maps_voice_and_output_format() {
        let mut request = SpeechSynthesisRequest::new("hello there", "asteria")
            .with_model("@cf/deepgram/aura-2-en");
        let plan = plan_speech(SpeechApi::Aura, &request).unwrap();
        assert_eq!(
            plan.body,
            json!({"text": "hello there", "speaker": "asteria", "encoding": "mp3"})
        );
        assert_eq!(plan.media_type, "audio/mpeg");

        request.output = AudioOutputSpec {
            format: AudioFormat::Wav,
            sample_rate_hz: Some(16000),
            channels: Some(1),
        };
        let plan = plan_speech(SpeechApi::Aura, &request).unwrap();
        assert_eq!(plan.body["encoding"], "linear16");
        assert_eq!(plan.body["container"], "wav");
        assert_eq!(plan.body["sample_rate"], 16000);
        assert_eq!(plan.media_type, "audio/wav");
        assert_eq!(plan.sample_rate_hz, Some(16000));

        request.output = AudioOutputSpec {
            format: AudioFormat::M4a,
            sample_rate_hz: None,
            channels: None,
        };
        assert!(plan_speech(SpeechApi::Aura, &request).is_err());

        // The speaker is a JSON value, but keep it to a bare name.
        let odd = SpeechSynthesisRequest::new("hi", "asteria\"}").with_model("@cf/deepgram/aura-1");
        assert!(plan_speech(SpeechApi::Aura, &odd).is_err());
    }

    #[test]
    fn melotts_body_uses_prompt_and_lang_and_drops_voice() {
        let mut request =
            SpeechSynthesisRequest::new("bonjour", "any").with_model("@cf/myshell-ai/melotts");
        request.language = Some("fr".to_string());
        assert!(plan_speech(SpeechApi::Melo, &request).is_err());
        request.request_options.unsupported_parameter_policy =
            UnsupportedParameterPolicy::WarnAndDrop;
        let plan = plan_speech(SpeechApi::Melo, &request).unwrap();
        assert_eq!(plan.body, json!({"prompt": "bonjour", "lang": "fr"}));
        assert_eq!(plan.warnings[0].parameter.as_deref(), Some("voice"));

        request.output.format = AudioFormat::Wav;
        assert!(plan_speech(SpeechApi::Melo, &request).is_err());
    }

    #[test]
    fn character_usage_prices_aura_from_the_reference_table() {
        let estimate = shared::resolved_cost_estimate(
            PROVIDER,
            "@cf/deepgram/aura-2-en",
            None,
            Some((UsageUnit::Characters, 1000.0)),
        )
        .unwrap();
        let mut warnings = Vec::new();
        let usage = character_usage(1000.0, Some(&estimate), &mut warnings);
        assert!((usage.estimated_cost.unwrap() - 0.03).abs() < 1e-12);
        assert!(warnings.is_empty());

        // MeloTTS bills per minute, unknown at submit.
        let estimate = shared::resolved_cost_estimate(
            PROVIDER,
            "@cf/myshell-ai/melotts",
            None,
            Some((UsageUnit::Characters, 1000.0)),
        )
        .unwrap();
        let usage = character_usage(1000.0, Some(&estimate), &mut warnings);
        assert!(usage.estimated_cost.is_none());
        assert_eq!(warnings[0].code, WarningCode::CostUnavailable);
    }

    #[test]
    fn whisper_response_yields_words_and_no_price() {
        let result = json!({
            "text": "hello world",
            "word_count": 2,
            "vtt": "WEBVTT\n",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9}
            ]
        });
        let estimate = shared::resolved_cost_estimate(PROVIDER, "@cf/openai/whisper", None, None);
        let parsed = parse_transcription(
            TranscriptionApi::Whisper,
            &result,
            estimate.as_ref(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(parsed.text, "hello world");
        assert_eq!(parsed.words.len(), 2);
        assert_eq!(parsed.words[1].word, "world");
        assert!(parsed.duration_secs.is_none());
        assert!(parsed.usage.unwrap().estimated_cost.is_none());
        assert_eq!(parsed.warnings[0].code, WarningCode::CostUnavailable);
    }

    #[test]
    fn whisper_turbo_response_yields_segments_language_and_duration_cost() {
        let result = json!({
            "text": "one two",
            "word_count": 2,
            "usage": {"neurons": 93.27},
            "transcription_info": {"language": "en", "language_probability": 0.99, "duration": 120.0},
            "segments": [
                {"start": 0.0, "end": 1.0, "text": " one", "words": [{"word": " one", "start": 0.0, "end": 0.4}]},
                {"start": 1.0, "end": 2.0, "text": "two ", "words": [{"word": "two", "start": 1.1, "end": 1.5}]}
            ]
        });
        let estimate = shared::resolved_cost_estimate(
            PROVIDER,
            "@cf/openai/whisper-large-v3-turbo",
            None,
            None,
        );
        let parsed = parse_transcription(
            TranscriptionApi::WhisperTurbo,
            &result,
            estimate.as_ref(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(parsed.language.as_deref(), Some("en"));
        assert_eq!(parsed.duration_secs, Some(120.0));
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].text, "one");
        assert_eq!(parsed.words.len(), 2);
        // Two minutes at $0.000513 per minute.
        let usage = parsed.usage.unwrap();
        assert!((usage.estimated_cost.unwrap() - 0.001_026).abs() < 1e-9);
        assert!((usage.provider_reported_cost.unwrap() - 93.27 * NEURON_USD).abs() < 1e-12);
        assert_eq!(parsed.words[0].word, "one");
        assert_eq!(usage.line_items[0].unit, UsageUnit::AudioSeconds);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn nova3_response_reads_deepgram_alternatives() {
        let result = json!({
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "neurons": 24.01419},
            "results": {"channels": [{
                "alternatives": [{
                    "transcript": "good morning",
                    "confidence": 0.98,
                    "languages": ["en"],
                    "words": [
                        {"word": "good", "start": 0.1, "end": 0.3, "speaker": 0},
                        {"word": "morning", "start": 0.4, "end": 0.9, "speaker": 0}
                    ]
                }]
            }]}
        });
        let estimate = shared::resolved_cost_estimate(PROVIDER, "@cf/deepgram/nova-3", None, None);
        let parsed = parse_transcription(
            TranscriptionApi::Nova3,
            &result,
            estimate.as_ref(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(parsed.text, "good morning");
        assert_eq!(parsed.language.as_deref(), Some("en"));
        assert_eq!(parsed.words[0].speaker.as_deref(), Some("0"));
        assert!(parsed.duration_secs.is_none());
        // No duration, so no estimate, but the billed neurons price it exactly.
        let usage = parsed.usage.unwrap();
        assert!(usage.estimated_cost.is_none());
        assert!((usage.provider_reported_cost.unwrap() - 24.01419 * NEURON_USD).abs() < 1e-12);
        assert!(parsed.warnings.is_empty());

        assert!(parse_transcription(
            TranscriptionApi::Nova3,
            &json!({"results": {"channels": []}}),
            None,
            Vec::new()
        )
        .is_err());
    }

    #[test]
    fn nova3_query_carries_language_and_flags() {
        let mut request = TranscriptionRequest::new(MediaSource::Bytes {
            data: vec![0, 1, 2],
            media_type: "audio/wav".to_string(),
        })
        .with_model("@cf/deepgram/nova-3");
        request.language = Some("es".to_string());
        let options = CloudflareMediaOptions {
            detect_language: Some(false),
            punctuate: Some(true),
            ..Default::default()
        };
        assert_eq!(
            nova3_query(&request, &options),
            vec![
                ("language", "es".to_string()),
                ("detect_language", "false".to_string()),
                ("punctuate", "true".to_string())
            ]
        );
    }

    #[test]
    fn rest_envelope_is_unwrapped_and_failures_surface() {
        let ok = shared::CapturedResponse {
            status: reqwest::StatusCode::OK,
            headers: Default::default(),
            body: br#"{"result":{"image":"aGk="},"success":true,"errors":[],"messages":[]}"#
                .to_vec(),
        };
        assert_eq!(unwrap_result(&ok).unwrap()["image"], "aGk=");

        let failed = shared::CapturedResponse {
            status: reqwest::StatusCode::OK,
            headers: Default::default(),
            body: br#"{"result":null,"success":false,"errors":[{"code":5035,"message":"upgrade"}],"messages":[]}"#
                .to_vec(),
        };
        assert!(matches!(
            unwrap_result(&failed),
            Err(MediaError::Api { message, .. }) if message == "upgrade"
        ));
    }

    #[test]
    fn image_container_is_sniffed_from_bytes() {
        assert_eq!(
            sniff_image_type(b"\x89PNG\r\n\x1a\n....").as_deref(),
            Some("image/png")
        );
        assert_eq!(
            sniff_image_type(&[0xFF, 0xD8, 0xFF, 0xE0, 0, 0]).as_deref(),
            Some("image/jpeg")
        );
        assert_eq!(
            sniff_image_type(b"RIFF\0\0\0\0WEBPVP8 ").as_deref(),
            Some("image/webp")
        );
        assert!(sniff_image_type(b"not an image").is_none());
    }

    #[test]
    fn run_url_honours_override_and_validates_account() {
        let provider = CloudflareMediaProvider::new();
        std::env::set_var(API_BASE_ENV, "https://gateway.example/ai/run/");
        assert_eq!(
            provider.run_url("@cf/openai/whisper").unwrap(),
            "https://gateway.example/ai/run/@cf/openai/whisper"
        );
        std::env::remove_var(API_BASE_ENV);
        std::env::set_var(ACCOUNT_ID_ENV, "abc123");
        assert_eq!(
            provider.run_url("@cf/openai/whisper").unwrap(),
            "https://api.cloudflare.com/client/v4/accounts/abc123/ai/run/@cf/openai/whisper"
        );
        std::env::set_var(ACCOUNT_ID_ENV, "abc/../123");
        assert!(provider.run_url("@cf/openai/whisper").is_err());
        std::env::remove_var(ACCOUNT_ID_ENV);
    }

    #[test]
    fn descriptors_reflect_per_model_parameter_support() {
        let provider = CloudflareMediaProvider::new();
        let phoenix = ImageGenerationProvider::capabilities(&provider, "@cf/leonardo/phoenix-1.0");
        assert_eq!(phoenix.parameters.dimensions, CapabilitySupport::Supported);
        assert_eq!(
            phoenix.parameters.negative_prompt,
            CapabilitySupport::Supported
        );
        assert_eq!(phoenix.execution.polling, CapabilitySupport::Unsupported);

        let flux = ImageGenerationProvider::capabilities(
            &provider,
            "@cf/black-forest-labs/flux-1-schnell",
        );
        assert_eq!(flux.parameters.dimensions, CapabilitySupport::Unsupported);
        assert_eq!(flux.parameters.seed, CapabilitySupport::Unsupported);
        assert_eq!(phoenix.parameters.seed, CapabilitySupport::Supported);

        let aura = SpeechSynthesisProvider::capabilities(&provider, "@cf/deepgram/aura-1");
        assert_eq!(aura.parameters.output_format, CapabilitySupport::Supported);
        assert_eq!(aura.limits.supported_formats.len(), 6);

        let melo = SpeechSynthesisProvider::capabilities(&provider, "@cf/myshell-ai/melotts");
        assert_eq!(melo.limits.supported_formats, vec!["mp3".to_string()]);
    }
}
