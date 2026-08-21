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

//! Provider-independent media request, operation, artifact, and usage types.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::watch;

/// Provider-specific settings keyed by normalized provider name.
pub type ProviderOptions = BTreeMap<String, Value>;

/// A media input. Base64 is deliberately not a source variant: it is a wire encoding.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    Bytes {
        data: Vec<u8>,
        media_type: String,
    },
    Url {
        url: String,
        media_type: Option<String>,
    },
    File {
        path: PathBuf,
        media_type: Option<String>,
    },
    ProviderFile {
        id: String,
        media_type: Option<String>,
    },
    ObjectStorage {
        uri: String,
        media_type: Option<String>,
    },
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ArtifactSource {
    Inline(Vec<u8>),
    Url(String),
    ProviderFileId(String),
    ObjectStorageUri(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Dimensions {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MediaArtifact {
    pub kind: MediaKind,
    pub media_type: String,
    pub source: ArtifactSource,
    pub size_bytes: Option<u64>,
    pub dimensions: Option<Dimensions>,
    pub duration_secs: Option<f64>,
    pub frame_rate: Option<f32>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub expires_at: Option<u64>,
    #[serde(default)]
    pub metadata: Value,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MediaTask {
    TextToImage,
    ImageEdit,
    Inpainting,
    ImageVariation,
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
    VideoExtend,
    VideoEdit,
    TextToSpeech,
    SpeechToText,
    SpeechTranslation,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OperationStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    CancellationRequested,
    Cancelled,
    Expired,
}

impl OperationStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::Expired
        )
    }
}

/// Serializable, credential-free identity for resuming an upstream operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobHandle {
    pub provider: String,
    pub model: String,
    pub task: MediaTask,
    pub remote_id: String,
    /// Optional caller-supplied rate snapshot used only for transparent estimates.
    #[serde(default)]
    pub cost_estimate: Option<CostEstimate>,
    /// Contract warnings produced at submission and carried across restart.
    #[serde(default)]
    pub warnings: Vec<ProviderWarning>,
    /// Requested output count retained so resumed jobs can report partial output.
    #[serde(default)]
    pub expected_outputs: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CostEstimate {
    pub unit: UsageUnit,
    pub usd_per_unit: f64,
    /// Known request quantity, such as input characters. Runtime/output units may be `None`.
    pub quantity: Option<f64>,
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    Unsupported,
    Authentication,
    Permission,
    InsufficientCredits,
    RateLimit,
    ContentPolicy,
    InvalidInput,
    RemoteJob,
    Cancelled,
    Expired,
    InvalidResponse,
    Transport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationFailure {
    pub category: FailureCategory,
    pub message: String,
    pub provider_code: Option<String>,
    pub http_status: Option<u16>,
    pub request_id: Option<String>,
    #[serde(default)]
    pub provider_metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Operation<T> {
    pub handle: Option<JobHandle>,
    pub status: OperationStatus,
    pub progress: Option<f32>,
    pub result: Option<T>,
    pub error: Option<GenerationFailure>,
    pub created_at: Option<u64>,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    #[serde(default)]
    pub provider_metadata: Value,
}

impl<T> Operation<T> {
    pub fn completed(result: T) -> Self {
        Self {
            handle: None,
            status: OperationStatus::Succeeded,
            progress: Some(1.0),
            result: Some(result),
            error: None,
            created_at: None,
            started_at: None,
            completed_at: None,
            provider_metadata: Value::Null,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnsupportedParameterPolicy {
    #[default]
    Error,
    WarnAndDrop,
}

/// Transport and local waiting controls; not model-generation parameters.
#[derive(Debug, Clone)]
pub struct RequestOptions {
    pub max_retries: u32,
    pub retry_backoff: Duration,
    pub submit_timeout: Option<Duration>,
    pub wait_timeout: Option<Duration>,
    pub polling_interval: Duration,
    pub cancellation_token: Option<watch::Receiver<bool>>,
    pub idempotency_key: Option<String>,
    pub extra_headers: Option<HashMap<String, String>>,
    pub unsupported_parameter_policy: UnsupportedParameterPolicy,
    pub max_source_bytes: usize,
    pub max_response_bytes: usize,
}

impl Default for RequestOptions {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_backoff: Duration::from_secs(1),
            submit_timeout: Some(Duration::from_secs(120)),
            wait_timeout: Some(Duration::from_secs(600)),
            polling_interval: Duration::from_secs(2),
            cancellation_token: None,
            idempotency_key: None,
            extra_headers: None,
            unsupported_parameter_policy: UnsupportedParameterPolicy::Error,
            max_source_bytes: 20 * 1024 * 1024,
            max_response_bytes: 100 * 1024 * 1024,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageGenerationMode {
    Generate,
    Edit,
    Inpaint,
    Variation,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VideoGenerationMode {
    TextToVideo,
    ImageToVideo,
    ReferenceToVideo,
    Extend,
    Edit,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputGeometry {
    Dimensions { width: u32, height: u32 },
    AspectRatio { width: u32, height: u32 },
}

impl OutputGeometry {
    pub fn as_api_string(self) -> String {
        match self {
            Self::Dimensions { width, height } => format!("{width}x{height}"),
            Self::AspectRatio { width, height } => format!("{width}:{height}"),
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Svg,
}

impl ImageFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
            Self::Svg => "svg",
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat {
    Mp4,
    Webm,
    Mov,
}

impl VideoFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mp4 => "mp4",
            Self::Webm => "webm",
            Self::Mov => "mov",
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Mp3,
    Pcm,
    Wav,
    Flac,
    Aac,
    Ogg,
    M4a,
}

impl AudioFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Pcm => "pcm",
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Aac => "aac",
            Self::Ogg => "ogg",
            Self::M4a => "m4a",
        }
    }

    pub fn media_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Pcm => "audio/pcm",
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Aac => "audio/aac",
            Self::Ogg => "audio/ogg",
            Self::M4a => "audio/mp4",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioOutputSpec {
    pub format: AudioFormat,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
}

impl Default for AudioOutputSpec {
    fn default() -> Self {
        Self {
            format: AudioFormat::Mp3,
            sample_rate_hz: None,
            channels: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub mode: ImageGenerationMode,
    pub source_images: Vec<MediaSource>,
    pub mask: Option<MediaSource>,
    pub count: Option<u32>,
    pub seed: Option<u64>,
    pub geometry: Option<OutputGeometry>,
    pub negative_prompt: Option<String>,
    pub output_format: Option<ImageFormat>,
    pub provider_options: ProviderOptions,
    pub request_options: RequestOptions,
}

impl ImageGenerationRequest {
    /// Create a request for use with [`crate::media::generate_image`].
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            model: String::new(),
            prompt: prompt.into(),
            mode: ImageGenerationMode::Generate,
            source_images: Vec::new(),
            mask: None,
            count: None,
            seed: None,
            geometry: None,
            negative_prompt: None,
            output_format: None,
            provider_options: BTreeMap::new(),
            request_options: RequestOptions::default(),
        }
    }

    /// Set the provider-native model for low-level trait calls.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Clone)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub mode: VideoGenerationMode,
    pub first_frame: Option<MediaSource>,
    pub last_frame: Option<MediaSource>,
    pub reference_images: Vec<MediaSource>,
    pub source_video: Option<MediaSource>,
    pub count: Option<u32>,
    pub seed: Option<u64>,
    pub duration_secs: Option<f64>,
    pub geometry: Option<OutputGeometry>,
    pub negative_prompt: Option<String>,
    pub output_format: Option<VideoFormat>,
    pub provider_options: ProviderOptions,
    pub request_options: RequestOptions,
}

impl VideoGenerationRequest {
    /// Create a request for use with [`crate::media::generate_video`].
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            model: String::new(),
            prompt: prompt.into(),
            mode: VideoGenerationMode::TextToVideo,
            first_frame: None,
            last_frame: None,
            reference_images: Vec::new(),
            source_video: None,
            count: None,
            seed: None,
            duration_secs: None,
            geometry: None,
            negative_prompt: None,
            output_format: None,
            provider_options: BTreeMap::new(),
            request_options: RequestOptions::default(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Clone)]
pub struct SpeechSynthesisRequest {
    pub model: String,
    pub text: String,
    pub voice: String,
    pub language: Option<String>,
    pub instructions: Option<String>,
    pub speed: Option<f32>,
    pub output: AudioOutputSpec,
    pub provider_options: ProviderOptions,
    pub request_options: RequestOptions,
}

impl SpeechSynthesisRequest {
    /// Create a request for use with [`crate::media::synthesize_speech`].
    pub fn new(text: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            model: String::new(),
            text: text.into(),
            voice: voice.into(),
            language: None,
            instructions: None,
            speed: None,
            output: AudioOutputSpec::default(),
            provider_options: BTreeMap::new(),
            request_options: RequestOptions::default(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TimestampGranularity {
    Segment,
    Word,
}

#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    pub model: String,
    pub audio: MediaSource,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub timestamp_granularities: Vec<TimestampGranularity>,
    pub provider_options: ProviderOptions,
    pub request_options: RequestOptions,
}

impl TranscriptionRequest {
    /// Create a request for use with [`crate::media::transcribe`].
    pub fn new(audio: MediaSource) -> Self {
        Self {
            model: String::new(),
            audio,
            language: None,
            prompt: None,
            timestamp_granularities: Vec::new(),
            provider_options: BTreeMap::new(),
            request_options: RequestOptions::default(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UsageUnit {
    Images,
    Megapixels,
    VideoSeconds,
    AudioSeconds,
    Characters,
    ComputeSeconds,
    InputTokens,
    OutputTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageLineItem {
    pub unit: UsageUnit,
    pub quantity: f64,
    pub cost: Option<f64>,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MediaUsage {
    #[serde(default)]
    pub line_items: Vec<UsageLineItem>,
    /// Authoritative amount reported by the upstream provider for this request.
    pub provider_reported_cost: Option<f64>,
    /// Locally computed amount. It is never presented as provider-reported cost.
    pub estimated_cost: Option<f64>,
    pub currency: String,
    #[serde(default)]
    pub metadata: Value,
}

impl Default for MediaUsage {
    fn default() -> Self {
        Self {
            line_items: Vec::new(),
            provider_reported_cost: None,
            estimated_cost: None,
            currency: "USD".to_string(),
            metadata: Value::Null,
        }
    }
}

impl MediaUsage {
    pub fn best_available_cost(&self) -> Option<f64> {
        self.provider_reported_cost.or(self.estimated_cost)
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WarningCode {
    UnsupportedParameterDropped,
    ParameterCoerced,
    BestEffortSeed,
    PartialOutput,
    OutputFormatChanged,
    UsageUnavailable,
    CostUnavailable,
    SchemaUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProviderWarning {
    pub code: WarningCode,
    pub message: String,
    pub parameter: Option<String>,
    #[serde(default)]
    pub provider_metadata: Value,
}

#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SafetyStatus {
    NotReported,
    Passed,
    PartiallyFiltered,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SafetyReport {
    pub status: SafetyStatus,
    pub filtered_count: u32,
    #[serde(default)]
    pub reasons: Vec<String>,
    #[serde(default)]
    pub provider_metadata: Value,
}

impl Default for SafetyReport {
    fn default() -> Self {
        Self {
            status: SafetyStatus::NotReported,
            filtered_count: 0,
            reasons: Vec::new(),
            provider_metadata: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageGenerationResult {
    pub artifacts: Vec<MediaArtifact>,
    pub usage: Option<MediaUsage>,
    #[serde(default)]
    pub warnings: Vec<ProviderWarning>,
    pub safety: SafetyReport,
    #[serde(default)]
    pub provider_metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoGenerationResult {
    pub artifacts: Vec<MediaArtifact>,
    pub usage: Option<MediaUsage>,
    #[serde(default)]
    pub warnings: Vec<ProviderWarning>,
    pub safety: SafetyReport,
    #[serde(default)]
    pub provider_metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechSynthesisResult {
    pub artifact: MediaArtifact,
    pub usage: Option<MediaUsage>,
    #[serde(default)]
    pub warnings: Vec<ProviderWarning>,
    #[serde(default)]
    pub provider_metadata: Value,
}

/// A genuine upstream byte stream. `next_chunk` yields bytes as the HTTP body
/// arrives; providers that only produce completed files must not return this.
#[derive(Debug)]
pub struct SpeechStream {
    pub media_type: String,
    pub generation_id: Option<String>,
    response: reqwest::Response,
}

impl SpeechStream {
    pub(crate) fn new(
        media_type: String,
        generation_id: Option<String>,
        response: reqwest::Response,
    ) -> Self {
        Self {
            media_type,
            generation_id,
            response,
        }
    }

    pub async fn next_chunk(&mut self) -> crate::media::MediaResult<Option<Vec<u8>>> {
        self.response
            .chunk()
            .await
            .map(|chunk| chunk.map(|bytes| bytes.to_vec()))
            .map_err(crate::media::MediaError::Transport)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptSegment {
    pub start_secs: f64,
    pub end_secs: f64,
    pub text: String,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptWord {
    pub start_secs: f64,
    pub end_secs: f64,
    pub word: String,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f64>,
    #[serde(default)]
    pub segments: Vec<TranscriptSegment>,
    #[serde(default)]
    pub words: Vec<TranscriptWord>,
    pub usage: Option<MediaUsage>,
    #[serde(default)]
    pub warnings: Vec<ProviderWarning>,
    #[serde(default)]
    pub provider_metadata: Value,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilitySupport {
    Supported,
    Unsupported,
    Unknown,
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    pub immediate: CapabilitySupport,
    pub persistent_jobs: CapabilitySupport,
    pub polling: CapabilitySupport,
    pub cancellation: CapabilitySupport,
    pub progress: CapabilitySupport,
    pub webhooks: CapabilitySupport,
    pub binary_streaming: CapabilitySupport,
    pub resumable: CapabilitySupport,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ParameterCapabilities {
    pub count: CapabilitySupport,
    pub seed: CapabilitySupport,
    pub dimensions: CapabilitySupport,
    pub aspect_ratio: CapabilitySupport,
    pub duration: CapabilitySupport,
    pub mask: CapabilitySupport,
    pub negative_prompt: CapabilitySupport,
    pub output_format: CapabilitySupport,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MediaLimits {
    pub max_input_bytes: Option<u64>,
    pub max_outputs: Option<u32>,
    pub max_duration_secs: Option<f64>,
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MediaModelDescriptor {
    pub provider: String,
    pub model: String,
    pub tasks: Vec<MediaTask>,
    pub input_modalities: Vec<Modality>,
    pub output_modalities: Vec<Modality>,
    pub execution: ExecutionCapabilities,
    pub parameters: ParameterCapabilities,
    pub limits: MediaLimits,
    pub provider_options_schema: Option<Value>,
}

pub type ImageCapabilities = MediaModelDescriptor;
pub type VideoCapabilities = MediaModelDescriptor;
pub type SpeechCapabilities = MediaModelDescriptor;
pub type TranscriptionCapabilities = MediaModelDescriptor;
