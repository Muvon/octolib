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

use super::errors::MediaResult;
use super::types::*;

#[async_trait::async_trait]
pub trait ImageGenerationProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;
    fn capabilities(&self, model: &str) -> ImageCapabilities;
    async fn submit_image(
        &self,
        request: ImageGenerationRequest,
    ) -> MediaResult<Operation<ImageGenerationResult>>;
    async fn poll_image(&self, handle: &JobHandle)
        -> MediaResult<Operation<ImageGenerationResult>>;
    async fn cancel_image(&self, handle: &JobHandle) -> MediaResult<()>;
}

#[async_trait::async_trait]
pub trait VideoGenerationProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;
    fn capabilities(&self, model: &str) -> VideoCapabilities;
    async fn submit_video(
        &self,
        request: VideoGenerationRequest,
    ) -> MediaResult<Operation<VideoGenerationResult>>;
    async fn poll_video(&self, handle: &JobHandle)
        -> MediaResult<Operation<VideoGenerationResult>>;
    async fn cancel_video(&self, handle: &JobHandle) -> MediaResult<()>;
}

#[async_trait::async_trait]
pub trait SpeechSynthesisProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;
    fn capabilities(&self, model: &str) -> SpeechCapabilities;
    async fn submit_speech(
        &self,
        request: SpeechSynthesisRequest,
    ) -> MediaResult<Operation<SpeechSynthesisResult>>;
    async fn poll_speech(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<SpeechSynthesisResult>>;
    async fn cancel_speech(&self, handle: &JobHandle) -> MediaResult<()>;

    /// Return a real upstream byte stream when the provider supports it.
    async fn stream_speech(&self, request: SpeechSynthesisRequest) -> MediaResult<SpeechStream> {
        Err(crate::media::MediaError::UnsupportedTask {
            provider: self.name().to_string(),
            model: request.model,
            task: MediaTask::TextToSpeech,
        })
    }
}

#[async_trait::async_trait]
pub trait TranscriptionProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_model(&self, model: &str) -> bool;
    fn capabilities(&self, model: &str) -> TranscriptionCapabilities;
    async fn submit_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> MediaResult<Operation<TranscriptionResult>>;
    async fn poll_transcription(
        &self,
        handle: &JobHandle,
    ) -> MediaResult<Operation<TranscriptionResult>>;
    async fn cancel_transcription(&self, handle: &JobHandle) -> MediaResult<()>;
}
