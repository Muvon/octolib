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

use super::errors::{MediaError, MediaResult};
use super::providers::{FalMediaProvider, OpenRouterMediaProvider, ReplicateMediaProvider};
use super::traits::*;

pub struct MediaProviderFactory;

impl MediaProviderFactory {
    pub fn parse_model(value: &str) -> MediaResult<(String, String)> {
        let value = value.trim();
        let (provider, model) = value
            .split_once(':')
            .ok_or_else(|| MediaError::InvalidModelFormat(value.to_string()))?;
        let provider = provider.trim().to_ascii_lowercase();
        let model = model.trim().to_string();
        if provider.is_empty() || model.is_empty() {
            return Err(MediaError::InvalidModelFormat(value.to_string()));
        }
        Ok((provider, model))
    }

    pub fn create_image_provider(name: &str) -> MediaResult<Box<dyn ImageGenerationProvider>> {
        match name.to_ascii_lowercase().as_str() {
            "fal" => Ok(Box::new(FalMediaProvider::new())),
            "openrouter" => Ok(Box::new(OpenRouterMediaProvider::new())),
            "replicate" => Ok(Box::new(ReplicateMediaProvider::new())),
            other => Err(MediaError::UnsupportedProvider(other.to_string())),
        }
    }

    pub fn get_image_provider_for_model(
        value: &str,
    ) -> MediaResult<(Box<dyn ImageGenerationProvider>, String)> {
        let (provider_name, model) = Self::parse_model(value)?;
        let provider = Self::create_image_provider(&provider_name)?;
        if !provider.supports_model(&model) {
            return Err(MediaError::UnsupportedTask {
                provider: provider_name,
                model,
                task: super::types::MediaTask::TextToImage,
            });
        }
        Ok((provider, model))
    }

    pub fn create_video_provider(name: &str) -> MediaResult<Box<dyn VideoGenerationProvider>> {
        match name.to_ascii_lowercase().as_str() {
            "fal" => Ok(Box::new(FalMediaProvider::new())),
            "openrouter" => Ok(Box::new(OpenRouterMediaProvider::new())),
            "replicate" => Ok(Box::new(ReplicateMediaProvider::new())),
            other => Err(MediaError::UnsupportedProvider(other.to_string())),
        }
    }

    pub fn get_video_provider_for_model(
        value: &str,
    ) -> MediaResult<(Box<dyn VideoGenerationProvider>, String)> {
        let (provider_name, model) = Self::parse_model(value)?;
        let provider = Self::create_video_provider(&provider_name)?;
        if !provider.supports_model(&model) {
            return Err(MediaError::UnsupportedTask {
                provider: provider_name,
                model,
                task: super::types::MediaTask::TextToVideo,
            });
        }
        Ok((provider, model))
    }

    pub fn create_speech_provider(name: &str) -> MediaResult<Box<dyn SpeechSynthesisProvider>> {
        match name.to_ascii_lowercase().as_str() {
            "fal" => Ok(Box::new(FalMediaProvider::new())),
            "openrouter" => Ok(Box::new(OpenRouterMediaProvider::new())),
            "replicate" => Ok(Box::new(ReplicateMediaProvider::new())),
            other => Err(MediaError::UnsupportedProvider(other.to_string())),
        }
    }

    pub fn get_speech_provider_for_model(
        value: &str,
    ) -> MediaResult<(Box<dyn SpeechSynthesisProvider>, String)> {
        let (provider_name, model) = Self::parse_model(value)?;
        let provider = Self::create_speech_provider(&provider_name)?;
        if !provider.supports_model(&model) {
            return Err(MediaError::UnsupportedTask {
                provider: provider_name,
                model,
                task: super::types::MediaTask::TextToSpeech,
            });
        }
        Ok((provider, model))
    }

    pub fn create_transcription_provider(
        name: &str,
    ) -> MediaResult<Box<dyn TranscriptionProvider>> {
        match name.to_ascii_lowercase().as_str() {
            "fal" => Ok(Box::new(FalMediaProvider::new())),
            "openrouter" => Ok(Box::new(OpenRouterMediaProvider::new())),
            "replicate" => Ok(Box::new(ReplicateMediaProvider::new())),
            other => Err(MediaError::UnsupportedProvider(other.to_string())),
        }
    }

    pub fn get_transcription_provider_for_model(
        value: &str,
    ) -> MediaResult<(Box<dyn TranscriptionProvider>, String)> {
        let (provider_name, model) = Self::parse_model(value)?;
        let provider = Self::create_transcription_provider(&provider_name)?;
        if !provider.supports_model(&model) {
            return Err(MediaError::UnsupportedTask {
                provider: provider_name,
                model,
                task: super::types::MediaTask::SpeechToText,
            });
        }
        Ok((provider, model))
    }

    pub fn supported_providers() -> &'static [&'static str] {
        &["fal", "openrouter", "replicate"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_provider_model() {
        let (provider, model) =
            MediaProviderFactory::parse_model(" openrouter : google/veo-3.1 ").unwrap();
        assert_eq!(provider, "openrouter");
        assert_eq!(model, "google/veo-3.1");
    }

    #[test]
    fn rejects_missing_parts() {
        assert!(MediaProviderFactory::parse_model("model-only").is_err());
        assert!(MediaProviderFactory::parse_model("replicate:").is_err());
    }

    #[test]
    fn task_factories_return_the_native_model() {
        let (provider, model) =
            MediaProviderFactory::get_video_provider_for_model("replicate:google/veo-model")
                .unwrap();
        assert_eq!(provider.name(), "replicate");
        assert_eq!(model, "google/veo-model");
    }
}
