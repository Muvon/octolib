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

//! Speech, transcription, and video against OpenRouter's cheapest endpoints.
//! Run with `OPENROUTER_API_KEY=... cargo run --example media_openrouter_audio_video`.

use octolib::{
    generate_video, synthesize_speech, transcribe, ArtifactSource, MediaSource,
    SpeechSynthesisRequest, TranscriptionRequest, VideoGenerationRequest,
};

const SPOKEN_TEXT: &str = "Octolib multimodal check.";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let speech = synthesize_speech(
        "openrouter:hexgrad/kokoro-82m",
        SpeechSynthesisRequest::new(SPOKEN_TEXT, "af_bella"),
    )
    .await?;
    let ArtifactSource::Inline(audio) = speech.artifact.source.clone() else {
        anyhow::bail!("expected inline speech audio");
    };
    println!(
        "speech: {} bytes of {}, cost {:?} USD",
        audio.len(),
        speech.artifact.media_type,
        speech
            .usage
            .as_ref()
            .and_then(|usage| usage.best_available_cost())
    );

    // Feed the generated audio straight back in: the transcript should echo it.
    let transcription = transcribe(
        "openrouter:openai/whisper-large-v3-turbo",
        TranscriptionRequest::new(MediaSource::Bytes {
            data: audio,
            media_type: speech.artifact.media_type.clone(),
        }),
    )
    .await?;
    println!(
        "transcript: {:?}, cost {:?} USD",
        transcription.text.trim(),
        transcription
            .usage
            .as_ref()
            .and_then(|usage| usage.best_available_cost())
    );

    let mut video_request = VideoGenerationRequest::new("A brass octopus waving one tentacle");
    video_request.duration_secs = Some(4.0);
    let video = generate_video("openrouter:bytedance/seedance-2.0-mini", video_request).await?;
    for artifact in &video.artifacts {
        match &artifact.source {
            ArtifactSource::Inline(bytes) => {
                println!("video: {} bytes of {}", bytes.len(), artifact.media_type)
            }
            ArtifactSource::Url(url) => println!("video: {url}"),
            _ => println!("video: provider-managed artifact"),
        }
    }
    println!(
        "video cost {:?} USD",
        video
            .usage
            .as_ref()
            .and_then(|usage| usage.best_available_cost())
    );
    Ok(())
}
