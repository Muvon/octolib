// Copyright 2025 Muvon Un Limited
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

//! Example demonstrating Moonshot Kimi K2.5 vision capabilities
//!
//! This example shows how to use the Kimi K2.5 model with images and videos.
//!
//! Supported formats:
//! - Images: png, jpeg, webp, gif
//! - Videos: mp4, mpeg, mov, avi, x-flv, mpg, webm, wmv, 3gpp
//!
//! Usage:
//! ```bash
//! export MOONSHOT_API_KEY="your_key"
//! cargo run --example moonshot_vision
//! ```

use octolib::llm::{
    ChatCompletionParams, ImageAttachment, ImageData, Message, ProviderFactory, SourceType,
    VideoAttachment, VideoData,
};
use std::fs;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Example 1: Image analysis
    println!("=== Example 1: Image Analysis ===\n");

    // Read and encode image
    let image_path = "examples/test_image.png";
    if let Ok(image_data) = fs::read(image_path) {
        let base64_image =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_data);

        let image_attachment = ImageAttachment {
            data: ImageData::Base64(base64_image),
            media_type: "image/png".to_string(),
            source_type: SourceType::File(PathBuf::from(image_path)),
            dimensions: None,
            size_bytes: Some(image_data.len() as u64),
        };

        let message = Message::user("Please describe what you see in this image.")
            .with_images(vec![image_attachment]);

        let (provider, model) = ProviderFactory::get_provider_for_model("moonshot:kimi-k2.5")?;

        let params = ChatCompletionParams::new(&[message], &model, 0.7, 1.0, 50, 4000);

        let response = provider.chat_completion(params).await?;

        println!("Response: {}", response.content);
        if let Some(usage) = &response.exchange.usage {
            println!(
                "Tokens: {} input, {} output",
                usage.input_tokens, usage.output_tokens
            );
            if let Some(cost) = usage.cost {
                println!("Cost: ${:.6}", cost);
            }
        }
    } else {
        println!("Note: Create examples/test_image.png to test image analysis");
    }

    // Example 2: Video analysis
    println!("\n=== Example 2: Video Analysis ===\n");

    let video_path = "examples/test_video.mp4";
    if let Ok(video_data) = fs::read(video_path) {
        let base64_video =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &video_data);

        let video_attachment = VideoAttachment {
            data: VideoData::Base64(base64_video),
            media_type: "video/mp4".to_string(),
            source_type: SourceType::File(PathBuf::from(video_path)),
            dimensions: None,
            size_bytes: Some(video_data.len() as u64),
            duration_secs: None,
        };

        let message = Message::user("Please describe the content of this video.")
            .with_videos(vec![video_attachment]);

        let (provider, model) = ProviderFactory::get_provider_for_model("moonshot:kimi-k2.5")?;

        let params = ChatCompletionParams::new(&[message], &model, 0.7, 1.0, 50, 4000);

        let response = provider.chat_completion(params).await?;

        println!("Response: {}", response.content);
        if let Some(usage) = &response.exchange.usage {
            println!(
                "Tokens: {} input, {} output",
                usage.input_tokens, usage.output_tokens
            );
            if let Some(cost) = usage.cost {
                println!("Cost: ${:.6}", cost);
            }
        }
    } else {
        println!("Note: Create examples/test_video.mp4 to test video analysis");
    }

    // Example 3: Multiple images and videos
    println!("\n=== Example 3: Multiple Media Files ===\n");

    #[allow(unused_assignments)]
    let mut attachments_message =
        Message::user("Compare these media files and describe what you see.");

    // Add multiple images if available
    if let Ok(image_data) = fs::read("examples/image1.png") {
        let base64_image =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_data);
        attachments_message = attachments_message.with_images(vec![ImageAttachment {
            data: ImageData::Base64(base64_image),
            media_type: "image/png".to_string(),
            source_type: SourceType::File(PathBuf::from("examples/image1.png")),
            dimensions: None,
            size_bytes: Some(image_data.len() as u64),
        }]);
    }

    // Add video if available
    if let Ok(video_data) = fs::read("examples/video1.mp4") {
        let base64_video =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &video_data);
        #[allow(unused_assignments)]
        {
            attachments_message = attachments_message.with_videos(vec![VideoAttachment {
                data: VideoData::Base64(base64_video),
                media_type: "video/mp4".to_string(),
                source_type: SourceType::File(PathBuf::from("examples/video1.mp4")),
                dimensions: None,
                size_bytes: Some(video_data.len() as u64),
                duration_secs: None,
            }]);
        }
    }

    println!("Note: Create examples/image1.png and examples/video1.mp4 to test multiple media");

    // Example 4: Using MessageBuilder
    println!("\n=== Example 4: Using MessageBuilder ===\n");

    let message = Message::builder()
        .role("user")
        .content("Analyze this image")
        .with_image(ImageAttachment {
            data: ImageData::Base64("...base64_data...".to_string()),
            media_type: "image/jpeg".to_string(),
            source_type: SourceType::Clipboard,
            dimensions: Some((1920, 1080)),
            size_bytes: Some(150000),
        })
        .with_video(VideoAttachment {
            data: VideoData::Base64("...base64_data...".to_string()),
            media_type: "video/mp4".to_string(),
            source_type: SourceType::File(PathBuf::from("video.mp4")),
            dimensions: Some((1920, 1080)),
            size_bytes: Some(5000000),
            duration_secs: Some(30.5),
        })
        .build()?;

    println!("Message created with builder pattern:");
    println!("- Role: {}", message.role);
    println!("- Content: {}", message.content);
    println!(
        "- Images: {}",
        message.images.as_ref().map(|i| i.len()).unwrap_or(0)
    );
    println!(
        "- Videos: {}",
        message.videos.as_ref().map(|v| v.len()).unwrap_or(0)
    );

    Ok(())
}
