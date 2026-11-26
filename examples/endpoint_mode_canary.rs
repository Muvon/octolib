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

//! Live smoke check for the `*_API_MODE` endpoint dispatch (needs the
//! provider's key, and costs a little). Everything goes through the public
//! API and the library's own provider code; the protocol is chosen by the
//! environment exactly as a user configures it.
//!
//! usage: `cargo run --example endpoint_mode_canary -- <provider:model> <text|image|image-url|tool|schema|json> [none|low|high|max]`

use octolib::llm::factory::ProviderFactory;
use octolib::llm::types::{
	ChatCompletionParams, FunctionDefinition, ImageAttachment, ImageData, Message, ReasoningEffort, SourceType,
	StructuredOutputRequest,
};

/// Served unchanged at [`RED_PNG_URL`] for the URL form of the image scenario.
/// The URL form is documented by both Messages contracts, but each route has
/// refused it live; the canary reports the outcome instead of asserting one.
const RED_PNG_URL: &str = "https://upload.wikimedia.org/wikipedia/commons/c/ca/1x1.png";

/// The image the `image` scenario asserts on: 96x96, three equal vertical bands,
/// left to right red, green, blue. Generated locally so no network asset can
/// change under the canary and the expected answer is exactly three words.
const STRIPE_PROMPT: &str =
	"This image is three vertical colour bands. Name the colour of each band, left to right, as three words separated by spaces.";

/// Build the striped PNG: no dependencies, one stored deflate block, CRCs and
/// the Adler-32 checksum computed here.
fn stripes_png() -> String {
	const SIZE: usize = 96;
	// One row is a filter byte (none) and SIZE RGB pixels; every row is equal.
	let mut row = Vec::with_capacity(1 + SIZE * 3);
	row.push(0_u8);
	for x in 0..SIZE {
		let band = match x {
			0..=31 => [255u8, 0, 0],
			32..=63 => [0, 255, 0],
			_ => [0, 0, 255],
		};
		row.extend_from_slice(&band);
	}
	let mut raw = Vec::with_capacity(SIZE * row.len());
	for _ in 0..SIZE {
		raw.extend_from_slice(&row);
	}
	let mut idat = vec![0x78, 0x01]; // zlib header: deflate, no preset dict
	let len = raw.len() as u16;
	idat.push(0x01); // final, stored
	idat.extend_from_slice(&len.to_le_bytes());
	idat.extend_from_slice(&(!len).to_le_bytes());
	idat.extend_from_slice(&raw);
	idat.extend_from_slice(&adler32(&raw).to_be_bytes());

	let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
	let mut ihdr = Vec::new();
	ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
	ihdr.extend_from_slice(&(SIZE as u32).to_be_bytes());
	ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, no interlace
	png.extend_from_slice(&png_chunk(b"IHDR", &ihdr));
	png.extend_from_slice(&png_chunk(b"IDAT", &idat));
	png.extend_from_slice(&png_chunk(b"IEND", &[]));
	base64(&png)
}

fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
	let mut chunk = Vec::with_capacity(data.len() + 12);
	chunk.extend_from_slice(&(data.len() as u32).to_be_bytes());
	chunk.extend_from_slice(kind);
	chunk.extend_from_slice(data);
	let mut crc_input = Vec::with_capacity(kind.len() + data.len());
	crc_input.extend_from_slice(kind);
	crc_input.extend_from_slice(data);
	chunk.extend_from_slice(&crc32(&crc_input).to_be_bytes());
	chunk
}

fn crc32(bytes: &[u8]) -> u32 {
	let mut crc = 0xFFFF_FFFFu32;
	for &byte in bytes {
		crc ^= byte as u32;
		for _ in 0..8 {
			let mask = (crc & 1).wrapping_neg();
			crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
		}
	}
	!crc
}

fn adler32(bytes: &[u8]) -> u32 {
	let (mut a, mut b) = (1u32, 0u32);
	for &byte in bytes {
		a = (a + byte as u32) % 65_521;
		b = (b + a) % 65_521;
	}
	(b << 16) | a
}

fn base64(bytes: &[u8]) -> String {
	const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
	for group in bytes.chunks(3) {
		let b = [group[0], *group.get(1).unwrap_or(&0), *group.get(2).unwrap_or(&0)];
		let packed = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
		out.push(ALPHABET[(packed >> 18) as usize & 63] as char);
		out.push(ALPHABET[(packed >> 12) as usize & 63] as char);
		out.push(if group.len() > 1 {
			ALPHABET[(packed >> 6) as usize & 63] as char
		} else {
			'='
		});
		out.push(if group.len() > 2 {
			ALPHABET[packed as usize & 63] as char
		} else {
			'='
		});
	}
	out
}

/// The answer must name the three bands in the order they appear. A 200 with
/// an unrelated word would otherwise pass, which is exactly the failure this
/// canary exists to catch.
fn verify_stripes(content: &str) -> Result<(), String> {
	let answer = content.to_lowercase();
	let position = |needle: &str| answer.find(needle);
	let (red, green, blue) = match (position("red"), position("green"), position("blue")) {
		(Some(red), Some(green), Some(blue)) => (red, green, blue),
		_ => return Err(format!("expected red, green and blue; got {content:?}")),
	};
	if red < green && green < blue {
		Ok(())
	} else {
		Err(format!("bands out of order in {content:?}"))
	}
}

#[tokio::main]
async fn main() {
	let args: Vec<String> = std::env::args().skip(1).collect();
	if args.len() < 2 {
		println!(
			"usage: endpoint_mode_canary <provider:model> <text|image|image-url|tool|schema|json> [none|low|high|max]"
		);
		std::process::exit(2);
	}
	let (model, scenario) = (&args[0], args[1].as_str());
	let effort = args.get(2).map(|raw| match raw.as_str() {
		"none" => ReasoningEffort::None,
		"low" => ReasoningEffort::Low,
		"high" => ReasoningEffort::High,
		_ => ReasoningEffort::Max,
	});

	let (provider, model_name) = match ProviderFactory::get_provider_for_model(model) {
		Ok(pair) => pair,
		Err(error) => {
			println!("CANARY ERR {model} | provider: {error:#}");
			std::process::exit(1);
		}
	};
	let mut params =
		ChatCompletionParams::new(&[Message::user("Reply with exactly the word pong.")], &model_name, 2048)
			.with_max_retries(0);
	match scenario {
		// `image` sends the striped PNG inline and asserts the reading;
		// `image-url` sends a 1x1 by URL, which is reported, not asserted.
		"image" | "image-url" => {
			let (data, prompt) = if scenario == "image-url" {
				(
					ImageData::Url(RED_PNG_URL.to_owned()),
					"What color is this image? Answer with one word.",
				)
			} else {
				(ImageData::Base64(stripes_png()), STRIPE_PROMPT)
			};
			let image = Message::user(prompt).with_images(vec![ImageAttachment {
				data,
				media_type: "image/png".to_owned(),
				source_type: SourceType::Url,
				dimensions: None,
				size_bytes: None,
			}]);
			params = ChatCompletionParams::new(&[image], &model_name, 2048).with_max_retries(0);
		}
		"tool" => {
			params = ChatCompletionParams::new(
				&[Message::user("What is the weather in Oslo right now? Call the tool.")],
				&model_name,
				2048,
			)
			.with_max_retries(0)
			.with_tools(vec![FunctionDefinition {
				name: "get_weather".to_owned(),
				description: "Current weather for a city".to_owned(),
				parameters: serde_json::json!({
					"type": "object",
					"properties": { "city": { "type": "string" } },
					"required": ["city"]
				}),
				cache_control: None,
				freeform: None,
			}]);
		}
		"schema" => {
			params = ChatCompletionParams::new(
				&[Message::user("Reply ONLY with a JSON array of three color names.")],
				&model_name,
				2048,
			)
			.with_max_retries(0)
			.with_structured_output(StructuredOutputRequest::json_schema(serde_json::json!({
					"type": "object",
					"properties": { "colors": { "type": "array", "items": { "type": "string" } } },
					"required": ["colors"],
				"additionalProperties": false
			})));
		}
		// A bare JSON-object request: the answer must be an object, not prose.
		"json" => {
			params = ChatCompletionParams::new(
				&[Message::user(
					"Reply ONLY with a JSON object that has a single key `ok`.",
				)],
				&model_name,
				2048,
			)
			.with_max_retries(0)
			.with_structured_output(StructuredOutputRequest::json());
		}
		_ => {}
	}
	if let Some(effort) = effort {
		params = params.with_reasoning_effort(effort);
	}

	let label = format!(
		"{model} {scenario} effort={:?} mode={:?}",
		effort,
		std::env::var(format!("{}_API_MODE", provider.name().to_uppercase())).ok()
	);
	let mut failed = false;
	match provider.chat_completion(params).await {
		Ok(response) => {
			let usage = response.exchange.usage;
			let tools = response.tool_calls.clone().map(|calls| {
				calls
					.iter()
					.map(|call| format!("{}{}", call.name, call.arguments))
					.collect::<Vec<_>>()
					.join(",")
			});
			println!(
				"CANARY ok {label} | content={:?} | thinking={} chars | tools={:?} | in={} cache_read={} out={} reasoning={} cost={:?}",
				response.content.chars().take(160).collect::<String>(),
				response.thinking.as_ref().map_or(0, |thinking| thinking.content.chars().count()),
				tools,
				usage.as_ref().map_or(0, |usage| usage.input_tokens),
				usage.as_ref().map_or(0, |usage| usage.cache_read_tokens),
				usage.as_ref().map_or(0, |usage| usage.output_tokens),
				usage.as_ref().map_or(0, |usage| usage.reasoning_tokens),
				usage.as_ref().and_then(|usage| usage.cost),
			);
			if scenario == "image" {
				match verify_stripes(&response.content) {
					Ok(()) => println!("CANARY assert ok {label} | red < green < blue"),
					Err(why) => {
						println!("CANARY FAIL {label} | {why}");
						failed = true;
					}
				}
			}
		}
		Err(error) => {
			println!("CANARY ERR {label} | {error:#}");
			failed = true;
		}
	}
	if failed {
		std::process::exit(1);
	}
}
