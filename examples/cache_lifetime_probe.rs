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

//! Measures how long each provider keeps an unused prompt cache.
//!
//! For every model and delay it sends a unique prompt once, waits the delay
//! with no other traffic on that prompt, sends it again and checks whether the
//! second call read from the cache. Requests go through octolib's own provider
//! code, so endpoints, cache markers and thinking settings match real calls.
//! The result table is the evidence `AiProvider::cache_lifetime` is set from;
//! see "Cache lifetime: measure it live" in AGENTS.md.
//!
//! Usage:
//! ```bash
//! # Every listed model whose provider key is set (about an hour):
//! cargo run --example cache_lifetime_probe
//! # Only models whose `provider:model` contains a filter:
//! cargo run --example cache_lifetime_probe -- alibaba kimi-k3
//! # Narrow a boundary:
//! cargo run --example cache_lifetime_probe -- alibaba:qwen3.7-plus --delays 30,45,60,75,90
//! ```
//!
//! Options: `--delays <seconds,...>`, `--samples <n>` (per delay),
//! `--prompt-tokens <n>` (overrides every model's prompt size), `--long-cache`
//! (asks providers with a long TTL for it). Models whose provider has no API
//! key are skipped.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory, ReasoningEffort};

/// Every model measured so far, with the prompt size its cache needs.
/// Gemini caches nothing under 4,096 tokens; Beam missed at ~5k and hit at ~36k.
const TARGETS: &[(&str, usize)] = &[
	("zai:glm-5.3", DEFAULT_PROMPT_TOKENS),
	("xiaomi:mimo-v2.6-pro", DEFAULT_PROMPT_TOKENS),
	("openai:gpt-6-luna", DEFAULT_PROMPT_TOKENS),
	("moonshot:kimi-k3.1-pro", DEFAULT_PROMPT_TOKENS),
	("moonshot:kimi-k3", DEFAULT_PROMPT_TOKENS),
	("moonshot:kimi-k2.7-code", DEFAULT_PROMPT_TOKENS),
	("moonshot:kimi-k2.6", DEFAULT_PROMPT_TOKENS),
	("alibaba:glm-5.3", DEFAULT_PROMPT_TOKENS),
	("alibaba:deepseek-v4.1-flash", DEFAULT_PROMPT_TOKENS),
	("alibaba:qwen3.8-flash", DEFAULT_PROMPT_TOKENS),
	("alibaba:qwen3.8-max", DEFAULT_PROMPT_TOKENS),
	("alibaba:qwen3.7-plus", DEFAULT_PROMPT_TOKENS),
	("google-studio:gemini-4-argon", DEFAULT_PROMPT_TOKENS),
	("google-studio:gemini-3.8-flash", DEFAULT_PROMPT_TOKENS),
	("google-studio:gemini-3.6-flash", DEFAULT_PROMPT_TOKENS),
	("google-studio:gemini-3.1-pro-preview", DEFAULT_PROMPT_TOKENS),
	("tinker:openai/gpt-oss-120b", DEFAULT_PROMPT_TOKENS),
	("tinker:openai/gpt-oss-20b", DEFAULT_PROMPT_TOKENS),
	("isoquant:deepseek-v4.1-flash-1017-fast", DEFAULT_PROMPT_TOKENS),
	("isoquant:glm-5.3-flash", DEFAULT_PROMPT_TOKENS),
	("reflection:Beam-1.0-501B-A23B", 32_000),
];

const DEFAULT_PROMPT_TOKENS: usize = 8_000;
const DEFAULT_DELAYS_SECS: &[u64] = &[60, 180, 300, 420, 600, 900, 1200, 1800, 2700, 3600];
const DEFAULT_SAMPLES: usize = 2;
/// Rough size of one generated prompt line, used to reach a token target.
const TOKENS_PER_LINE: usize = 12;
/// Room for a short answer plus the reasoning that always-thinking models spend.
const MAX_OUTPUT_TOKENS: u32 = 256;
const MAX_RETRIES: u32 = 2;
/// Seeds start this far apart so the first wave does not hit rate limits.
const SEED_SPACING: Duration = Duration::from_millis(500);

struct Options {
	filters: Vec<String>,
	delays: Vec<u64>,
	samples: usize,
	prompt_tokens: Option<usize>,
	long_cache: bool,
}

fn parse_options() -> anyhow::Result<Options> {
	let mut options = Options {
		filters: Vec::new(),
		delays: DEFAULT_DELAYS_SECS.to_vec(),
		samples: DEFAULT_SAMPLES,
		prompt_tokens: None,
		long_cache: false,
	};
	let mut args = std::env::args().skip(1);
	while let Some(arg) = args.next() {
		let mut value = || args.next().ok_or_else(|| anyhow::anyhow!("{arg} needs a value"));
		match arg.as_str() {
			"--delays" => {
				options.delays = value()?
					.split(',')
					.map(|delay| delay.trim().parse())
					.collect::<Result<_, _>>()?
			}
			"--samples" => options.samples = value()?.parse()?,
			"--prompt-tokens" => options.prompt_tokens = Some(value()?.parse()?),
			"--long-cache" => options.long_cache = true,
			flag if flag.starts_with("--") => anyhow::bail!("unknown option {flag}"),
			filter => options.filters.push(filter.to_lowercase()),
		}
	}
	Ok(options)
}

enum Outcome {
	Hit,
	Miss,
	Failed(String),
}

impl Outcome {
	fn symbol(&self) -> char {
		match self {
			Outcome::Hit => 'H',
			Outcome::Miss => 'm',
			Outcome::Failed(_) => 'E',
		}
	}
}

/// Sends the prompt and returns how many prompt tokens were read from the cache.
async fn cached_tokens(spec: &str, prompt: &str, long_cache: bool) -> anyhow::Result<u64> {
	let (provider, model) = ProviderFactory::get_provider_for_model(spec)?;
	let messages = vec![Message::system(prompt).with_cache_marker(), Message::user("Say ok.")];
	let params = ChatCompletionParams::new(&messages, &model, 0.7, 1.0, 50, MAX_OUTPUT_TOKENS)
		.with_reasoning_effort(ReasoningEffort::Low)
		.with_long_cache(long_cache)
		.with_max_retries(MAX_RETRIES);
	let response = provider.chat_completion(params).await?;
	Ok(response.exchange.usage.map_or(0, |usage| usage.cache_read_tokens))
}

/// The run stamp makes every prompt unique, so no earlier request can have cached it.
fn probe_prompt(stamp: &str, tokens: usize) -> String {
	let mut prompt = format!("Probe {stamp}.");
	for line in 0..tokens / TOKENS_PER_LINE {
		prompt.push_str(&format!(" Item {line} of list {stamp} is number {}.", line * 7));
	}
	prompt
}

async fn sample(spec: &'static str, prompt: String, start_after: Duration, delay: u64, long_cache: bool) -> Outcome {
	tokio::time::sleep(start_after).await;
	if let Err(error) = cached_tokens(spec, &prompt, long_cache).await {
		return Outcome::Failed(format!("seed: {error}"));
	}
	tokio::time::sleep(Duration::from_secs(delay)).await;
	match cached_tokens(spec, &prompt, long_cache).await {
		Ok(0) => Outcome::Miss,
		Ok(_) => Outcome::Hit,
		Err(error) => Outcome::Failed(format!("probe: {error}")),
	}
}

fn format_delay(secs: u64) -> String {
	if secs.is_multiple_of(60) {
		format!("{}m", secs / 60)
	} else {
		format!("{secs}s")
	}
}

/// The longest delay up to which every sample at every delay hit.
fn held_up_to(by_delay: &BTreeMap<u64, Vec<Outcome>>) -> Option<u64> {
	by_delay
		.iter()
		.take_while(|(_, outcomes)| outcomes.iter().all(|outcome| matches!(outcome, Outcome::Hit)))
		.map(|(delay, _)| *delay)
		.last()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	let options = parse_options()?;
	let run = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();

	let mut targets = Vec::new();
	for &(spec, default_tokens) in TARGETS {
		let lowered = spec.to_lowercase();
		if !options.filters.is_empty() && !options.filters.iter().any(|filter| lowered.contains(filter)) {
			continue;
		}
		let (provider, model) = ProviderFactory::get_provider_for_model(spec)?;
		if provider.get_api_key().is_err() {
			println!("skip {spec}: no API key");
			continue;
		}
		let current = provider.cache_lifetime(&model, options.long_cache);
		targets.push((spec, options.prompt_tokens.unwrap_or(default_tokens), current));
	}
	if targets.is_empty() {
		anyhow::bail!("no model matches the filters and has an API key");
	}

	let mut tasks = Vec::new();
	for &(spec, tokens, _) in &targets {
		for &delay in &options.delays {
			for index in 0..options.samples {
				let prompt = probe_prompt(&format!("{run}-{spec}-{delay}-{index}"), tokens);
				let start_after = SEED_SPACING * tasks.len() as u32;
				let task = tokio::spawn(sample(spec, prompt, start_after, delay, options.long_cache));
				tasks.push((spec, delay, task));
			}
		}
	}
	let longest = options.delays.iter().max().copied().unwrap_or(0);
	println!(
		"{} calls across {} models; done in about {}",
		tasks.len() * 2,
		targets.len(),
		format_delay(longest + (SEED_SPACING * tasks.len() as u32).as_secs())
	);

	let mut results: BTreeMap<&str, BTreeMap<u64, Vec<Outcome>>> = BTreeMap::new();
	for (spec, delay, task) in tasks {
		let outcome = task.await.unwrap_or_else(|error| Outcome::Failed(error.to_string()));
		results.entry(spec).or_default().entry(delay).or_default().push(outcome);
	}

	println!("\nH = cache hit, m = miss, E = error; one letter per sample");
	for (spec, _, current) in &targets {
		let by_delay = &results[spec];
		let cells: Vec<String> = by_delay
			.iter()
			.map(|(delay, outcomes)| {
				let symbols: String = outcomes.iter().map(Outcome::symbol).collect();
				format!("{}:{symbols}", format_delay(*delay))
			})
			.collect();
		let held = held_up_to(by_delay).map_or("never".to_string(), format_delay);
		let current = current.map_or("None".to_string(), |lifetime| format_delay(lifetime.as_secs()));
		println!(
			"{spec}\n  {}\n  every sample hit up to: {held}; cache_lifetime now: {current}",
			cells.join(" ")
		);
		for outcome in by_delay.values().flatten() {
			if let Outcome::Failed(error) = outcome {
				println!("  error: {error}");
			}
		}
	}
	Ok(())
}
