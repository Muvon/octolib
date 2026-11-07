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
//! key are skipped. Once every sample of a model missed at two delays in a row,
//! its longer delays are cancelled: the cache is gone and waiting adds nothing.

use std::collections::BTreeMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use octolib::llm::{ChatCompletionParams, Message, ProviderFactory, ReasoningEffort};
use tokio::sync::watch;
use tokio::task::JoinSet;

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
	("tinker:inkling", DEFAULT_PROMPT_TOKENS),
	("tinker:thinkingmachines/Inkling-2-Small", DEFAULT_PROMPT_TOKENS),
	(
		"tinker:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
		DEFAULT_PROMPT_TOKENS,
	),
	("isoquant:deepseek-v4.1-flash-1017-fast", DEFAULT_PROMPT_TOKENS),
	("isoquant:glm-5.3-flash", DEFAULT_PROMPT_TOKENS),
	("reflection:Beam-1.0-501B-A23B", 32_000),
	("inception:mercury-3", DEFAULT_PROMPT_TOKENS),
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
/// One delay where every sample missed can be routing; this many in a row mean the cache
/// is gone, so the model's longer delays are cancelled instead of waited out.
const STOP_AFTER_MISSED_DELAYS: usize = 2;

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
	options.delays.sort_unstable();
	options.delays.dedup();
	Ok(options)
}

enum Outcome {
	Hit,
	Miss,
	Cancelled,
	Failed(String),
}

impl Outcome {
	fn symbol(&self) -> char {
		match self {
			Outcome::Hit => 'H',
			Outcome::Miss => 'm',
			Outcome::Cancelled => '-',
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

async fn sample(
	spec: &'static str,
	prompt: String,
	start_after: Duration,
	delay: u64,
	long_cache: bool,
	mut stop: watch::Receiver<bool>,
) -> Outcome {
	tokio::select! {
		_ = tokio::time::sleep(start_after) => {}
		_ = stop.wait_for(|stopped| *stopped) => return Outcome::Cancelled,
	}
	if let Err(error) = cached_tokens(spec, &prompt, long_cache).await {
		return Outcome::Failed(format!("seed: {error}"));
	}
	tokio::select! {
		_ = tokio::time::sleep(Duration::from_secs(delay)) => {}
		_ = stop.wait_for(|stopped| *stopped) => return Outcome::Cancelled,
	}
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

/// True once every sample missed at `STOP_AFTER_MISSED_DELAYS` consecutive delays,
/// counting only from the shortest delay up to the first one still running.
fn cache_gone(by_delay: &BTreeMap<u64, Vec<Outcome>>, delays: &[u64], samples: usize) -> bool {
	let mut missed_in_a_row = 0;
	for delay in delays {
		let Some(outcomes) = by_delay.get(delay).filter(|outcomes| outcomes.len() == samples) else {
			return false;
		};
		if outcomes.iter().all(|outcome| matches!(outcome, Outcome::Miss)) {
			missed_in_a_row += 1;
			if missed_in_a_row == STOP_AFTER_MISSED_DELAYS {
				return true;
			}
		} else {
			missed_in_a_row = 0;
		}
	}
	false
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

	let stops: BTreeMap<&str, watch::Sender<bool>> = targets
		.iter()
		.map(|(spec, _, _)| (*spec, watch::channel(false).0))
		.collect();
	let mut tasks = JoinSet::new();
	let mut started: u32 = 0;
	for &(spec, tokens, _) in &targets {
		for &delay in &options.delays {
			for index in 0..options.samples {
				let prompt = probe_prompt(&format!("{run}-{spec}-{delay}-{index}"), tokens);
				let start_after = SEED_SPACING * started;
				let stop = stops[spec].subscribe();
				let long_cache = options.long_cache;
				tasks.spawn(async move {
					let outcome = sample(spec, prompt, start_after, delay, long_cache, stop).await;
					(spec, delay, outcome)
				});
				started += 1;
			}
		}
	}
	let longest = options.delays.last().copied().unwrap_or(0);
	println!(
		"up to {} calls across {} models; done in at most {}",
		started * 2,
		targets.len(),
		format_delay(longest + (SEED_SPACING * started).as_secs())
	);

	let mut results: BTreeMap<&str, BTreeMap<u64, Vec<Outcome>>> = BTreeMap::new();
	while let Some(joined) = tasks.join_next().await {
		let (spec, delay, outcome) = joined?;
		let by_delay = results.entry(spec).or_default();
		by_delay.entry(delay).or_default().push(outcome);
		if cache_gone(by_delay, &options.delays, options.samples) {
			stops[spec].send_replace(true);
		}
	}

	println!("\nH = cache hit, m = miss, E = error, - = cancelled after the cache was gone");
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
