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

//! # hindsight
//!
//! A small, calibrated, non-generative verifier for coding-agent sessions. It
//! reads the canonical event trace of a session ([`trace::Event`]) and
//! predicts, at a user-turn boundary, whether the user's next turn will be a
//! correction and of which class (not done, unrequested, rule violation, …).
//! Trained on hindsight labels — what users actually did next — never on an
//! agent's own claims.
//!
//! The weights are two ONNX graphs exported by the hindsight repo's
//! `gpu/export_onnx.py`: an event encoder (UniXCoder, mean-pooled) and the
//! level-2 core (a GRU over the last `window` event vectors with a readout
//! conditioned on the latest user request) with its calibrated heads. They
//! ship with the tokenizer and a `hindsight.json` describing the feature
//! layout, the window and the Platt parameters, under `onnx/` in the
//! Hugging Face repo or in any local directory.
//!
//! Everything runs on CPU through ONNX Runtime: roughly 30 ms to encode an
//! event (once, when it happens) and a few milliseconds to score a boundary.
//! No labelers, no network at inference.
//!
//! ```rust,no_run
//! # async fn example() -> anyhow::Result<()> {
//! use octolib::hindsight::{trace::Event, Hindsight, Session};
//! let model = std::sync::Arc::new(Hindsight::load("muvon/hindsight").await?);
//! let mut session = Session::new(model);
//! session.push(&Event::user_turn("fix the failing test"))?;
//! session.push(&Event::assistant("Done, the test passes now."))?;
//! let scores = session.score()?;
//! println!("p(next turn is a correction) = {:.2}", scores.p_correction);
//! # Ok(())
//! # }
//! ```

pub mod trace;

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::{Session as OrtSession, SessionInputValue};
use ort::value::Tensor;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use trace::{
    event_feat, event_text, Event, FEAT_DIM, HARNESS_EVENTS, KINDS, N_LABELS, OPS, STATUSES,
};

/// The files of one export, relative to its directory.
pub const FILES: [&str; 4] = [
    "hindsight.json",
    "tokenizer.json",
    "encoder.onnx",
    "level2.onnx",
];
/// Where the files live inside a Hugging Face model repo.
const HF_PREFIX: &str = "onnx";
/// `hindsight.json` layout this module reads.
const FORMAT: u32 = 1;
/// ORT threads per graph — enough to keep one encode under ~50 ms on a
/// laptop core without starving the agent process.
const THREADS: usize = 4;
/// Whole-file download attempts after a transient failure (each resumes).
const DOWNLOAD_RETRIES: usize = 3;
const RETRY_WAIT: std::time::Duration = std::time::Duration::from_secs(5);
/// How long to wait for another process's download of the same file before
/// giving up: attempts × interval ≈ 10 minutes, a 500 MB graph on a slow link.
const LOCK_WAIT_ATTEMPTS: usize = 40;
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(15);

/// `hindsight.json`: what the export needs the caller to know.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub format: u32,
    pub model: String,
    pub d_vec: usize,
    pub d_feat: usize,
    pub window: usize,
    pub max_len: usize,
    pub max_chars: usize,
    pub summary: bool,
    pub req: bool,
    pub sources: Vec<String>,
    /// Source-embedding slot for a harness outside the training sources.
    pub unknown_source: usize,
    pub kinds: Vec<String>,
    pub ops: Vec<String>,
    pub statuses: Vec<String>,
    pub harness_events: Vec<String>,
    pub n_labels: usize,
    /// Correction classes, in head order.
    pub groups: Vec<String>,
    /// Platt `[a, b]` per head: `corr` and `grp_<group>` for the groups that
    /// had enough calibration positives.
    pub platt: BTreeMap<String, [f32; 2]>,
}

/// The calibrated verdict at one boundary.
#[derive(Debug, Clone)]
pub struct Scores {
    pub logit_correction: f32,
    /// Probability that the next user turn is a correction of any class.
    pub p_correction: f32,
    /// Raw class-head logits, in `ModelConfig::groups` order.
    pub logits: Vec<f32>,
    /// Calibrated probability per class that has Platt parameters.
    pub classes: Vec<(String, f32)>,
}

impl Scores {
    pub fn class(&self, name: &str) -> Option<f32> {
        self.classes
            .iter()
            .find(|(g, _)| g == name)
            .map(|(_, p)| *p)
    }

    /// One-line rendering for logs: `p_corr=0.31 not_done=0.22 …`.
    pub fn summary(&self) -> String {
        let mut s = format!("p_corr={:.2}", self.p_correction);
        for (name, p) in &self.classes {
            s.push_str(&format!(" {name}={p:.2}"));
        }
        s
    }
}

/// A loaded model: two ORT sessions and the tokenizer. Share it behind an
/// `Arc`; every method takes `&self`.
pub struct Hindsight {
    config: ModelConfig,
    tokenizer: tokenizers::Tokenizer,
    // `Session::run` takes `&mut self`, so each graph is serialized.
    encoder: Mutex<OrtSession>,
    level2: Mutex<OrtSession>,
}

impl Hindsight {
    /// Load an export from a local directory or a Hugging Face repo
    /// (`org/repo`, files under `onnx/`; `org/repo#dir` for another prefix).
    /// A private repo needs `HF_TOKEN` in the environment.
    pub async fn load(spec: &str) -> Result<Self> {
        let spec = spec.trim();
        if Path::new(spec).is_dir() {
            return Self::load_dir(Path::new(spec));
        }
        let (repo_id, prefix) = spec
            .split_once('#')
            .map_or((spec, HF_PREFIX), |(r, p)| (r.trim(), p.trim()));
        let cache_dir =
            crate::storage::get_model_cache_dir().context("Failed to get model cache directory")?;
        std::env::set_var("HF_HOME", &cache_dir);
        let mut builder = hf_hub::api::tokio::ApiBuilder::new().with_progress(false);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            builder = builder.with_token(Some(token));
        }
        let api = builder
            .build()
            .context("Failed to initialize HuggingFace API")?;
        let repo = api.repo(hf_hub::Repo::new(
            repo_id.to_string(),
            hf_hub::RepoType::Model,
        ));
        let mut dir = None;
        for name in FILES {
            let file = format!("{prefix}/{name}");
            let (mut waits, mut retries) = (0, 0);
            let path = loop {
                match repo.get(&file).await {
                    Ok(path) => break path,
                    // Another process (a second agent session started at the
                    // same time) is downloading this file; hf-hub gives up on
                    // its lock after five seconds. Wait for it: when it is
                    // done the file is in the shared cache.
                    Err(hf_hub::api::tokio::ApiError::LockAcquisition(_))
                        if waits < LOCK_WAIT_ATTEMPTS =>
                    {
                        waits += 1;
                        tokio::time::sleep(LOCK_WAIT).await;
                    }
                    // The encoder graph is 500 MB and the tokio client does
                    // no retries of its own; a new attempt resumes from the
                    // partial file hf-hub keeps.
                    Err(error) if retries < DOWNLOAD_RETRIES => {
                        tracing::debug!("hindsight: download of {file} failed, retrying: {error}");
                        retries += 1;
                        tokio::time::sleep(RETRY_WAIT).await;
                    }
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("Failed to download {file} from {repo_id}"))
                    }
                }
            };
            dir = path.parent().map(Path::to_path_buf);
        }
        Self::load_dir(&dir.context("hf_hub returned no file path")?)
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join(FILES[0]);
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let config: ModelConfig = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if config.format != FORMAT {
            bail!(
                "hindsight export format {} is not the supported {FORMAT}",
                config.format
            );
        }
        let layout_ok = config.d_feat == FEAT_DIM
            && config.n_labels == N_LABELS
            && config.kinds == KINDS
            && config.ops == OPS
            && config.statuses == STATUSES
            && config.harness_events == HARNESS_EVENTS
            && config.window > 0
            && config.unknown_source == config.sources.len();
        if !layout_ok {
            bail!(
                "hindsight export {} has a feature layout this build does not render",
                config.model
            );
        }
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join(FILES[1]))
            .map_err(|e| anyhow!("Failed to load hindsight tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: config.max_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow!("Failed to configure hindsight tokenizer: {e}"))?;
        tokenizer.with_padding(None);
        let session = |name: &str| -> Result<OrtSession> {
            OrtSession::builder()
                .map_err(ort_error)?
                .with_intra_threads(THREADS)
                .map_err(ort_error)?
                .commit_from_file(dir.join(name))
                .map_err(ort_error)
                .with_context(|| format!("Failed to load hindsight graph {name}"))
        };
        Ok(Self {
            encoder: Mutex::new(session(FILES[2])?),
            level2: Mutex::new(session(FILES[3])?),
            config,
            tokenizer,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The event vector: tokenized at `max_len`, mean-pooled, L2-normalized.
    pub fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("hindsight tokenizer: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| i64::from(x)).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| i64::from(x))
            .collect();
        let n = ids.len();
        let inputs = vec![
            input(
                "input_ids",
                Tensor::from_array(([1usize, n], ids)).map_err(ort_error)?,
            ),
            input(
                "attention_mask",
                Tensor::from_array(([1usize, n], mask)).map_err(ort_error)?,
            ),
        ];
        let mut session = self
            .encoder
            .lock()
            .map_err(|e| anyhow!("hindsight encoder mutex poisoned: {e}"))?;
        let outputs = session.run(inputs).map_err(ort_error)?;
        let (_, data) = outputs
            .get("embedding")
            .context("hindsight encoder returned no embedding")?
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        Ok(data.to_vec())
    }

    /// Score the boundary after the last row of `vecs`/`feats` (one row per
    /// event, oldest first): the window is the last `window` rows, the task
    /// is the first event, `request` the index of the latest user turn.
    pub fn score(&self, vecs: &[Vec<f32>], feats: &[Vec<f32>], request: usize) -> Result<Scores> {
        let cfg = &self.config;
        let n = vecs.len();
        if n == 0 || feats.len() != n || request >= n {
            bail!(
                "hindsight score: {n} vectors, {} features, request {request}",
                feats.len()
            );
        }
        let (w, d, df) = (cfg.window, cfg.d_vec, cfg.d_feat);
        let start = n.saturating_sub(w);
        let len = n - start;
        let mut vec = vec![0f32; w * d];
        let mut feat = vec![0f32; w * df];
        for (row, k) in (start..n).enumerate() {
            if vecs[k].len() != d || feats[k].len() != df {
                bail!("hindsight score: row {k} has the wrong width");
            }
            vec[row * d..(row + 1) * d].copy_from_slice(&vecs[k]);
            feat[row * df..(row + 1) * df].copy_from_slice(&feats[k]);
        }
        let mut inputs = vec![
            input(
                "vec",
                Tensor::from_array(([1usize, w, d], vec)).map_err(ort_error)?,
            ),
            input(
                "feat",
                Tensor::from_array(([1usize, w, df], feat)).map_err(ort_error)?,
            ),
            input(
                "task",
                Tensor::from_array(([1usize, d], vecs[0].clone())).map_err(ort_error)?,
            ),
            input(
                "src",
                Tensor::from_array(([1usize], vec![cfg.unknown_source as i64]))
                    .map_err(ort_error)?,
            ),
            input(
                "lengths",
                Tensor::from_array(([1usize], vec![len as i64])).map_err(ort_error)?,
            ),
        ];
        if cfg.summary {
            // Mean of every event before the window; zeros when the window
            // starts at the session start.
            let mut summ = vec![0f32; d];
            if start > 0 {
                for v in &vecs[..start] {
                    for (s, x) in summ.iter_mut().zip(v) {
                        *s += x;
                    }
                }
                for s in &mut summ {
                    *s /= start as f32;
                }
            }
            inputs.push(input(
                "summ",
                Tensor::from_array(([1usize, d], summ)).map_err(ort_error)?,
            ));
        }
        if cfg.req {
            inputs.push(input(
                "req",
                Tensor::from_array(([1usize, d], vecs[request].clone())).map_err(ort_error)?,
            ));
        }
        let mut session = self
            .level2
            .lock()
            .map_err(|e| anyhow!("hindsight level2 mutex poisoned: {e}"))?;
        let outputs = session.run(inputs).map_err(ort_error)?;
        let (_, corr) = outputs
            .get("corr")
            .context("hindsight level2 returned no corr")?
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        let (_, grp) = outputs
            .get("grp")
            .context("hindsight level2 returned no grp")?
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        let logit_correction = corr[0];
        let logits = grp.to_vec();
        let platt = |head: &str, logit: f32| {
            cfg.platt
                .get(head)
                .map(|[a, b]| 1.0 / (1.0 + (-(a * logit + b)).exp()))
        };
        let p_correction = platt("corr", logit_correction)
            .context("hindsight export carries no Platt parameters for corr")?;
        let classes = cfg
            .groups
            .iter()
            .zip(&logits)
            .filter_map(|(g, &l)| platt(&format!("grp_{g}"), l).map(|p| (g.clone(), p)))
            .collect();
        Ok(Scores {
            logit_correction,
            p_correction,
            logits,
            classes,
        })
    }
}

fn input<'v>(
    name: &'static str,
    tensor: impl Into<SessionInputValue<'v>>,
) -> (Cow<'static, str>, SessionInputValue<'v>) {
    (Cow::Borrowed(name), tensor.into())
}

// ort's error types are generic over a recoverable payload that is not
// `Send`, so they are flattened to their message before entering anyhow.
fn ort_error(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("onnxruntime: {error}")
}

/// One session's trace as the model sees it: events are rendered, featurized
/// and encoded when pushed; `score` judges the boundary after the last one.
pub struct Session {
    model: Arc<Hindsight>,
    vecs: Vec<Vec<f32>>,
    feats: Vec<Vec<f32>>,
    /// Index of the latest genuine user turn — the request the readout
    /// conditions on.
    request: usize,
}

impl Session {
    pub fn new(model: Arc<Hindsight>) -> Self {
        Self {
            model,
            vecs: Vec::new(),
            feats: Vec::new(),
            request: 0,
        }
    }

    pub fn model(&self) -> &Arc<Hindsight> {
        &self.model
    }

    pub fn len(&self) -> usize {
        self.vecs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vecs.is_empty()
    }

    /// Append one event. The first event of a session must be a user turn —
    /// it is the task vector every boundary is scored against.
    pub fn push(&mut self, event: &Event) -> Result<()> {
        let k = self.vecs.len();
        if k == 0 && !matches!(event, Event::UserTurn { .. }) {
            bail!(
                "hindsight: a session starts with a user turn, got {}",
                event.kind()
            );
        }
        let is_last = matches!(event, Event::SessionEnd { .. });
        let text = event_text(event, self.model.config.max_chars);
        self.vecs.push(self.model.encode(&text)?);
        self.feats.push(event_feat(event, k, is_last));
        if matches!(event, Event::UserTurn { .. }) {
            self.request = k;
        }
        Ok(())
    }

    /// Score the boundary after the last pushed event.
    pub fn score(&self) -> Result<Scores> {
        self.model.score(&self.vecs, &self.feats, self.request)
    }
}
