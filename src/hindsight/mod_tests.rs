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

//! Conformance against the Python reference. `golden.json` is written by the
//! hindsight repo's `gpu/export_onnx.py --golden`: two synthetic sessions with
//! the event texts, feature vectors and per-boundary logits the reference
//! computes (label slots zero, unknown source). The rendering tests run
//! everywhere; the logit test needs the export on disk
//! (`HINDSIGHT_MODEL_DIR=<dir> cargo test -- --ignored hindsight`).

use super::trace::{
    clip, event_feat, event_text, op_for, parse_result_text, py_json, Event, Op, Tests, FEAT_DIM,
};
use super::{Hindsight, Session};
use serde::Deserialize;
use std::sync::Arc;

const GOLDEN: &str = include_str!("golden.json");

#[derive(Deserialize)]
struct Fixture {
    config: FixtureConfig,
    sessions: Vec<FixtureSession>,
}

#[derive(Deserialize)]
struct FixtureConfig {
    max_chars: usize,
}

#[derive(Deserialize)]
struct FixtureSession {
    events: Vec<Event>,
    texts: Vec<String>,
    feats: Vec<Vec<f32>>,
    boundaries: Vec<Boundary>,
}

#[derive(Deserialize)]
struct Boundary {
    t: usize,
    logit_corr: f32,
    logits_grp: Vec<f32>,
    p_corr: f32,
}

fn fixture() -> Fixture {
    serde_json::from_str(GOLDEN).expect("golden.json parses")
}

#[test]
fn event_text_matches_reference() {
    let fx = fixture();
    for session in &fx.sessions {
        assert_eq!(session.events.len(), session.texts.len());
        for (k, (event, expected)) in session.events.iter().zip(&session.texts).enumerate() {
            assert_eq!(
                &event_text(event, fx.config.max_chars),
                expected,
                "event {k}"
            );
        }
    }
}

#[test]
fn event_feat_matches_reference() {
    let fx = fixture();
    for session in &fx.sessions {
        let n = session.events.len();
        for (k, (event, expected)) in session.events.iter().zip(&session.feats).enumerate() {
            let got = event_feat(event, k, k == n - 1);
            assert_eq!(got.len(), FEAT_DIM);
            assert_eq!(expected.len(), FEAT_DIM);
            for (i, (g, e)) in got.iter().zip(expected).enumerate() {
                assert!((g - e).abs() <= 1e-4, "event {k} feature {i}: {g} vs {e}");
            }
        }
    }
}

#[test]
fn canonical_events_round_trip_through_serde() {
    let fx = fixture();
    for session in &fx.sessions {
        for event in &session.events {
            let json = serde_json::to_string(event).unwrap();
            let back: Event = serde_json::from_str(&json).unwrap();
            assert_eq!(event_text(&back, 1500), event_text(event, 1500));
        }
    }
}

#[test]
fn clip_keeps_a_third_then_two_thirds_in_chars() {
    assert_eq!(clip("abc", 3), "abc");
    assert_eq!(clip("abcdefghij", 6), "ab\n...[clipped]...\nghij");
    // Unicode scalar values, not bytes: ten two-byte chars fit in ten.
    assert_eq!(clip("ééééééééééé", 10), "ééé\n...[clipped]...\nééééééé");
}

#[test]
fn py_json_renders_like_python() {
    let v = serde_json::json!({"cmd": "ls -la", "n": 2, "ok": true, "path": null, "text": "ünï \"q\" \\ \n ✓ 😀", "list": [1, "a"]});
    // Python escapes every non-ASCII char as \u + four hex digits (surrogate
    // pairs above the BMP); spelled in pieces so no tool or editor decodes them.
    let expected = concat!(
        r#"{"cmd": "ls -la", "list": [1, "a"], "n": 2, "ok": true, "path": null, "text": ""#,
        "\\u",
        "00fcn",
        "\\u",
        "00ef",
        r#" \"q\" \\ \n "#,
        "\\u",
        "2713 ",
        "\\u",
        "d83d",
        "\\u",
        "de00",
        r#""}"#
    );
    assert_eq!(py_json(&v), expected);
}

#[test]
fn parse_result_text_matches_reference_regexes() {
    let p = parse_result_text("Traceback (most recent call last):\n  File x\nValueError: bad\nScript completed successfully\n2 passed 1 failed 3 errors");
    assert_eq!(
        p.tests,
        Some(Tests {
            passed: 2,
            failed: 1,
            errors: 3
        })
    );
    assert_eq!(p.traceback, Some(true));
    assert_eq!(p.exception.as_deref(), Some("ValueError"));
    assert_eq!(p.script_ok, Some(true));
    let p = parse_result_text("12 passed, 1 error in 0.3s\nKeyErrorX: no\nTimeoutException\n");
    assert_eq!(
        p.tests,
        Some(Tests {
            passed: 12,
            failed: 0,
            errors: 1
        })
    );
    assert_eq!(p.exception.as_deref(), Some("TimeoutException"));
    assert_eq!(p.traceback, None);
    let p = parse_result_text("all good");
    assert_eq!(p.tests, None);
    assert_eq!(p.exception, None);
}

#[test]
fn op_for_follows_tool_name_and_command() {
    assert_eq!(op_for("Read", None), Op::Read);
    assert_eq!(op_for("Bash", Some("git status")), Op::Git);
    assert_eq!(op_for("Bash", Some(" gh pr view 1")), Op::Git);
    assert_eq!(op_for("shell", Some("curl https://x")), Op::External);
    assert_eq!(op_for("shell", Some("aws s3 ls")), Op::External);
    assert_eq!(op_for("shell", Some("cargo test")), Op::Run);
    assert_eq!(op_for("shell", Some("echo gitx")), Op::Run);
    assert_eq!(op_for("mcp__octofs__view", None), Op::External);
    assert_eq!(op_for("unknown_tool", None), Op::Other);
}

#[test]
fn tool_call_constructor_clips_and_stringifies_args() {
    let args = serde_json::json!({"command": "cargo test", "n": 3});
    let Event::ToolCall { op, args, .. } = Event::tool_call("shell", &args, None) else {
        panic!("tool call")
    };
    assert_eq!(op, Op::Run);
    assert_eq!(args["n"], serde_json::json!("3"));
}

/// Needs the export: `HINDSIGHT_MODEL_DIR=<dir> cargo test -- --ignored hindsight`.
#[test]
#[ignore]
fn golden_logits_match_reference() {
    let dir = std::env::var("HINDSIGHT_MODEL_DIR").expect("HINDSIGHT_MODEL_DIR");
    let model = Arc::new(Hindsight::load_dir(std::path::Path::new(&dir)).unwrap());
    let fx = fixture();
    let mut worst = 0f32;
    for session in &fx.sessions {
        let mut live = Session::new(model.clone());
        let mut boundaries = session.boundaries.iter().peekable();
        for (k, event) in session.events.iter().enumerate() {
            if let Some(b) = boundaries.peek() {
                if b.t == k {
                    let scores = live.score().unwrap();
                    worst = worst
                        .max((scores.logit_correction - b.logit_corr).abs())
                        .max((scores.p_correction - b.p_corr).abs());
                    for (g, e) in scores.logits.iter().zip(&b.logits_grp) {
                        worst = worst.max((g - e).abs());
                    }
                    boundaries.next();
                }
            }
            live.push(event).unwrap();
        }
        assert!(boundaries.next().is_none(), "every boundary scored");
    }
    assert!(worst < 2e-3, "worst |delta| {worst}");
}
