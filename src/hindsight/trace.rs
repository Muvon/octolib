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

//! Canonical event trace of a coding-agent session: the input contract of the
//! hindsight model, ported from the reference `schema.py` and
//! `gpu/prepare_events.py` of the hindsight repo. The text rendered for the
//! encoder and the structural feature vector must match the Python reference
//! exactly — the model was trained on that rendering — and the conformance
//! fixture in `mod_tests.rs` pins both.
//!
//! Lengths are in Unicode scalar values (Python `len`), never bytes.

use serde::{Deserialize, Serialize};

pub const KINDS: [&str; 8] = [
    "user_turn",
    "assistant",
    "thinking",
    "tool_call",
    "tool_result",
    "diff",
    "harness",
    "session_end",
];
pub const OPS: [&str; 8] = [
    "read", "edit", "write", "run", "search", "git", "external", "other",
];
pub const STATUSES: [&str; 4] = ["ok", "error", "denied", "interrupted"];
pub const HARNESS_EVENTS: [&str; 7] = [
    "interrupt",
    "denial",
    "pivot",
    "gate_verdict",
    "permission_prompt",
    "compaction",
    "steer",
];
/// Tier-1 labeler slots at the tail of the feature vector. The deployed model
/// is trained with them zero, so the port never fills them; they stay in the
/// layout because the graph input has this width.
pub const N_LABELS: usize = 30;
const NUMERIC_FEATS: usize = 12;
pub const FEAT_DIM: usize =
    KINDS.len() + OPS.len() + STATUSES.len() + HARNESS_EVENTS.len() + NUMERIC_FEATS + N_LABELS;
/// Text a normalizer keeps per user/assistant/tool text (`schema.MAX_TEXT`).
pub const MAX_TEXT: usize = 4000;
/// Text kept per diff hunk (`schema.MAX_HUNK`).
pub const MAX_HUNK: usize = 2000;
/// Text kept per tool argument value (`schema.MAX_ARG`).
pub const MAX_ARG: usize = 500;
/// Chars of the tool-argument JSON the encoder sees when a call has no command.
const ARGS_JSON_CHARS: usize = 400;
const HARNESS_TEXT_CHARS: usize = 300;
const CLIP_MARKER: &str = "\n...[clipped]...\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Read,
    Edit,
    Write,
    Run,
    Search,
    Git,
    External,
    Other,
}

impl Op {
    pub fn as_str(self) -> &'static str {
        OPS[self as usize]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Ok,
    Error,
    Denied,
    Interrupted,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        STATUSES[self as usize]
    }
}

/// Test counts parsed from a tool result (`schema.parse_result_text`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tests {
    pub passed: u64,
    pub failed: u64,
    pub errors: u64,
}

/// Structured view of a tool result. Every field is absent when not applicable.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Parsed {
    #[serde(default)]
    pub tests: Option<Tests>,
    #[serde(default)]
    pub traceback: Option<bool>,
    #[serde(default)]
    pub exception: Option<String>,
    #[serde(default)]
    pub script_ok: Option<bool>,
}

/// One canonical event. Serializes as the JSON the hindsight normalizers write
/// (`kind` tag, snake_case), minus bookkeeping the model never reads (`t`,
/// call ids, turn kinds), which is ignored on input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event {
    UserTurn {
        text: String,
    },
    Assistant {
        text: String,
    },
    Thinking {
        text: String,
    },
    ToolCall {
        tool: String,
        op: Op,
        #[serde(default)]
        path: Option<String>,
        /// Argument values are strings (a normalizer clips each to `MAX_ARG`).
        #[serde(default)]
        args: serde_json::Value,
    },
    ToolResult {
        status: Status,
        #[serde(default)]
        parsed: Parsed,
        /// Length of the result before clipping.
        #[serde(default)]
        raw_len: u64,
        #[serde(default)]
        text: String,
    },
    Diff {
        path: String,
        added: u64,
        removed: u64,
        #[serde(default)]
        hunks: Vec<String>,
    },
    Harness {
        event: String,
        #[serde(default)]
        detail: serde_json::Value,
        #[serde(default)]
        text: Option<String>,
    },
    SessionEnd {
        #[serde(default)]
        end_kind: String,
    },
}

impl Event {
    /// A genuine user turn, clipped as a normalizer stores it.
    pub fn user_turn(text: &str) -> Self {
        Event::UserTurn {
            text: clip(text, MAX_TEXT),
        }
    }

    pub fn assistant(text: &str) -> Self {
        Event::Assistant {
            text: clip(text, MAX_TEXT),
        }
    }

    pub fn thinking(text: &str) -> Self {
        Event::Thinking {
            text: clip(text, MAX_TEXT),
        }
    }

    /// A tool call as a normalizer records it: the op inferred from the tool
    /// name and command text, every argument value clipped to a string.
    pub fn tool_call(tool: &str, args: &serde_json::Value, path: Option<String>) -> Self {
        let command = args
            .get("command")
            .or_else(|| args.get("cmd"))
            .and_then(serde_json::Value::as_str);
        Self::tool_call_with_op(tool, op_for(tool, command), args, path)
    }

    /// [`Event::tool_call`] with the op decided by the caller — for harnesses
    /// whose tool names `op_for` does not know.
    pub fn tool_call_with_op(
        tool: &str,
        op: Op,
        args: &serde_json::Value,
        path: Option<String>,
    ) -> Self {
        let args = match args.as_object() {
            Some(map) => serde_json::Value::Object(
                map.iter()
                    .map(|(k, v)| {
                        let text = match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => py_json(other),
                        };
                        (k.clone(), serde_json::Value::String(clip(&text, MAX_ARG)))
                    })
                    .collect(),
            ),
            None => serde_json::Value::Object(Default::default()),
        };
        Event::ToolCall {
            tool: tool.to_string(),
            op,
            path,
            args,
        }
    }

    /// A tool result as a normalizer records it: parsed, measured, clipped.
    pub fn tool_result(text: &str, status: Status) -> Self {
        Event::ToolResult {
            status,
            parsed: parse_result_text(text),
            raw_len: text.chars().count() as u64,
            text: clip(text, MAX_TEXT),
        }
    }

    pub fn diff(path: &str, added: u64, removed: u64, hunks: Vec<String>) -> Self {
        Event::Diff {
            path: path.to_string(),
            added,
            removed,
            hunks: hunks.iter().map(|h| clip(h, MAX_HUNK)).collect(),
        }
    }

    /// A harness event (`HARNESS_EVENTS`) with the detail kind and text the
    /// normalizers record for it.
    pub fn harness(event: &str, kind: &str, text: &str) -> Self {
        Event::Harness {
            event: event.to_string(),
            detail: serde_json::json!({ "kind": kind, "text": text }),
            text: None,
        }
    }

    pub fn kind(&self) -> &'static str {
        KINDS[self.kind_index()]
    }

    fn kind_index(&self) -> usize {
        match self {
            Event::UserTurn { .. } => 0,
            Event::Assistant { .. } => 1,
            Event::Thinking { .. } => 2,
            Event::ToolCall { .. } => 3,
            Event::ToolResult { .. } => 4,
            Event::Diff { .. } => 5,
            Event::Harness { .. } => 6,
            Event::SessionEnd { .. } => 7,
        }
    }

    /// The `text` field the reference reads on any kind (empty when absent).
    fn text(&self) -> &str {
        match self {
            Event::UserTurn { text } | Event::Assistant { text } | Event::Thinking { text } => text,
            Event::ToolResult { text, .. } => text,
            Event::Harness { text, .. } => text.as_deref().unwrap_or(""),
            _ => "",
        }
    }
}

/// `schema.clip`: keep the first third and the last two thirds of `n` chars.
pub fn clip(s: &str, n: usize) -> String {
    let len = s.chars().count();
    if len <= n {
        return s.to_string();
    }
    let head = n / 3;
    let tail = n - head;
    let mut out: String = s.chars().take(head).collect();
    out.push_str(CLIP_MARKER);
    out.extend(s.chars().skip(len - tail));
    out
}

/// The text the encoder sees for one event (`prepare_events.event_text`).
pub fn event_text(e: &Event, max_chars: usize) -> String {
    match e {
        Event::UserTurn { text } => format!("USER\n{}", clip(text, max_chars)),
        Event::Assistant { text } => format!("AGENT\n{}", clip(text, max_chars)),
        Event::Thinking { text } => format!("THINK\n{}", clip(text, max_chars / 3)),
        Event::ToolCall {
            tool,
            op,
            path,
            args,
        } => {
            // Python: `args.get("command") or args.get("cmd") or json.dumps(args)[:400]`
            // — an empty string falls through like a missing key.
            let a = match non_empty(args.get("command")).or_else(|| non_empty(args.get("cmd"))) {
                Some(command) => command.to_string(),
                None => py_json(args).chars().take(ARGS_JSON_CHARS).collect(),
            };
            format!(
                "CALL {} op={} path={}\n{}",
                tool,
                op.as_str(),
                path.as_deref().unwrap_or(""),
                clip(&a, max_chars / 2)
            )
        }
        Event::ToolResult {
            status,
            parsed,
            text,
            ..
        } => {
            let mut flags = Vec::new();
            if parsed.traceback == Some(true) {
                flags.push("traceback");
            }
            if parsed.script_ok == Some(true) {
                flags.push("script_ok");
            }
            let tests = match &parsed.tests {
                Some(t) => format!(" tests={}p/{}f/{}e", t.passed, t.failed, t.errors),
                None => String::new(),
            };
            format!(
                "RESULT status={}{} {}\n{}",
                status.as_str(),
                tests,
                flags.join(" "),
                clip(text, max_chars / 2)
            )
        }
        Event::Diff {
            path,
            added,
            removed,
            hunks,
        } => format!(
            "DIFF {} +{} -{}\n{}",
            path,
            added,
            removed,
            clip(&hunks.join("\n"), max_chars)
        ),
        Event::Harness { event, detail, .. } => {
            let kind = detail
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let text = ["text", "path", "condition"]
                .iter()
                .find_map(|key| non_empty(detail.get(key)))
                .unwrap_or("");
            format!(
                "HARNESS {} {}\n{}",
                event,
                kind,
                clip(text, HARNESS_TEXT_CHARS)
            )
        }
        Event::SessionEnd { .. } => "END".to_string(),
    }
}

fn non_empty(v: Option<&serde_json::Value>) -> Option<&str> {
    v.and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
}

/// The structural feature vector of event `k` (`prepare_events.event_feat`),
/// tier-1 label slots zero. `is_last` is the reference's `k == n - 1`, true
/// only for the closing `session_end` of a complete session.
pub fn event_feat(e: &Event, k: usize, is_last: bool) -> Vec<f32> {
    let mut f = vec![0f64; FEAT_DIM];
    f[e.kind_index()] = 1.0;
    let mut i = KINDS.len();
    if let Event::ToolCall { op, .. } = e {
        f[i + *op as usize] = 1.0;
    }
    i += OPS.len();
    if let Event::ToolResult { status, .. } = e {
        f[i + *status as usize] = 1.0;
    }
    i += STATUSES.len();
    if let Event::Harness { event, .. } = e {
        if let Some(index) = HARNESS_EVENTS.iter().position(|h| h == event) {
            f[i + index] = 1.0;
        }
    }
    i += HARNESS_EVENTS.len();
    let (raw_len, parsed) = match e {
        Event::ToolResult {
            raw_len, parsed, ..
        } => (*raw_len, parsed.clone()),
        _ => (0, Parsed::default()),
    };
    let (added, removed, hunks) = match e {
        Event::Diff {
            added,
            removed,
            hunks,
            ..
        } => (*added, *removed, hunks.len()),
        _ => (0, 0, 0),
    };
    let tests = parsed.tests.clone().unwrap_or_default();
    let ln1p = |x: u64| (x as f64).ln_1p();
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let numeric = [
        ln1p(e.text().chars().count() as u64) / 10.0,
        ln1p(raw_len) / 10.0,
        ln1p(added) / 6.0,
        ln1p(removed) / 6.0,
        flag(parsed.traceback == Some(true)),
        flag(tests.failed > 0),
        flag(tests.passed > 0 && tests.failed == 0),
        flag(parsed.script_ok == Some(true)),
        flag(parsed.exception.as_deref().is_some_and(|x| !x.is_empty())),
        ln1p(k as u64) / 8.0,
        flag(is_last),
        ln1p(hunks as u64) / 3.0,
    ];
    f[i..i + NUMERIC_FEATS].copy_from_slice(&numeric);
    f.into_iter()
        .map(|x| ((x * 10_000.0).round() / 10_000.0) as f32)
        .collect()
}

/// `json.dumps(value)` as Python renders it: `", "` and `": "` separators,
/// non-ASCII escaped as `\uXXXX`. Map keys come out in serde_json map order.
pub fn py_json(v: &serde_json::Value) -> String {
    let mut out = String::new();
    write_py_json(&mut out, v);
    out
}

fn write_py_json(out: &mut String, v: &serde_json::Value) {
    use serde_json::Value;
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&n.to_string()),
        Value::String(s) => write_py_string(out, s),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_string(out, key);
                out.push_str(": ");
                write_py_json(out, value);
            }
            out.push('}');
        }
    }
}

fn write_py_string(out: &mut String, s: &str) {
    use std::fmt::Write as _;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{:04x}", unit);
                }
            }
        }
    }
    out.push('"');
}

const OP_BY_TOOL: &[(&str, Op)] = &[
    ("read", Op::Read),
    ("Read", Op::Read),
    ("cat", Op::Read),
    ("open", Op::Read),
    ("view", Op::Read),
    ("read_file", Op::Read),
    ("NotebookRead", Op::Read),
    ("edit", Op::Edit),
    ("Edit", Op::Edit),
    ("str_replace", Op::Edit),
    ("str_replace_editor", Op::Edit),
    ("apply_patch", Op::Edit),
    ("MultiEdit", Op::Edit),
    ("NotebookEdit", Op::Edit),
    ("write", Op::Write),
    ("Write", Op::Write),
    ("create", Op::Write),
    ("create_file", Op::Write),
    ("bash", Op::Run),
    ("Bash", Op::Run),
    ("shell", Op::Run),
    ("run_terminal_cmd", Op::Run),
    ("execute", Op::Run),
    ("python", Op::Run),
    ("grep", Op::Search),
    ("Grep", Op::Search),
    ("Glob", Op::Search),
    ("glob", Op::Search),
    ("search", Op::Search),
    ("search_dir", Op::Search),
    ("search_file", Op::Search),
    ("find_file", Op::Search),
    ("codebase_search", Op::Search),
    ("list_dir", Op::Search),
    ("LS", Op::Search),
    ("WebFetch", Op::External),
    ("WebSearch", Op::External),
    ("web_search", Op::External),
    ("fetch", Op::External),
];
/// `schema.EXTERNAL_RE` alternatives, matched at word boundaries.
const EXTERNAL_WORDS: &[&str] = &[
    "curl",
    "wget",
    "ssh",
    "scp",
    "rsync",
    "npm publish",
    "cargo publish",
    "pip upload",
    "twine",
    "docker push",
    "kubectl",
    "terraform",
    "aws ",
    "gcloud ",
    "az ",
    "vercel",
    "netlify",
    "fly deploy",
    "heroku",
    "wrangler",
    "gh pr ",
    "gh release",
    "gh repo",
    "gh api",
];

/// `schema.op_for`: the op of a tool call from its name and, for shell tools,
/// its command text. Tool names of MCP servers (`mcp__…`) are external.
pub fn op_for(tool: &str, command: Option<&str>) -> Op {
    let base = OP_BY_TOOL
        .iter()
        .find(|(name, _)| *name == tool)
        .map(|(_, op)| *op);
    match (base, command) {
        (Some(Op::Run), Some(cmd)) if !cmd.is_empty() => {
            if is_git_command(cmd) {
                Op::Git
            } else if EXTERNAL_WORDS.iter().any(|w| contains_word(cmd, w)) {
                Op::External
            } else {
                Op::Run
            }
        }
        (Some(op), _) => op,
        (None, _) if tool.starts_with("mcp__") => Op::External,
        (None, _) => Op::Other,
    }
}

/// `^\s*(git|gh)\s+\S`
fn is_git_command(cmd: &str) -> bool {
    let rest = cmd.trim_start();
    for prefix in ["git", "gh"] {
        if let Some(after) = rest.strip_prefix(prefix) {
            let mut chars = after.chars();
            if chars.next().is_some_and(char::is_whitespace) {
                return chars.skip_while(|c| c.is_whitespace()).next().is_some();
            }
        }
    }
    false
}

fn is_word(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_alphanumeric() || c == '_')
}

/// `\bword\b` — a boundary is a change of word-ness on each side of the match.
fn contains_word(text: &str, word: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let needle: Vec<char> = word.chars().collect();
    if needle.is_empty() || chars.len() < needle.len() {
        return false;
    }
    (0..=chars.len() - needle.len()).any(|start| {
        let end = start + needle.len();
        chars[start..end] == needle[..]
            && is_word(start.checked_sub(1).map(|i| chars[i])) != is_word(Some(chars[start]))
            && is_word(Some(chars[end - 1])) != is_word(chars.get(end).copied())
    })
}

const TRACEBACK: &str = "Traceback (most recent call last)";
const SCRIPT_OK: &str = "Script completed successfully";

/// `schema.parse_result_text`: test counts, traceback, last exception name,
/// and the SWE-agent script marker.
pub fn parse_result_text(text: &str) -> Parsed {
    let mut exception = None;
    for line in text.lines() {
        let word: String = line
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        let named = ["Error", "Exception"]
            .iter()
            .any(|suffix| word.len() > suffix.len() && word.ends_with(suffix));
        if named {
            exception = Some(word);
        }
    }
    let passed = count_before(text, &[" passed"]);
    let failed = count_before(text, &[" failed"]);
    let errors = count_before(text, &[" errors", " error"]);
    let tests = (passed + failed + errors > 0).then_some(Tests {
        passed,
        failed,
        errors,
    });
    Parsed {
        tests,
        traceback: text.contains(TRACEBACK).then_some(true),
        exception,
        script_ok: text.contains(SCRIPT_OK).then_some(true),
    }
}

/// Sum of every digit run followed by one of `suffixes` at a word boundary
/// (`(\d+) passed`, `(\d+) errors?\b`).
fn count_before(text: &str, suffixes: &[&str]) -> u64 {
    let chars: Vec<char> = text.chars().collect();
    let mut total = 0;
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        let rest: String = chars[i..].iter().take(16).collect();
        let matched = suffixes.iter().any(|suffix| {
            rest.starts_with(suffix) && !is_word(chars.get(i + suffix.chars().count()).copied())
        });
        if matched {
            let digits: String = chars[start..i].iter().collect();
            total += digits.parse::<u64>().unwrap_or(u64::MAX);
        }
    }
    total
}
