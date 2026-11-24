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

//! Interactive terminal input for command-line setup.

use anyhow::{bail, Result};
use crossterm::{
	cursor,
	event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
	execute, terminal,
};
use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;
use std::io::{self, Write};

/// What the user did in `pick_index`.
pub enum Picked {
	Index(usize),
	/// Esc, or Enter with nothing matching the filter.
	Back,
	/// Ctrl-C or Ctrl-D: raw mode swallows the signal, so the caller decides how to stop.
	Quit,
}

/// Pick a label with fuzzy filtering.
/// Title and labels may carry ANSI colors: matching and row widths use the plain text,
/// and the highlighted row is drawn plain so its inverse video is not cut by color resets.
pub fn pick_index(title: &str, labels: &[String], mut selected: usize) -> Result<Picked> {
	const EXTRA_ROWS: usize = 4;
	let matcher = SkimMatcherV2::default();
	let plain_labels: Vec<String> = labels.iter().map(|label| strip_ansi(label)).collect();
	let mut query: Vec<char> = Vec::new();
	let mut cursor_pos: usize = 0; // char index into query, 0..=len
	let mut stdout = std::io::stdout();

	terminal::enable_raw_mode()?;
	let result = (|| -> Result<Picked> {
		execute!(stdout, terminal::EnterAlternateScreen, cursor::Hide)?;
		loop {
			let query_str: String = query.iter().collect();
			// An empty query preserves caller ordering and its initial selection.
			let filtered: Vec<(usize, &String)> = if query.is_empty() {
				labels.iter().enumerate().collect()
			} else {
				let mut scored: Vec<(i64, usize, &String)> = labels
					.iter()
					.enumerate()
					.filter_map(|(i, label)| matcher.fuzzy_match(&plain_labels[i], &query_str).map(|s| (s, i, label)))
					.collect();
				scored.sort_by_key(|b| std::cmp::Reverse(b.0));
				scored.into_iter().map(|(_, i, label)| (i, label)).collect()
			};
			if selected >= filtered.len() {
				selected = filtered.len().saturating_sub(1);
			}

			let (columns, rows) = terminal::size()?;
			anyhow::ensure!(columns > 0, "terminal width must be positive");
			let title_rows: usize = title
				.lines()
				.map(|line| strip_ansi(line).chars().count().max(1).div_ceil(usize::from(columns)))
				.sum();
			execute!(stdout, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0, 0))?;
			for line in title.lines() {
				write!(stdout, "{line}\r\n")?;
			}
			write!(stdout, "> {}\r\n\r\n", query_str)?;

			let max_visible = usize::from(rows).saturating_sub(title_rows + EXTRA_ROWS).max(1);
			// Keep the selection inside the visible window.
			let offset = if selected >= max_visible {
				selected + 1 - max_visible
			} else {
				0
			};
			for (i, (label_index, label)) in filtered.iter().skip(offset).take(max_visible).enumerate() {
				let idx = offset + i;
				if idx == selected {
					write!(stdout, "\x1b[7m> {}\x1b[0m\r\n", plain_labels[*label_index])?;
				} else {
					write!(stdout, "  {}\r\n", label)?;
				}
			}
			if filtered.is_empty() {
				write!(stdout, "  (no matches — Enter cancels)\r\n")?;
			}
			// Place the cursor in the query line at its edit position.
			execute!(
				stdout,
				cursor::Show,
				cursor::MoveTo(2 + cursor_pos as u16, title_rows as u16)
			)?;
			stdout.flush()?;

			if let Event::Key(KeyEvent {
				code, modifiers, kind, ..
			}) = event::read()?
			{
				if kind == KeyEventKind::Release {
					continue;
				}
				let ctrl = modifiers.contains(KeyModifiers::CONTROL);
				match code {
					KeyCode::Esc => return Ok(Picked::Back),
					KeyCode::Enter => {
						return Ok(filtered.get(selected).map_or(Picked::Back, |(i, _)| Picked::Index(*i)));
					}
					KeyCode::Up => selected = selected.saturating_sub(1),
					KeyCode::Down => {
						if selected + 1 < filtered.len() {
							selected += 1;
						}
					}
					// ── readline-style editing ─────────────────────────
					KeyCode::Char('c' | 'd') if ctrl => return Ok(Picked::Quit),
					KeyCode::Char('n') if ctrl => {
						if selected + 1 < filtered.len() {
							selected += 1;
						}
					}
					KeyCode::Char('p') if ctrl => selected = selected.saturating_sub(1),
					KeyCode::Char('a') if ctrl => cursor_pos = 0,
					KeyCode::Char('e') if ctrl => cursor_pos = query.len(),
					KeyCode::Char('u') if ctrl => {
						query.drain(..cursor_pos);
						cursor_pos = 0;
						selected = 0;
					}
					KeyCode::Char('k') if ctrl => {
						query.truncate(cursor_pos);
						selected = 0;
					}
					KeyCode::Char('w') if ctrl => {
						// kill word before cursor: trailing spaces, then the word
						let mut start = cursor_pos;
						while start > 0 && query[start - 1] == ' ' {
							start -= 1;
						}
						while start > 0 && query[start - 1] != ' ' {
							start -= 1;
						}
						query.drain(start..cursor_pos);
						cursor_pos = start;
						selected = 0;
					}
					KeyCode::Left => cursor_pos = cursor_pos.saturating_sub(1),
					KeyCode::Right => cursor_pos = (cursor_pos + 1).min(query.len()),
					KeyCode::Home => cursor_pos = 0,
					KeyCode::End => cursor_pos = query.len(),
					KeyCode::Backspace => {
						if cursor_pos > 0 {
							query.remove(cursor_pos - 1);
							cursor_pos -= 1;
							selected = 0;
						}
					}
					KeyCode::Delete => {
						if cursor_pos < query.len() {
							query.remove(cursor_pos);
							selected = 0;
						}
					}
					KeyCode::Char(c) if !ctrl => {
						query.insert(cursor_pos, c);
						cursor_pos += 1;
						selected = 0;
					}
					_ => {}
				}
			}
		}
	})();
	let restore_screen = execute!(stdout, cursor::Show, terminal::LeaveAlternateScreen);
	let restore_input = terminal::disable_raw_mode();
	restore_screen?;
	restore_input?;
	result
}

/// Read a hidden value. Returns `None` when the user presses Esc.
pub fn read_secret(name: &str) -> Result<Option<String>> {
	terminal::enable_raw_mode()?;
	let result = prompt_secret(name, &mut io::stderr(), event::read);
	terminal::disable_raw_mode()?;
	writeln!(io::stderr())?;
	result
}

fn prompt_secret(
	name: &str,
	output: &mut impl Write,
	mut read_event: impl FnMut() -> io::Result<Event>,
) -> Result<Option<String>> {
	loop {
		write!(output, "{name} (hidden, Esc to go back): ")?;
		output.flush()?;
		let mut value = String::new();
		loop {
			if let Event::Key(key) = read_event()? {
				if key.kind == KeyEventKind::Release {
					continue;
				}
				match key.code {
					KeyCode::Enter => {
						if value.trim().is_empty() {
							write!(output, "\r\n{name}: a non-empty value is required\r\n")?;
							break;
						}
						return Ok(Some(value.trim().to_string()));
					}
					KeyCode::Esc => return Ok(None),
					KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
						bail!("{name}: input cancelled");
					}
					KeyCode::Backspace => {
						value.pop();
					}
					KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => value.push(c),
					_ => {}
				}
			}
		}
	}
}

/// Strip ANSI CSI escape sequences from a string.
fn strip_ansi(s: &str) -> String {
	let mut out = String::with_capacity(s.len());
	let mut in_esc = false;
	for c in s.chars() {
		if in_esc {
			if c.is_ascii_alphabetic() {
				in_esc = false;
			}
			continue;
		}
		if c == '\x1b' {
			in_esc = true;
			continue;
		}
		out.push(c);
	}
	out
}

#[cfg(test)]
#[path = "terminal_tests.rs"]
mod tests;
