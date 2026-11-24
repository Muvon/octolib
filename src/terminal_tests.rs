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

use super::*;

#[test]
fn configure_secret_reprompts_empty_input_without_echoing_values() {
	for name in ["PROVIDER_API_KEY", "PROVIDER_REQUIRED_EXTRA"] {
		let mut events = [
			KeyCode::Enter,
			KeyCode::Char(' '),
			KeyCode::Enter,
			KeyCode::Char('s'),
			KeyCode::Char('e'),
			KeyCode::Char('c'),
			KeyCode::Char('r'),
			KeyCode::Char('e'),
			KeyCode::Char('t'),
			KeyCode::Enter,
		]
		.into_iter();
		let mut output = Vec::new();
		let value = prompt_secret(name, &mut output, || {
			Ok(Event::Key(crossterm::event::KeyEvent::new(
				events.next().expect("input must finish with a non-empty value"),
				KeyModifiers::NONE,
			)))
		})
		.expect("required value entered");
		assert_eq!(value.as_deref(), Some("secret"));
		assert!(events.next().is_none());
		assert_eq!(
			String::from_utf8(output).unwrap(),
			format!(
				"{name} (hidden, Esc to go back): \r\n{name}: a non-empty value is required\r\n\
			{name} (hidden, Esc to go back): \r\n{name}: a non-empty value is required\r\n\
			{name} (hidden, Esc to go back): "
			)
		);
	}
}

#[test]
fn configure_secret_trims_surrounding_whitespace() {
	let mut events = [
		KeyCode::Char(' '),
		KeyCode::Char('a'),
		KeyCode::Char('b'),
		KeyCode::Char(' '),
		KeyCode::Enter,
	]
	.into_iter();
	let mut output = Vec::new();
	let value = prompt_secret("PROVIDER_API_KEY", &mut output, || {
		Ok(Event::Key(crossterm::event::KeyEvent::new(
			events.next().expect("input must finish with Enter"),
			KeyModifiers::NONE,
		)))
	})
	.expect("value entered");
	assert_eq!(value.as_deref(), Some("ab"));
	assert!(events.next().is_none());
	assert_eq!(
		String::from_utf8(output).unwrap(),
		"PROVIDER_API_KEY (hidden, Esc to go back): "
	);
}

#[test]
fn configure_secret_esc_goes_back() {
	let mut output = Vec::new();
	let value = prompt_secret("PROVIDER_API_KEY", &mut output, || {
		Ok(Event::Key(crossterm::event::KeyEvent::new(
			KeyCode::Esc,
			KeyModifiers::NONE,
		)))
	})
	.expect("Esc is not an error");
	assert_eq!(value, None);
}

#[test]
fn configure_secret_propagates_input_errors() {
	let mut output = Vec::new();
	let err = prompt_secret("PROVIDER_REQUIRED_EXTRA", &mut output, || {
		Err(io::Error::new(io::ErrorKind::UnexpectedEof, "terminal input ended"))
	})
	.expect_err("input failure must end setup");
	assert_eq!(err.to_string(), "terminal input ended");
}
