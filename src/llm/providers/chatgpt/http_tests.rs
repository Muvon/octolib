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

use super::super::build_request;
use super::*;
use crate::llm::types::Message;

#[test]
fn cache_key_names_the_session_so_the_plan_route_keeps_it() {
	let messages = vec![Message::user("hi")];
	let params = ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000);
	let mut turn_states = TurnStateStore::default();
	assert_eq!(request_headers_with_store(&params, &mut turn_states), None);

	let params = params.with_prompt_cache_key("s-1");
	assert_eq!(build_request(&params)["prompt_cache_key"], "s-1");
	assert_eq!(
		request_headers_with_store(&params, &mut turn_states)
			.unwrap()
			.get(SESSION_HEADER),
		Some(&"s-1".to_string())
	);

	let caller = HashMap::from([(SESSION_HEADER.to_string(), "caller".to_string())]);
	let params = params.with_extra_headers(caller);
	assert_eq!(
		request_headers_with_store(&params, &mut turn_states)
			.unwrap()
			.get(SESSION_HEADER),
		Some(&"caller".to_string())
	);
}

#[test]
fn turn_state_is_echoed_on_the_second_request_of_a_turn() {
	let mut turn_states = TurnStateStore::default();
	let user_messages = vec![Message::user("start")];
	let user_params =
		ChatCompletionParams::new(&user_messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000).with_prompt_cache_key("session");

	let headers = request_headers_with_store(&user_params, &mut turn_states).unwrap();
	assert!(!headers.contains_key(TURN_STATE_HEADER));
	remember_turn_state_with_store(Some("session"), Some("first-turn-state"), &mut turn_states);

	let continued_messages = vec![Message::user("start"), Message::assistant("working")];
	let continued_params = ChatCompletionParams::new(&continued_messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000)
		.with_prompt_cache_key("session");
	let headers = request_headers_with_store(&continued_params, &mut turn_states).unwrap();
	assert_eq!(
		headers.get(TURN_STATE_HEADER).map(String::as_str),
		Some("first-turn-state")
	);
}

#[test]
fn first_turn_state_wins_when_later_responses_return_another() {
	let mut turn_states = TurnStateStore::default();
	remember_turn_state_with_store(Some("session"), Some("first"), &mut turn_states);
	remember_turn_state_with_store(Some("session"), Some("later"), &mut turn_states);

	assert_eq!(turn_states.get("session").as_deref(), Some("first"));
}

#[test]
fn new_user_turn_clears_the_stored_turn_state() {
	let mut turn_states = TurnStateStore::default();
	turn_states.store_first("session", "previous-turn");
	let messages = vec![Message::user("next turn")];
	let params =
		ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000).with_prompt_cache_key("session");

	let headers = request_headers_with_store(&params, &mut turn_states).unwrap();

	assert!(!headers.contains_key(TURN_STATE_HEADER));
	assert_eq!(turn_states.get("session"), None);
}

#[test]
fn requests_without_cache_keys_have_no_turn_state_affinity() {
	let mut turn_states = TurnStateStore::default();
	let messages = vec![Message::assistant("continue")];
	let caller_headers = HashMap::from([
		(TURN_STATE_HEADER.to_string(), "caller-token".to_string()),
		("x-test".to_string(), "kept".to_string()),
	]);
	let params =
		ChatCompletionParams::new(&messages, "gpt-6.1-sol", 0.7, 1.0, 50, 1000).with_extra_headers(caller_headers);

	let headers = request_headers_with_store(&params, &mut turn_states).unwrap();
	remember_turn_state_with_store(None, Some("response-token"), &mut turn_states);

	assert!(!headers.contains_key(TURN_STATE_HEADER));
	assert_eq!(headers.get("x-test").map(String::as_str), Some("kept"));
	assert!(turn_states.entries.is_empty());
}

#[test]
fn turn_state_store_evicts_the_least_recently_used_key_at_capacity() {
	let mut turn_states = TurnStateStore::default();
	for index in 0..TURN_STATE_STORE_CAPACITY {
		turn_states.store_first(&format!("session-{index}"), &format!("token-{index}"));
	}
	assert_eq!(turn_states.get("session-0").as_deref(), Some("token-0"));

	turn_states.store_first("overflow", "new-token");

	assert_eq!(turn_states.entries.len(), TURN_STATE_STORE_CAPACITY);
	assert_eq!(turn_states.get("session-1"), None);
	assert_eq!(turn_states.get("session-0").as_deref(), Some("token-0"));
	assert_eq!(turn_states.get("overflow").as_deref(), Some("new-token"));
}

#[test]
fn sse_accepts_multiline_data_and_requires_a_completed_event() {
	let body = concat!(
		": keepalive\r\n\r\n",
		"event: response.completed\r\n",
		"data: {\"type\":\"response.completed\",\r\n",
		"data: \"response\":{\"id\":\"resp_http\",\"output\":[]}}\r\n\r\n"
	);
	assert_eq!(merge_stream(body).unwrap()["id"], "resp_http");
	assert!(merge_stream(": keepalive\n\n")
		.unwrap_err()
		.is::<RetryableStreamError>());
	assert!(merge_stream("data: invalid JSON\n\n").is_err());
}
