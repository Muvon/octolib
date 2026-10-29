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

//! Process-wide HTTP client shared by every octolib provider.
//!
//! Lives outside `llm` so the media adapters reuse the same connection pool
//! without pulling in the chat providers.

use arc_swap::ArcSwap;
use std::sync::LazyLock;
use std::time::Duration;

const DEFAULT_USER_AGENT: &str = concat!("octolib/", env!("CARGO_PKG_VERSION"));

/// `User-Agent` sent with every upstream request. reqwest sends none by default
/// and some upstreams (OpenCode) reject anonymous requests, so octolib always
/// identifies itself; embedding apps rename it with [`set_user_agent`].
static USER_AGENT: LazyLock<ArcSwap<String>> = LazyLock::new(|| ArcSwap::from_pointee(DEFAULT_USER_AGENT.to_string()));

/// Identify the calling application in the `User-Agent` of every upstream
/// request, replacing the `octolib/<version>` default.
///
/// Call once at startup: it rebuilds the shared HTTP client, so in-flight
/// requests keep the old client and pooled connections are dropped. Values that
/// are empty or not a valid header are ignored.
pub fn set_user_agent(user_agent: impl Into<String>) {
	let user_agent = user_agent.into();
	if user_agent.trim().is_empty() || reqwest::header::HeaderValue::from_str(&user_agent).is_err() {
		tracing::warn!(user_agent = %user_agent, "ignoring invalid user agent");
		return;
	}
	USER_AGENT.store(std::sync::Arc::new(user_agent));
	refresh_http_client();
}

/// Reqwest owns connection health and eviction within this shared pool.
/// Request failures must not discard healthy connections to unrelated hosts.
/// Total request deadlines are supplied by callers; connection establishment
/// is bounded separately. HTTP/2 PINGs stay disabled because upstreams have
/// different keepalive policies; TCP keepalive remains enabled.
static HTTP_CLIENT: LazyLock<ArcSwap<reqwest::Client>> = LazyLock::new(|| ArcSwap::from_pointee(build_http_client()));

fn build_http_client() -> reqwest::Client {
	reqwest::Client::builder()
		.user_agent(USER_AGENT.load().as_str())
		.connect_timeout(Duration::from_secs(20))
		.tcp_keepalive(Duration::from_secs(10))
		.tcp_keepalive_interval(Duration::from_secs(5))
		.tcp_nodelay(true)
		.pool_idle_timeout(Duration::from_secs(15))
		.build()
		.expect("failed to build HTTP client")
}

/// Returns a cloned handle to the process-wide shared HTTP client.
///
/// Existing clones retain their pool when application configuration changes.
#[cfg(any(feature = "llm", feature = "media", feature = "evaluation"))]
pub(crate) fn http_client() -> reqwest::Client {
	// load_full() clones the Arc (cheap atomic increment),
	// then dereference and clone the Client (cheap — Client is Arc internally)
	(*HTTP_CLIENT.load_full()).clone()
}

/// Rebuild after application configuration changes. Existing requests retain
/// their client until completion.
fn refresh_http_client() {
	let fresh = build_http_client();
	HTTP_CLIENT.store(std::sync::Arc::new(fresh));
	tracing::debug!("HTTP client refreshed with new connection pool");
}
