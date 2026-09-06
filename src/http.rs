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
static USER_AGENT: LazyLock<ArcSwap<String>> =
    LazyLock::new(|| ArcSwap::from_pointee(DEFAULT_USER_AGENT.to_string()));

/// Identify the calling application in the `User-Agent` of every upstream
/// request, replacing the `octolib/<version>` default.
///
/// Call once at startup: it rebuilds the shared HTTP client, so in-flight
/// requests keep the old client and pooled connections are dropped. Values that
/// are empty or not a valid header are ignored.
pub fn set_user_agent(user_agent: impl Into<String>) {
    let user_agent = user_agent.into();
    if user_agent.trim().is_empty() || reqwest::header::HeaderValue::from_str(&user_agent).is_err()
    {
        tracing::warn!(user_agent = %user_agent, "ignoring invalid user agent");
        return;
    }
    USER_AGENT.store(std::sync::Arc::new(user_agent));
    refresh_http_client();
}

/// Process-wide shared HTTP client, swappable on connection errors.
///
/// `reqwest::Client` holds a connection pool internally — reusing it across
/// all provider requests enables connection keep-alive, HTTP/2 multiplexing
/// (when the server supports it via ALPN), and avoids the per-request TLS
/// handshake overhead that causes connection-reset errors under load.
///
/// When a connection error is detected (DNS failure, TCP reset, TLS handshake
/// failure, network unreachable), `refresh_http_client()` atomically swaps
/// in a fresh client with a new connection pool, so subsequent retries don't
/// reuse stale/broken connections.
///
/// # HTTP stack tuning
///
/// Total request timeouts are applied per call via `apply_request_timeout()`.
/// The client itself only limits connection establishment, so slow LLM
/// generation is not affected by the connect timeout.
///
/// **Transport / pool reliability**
/// - `connect_timeout(20s)`: bound DNS, TCP, and TLS establishment without
///   limiting how long an established LLM request may run
/// - `tcp_keepalive(10s)` + `tcp_keepalive_interval(5s)`: OS-level probes
///   detect dead connections before reuse. The first probe fires at 10s —
///   before `pool_idle_timeout` evicts the connection — so stale sockets are
///   surfaced and removed rather than reused. Subsequent probes every 5s
///   catch connections that go bad while idle in the pool.
/// - `tcp_nodelay(true)`: disable Nagle's algorithm — request bodies ship
///   immediately instead of waiting for ACK coalescing (lower latency)
/// - `pool_idle_timeout(15s)`: evict idle pooled connections before NAT/firewall
///   or the upstream edge silently drops them. Some upstream edges (notably
///   CN-hosted endpoints like DeepSeek / Moonshot, and Alibaba Token Plan NLBs
///   in ap-southeast-1) close idle keep-alive connections aggressively; reusing
///   such a half-closed socket produces "error sending request" / TCP RST
///   mid-write. 15s is short enough to stay ahead of most NLB idle timeouts
///   (typically 60s) while still allowing connection reuse for rapid
///   successive requests.
///
/// **HTTP/2 keep-alive (only takes effect when ALPN negotiates h2)**
/// - `http2_keep_alive_interval(10s)`: PING frames detect dead peers
///   proactively and prevent NAT/firewall idle-timeout from silently dropping
///   the multiplexed connection. 10s is well within any NLB idle timeout and
///   shorter than `pool_idle_timeout` so stale h2 connections are torn down
///   before reuse. PING frames count as data transfer for L4 NLB idle
///   timeouts, keeping the connection alive.
/// - `http2_keep_alive_while_idle(true)`: keep PINGing even with no active streams
/// - `http2_keep_alive_timeout(10s)`: drop conn if PING unACKed within 10s
///
/// **Design principle**: keepalive intervals (10s) < pool_idle_timeout (15s) <
/// NLB idle timeout (typically 60s). This ensures stale connections are
/// probed and removed before they can be reused, regardless of whether the
/// connection is HTTP/2 (PING-based detection) or HTTP/1.1 (TCP keepalive
/// probe-based detection).
static HTTP_CLIENT: LazyLock<ArcSwap<reqwest::Client>> =
    LazyLock::new(|| ArcSwap::from_pointee(build_http_client()));

fn build_http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT.load().as_str())
        .connect_timeout(Duration::from_secs(20))
        .tcp_keepalive(Duration::from_secs(10))
        .tcp_keepalive_interval(Duration::from_secs(5))
        .tcp_nodelay(true)
        .pool_idle_timeout(Duration::from_secs(15))
        .http2_keep_alive_interval(Duration::from_secs(10))
        .http2_keep_alive_while_idle(true)
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .build()
        .expect("failed to build HTTP client")
}

/// Returns a cloned handle to the process-wide shared HTTP client.
///
/// `reqwest::Client` is internally `Arc`-based, so cloning is cheap and
/// always points to the current client (even after `refresh_http_client()`
/// swaps the global).
#[cfg(any(feature = "llm", feature = "media"))]
pub(crate) fn http_client() -> reqwest::Client {
    // load_full() clones the Arc (cheap atomic increment),
    // then dereference and clone the Client (cheap — Client is Arc internally)
    (*HTTP_CLIENT.load_full()).clone()
}

/// Atomically replace the shared HTTP client with a fresh instance.
///
/// Call this when a connection error is detected (DNS failure, TCP reset,
/// TLS handshake failure, network unreachable). The new client gets a fresh
/// connection pool, so subsequent requests — including retry attempts —
/// won't reuse stale/broken connections from the old pool.
///
/// The old client is dropped once all outstanding references to it are gone,
/// which closes its idle connections.
pub(crate) fn refresh_http_client() {
    let fresh = build_http_client();
    HTTP_CLIENT.store(std::sync::Arc::new(fresh));
    tracing::debug!("HTTP client refreshed with new connection pool");
}
