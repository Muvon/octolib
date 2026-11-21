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
#[cfg(target_os = "linux")]
use hickory_resolver::{config::LookupIpStrategy, TokioResolver};
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::time::Instant;
use tower::{Layer, Service};

pub(crate) const CONNECT_TIMEOUT: Duration = Duration::from_secs(20);

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
static HTTP_CLIENT: LazyLock<ArcSwap<reqwest::Client>> =
	LazyLock::new(|| ArcSwap::from_pointee(build_http_client(false)));

// WebSocket upgrades require HTTP/1.1, including during TLS ALPN negotiation.
#[cfg(feature = "websockets")]
static WEBSOCKET_CLIENT: LazyLock<ArcSwap<reqwest::Client>> =
	LazyLock::new(|| ArcSwap::from_pointee(build_http_client(true)));

/// A lookup or connect at or above this is logged at `warn`: normal ones take
/// milliseconds, so a full second points at the network or resolver.
const SLOW_NETWORK_STEP: Duration = Duration::from_secs(1);

/// Times every lookup so slow DNS shows up in the log.
struct TimedResolver<R> {
	inner: R,
}

impl<R: Resolve> Resolve for TimedResolver<R> {
	fn resolve(&self, name: Name) -> Resolving {
		let host = name.as_str().to_string();
		let lookup = self.inner.resolve(name);
		Box::pin(async move {
			let started = Instant::now();
			let result = lookup.await;
			let elapsed = started.elapsed();
			if elapsed >= SLOW_NETWORK_STEP {
				tracing::warn!(host = %host, elapsed_ms = elapsed.as_millis() as u64, "slow DNS lookup");
			} else {
				tracing::debug!(host = %host, elapsed_ms = elapsed.as_millis() as u64, "DNS lookup");
			}
			result
		})
	}
}

/// How a host name becomes addresses for a new connection.
enum Lookup {
	/// hickory, which caches answers for their TTL, so a new connection after
	/// the pool's idle timeout does not repeat a slow lookup (5 s per lookup in
	/// Docker on macOS, 2026-11-17). Linux only: on macOS and Windows it would
	/// skip per-domain DNS (VPN split DNS, `/etc/resolver/*`, `.local` names).
	#[cfg(target_os = "linux")]
	Cached(Box<TokioResolver>),
	/// The operating system's lookup, as reqwest does by default.
	System,
}

impl Lookup {
	#[cfg(target_os = "linux")]
	fn for_platform() -> Self {
		let built = TokioResolver::builder_tokio().and_then(|mut builder| {
			// Same as reqwest's own hickory integration: ask for both families so
			// connection racing (happy eyeballs) has addresses to race.
			builder.options_mut().ip_strategy = LookupIpStrategy::Ipv4AndIpv6;
			builder.build()
		});
		match built {
			Ok(resolver) => Lookup::Cached(Box::new(resolver)),
			// Without a readable /etc/resolv.conf the system lookup still serves
			// names from /etc/hosts, such as localhost.
			Err(error) => {
				tracing::warn!(%error, "cannot set up the DNS cache; host names resolve without a cache");
				Lookup::System
			}
		}
	}

	#[cfg(not(target_os = "linux"))]
	fn for_platform() -> Self {
		Lookup::System
	}
}

impl Resolve for Lookup {
	fn resolve(&self, name: Name) -> Resolving {
		match self {
			#[cfg(target_os = "linux")]
			Lookup::Cached(resolver) => {
				let resolver = TokioResolver::clone(resolver);
				Box::pin(async move {
					let lookup = resolver.lookup_ip(name.as_str()).await?;
					let addrs: Addrs = Box::new(lookup.into_iter().map(|ip| SocketAddr::new(ip, 0)));
					Ok(addrs)
				})
			}
			Lookup::System => Box::pin(async move {
				let found: Vec<SocketAddr> = tokio::net::lookup_host((name.as_str(), 0)).await?.collect();
				let addrs: Addrs = Box::new(found.into_iter());
				Ok(addrs)
			}),
		}
	}
}

/// Times each new connection (DNS + TCP + TLS) and warns when it is slow.
#[derive(Clone)]
struct ConnectTimingLayer;

impl<S> Layer<S> for ConnectTimingLayer {
	type Service = ConnectTiming<S>;

	fn layer(&self, inner: S) -> Self::Service {
		ConnectTiming(inner)
	}
}

#[derive(Clone)]
struct ConnectTiming<S>(S);

impl<S, Target> Service<Target> for ConnectTiming<S>
where
	S: Service<Target>,
	S::Future: Send + 'static,
	S::Response: Send + 'static,
	S::Error: Send + 'static,
	Target: std::fmt::Debug,
{
	type Response = S::Response;
	type Error = S::Error;
	type Future = Pin<Box<dyn Future<Output = Result<S::Response, S::Error>> + Send>>;

	fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
		self.0.poll_ready(cx)
	}

	fn call(&mut self, target: Target) -> Self::Future {
		// reqwest hides the target type; its Debug output carries the URI.
		let described = format!("{target:?}");
		let started = Instant::now();
		let connecting = self.0.call(target);
		Box::pin(async move {
			let result = connecting.await;
			let elapsed = started.elapsed();
			if elapsed >= SLOW_NETWORK_STEP {
				tracing::warn!(target = %described, elapsed_ms = elapsed.as_millis() as u64, "slow connect");
			}
			result
		})
	}
}

fn build_http_client(http1_only: bool) -> reqwest::Client {
	let builder = reqwest::Client::builder()
		.user_agent(USER_AGENT.load().as_str())
		.dns_resolver(Arc::new(TimedResolver {
			inner: Lookup::for_platform(),
		}))
		.connector_layer(ConnectTimingLayer)
		.connect_timeout(CONNECT_TIMEOUT)
		.tcp_keepalive(Duration::from_secs(10))
		.tcp_keepalive_interval(Duration::from_secs(5))
		.tcp_nodelay(true)
		.pool_idle_timeout(Duration::from_secs(15));
	let builder = if http1_only { builder.http1_only() } else { builder };
	builder.build().expect("failed to build HTTP client")
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
	let fresh = build_http_client(false);
	HTTP_CLIENT.store(std::sync::Arc::new(fresh));
	#[cfg(feature = "websockets")]
	WEBSOCKET_CLIENT.store(std::sync::Arc::new(build_http_client(true)));
	tracing::debug!("HTTP client refreshed with new connection pool");
}

#[cfg(feature = "websockets")]
pub(crate) fn websocket_client() -> reqwest::Client {
	(*WEBSOCKET_CLIENT.load_full()).clone()
}

#[cfg(feature = "websockets")]
pub(crate) fn user_agent() -> std::sync::Arc<String> {
	USER_AGENT.load_full()
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::io;
	use std::sync::Mutex;
	use tokio::io::{AsyncReadExt, AsyncWriteExt};

	struct SlowResolver;

	impl Resolve for SlowResolver {
		fn resolve(&self, _name: Name) -> Resolving {
			Box::pin(async {
				tokio::time::sleep(SLOW_NETWORK_STEP * 2).await;
				let addrs: Addrs = Box::new(std::iter::empty());
				Ok(addrs)
			})
		}
	}

	#[derive(Clone)]
	struct LogBuffer(Arc<Mutex<Vec<u8>>>);

	impl io::Write for LogBuffer {
		fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
			self.0.lock().unwrap().extend_from_slice(buf);
			Ok(buf.len())
		}

		fn flush(&mut self) -> io::Result<()> {
			Ok(())
		}
	}

	#[tokio::test(start_paused = true)]
	async fn slow_lookup_is_reported_as_warning() {
		let buffer = LogBuffer(Arc::default());
		let writer = buffer.clone();
		let subscriber = tracing_subscriber::fmt()
			.with_max_level(tracing::Level::DEBUG)
			.with_ansi(false)
			.with_writer(move || writer.clone())
			.finish();
		let _guard = tracing::subscriber::set_default(subscriber);

		let resolver = TimedResolver { inner: SlowResolver };
		let _addrs = resolver.resolve("slow.example".parse().unwrap()).await.unwrap();

		let log = String::from_utf8(buffer.0.lock().unwrap().clone()).unwrap();
		assert!(log.contains("WARN"), "{log}");
		assert!(log.contains("slow.example"), "{log}");
	}

	// The shared client is built inside a `LazyLock`, possibly outside any runtime.
	#[test]
	fn client_builds_without_a_runtime() {
		build_http_client(false);
	}

	#[tokio::test]
	async fn system_lookup_resolves_localhost() {
		let addrs: Vec<SocketAddr> = Lookup::System
			.resolve("localhost".parse().unwrap())
			.await
			.unwrap()
			.collect();
		assert!(addrs.iter().any(|addr| addr.ip().is_loopback()), "{addrs:?}");
	}

	#[tokio::test]
	async fn client_resolves_localhost() {
		let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
		let port = listener.local_addr().unwrap().port();
		tokio::spawn(async move {
			let (mut socket, _) = listener.accept().await.unwrap();
			let mut request = [0u8; 1024];
			let _ = socket.read(&mut request).await;
			socket
				.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
				.await
				.unwrap();
		});

		let url = format!("http://localhost:{port}/");
		let body = build_http_client(false)
			.get(url)
			.send()
			.await
			.unwrap()
			.text()
			.await
			.unwrap();
		assert_eq!(body, "ok");
	}
}
