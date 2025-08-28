// Copyright 2025 Muvon Un Limited
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

//! Type-safe configuration types for octolib

use crate::errors::{ConfigError, ConfigResult};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Type-safe cache TTL configuration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheTTL {
	/// Cache for specified number of minutes
	Minutes(u32),
	/// Cache for specified number of hours
	Hours(u32),
	/// Cache for specified number of seconds (for fine-grained control)
	Seconds(u32),
}

impl CacheTTL {
	/// Convert to Duration for internal calculations
	pub fn to_duration(&self) -> Duration {
		match self {
			CacheTTL::Seconds(s) => Duration::from_secs(*s as u64),
			CacheTTL::Minutes(m) => Duration::from_secs(*m as u64 * 60),
			CacheTTL::Hours(h) => Duration::from_secs(*h as u64 * 3600),
		}
	}

	/// Create from Duration
	pub fn from_duration(duration: Duration) -> Self {
		let total_seconds = duration.as_secs();

		if total_seconds % 3600 == 0 && total_seconds >= 3600 {
			CacheTTL::Hours((total_seconds / 3600) as u32)
		} else if total_seconds % 60 == 0 && total_seconds >= 60 {
			CacheTTL::Minutes((total_seconds / 60) as u32)
		} else {
			CacheTTL::Seconds(total_seconds as u32)
		}
	}

	/// Parse from string format (e.g., "5m", "1h", "30s")
	pub fn from_string(s: &str) -> ConfigResult<Self> {
		if s.is_empty() {
			return Err(ConfigError::InvalidValue {
				field: "cache_ttl".to_string(),
				value: s.to_string(),
			});
		}

		let (number_part, unit_part) = if let Some(pos) = s.chars().position(|c| c.is_alphabetic())
		{
			(&s[..pos], &s[pos..])
		} else {
			return Err(ConfigError::InvalidDurationFormat {
				format: s.to_string(),
			});
		};

		let number: u32 = number_part
			.parse()
			.map_err(|_| ConfigError::InvalidDurationFormat {
				format: s.to_string(),
			})?;

		match unit_part.to_lowercase().as_str() {
			"s" | "sec" | "second" | "seconds" => Ok(CacheTTL::Seconds(number)),
			"m" | "min" | "minute" | "minutes" => Ok(CacheTTL::Minutes(number)),
			"h" | "hr" | "hour" | "hours" => Ok(CacheTTL::Hours(number)),
			_ => Err(ConfigError::InvalidDurationFormat {
				format: s.to_string(),
			}),
		}
	}

	/// Create short TTL (5 minutes)
	pub fn short() -> Self {
		CacheTTL::Minutes(5)
	}

	/// Create long TTL (1 hour)
	pub fn long() -> Self {
		CacheTTL::Hours(1)
	}

	/// Create from configuration flag
	pub fn from_long_cache_flag(use_long_cache: bool) -> Self {
		if use_long_cache {
			Self::long()
		} else {
			Self::short()
		}
	}

	/// Check if this TTL is considered "long" (>= 1 hour)
	pub fn is_long(&self) -> bool {
		self.to_duration() >= Duration::from_secs(3600)
	}

	/// Check if this TTL is considered "short" (< 1 hour)
	pub fn is_short(&self) -> bool {
		!self.is_long()
	}
}

impl std::fmt::Display for CacheTTL {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		match self {
			CacheTTL::Seconds(s) => write!(f, "{}s", s),
			CacheTTL::Minutes(m) => write!(f, "{}m", m),
			CacheTTL::Hours(h) => write!(f, "{}h", h),
		}
	}
}

impl Default for CacheTTL {
	fn default() -> Self {
		CacheTTL::Minutes(5)
	}
}

/// Configuration for cache control
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheConfig {
	/// TTL for cache entries
	pub ttl: CacheTTL,
	/// Cache type (ephemeral, persistent, etc.)
	pub cache_type: CacheType,
}

/// Type of cache to use
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CacheType {
	/// Ephemeral cache (temporary)
	#[default]
	Ephemeral,
	/// Persistent cache (survives restarts)
	Persistent,
}

impl CacheType {
	pub fn to_string(&self) -> &'static str {
		match self {
			CacheType::Ephemeral => "ephemeral",
			CacheType::Persistent => "persistent",
		}
	}
}

impl CacheConfig {
	/// Create new cache config
	pub fn new(ttl: CacheTTL, cache_type: CacheType) -> Self {
		Self { ttl, cache_type }
	}

	/// Create ephemeral cache config with TTL
	pub fn ephemeral(ttl: CacheTTL) -> Self {
		Self::new(ttl, CacheType::Ephemeral)
	}

	/// Create persistent cache config with TTL
	pub fn persistent(ttl: CacheTTL) -> Self {
		Self::new(ttl, CacheType::Persistent)
	}

	/// Convert to JSON value for provider APIs
	pub fn to_json(&self) -> serde_json::Value {
		serde_json::json!({
			"type": self.cache_type.to_string(),
			"ttl": self.ttl.to_string()
		})
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_cache_ttl_string_conversion() {
		assert_eq!(CacheTTL::Minutes(5).to_string(), "5m");
		assert_eq!(CacheTTL::Hours(1).to_string(), "1h");
		assert_eq!(CacheTTL::Seconds(30).to_string(), "30s");
	}

	#[test]
	fn test_cache_ttl_duration_conversion() {
		let ttl = CacheTTL::Minutes(5);
		assert_eq!(ttl.to_duration(), Duration::from_secs(300));

		let ttl = CacheTTL::Hours(2);
		assert_eq!(ttl.to_duration(), Duration::from_secs(7200));
	}

	#[test]
	fn test_cache_ttl_from_duration() {
		let duration = Duration::from_secs(3600); // 1 hour
		assert_eq!(CacheTTL::from_duration(duration), CacheTTL::Hours(1));

		let duration = Duration::from_secs(300); // 5 minutes
		assert_eq!(CacheTTL::from_duration(duration), CacheTTL::Minutes(5));

		let duration = Duration::from_secs(45); // 45 seconds
		assert_eq!(CacheTTL::from_duration(duration), CacheTTL::Seconds(45));
	}

	#[test]
	fn test_cache_ttl_from_string() {
		assert_eq!(CacheTTL::from_string("5m").unwrap(), CacheTTL::Minutes(5));
		assert_eq!(CacheTTL::from_string("1h").unwrap(), CacheTTL::Hours(1));
		assert_eq!(CacheTTL::from_string("30s").unwrap(), CacheTTL::Seconds(30));

		// Test case insensitive
		assert_eq!(CacheTTL::from_string("5M").unwrap(), CacheTTL::Minutes(5));
		assert_eq!(CacheTTL::from_string("1H").unwrap(), CacheTTL::Hours(1));

		// Test full words
		assert_eq!(
			CacheTTL::from_string("5minutes").unwrap(),
			CacheTTL::Minutes(5)
		);
		assert_eq!(CacheTTL::from_string("1hour").unwrap(), CacheTTL::Hours(1));
	}

	#[test]
	fn test_cache_ttl_from_string_errors() {
		assert!(CacheTTL::from_string("").is_err());
		assert!(CacheTTL::from_string("5").is_err());
		assert!(CacheTTL::from_string("5x").is_err());
		assert!(CacheTTL::from_string("abc").is_err());
	}

	#[test]
	fn test_cache_ttl_predicates() {
		assert!(CacheTTL::Hours(1).is_long());
		assert!(CacheTTL::Hours(2).is_long());
		assert!(CacheTTL::Minutes(59).is_short());
		assert!(CacheTTL::Minutes(5).is_short());
	}

	#[test]
	fn test_cache_config_json() {
		let config = CacheConfig::ephemeral(CacheTTL::Minutes(5));
		let json = config.to_json();

		assert_eq!(json["type"], "ephemeral");
		assert_eq!(json["ttl"], "5m");
	}

	#[test]
	fn test_cache_ttl_from_flag() {
		assert_eq!(CacheTTL::from_long_cache_flag(true), CacheTTL::Hours(1));
		assert_eq!(CacheTTL::from_long_cache_flag(false), CacheTTL::Minutes(5));
	}
}
