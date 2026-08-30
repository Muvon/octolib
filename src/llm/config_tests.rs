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
