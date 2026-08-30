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
fn test_is_retryable_status_429_rate_limit() {
    // 429 (rate limit) should be retryable
    assert!(is_retryable_status(429), "429 should be retryable");
}

#[test]
fn test_is_retryable_status_5xx_server_errors() {
    // All 5xx server errors should be retryable
    assert!(is_retryable_status(500), "500 should be retryable");
    assert!(is_retryable_status(501), "501 should be retryable");
    assert!(is_retryable_status(502), "502 should be retryable");
    assert!(is_retryable_status(503), "503 should be retryable");
    assert!(is_retryable_status(504), "504 should be retryable");
    assert!(is_retryable_status(599), "599 should be retryable");
}

#[test]
fn test_is_retryable_status_4xx_client_errors_not_retryable() {
    // 4xx client errors should NOT be retryable (except 429)
    assert!(!is_retryable_status(400), "400 should not be retryable");
    assert!(!is_retryable_status(401), "401 should not be retryable");
    assert!(!is_retryable_status(403), "403 should not be retryable");
    assert!(!is_retryable_status(404), "404 should not be retryable");
    assert!(!is_retryable_status(405), "405 should not be retryable");
    assert!(!is_retryable_status(408), "408 should not be retryable");
    assert!(!is_retryable_status(418), "418 should not be retryable");
}

#[test]
fn test_is_retryable_status_2xx_success_not_retryable() {
    // 2xx success codes should NOT be retryable (they're already successful)
    assert!(!is_retryable_status(200), "200 should not be retryable");
    assert!(!is_retryable_status(201), "201 should not be retryable");
    assert!(!is_retryable_status(204), "204 should not be retryable");
}

#[test]
fn test_is_retryable_status_3xx_redirect_not_retryable() {
    // 3xx redirect codes should NOT be retryable
    assert!(!is_retryable_status(301), "301 should not be retryable");
    assert!(!is_retryable_status(302), "302 should not be retryable");
    assert!(!is_retryable_status(304), "304 should not be retryable");
}

#[test]
fn test_is_retryable_status_1xx_informational_not_retryable() {
    // 1xx informational codes should NOT be retryable
    assert!(!is_retryable_status(100), "100 should not be retryable");
    assert!(!is_retryable_status(101), "101 should not be retryable");
}
