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

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "octolib-chatgpt-{}-{}",
        tag,
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn registration(expires_at: u64) -> Registration {
    Registration {
        client_id: "client-1".to_string(),
        subject: "user-1".to_string(),
        email: Some("dev@example.com".to_string()),
        id_token: "id".to_string(),
        access_token: "access".to_string(),
        refresh_token: "refresh".to_string(),
        expires_at,
        scope: SCOPE.to_string(),
    }
}

#[test]
fn host_id_is_created_once_and_reused() {
    let dir = temp_dir("host");
    let first = host_id(&dir).unwrap();
    assert!(first.starts_with("urn:uuid:"), "{}", first);
    assert_eq!(host_id(&dir).unwrap(), first);
}

#[test]
fn registration_round_trips_owner_only() {
    let path = temp_dir("registration").join(AUTH_FILE);
    assert!(load_registration(&path).unwrap().is_none());

    save_registration(&path, &registration(42)).unwrap();
    let loaded = load_registration(&path)
        .unwrap()
        .expect("saved registration");
    assert_eq!(loaded.client_id, "client-1");
    assert_eq!(loaded.refresh_token, "refresh");
    assert_eq!(loaded.expires_at, 42);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
}

#[test]
fn missing_registration_means_not_signed_in() {
    let path = temp_dir("missing").join(AUTH_FILE);
    let error = require_registration(&path).err().expect("no registration");
    assert!(
        error.to_string().contains("Not signed in with ChatGPT"),
        "{}",
        error
    );
}

#[test]
fn tokens_refresh_inside_the_margin() {
    let registration = registration(1_000);
    assert!(!registration.expires_soon(1_000 - REFRESH_MARGIN_SECS - 1));
    assert!(registration.expires_soon(1_000 - REFRESH_MARGIN_SECS));
}

#[test]
fn id_token_claims_come_from_the_payload() {
    let payload =
        URL_SAFE_NO_PAD.encode(r#"{"sub":"user-1","nonce":"n-1","email":"dev@example.com"}"#);
    let claims = id_token_claims(&format!("e30.{}.signature", payload)).unwrap();
    assert_eq!(claims.sub, "user-1");
    assert_eq!(claims.nonce, "n-1");
    assert_eq!(claims.email.as_deref(), Some("dev@example.com"));

    assert!(id_token_claims("not-a-jwt").is_err());
}

#[tokio::test]
async fn callback_server_survives_idle_and_unrelated_connections() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();

    let browser = tokio::spawn(async move {
        // A speculative connection that never sends a request.
        let _idle = TcpStream::connect(address).await.unwrap();

        let mut favicon = TcpStream::connect(address).await.unwrap();
        favicon
            .write_all(b"GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .await
            .unwrap();
        let mut not_found = String::new();
        favicon.read_to_string(&mut not_found).await.unwrap();
        assert!(not_found.starts_with("HTTP/1.1 404"), "{}", not_found);

        let mut callback = TcpStream::connect(address).await.unwrap();
        callback
            .write_all(
                b"GET /callback?code=abc&state=s-1&client_id=client-9 HTTP/1.1\r\nHost: localhost\r\n\r\n",
            )
            .await
            .unwrap();
        let mut page = String::new();
        callback.read_to_string(&mut page).await.unwrap();
        assert!(page.starts_with("HTTP/1.1 200"), "{}", page);
    });

    let params = wait_for_callback(&listener).await.unwrap();
    browser.await.unwrap();

    assert_eq!(params["code"], "abc");
    assert_eq!(params["state"], "s-1");
    assert_eq!(params["client_id"], "client-9");
}
