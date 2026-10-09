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

// Consume complete requests so refresh grant bodies and replayed headers can be asserted.
async fn mock_server(
    responses: Vec<(u16, serde_json::Value)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for (status, body) in responses {
            let (mut stream, _) = listener.accept().await.unwrap();
            requests.push(read_http_request(&mut stream).await);
            let body = body.to_string();
            stream.write_all(format!(
                "HTTP/1.1 {} Response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status, body.len(), body
            ).as_bytes()).await.unwrap();
        }
        requests
    });
    (format!("http://{}", address), server)
}

async fn read_http_request(stream: &mut TcpStream) -> String {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await.unwrap();
        assert!(read > 0, "request ended early");
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(end) = buffer.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
            let headers = std::str::from_utf8(&buffer[..end]).unwrap();
            let length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().unwrap())
                })
                .unwrap_or(0);
            if buffer.len() >= end + 4 + length {
                return String::from_utf8(buffer).unwrap();
            }
        }
    }
}

fn rotated_tokens() -> serde_json::Value {
    serde_json::json!({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})
}

#[tokio::test]
async fn rejected_access_token_refreshes_and_replays_the_request() {
    let path = temp_dir("rejected").join(AUTH_FILE);
    save_registration(&path, &registration(unix_now().unwrap() + 3600)).unwrap();
    let (url, server) = mock_server(vec![
        (401, serde_json::json!({"error": "invalid_token"})),
        (200, rotated_tokens()),
        (200, serde_json::json!({"models": []})),
    ])
    .await;
    let headers = HashMap::from([("x-test".to_string(), "kept".to_string())]);
    let response = send_authenticated_at(
        &path,
        &format!("{}/token", url),
        shared::http_client()
            .post(format!("{}/models", url))
            .json(&serde_json::json!({"input": "hello"})),
        Some(Duration::from_secs(5)),
        Some(&headers),
    )
    .await
    .unwrap();
    assert!(response.status.is_success());
    let requests = server.await.unwrap();
    assert!(requests[0].contains("authorization: Bearer access\r\n"));
    assert!(requests[2].contains("authorization: Bearer new-access\r\n"));
    for index in [0, 2] {
        assert!(requests[index].contains("x-test: kept\r\n"));
        assert!(requests[index].ends_with(r#"{"input":"hello"}"#));
    }
    let form: HashMap<_, _> = reqwest::Url::parse(&format!(
        "http://localhost/?{}",
        requests[1].split_once("\r\n\r\n").unwrap().1
    ))
    .unwrap()
    .query_pairs()
    .into_owned()
    .collect();
    assert_eq!(form["grant_type"], "refresh_token");
    assert_eq!(form["client_id"], "client-1");
    assert_eq!(form["refresh_token"], "refresh");
    assert_eq!(form["resource"], RESOURCE);
    assert_eq!(
        require_registration(&path).unwrap().refresh_token,
        "new-refresh"
    );
}

#[tokio::test]
async fn a_second_unauthorized_response_is_not_retried() {
    let path = temp_dir("unauthorized").join(AUTH_FILE);
    save_registration(&path, &registration(unix_now().unwrap() + 3600)).unwrap();
    let (url, server) = mock_server(vec![
        (401, serde_json::json!({})),
        (200, rotated_tokens()),
        (401, serde_json::json!({})),
    ])
    .await;
    let response = send_authenticated_at(&path, &url, shared::http_client().get(&url), None, None)
        .await
        .unwrap();
    assert_eq!(response.status, reqwest::StatusCode::UNAUTHORIZED);
    assert_eq!(server.await.unwrap().len(), 3);
}

#[tokio::test]
async fn non_authentication_errors_do_not_rotate_credentials() {
    let path = temp_dir("forbidden").join(AUTH_FILE);
    save_registration(&path, &registration(unix_now().unwrap() + 3600)).unwrap();
    let (url, server) = mock_server(vec![(403, serde_json::json!({"error": "usage_limit"}))]).await;
    let response = send_authenticated_at(&path, &url, shared::http_client().get(&url), None, None)
        .await
        .unwrap();
    assert_eq!(response.status, reqwest::StatusCode::FORBIDDEN);
    assert_eq!(server.await.unwrap().len(), 1);
    assert_eq!(
        require_registration(&path).unwrap().refresh_token,
        "refresh"
    );
}

#[tokio::test]
async fn refresh_failure_preserves_credentials_and_requires_sign_in() {
    let path = temp_dir("failed-refresh").join(AUTH_FILE);
    save_registration(&path, &registration(unix_now().unwrap() + 3600)).unwrap();
    let (url, server) = mock_server(vec![
        (401, serde_json::json!({})),
        (400, serde_json::json!({"error": "invalid_grant"})),
    ])
    .await;
    let error = crate::llm::retry::retry_with_exponential_backoff(
        || {
            let path = path.clone();
            let url = url.clone();
            Box::pin(async move {
                match send_authenticated_at(
                    &path,
                    &url,
                    shared::http_client().get(&url),
                    None,
                    None,
                )
                .await
                {
                    Ok(response) => Ok(Ok::<_, anyhow::Error>(response)),
                    Err(error) if error.is::<reqwest::Error>() => Err(error),
                    Err(error) => Ok(Err(error)),
                }
            })
        },
        3,
        Duration::ZERO,
        None,
        || anyhow::anyhow!("cancelled"),
        |_| false,
        |_| false,
    )
    .await
    .unwrap()
    .err()
    .unwrap();
    assert!(error.to_string().contains("--force"));
    assert_eq!(server.await.unwrap().len(), 2);
    assert_eq!(
        require_registration(&path).unwrap().refresh_token,
        "refresh"
    );
}

#[tokio::test]
async fn concurrent_expiry_and_rejection_refresh_only_once() {
    for (tag, expiry, rejected) in [
        ("expired", 0, None),
        (
            "concurrent-rejected",
            unix_now().unwrap() + 3600,
            Some("access"),
        ),
    ] {
        let path = temp_dir(tag).join(AUTH_FILE);
        save_registration(&path, &registration(expiry)).unwrap();
        let (url, server) = mock_server(vec![(200, rotated_tokens())]).await;
        let (first, second) = tokio::join!(
            access_token_at(&path, &url, rejected),
            access_token_at(&path, &url, rejected)
        );
        assert_eq!(first.unwrap(), "new-access");
        assert_eq!(second.unwrap(), "new-access");
        assert_eq!(server.await.unwrap().len(), 1);
        assert_eq!(
            require_registration(&path).unwrap().refresh_token,
            "new-refresh"
        );
    }
}

#[tokio::test]
async fn waiting_refresh_uses_replacement_login_credentials() {
    use std::future::{poll_fn, Future};
    use std::task::Poll;

    let path = temp_dir("replacement-login").join(AUTH_FILE);
    save_registration(&path, &registration(unix_now().unwrap() + 3600)).unwrap();
    let lock = lock_refresh(&path).await.unwrap();
    let mut refresh = Box::pin(access_token_at(&path, "http://127.0.0.1:1", Some("access")));
    // Poll through the initial read into the blocked lock acquisition, without sleeps.
    poll_fn(|cx| {
        assert!(refresh.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    let mut replacement = registration(unix_now().unwrap() + 3600);
    replacement.access_token = "login-access".to_string();
    replacement.refresh_token = "login-refresh".to_string();
    save_registration(&path, &replacement).unwrap();
    drop(lock);
    assert_eq!(refresh.await.unwrap(), "login-access");
    assert_eq!(
        require_registration(&path).unwrap().refresh_token,
        "login-refresh"
    );
}

#[tokio::test]
async fn login_holds_the_refresh_lock_during_token_exchange() {
    let path = temp_dir("login-lock").join(AUTH_FILE);
    save_registration(&path, &registration(0)).unwrap();
    let token_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let token_url = format!("http://{}", token_listener.local_addr().unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let callback_url = format!(
        "http://{}/callback?code=abc&state=s-1&client_id=client-9",
        listener.local_addr().unwrap()
    );
    let pending = PendingLogin {
        listener,
        authorize_url: String::new(),
        redirect_uri: "http://localhost/callback".to_string(),
        state: "s-1".to_string(),
        nonce: "n-1".to_string(),
        code_verifier: "verifier".to_string(),
        saved_client_id: None,
        auth_path: path.clone(),
    };
    let server_path = path.clone();
    let server = tokio::spawn(async move {
        let (mut stream, _) = token_listener.accept().await.unwrap();
        let request = read_http_request(&mut stream).await;
        assert!(request.contains("grant_type=authorization_code"));
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .open(server_path.with_file_name(format!(".{AUTH_FILE}.lock")))
            .unwrap();
        assert!(
            fs4::FileExt::try_lock_exclusive(&lock).is_err(),
            "login must hold the same lock as refresh"
        );
        let id_token = format!(
            "e30.{}.signature",
            URL_SAFE_NO_PAD.encode(r#"{"sub":"user-1","nonce":"n-1"}"#)
        );
        let body = serde_json::json!({
            "access_token": "login-access", "refresh_token": "login-refresh", "id_token": id_token,
            "expires_in": 3600, "scope": PLAN_USAGE_SCOPE,
        })
        .to_string();
        stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(), body).as_bytes()).await.unwrap();
    });
    let (account, callback) = tokio::join!(
        pending.finish_with_token_url(&token_url),
        shared::http_client().get(callback_url).send()
    );
    account.unwrap();
    assert!(callback.unwrap().status().is_success());
    server.await.unwrap();
    assert_eq!(
        require_registration(&path).unwrap().refresh_token,
        "login-refresh"
    );
    let lock = lock_refresh(&path).await.unwrap();
    drop(lock);
}
