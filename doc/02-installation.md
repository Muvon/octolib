# Octolib: Installation and Setup

## Requirements

- Rust 1.70+ (Recommended: Latest stable version)
- Cargo
- Internet connection for dependency downloads

## Installation Methods

### 1. Git Repository (Recommended)

Add to your `Cargo.toml`:

```toml
[dependencies]
octolib = { git = "https://github.com/muvon/octolib" }
```

### 2. Specific Version from Git

Pin to a specific commit or branch:

```toml
[dependencies]
octolib = { git = "https://github.com/muvon/octolib", branch = "main" }
# Or specific commit
octolib = { git = "https://github.com/muvon/octolib", rev = "abc123" }
```

### 3. Local Path (Development)

```toml
[dependencies]
octolib = { path = "/path/to/local/octolib" }
```

## Environment Setup

### API Keys

Each provider requires its specific API key:

```bash
# OpenAI
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# OpenRouter
export OPENROUTER_API_KEY="your_openrouter_api_key"

# Google Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
```

### Rust Project Setup

1. Create a new Rust project:
```bash
cargo new my_ai_project
cd my_ai_project
```

2. Add Octolib to `Cargo.toml`

3. Enable async runtime in `Cargo.toml`:
```toml
[dependencies]
tokio = { version = "1.45.1", features = ["full"] }
```

## Async Runtime

Octolib is designed to work with async Rust. We recommend using Tokio:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Your async code here
    Ok(())
}
```

## Troubleshooting

### Common Issues

1. **Dependency Conflicts**
   - Ensure compatible versions of `serde`, `reqwest`, and `tokio`
   - Use `cargo update` to resolve minor version conflicts

2. **API Key Errors**
   - Double-check environment variable names
   - Verify API key validity with provider's dashboard

3. **SSL/TLS Issues**
   - Ensure up-to-date CA certificates
   - Use `rustls-tls` feature if needed

## Development Dependencies

For development and testing:

```toml
[dev-dependencies]
anyhow = "1.0.98"
tokio = { version = "1.45.1", features = ["test-util", "macros"] }
```

## Minimum Supported Rust Version (MSRV)

- Rust 1.70.0
- Recommended: Latest stable Rust version

## Security Recommendations

- Never commit API keys to version control
- Use environment variables or secure secret management
- Rotate API keys periodically
- Set up API key restrictions in provider dashboards

## Next Steps

- [Advanced Usage](03-advanced-usage.md)
- [Provider Configuration](04-advanced-guide.md)

## Contribution

Found an installation issue? [Open an Issue](https://github.com/Muvon/octolib/issues)
