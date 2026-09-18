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

//! Cloudflare Workers AI provider implementation
//!
//! Uses the OpenAI-compatible endpoint at
//! `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions`.
//!
//! Authentication: Requires API token and Account ID.
//!
//! **How to get credentials:**
//! 1. Cloudflare Dashboard → My Profile → API Tokens
//! 2. Create Token → Use template "Workers AI" or create custom with Workers AI permissions
//! 3. Copy the API token
//! 4. Get Account ID from Cloudflare Dashboard → Workers & Pages (in URL or sidebar)
//! 5. Set environment variables:
//!    - export CLOUDFLARE_API_KEY="your-api-token"
//!    - export CLOUDFLARE_ACCOUNT_ID="your-account-id"
//!    - export CLOUDFLARE_API_URL="..." (optional endpoint override)
//!
//! The API token is sent as a Bearer token in the Authorization header.
//!
//! Frontier models (DeepSeek V4, GLM-5.x, Kimi K2.6/K2.7) require the Workers
//! Paid plan or prepaid AI Gateway credits; on the Free plan they return
//! HTTP 403 with Cloudflare error 5035.
//!
//! The account's Text Generation catalog is fetched once per process from
//! `GET /accounts/{account_id}/ai/models/search`, either through [`preload`]
//! or lazily on the first chat call, and then drives model existence, context
//! window, vision, function calling and pricing. The compiled tables below are
//! the offline fallback and the source for anything the catalog omits.

use crate::llm::providers::openai_compat::{
    chat_completion as openai_compat_chat_completion, get_api_url, OpenAiCompatConfig,
};
use crate::llm::traits::AiProvider;
use crate::llm::types::{ChatCompletionParams, ModelPricing, ProviderResponse, ReasoningEffort};
use crate::llm::utils::{get_model_pricing, normalize_model_name, PricingTuple};
use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;
use std::env;
use tokio::sync::OnceCell;

const CLOUDFLARE_API_BASE: &str = "https://api.cloudflare.com/client/v4/accounts";
const CATALOG_TASK: &str = "Text Generation";
const CATALOG_PAGE_SIZE: usize = 100;
const CATALOG_MAX_PAGES: usize = 20;

/// One Text Generation entry from the account's model catalog.
#[derive(Debug, Clone)]
pub struct CatalogModel {
    pub id: String,
    pub context_window: Option<usize>,
    pub vision: bool,
    pub function_calling: bool,
    pub pricing: Option<ModelPricing>,
    /// `reasoning_effort.supported_efforts`; empty when the model has no effort knob.
    pub supported_efforts: Vec<String>,
    /// `reasoning_effort.normalizes_to`: levels Cloudflare maps itself.
    pub normalized_efforts: Vec<String>,
}

/// Process-wide catalog, filled by [`preload`] or the first chat call.
static CATALOG: OnceCell<Vec<CatalogModel>> = OnceCell::const_new();

#[derive(Deserialize)]
struct SearchResponse {
    #[serde(default)]
    result: Vec<SearchEntry>,
}

/// Catalog item shape: `name` is the `@cf/...` id, `properties` are
/// `{property_id, value}` pairs whose values are strings ("true", "262144")
/// except `price`, an array of `{unit, price, currency}`.
#[derive(Deserialize)]
struct SearchEntry {
    name: String,
    task: Option<SearchTask>,
    #[serde(default)]
    properties: Vec<SearchProperty>,
}

#[derive(Deserialize)]
struct SearchTask {
    name: String,
}

#[derive(Deserialize)]
struct SearchProperty {
    property_id: String,
    #[serde(default)]
    value: Value,
}

fn parse_catalog(entries: Vec<SearchEntry>) -> Vec<CatalogModel> {
    entries
        .into_iter()
        .filter(|entry| {
            entry
                .task
                .as_ref()
                .is_some_and(|task| task.name == CATALOG_TASK)
        })
        .map(|entry| {
            let property = |id: &str| {
                entry
                    .properties
                    .iter()
                    .find(|property| property.property_id == id)
                    .map(|property| &property.value)
            };
            let flag = |id: &str| property(id).and_then(Value::as_str) == Some("true");
            let context_window = property("context_window")
                .and_then(Value::as_str)
                .and_then(|value| value.parse().ok());
            let pricing = property("price")
                .and_then(Value::as_array)
                .and_then(|rates| {
                    let rate = |unit: &str| {
                        rates
                            .iter()
                            .find(|rate| rate.get("unit").and_then(Value::as_str) == Some(unit))
                            .and_then(|rate| rate.get("price"))
                            .and_then(Value::as_f64)
                    };
                    let input = rate("per M input tokens")?;
                    let output = rate("per M output tokens")?;
                    let cached = rate("per M cached input tokens").unwrap_or(input);
                    Some(ModelPricing::new(input, output, input, cached))
                });
            let strings = |value: Option<&Value>| -> Vec<String> {
                value
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let effort = property("reasoning_effort");
            CatalogModel {
                id: entry.name,
                context_window,
                vision: flag("vision"),
                function_calling: flag("function_calling"),
                pricing,
                supported_efforts: strings(effort.and_then(|e| e.get("supported_efforts"))),
                normalized_efforts: effort
                    .and_then(|e| e.get("normalizes_to"))
                    .and_then(Value::as_object)
                    .map(|map| map.keys().cloned().collect())
                    .unwrap_or_default(),
            }
        })
        .collect()
}

async fn fetch_catalog(api_token: &str, account_id: &str) -> Result<Vec<CatalogModel>> {
    let url = format!("{CLOUDFLARE_API_BASE}/{account_id}/ai/models/search");
    let mut models = Vec::new();
    for page in 1..=CATALOG_MAX_PAGES {
        let response = super::shared::http_client()
            .get(&url)
            .query(&[
                ("task", CATALOG_TASK),
                ("per_page", &CATALOG_PAGE_SIZE.to_string()),
                ("page", &page.to_string()),
            ])
            .header("Authorization", format!("Bearer {api_token}"))
            .send()
            .await
            .context("Failed to fetch Cloudflare model catalog")?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Cloudflare models API error {status}: {text}"));
        }
        let list: SearchResponse = response
            .json()
            .await
            .context("Failed to parse Cloudflare model catalog")?;
        let count = list.result.len();
        models.extend(parse_catalog(list.result));
        if count < CATALOG_PAGE_SIZE {
            break;
        }
    }
    Ok(models)
}

/// Fetch the account's Text Generation catalog once per process. Idempotent:
/// the first chat call does the same lazily, and a failure leaves the compiled
/// tables in charge until the next attempt.
pub async fn preload() -> Result<()> {
    let provider = CloudflareWorkersAiProvider::new();
    let api_token = provider.get_api_token()?;
    let account_id = provider.get_account_id()?;
    CATALOG
        .get_or_try_init(|| async move { fetch_catalog(&api_token, &account_id).await })
        .await
        .map(|_| ())
}

fn catalog_model<'a>(catalog: &'a [CatalogModel], model: &str) -> Option<&'a CatalogModel> {
    let normalized = normalize_model_name(model);
    catalog
        .iter()
        .find(|entry| normalize_model_name(&entry.id) == normalized)
}

fn cached_model(model: &str) -> Option<&'static CatalogModel> {
    CATALOG
        .get()
        .and_then(|catalog| catalog_model(catalog, model))
}

/// Cloudflare Workers AI provider
#[derive(Debug, Clone)]
pub struct CloudflareWorkersAiProvider;

impl Default for CloudflareWorkersAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudflareWorkersAiProvider {
    pub fn new() -> Self {
        Self
    }

    /// Get Cloudflare API token
    fn get_api_token(&self) -> Result<String> {
        env::var(CLOUDFLARE_API_KEY_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cloudflare API token not found. Set {} environment variable.\n\
                To create an API token:\n\
                1. Cloudflare Dashboard → My Profile → API Tokens\n\
                2. Create Token → Use 'Workers AI' template or create custom\n\
                3. Ensure token has Workers AI permissions",
                CLOUDFLARE_API_KEY_ENV
            )
        })
    }

    /// Get Cloudflare Account ID
    fn get_account_id(&self) -> Result<String> {
        env::var(CLOUDFLARE_ACCOUNT_ID_ENV).map_err(|_| {
            anyhow::anyhow!(
                "Cloudflare Account ID not found. Set {} environment variable.\n\
                Find your Account ID in Cloudflare Dashboard → Workers & Pages (in URL or sidebar)",
                CLOUDFLARE_ACCOUNT_ID_ENV
            )
        })
    }
}

const CLOUDFLARE_API_KEY_ENV: &str = "CLOUDFLARE_API_KEY";
const CLOUDFLARE_ACCOUNT_ID_ENV: &str = "CLOUDFLARE_ACCOUNT_ID";
const CLOUDFLARE_API_URL_ENV: &str = "CLOUDFLARE_API_URL";

/// Cloudflare Workers AI text-generation prices per 1M tokens, verified
/// Sep 18, 2026 against the pricing page and each model page. Prompt caching
/// is implicit: cache writes bill at the input rate and hits at the cached
/// rate; models without a cached rate bill hits at the input rate.
/// Longer IDs precede the IDs they contain because lookup is a substring match.
/// Format: (model ID, input, output, cache write, cached input).
const PRICING: &[PricingTuple] = &[
    ("@cf/meta/llama-3.2-1b-instruct", 0.027, 0.201, 0.027, 0.027),
    (
        "@cf/meta/llama-3.2-3b-instruct",
        0.0509,
        0.335,
        0.0509,
        0.0509,
    ),
    (
        "@cf/meta/llama-3.2-11b-vision-instruct",
        0.0485,
        0.676,
        0.0485,
        0.0485,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8-fast",
        0.045,
        0.384,
        0.045,
        0.045,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8",
        0.152,
        0.287,
        0.152,
        0.152,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-awq",
        0.123,
        0.266,
        0.123,
        0.123,
    ),
    ("@cf/meta/llama-3.1-8b-instruct", 0.282, 0.827, 0.282, 0.282),
    (
        "@cf/meta/llama-3.1-70b-instruct-fp8-fast",
        0.293,
        2.253,
        0.293,
        0.293,
    ),
    (
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        0.293,
        2.253,
        0.293,
        0.293,
    ),
    (
        "@cf/meta/llama-3-8b-instruct-awq",
        0.123,
        0.266,
        0.123,
        0.123,
    ),
    ("@cf/meta/llama-3-8b-instruct", 0.282, 0.827, 0.282, 0.282),
    ("@cf/meta/llama-2-7b-chat-fp16", 0.556, 6.667, 0.556, 0.556),
    ("@cf/meta/llama-guard-3-8b", 0.484, 0.030, 0.484, 0.484),
    (
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        0.270,
        0.850,
        0.270,
        0.270,
    ),
    (
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        0.497,
        4.881,
        0.497,
        0.497,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        0.440,
        1.320,
        0.440,
        0.014,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        1.320,
        3.960,
        1.320,
        0.044,
    ),
    (
        "@cf/mistral/mistral-7b-instruct-v0.1",
        0.110,
        0.190,
        0.110,
        0.110,
    ),
    (
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        0.351,
        0.555,
        0.351,
        0.351,
    ),
    ("@cf/google/gemma-3-12b-it", 0.345, 0.556, 0.345, 0.345),
    ("@cf/google/gemma-4-26b-a4b-it", 0.100, 0.300, 0.100, 0.100),
    (
        "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
        0.351,
        0.555,
        0.351,
        0.351,
    ),
    ("@cf/qwen/qwq-32b", 0.660, 1.000, 0.660, 0.660),
    (
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        0.660,
        1.000,
        0.660,
        0.660,
    ),
    ("@cf/qwen/qwen3-30b-a3b-fp8", 0.0509, 0.335, 0.0509, 0.0509),
    // The cached rate is on the model page only; the pricing page omits it.
    ("@cf/qwen/qwen3.8-27b", 0.450, 3.200, 0.450, 0.050),
    ("@cf/openai/gpt-oss-120b", 0.350, 0.750, 0.350, 0.350),
    ("@cf/openai/gpt-oss-20b", 0.200, 0.300, 0.200, 0.200),
    (
        "@cf/ibm-granite/granite-4.0-h-micro",
        0.017,
        0.112,
        0.017,
        0.017,
    ),
    ("@cf/zai-org/glm-4.7-flash", 0.0605, 0.400, 0.0605, 0.0605),
    ("@cf/zai-org/glm-5.2", 1.400, 4.400, 1.400, 0.260),
    ("@cf/zai-org/glm-5.3-flash", 0.150, 0.500, 0.150, 0.030),
    ("@cf/zai-org/glm-5.3", 1.400, 4.400, 1.400, 0.260),
    (
        "@cf/nvidia/nemotron-3-120b-a12b",
        0.500,
        1.500,
        0.500,
        0.500,
    ),
    ("@cf/moonshotai/kimi-k2.5", 0.600, 3.000, 0.600, 0.100),
    ("@cf/moonshotai/kimi-k2.6", 0.950, 4.000, 0.950, 0.160),
    ("@cf/moonshotai/kimi-k2.7-code", 0.950, 4.000, 0.950, 0.190),
];

/// Per-model facts from the Workers AI model pages, verified Sep 18, 2026.
/// The last flag marks models whose request schema is the OpenAI chat shape
/// (`tool_choice` incl. "required", `reasoning_effort` low|medium|high,
/// `response_format` json_schema); the rest use the legacy Workers AI schema.
/// Longer IDs precede the IDs they contain because lookup is a substring match.
/// Format: (model ID, context window, vision, function calling, OpenAI-shaped schema).
type ModelFacts = (&'static str, usize, bool, bool, bool);
const MODELS: &[ModelFacts] = &[
    (
        "@cf/meta/llama-3.2-1b-instruct",
        60_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.2-3b-instruct",
        80_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.2-11b-vision-instruct",
        128_000,
        true,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.1-8b-instruct-fp8",
        32_000,
        false,
        false,
        false,
    ),
    (
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        24_000,
        false,
        true,
        false,
    ),
    ("@cf/meta/llama-guard-3-8b", 131_072, false, false, false),
    (
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        131_000,
        true,
        true,
        false,
    ),
    (
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        80_000,
        false,
        false,
        false,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-flash-0731",
        1_048_576,
        false,
        true,
        true,
    ),
    (
        "@cf/deepseek-ai/deepseek-v4-pro-0813",
        1_048_576,
        false,
        true,
        true,
    ),
    (
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        128_000,
        false,
        true,
        false,
    ),
    ("@cf/google/gemma-4-26b-a4b-it", 256_000, true, true, true),
    (
        "@cf/aisingapore/gemma-sea-lion-v4-27b-it",
        128_000,
        false,
        false,
        false,
    ),
    ("@cf/qwen/qwq-32b", 24_000, false, false, false),
    (
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        32_768,
        false,
        false,
        false,
    ),
    ("@cf/qwen/qwen3-30b-a3b-fp8", 32_768, false, true, false),
    ("@cf/qwen/qwen3.8-27b", 262_144, true, true, true),
    ("@cf/openai/gpt-oss-120b", 128_000, false, true, false),
    ("@cf/openai/gpt-oss-20b", 128_000, false, true, false),
    (
        "@cf/ibm-granite/granite-4.0-h-micro",
        131_000,
        false,
        true,
        false,
    ),
    ("@cf/zai-org/glm-4.7-flash", 131_072, false, true, true),
    ("@cf/zai-org/glm-5.2", 262_144, false, true, true),
    ("@cf/zai-org/glm-5.3-flash", 1_310_720, true, true, true),
    ("@cf/zai-org/glm-5.3", 1_310_720, false, true, true),
    (
        "@cf/nvidia/nemotron-3-120b-a12b",
        256_000,
        false,
        true,
        true,
    ),
    ("@cf/moonshotai/kimi-k2.6", 262_144, true, true, true),
    ("@cf/moonshotai/kimi-k2.7-code", 262_144, true, true, true),
];

fn model_facts(model: &str) -> Option<&'static ModelFacts> {
    let normalized = normalize_model_name(model);
    MODELS
        .iter()
        .find(|(id, ..)| normalized.contains(&normalize_model_name(id)))
}

fn cloudflare_model_pricing(model: &str) -> Option<crate::llm::types::ModelPricing> {
    let (input, output, cache_write, cache_read) = get_model_pricing(model, PRICING)?;
    Some(crate::llm::types::ModelPricing::new(
        input,
        output,
        cache_write,
        cache_read,
    ))
}

fn default_cloudflare_api_url(account_id: &str) -> String {
    format!(
        "https://api.cloudflare.com/client/v4/accounts/{}/ai/v1/chat/completions",
        account_id
    )
}

const EFFORT_LADDER: [(ReasoningEffort, &str); 5] = [
    (ReasoningEffort::Low, "low"),
    (ReasoningEffort::Medium, "medium"),
    (ReasoningEffort::High, "high"),
    (ReasoningEffort::XHigh, "xhigh"),
    (ReasoningEffort::Max, "max"),
];

/// The `reasoning_effort` value to send for a catalog entry. A supported level
/// goes verbatim, and so does one Cloudflare documents it normalizes itself
/// (the catalog lists e.g. `medium -> max` for GLM-5.3). Anything else floors
/// to the nearest supported level below, or the lowest supported level when
/// nothing is below. `None` when the model has no effort knob, which leaves
/// the shared ladder in charge.
fn select_effort(entry: &CatalogModel, effort: ReasoningEffort) -> Option<&'static str> {
    if entry.supported_efforts.is_empty() {
        return None;
    }
    let supported = |name: &str| entry.supported_efforts.iter().any(|s| s == name);
    let index = EFFORT_LADDER
        .iter()
        .position(|(level, _)| *level == effort)?;
    let requested = EFFORT_LADDER[index].1;
    if supported(requested) || entry.normalized_efforts.iter().any(|s| s == requested) {
        return Some(requested);
    }
    EFFORT_LADDER[..index]
        .iter()
        .rev()
        .chain(EFFORT_LADDER[index + 1..].iter())
        .map(|(_, name)| *name)
        .find(|name| supported(name))
}

/// Catalog-driven effort for a Cloudflare model once the catalog is loaded.
pub(crate) fn catalog_reasoning_effort(
    model: &str,
    effort: ReasoningEffort,
) -> Option<&'static str> {
    cached_model(model).and_then(|entry| select_effort(entry, effort))
}

#[async_trait::async_trait]
impl AiProvider for CloudflareWorkersAiProvider {
    fn name(&self) -> &str {
        "cloudflare"
    }

    fn supports_model(&self, model: &str) -> bool {
        if model.is_empty() {
            return false;
        }
        // Exact catalog membership once it is loaded; anything goes before.
        CATALOG
            .get()
            .map(|catalog| catalog_model(catalog, model).is_some())
            .unwrap_or(true)
    }

    fn get_api_key(&self) -> Result<String> {
        // Cloudflare Workers AI requires both API token and account ID
        let api_token = self.get_api_token()?;
        let _account_id = self.get_account_id()?; // Validate it exists
        Ok(api_token) // Return API token as the "API key"
    }

    fn supports_required_tool_choice(&self, model: &str) -> bool {
        model_facts(model).is_some_and(|(_, _, _, _, openai_schema)| *openai_schema)
    }

    fn supports_caching(&self, model: &str) -> bool {
        cached_model(model)
            .and_then(|entry| entry.pricing)
            .or_else(|| cloudflare_model_pricing(model))
            .map(|pricing| pricing.cache_read_price_per_1m < pricing.input_price_per_1m)
            .unwrap_or(false)
    }

    fn supports_vision(&self, model: &str) -> bool {
        if let Some(entry) = cached_model(model) {
            return entry.vision;
        }
        if let Some((_, _, vision, _, _)) = model_facts(model) {
            return *vision;
        }
        // Check Cloudflare-specific naming patterns first
        let model_lower = normalize_model_name(model);
        if model_lower.contains("vision") || model_lower.contains("@cf/llava") {
            return true;
        }
        // Fall back to reference capabilities for the underlying model
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.vision)
            .unwrap_or(false)
    }

    /// Legacy-schema models reach a schema only through a forced tool call (see
    /// `enforces_response_schema`), so structured output needs function calling.
    fn supports_structured_output(&self, model: &str) -> bool {
        if let Some(entry) = cached_model(model) {
            return entry.function_calling;
        }
        match model_facts(model) {
            Some((_, _, _, function_calling, _)) => *function_calling,
            None => crate::llm::reference_models::get_reference_capabilities(model)
                .map(|c| c.structured_output)
                .unwrap_or(false),
        }
    }

    /// OpenAI-shaped models decode against `response_format` json_schema: a
    /// prompt demanding prose still got the single value an enum-only schema
    /// allowed, 12 of 12 times (GLM-5.3 Flash, DeepSeek V4 Flash, Kimi K2.6,
    /// Sep 18, 2026). The docs' "can't guarantee" caveat surfaces as an error
    /// ("JSON Mode couldn't be met"), not as output of another shape. Legacy
    /// models fall back to the forced tool call.
    fn enforces_response_schema(&self, model: &str) -> bool {
        model_facts(model).is_some_and(|(_, _, _, _, openai_schema)| *openai_schema)
    }

    fn get_model_pricing(&self, model: &str) -> Option<ModelPricing> {
        cached_model(model)
            .and_then(|entry| entry.pricing)
            .or_else(|| cloudflare_model_pricing(model))
            .or_else(|| crate::llm::reference_models::get_reference_pricing(model))
    }

    fn get_max_input_tokens(&self, model: &str) -> usize {
        if let Some(context_window) = cached_model(model).and_then(|entry| entry.context_window) {
            return context_window;
        }
        if let Some((_, context_window, ..)) = model_facts(model) {
            return *context_window;
        }
        // Use reference capabilities for model-specific context windows
        crate::llm::reference_models::get_reference_capabilities(model)
            .map(|c| c.max_input_tokens)
            .unwrap_or(4_096) // Conservative default for Cloudflare's smaller models
    }

    async fn chat_completion(&self, params: ChatCompletionParams) -> Result<ProviderResponse> {
        let api_key = self.get_api_key()?;
        let account_id = self.get_account_id()?;
        let api_url = get_api_url(
            CLOUDFLARE_API_URL_ENV,
            &default_cloudflare_api_url(&account_id),
        );

        // Lazy catalog load on first call; errors are ignored and retried next call.
        let token = api_key.clone();
        let account = account_id.clone();
        let _ = CATALOG
            .get_or_try_init(|| async move { fetch_catalog(&token, &account).await })
            .await;

        let model = params.model.clone();
        let mut response = openai_compat_chat_completion(
            OpenAiCompatConfig {
                provider_name: "cloudflare",
                usage_fallback_cost: None,
                use_response_cost: true,
                enforces_response_schema: self.enforces_response_schema(&model),
                supports_required_tool_choice: self.supports_required_tool_choice(&model),
            },
            api_key,
            api_url,
            params,
        )
        .await?;

        if let Some(ref mut usage) = response.exchange.usage {
            if usage.cost.is_none() {
                if let Some(pricing) = self.get_model_pricing(&model) {
                    usage.cost = Some(pricing.calculate_cost(
                        usage.input_tokens,
                        usage.cache_write_tokens,
                        usage.cache_read_tokens,
                        usage.billable_output_tokens(),
                    ));
                }
            }
        }

        Ok(response)
    }
}

#[cfg(test)]
#[path = "cloudflare_tests.rs"]
mod tests;
