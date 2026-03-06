# Changelog

## [0.10.1] - 2026-03-06

### 📋 Release Summary

This release expands embedding capabilities with OpenRouter provider support and delivers more reliable model availability through live API integration (839ad88e, 08db93d4).


### ✨ New Features & Enhancements

- **embedding**: add OpenRouter provider `839ad88e`

### 🔧 Improvements & Optimizations

- **openrouter**: replace hardcoded model list with live API fetch `08db93d4`

### 📊 Release Summary

**Total commits**: 2 across 2 categories

✨ **1** new feature - *Enhanced functionality*
🔧 **1** improvement - *Better performance & code quality*

## [0.10.0] - 2026-03-05

### 📋 Release Summary

This release expands reranking capabilities with new Mixedbread, HuggingFace, and Cohere v4 providers (f75d35d2). Multiple provider integrations are now more reliable with fixes for structured outputs and API compatibility across Minimax, Moonshot, Ollama, and Voyage (5366a204, b8bd46aa, 77512136, 6d551189).


### ✨ New Features & Enhancements

- **reranker**: add Mixedbread, HuggingFace providers and Cohere v4 `f75d35d2`

### 🐛 Bug Fixes & Stability

- **minimax**: correct structured output support test `5366a204`
- **minimax**: correct structured output support flag `b8bd46aa`
- **llm**: correct schema handling for Moonshot and Ollama providers `77512136`
- **voyage**: adapt to new API response structure `6d551189`

### 🔄 Other Changes

- update dependencies `f67e17f2`
- update dependencies `5e162605`

### 📊 Release Summary

**Total commits**: 7 across 3 categories

✨ **1** new feature - *Enhanced functionality*
🐛 **4** bug fixes - *Improved stability*
🔄 **2** other changes - *Maintenance & tooling*

## [0.9.3] - 2026-02-19

### 📋 Release Summary

This release improves reliability with automatic retry logic for network issues and delivers faster embedding performance through optimized processing. Provider management has been streamlined for better efficiency.


### 🔧 Improvements & Optimizations

- **providers**: extract cache and tool utilities `9d5939a9`
- **embedding**: optimize tokenizer and parsing logic `8bd33aaf`

### 🐛 Bug Fixes & Stability

- **llm**: add retry logic for HTTP errors `378054ae`

### 📊 Release Summary

**Total commits**: 3 across 2 categories

🔧 **2** improvements - *Better performance & code quality*
🐛 **1** bug fix - *Improved stability*

## [0.9.2] - 2026-02-18

### 📋 Release Summary

This release corrects pricing for GLM-4.7 flash models, ensuring they are now recognized as free across all supported providers.


### 🐛 Bug Fixes & Stability

- **zai**: set GLM-4.7 flash models to free pricing `9ca66b0a`

### 📊 Release Summary

**Total commits**: 1 across 1 categories

🐛 **1** bug fix - *Improved stability*

## [0.9.1] - 2026-02-15

### 📋 Release Summary

This release adds support for zero-cost proxy providers, allowing seamless integration with services that don’t charge per request.


### ✨ New Features & Enhancements

- **llm**: add zero pricing for proxy providers `f4f6d8a9`

### 📊 Release Summary

**Total commits**: 1 across 1 categories

✨ **1** new feature - *Enhanced functionality*

## [0.9.0] - 2026-02-15

### 📋 Release Summary

This release expands AI provider support with Google Vertex AI, Amazon Bedrock, Cerebras AI, and Ollama integrations, plus video capabilities for OpenRouter and Kimi K2.5 (cdb22f47, 6346769b, 6de74c29, 5e5cd014, d0a742b0). Token counting accuracy and model compatibility are improved across Anthropic, Moonshot, and ZAI providers (e6585b41, c2c14ce2, 7d6f693b, 46a52427).


### ✨ New Features & Enhancements

- **llm**: add video support to openrouter and enable local providers by default `5e5cd014`
- **vision**: enable video attachments for Kimi K2.5 `d0a742b0`
- **providers**: add Google Vertex AI and Amazon Bedrock support `cdb22f47`
- **provider**: add Cerebras AI support `6346769b`
- **llm**: add Ollama provider with OpenAI-compatible endpoint `6de74c29`

### 🐛 Bug Fixes & Stability

- **zai**: estimate reasoning tokens from thinking block `e6585b41`
- **anthropic**: exclude opus 4.6 from temperature and top p support `c2c14ce2`
- **moonshot**: correct field name from input_tokens to prompt_tokens `7d6f693b`
- **doc**: correct URL formatting in rustdoc comments `5ffc4d5f`
- **openai_compat**: resolve clippy warning for or_else usage `6e2166ea`
- **zai**: correct input token counting to exclude cached reads `46a52427`

### 📊 Release Summary

**Total commits**: 11 across 2 categories

✨ **5** new features - *Enhanced functionality*
🐛 **6** bug fixes - *Improved stability*

## [0.8.3] - 2026-02-14

### 📋 Release Summary

This release improves cost tracking accuracy for cached requests and unifies pricing structures across all AI providers (d93c6091, 1359bdf5, c57d862c, f6872ead). Enhanced model validation and updated pre-commit hooks ensure more reliable provider integrations (3bc7ae08, 62a4e00a).


### 🔧 Improvements & Optimizations

- **llm**: unify pricing table to 5-tuple format `c57d862c`
- **llm**: unify pricing structure across providers `f6872ead`
- **llm**: rename prompt_tokens to input_tokens and add cache fields `2f7f6e66`

### 🐛 Bug Fixes & Stability

- **zai**: handle cache read tokens in cost calculation `d93c6091`
- **anthropic**: restore dot notation model aliases `dfb56d78`
- **llm**: correct token calculation logic for cached requests `1359bdf5`

### 🔄 Other Changes

- **release**: 0.8.3" `549abe53`
- **release**: 0.8.3 `371da6c0`
- **pre-commit**: update hooks and model validation `3bc7ae08`
- **pre-commit**: add cargo doc check to precommit hooks `62a4e00a`

### 📊 Release Summary

**Total commits**: 10 across 3 categories

🔧 **3** improvements - *Better performance & code quality*
🐛 **3** bug fixes - *Improved stability*
🔄 **4** other changes - *Maintenance & tooling*

## [0.8.2] - 2026-02-13

### 📋 Release Summary

This release adds MiniMax-M2.5 and the latest February 2026 models with refreshed pricing, expanding your provider choices. All cached-input costs are now tracked accurately, so usage reports and budgets reflect real spend.


### ✨ New Features & Enhancements

- **minimax**: add MiniMax-M2.5 model support with updated pricing `ffe2326c`
- **pricing**: add latest model pricing for Feb 2026 `52b76516`

### 🐛 Bug Fixes & Stability

- **openai**: add cached input pricing and update model support `52a5218f`
- **llm**: correct cached token calculation for providers `b46e3c7d`

### 📊 Release Summary

**Total commits**: 4 across 2 categories

✨ **2** new features - *Enhanced functionality*
🐛 **2** bug fixes - *Improved stability*

## [0.8.1] - 2026-02-11

### 📋 Release Summary

This release improves cost tracking accuracy by fixing token caching calculations for Moonshot and other providers (956423a9, ee7e0094). Enhanced error handling ensures more reliable provider operations (118c6b46).


### 🔧 Improvements & Optimizations

- **openrouter**: replace panics with error handling `118c6b46`

### 🐛 Bug Fixes & Stability

- **moonshot**: fix cached_tokens detection and remove deprecated manual caching `956423a9`
- **llm**: correct cache token calculation for pricing `ee7e0094`

### 📊 Release Summary

**Total commits**: 3 across 2 categories

🔧 **1** improvement - *Better performance & code quality*
🐛 **2** bug fixes - *Improved stability*

## [0.8.0] - 2026-02-10

### 📋 Release Summary

This release adds comprehensive Moonshot AI provider support with pricing, context caching, and reasoning capabilities for advanced thinking models (ee8c7233, 23285017, 622f9bf5, 5b3150a1). All providers now include model pricing support for better cost tracking and transparency (56fec226, 3ff3d62f). Documentation has been expanded with new provider guides and enhanced reranking/thinking support details (534f8f94, f21a41a9).


### ✨ New Features & Enhancements

- **moonshot**: add new model pricing and support `ee8c7233`
- **moonshot**: add context caching support `23285017`
- **llm**: add pricing function to minimax, moonshot, and zai providers `56fec226`
- **moonshot**: add reasoning_content support for kimi-k2.5 thinking mode `622f9bf5`
- **pricing**: add model pricing support to all providers `3ff3d62f`
- **providers**: add Moonshot AI provider support `5b3150a1`

### 📚 Documentation & Examples

- expand reranking and thinking support documentation `534f8a94`
- add Moonshot, Cohere, Jina, FastEmbed providers `f21a41a9`

### 🔄 Other Changes

- **ci**: remove disk space cleanup step `31de438c`
- **ci**: use v10 version of maximize-build-space `68dfba39`
- bump Rust version to 1.92.0 `6a2b74e2`
- remove redundant cargo clean step `5951d57d`
- remove CARGO_TARGET_DIR environment variable `25d339be`
- **coverage**: fix tarpaulin configuration for CI `7799dda2`
- **deps**: update deps versions `2464c2eb`

### 📊 Release Summary

**Total commits**: 15 across 3 categories

✨ **6** new features - *Enhanced functionality*
📚 **2** documentation updates - *Better developer experience*
🔄 **7** other changes - *Maintenance & tooling*

## [0.7.0] - 2026-02-03

### 📋 Release Summary

This release introduces cross-encoder reranking capabilities with new provider support for Cohere, Jina, and FastEmbed. A bug fix replaces a deprecated FastEmbed model with updated API syntax. Documentation and tests accompany the new reranking functionality.


### ✨ New Features & Enhancements

- **reranker**: add Cohere, Jina, and FastEmbed providers `6a940561`
- **reranker**: add cross-encoder reranking module `3d3edd32`

### 🐛 Bug Fixes & Stability

- **fastembed**: replace deprecated model and fix API syntax `b6a2cb72`

### 📚 Documentation & Examples

- add reranker docs and reorganize file order `8382acd4`

### 🔄 Other Changes

- test(reranker): make provider tests resilient in CI `55191b94`

### 📊 Release Summary

**Total commits**: 5 across 4 categories

✨ **2** new features - *Enhanced functionality*
🐛 **1** bug fix - *Improved stability*
📚 **1** documentation update - *Better developer experience*
🔄 **1** other change - *Maintenance & tooling*

## [0.6.0] - 2026-02-01

### 📋 Release Summary

This release expands multi-provider support with new Codex AI integration and local LLM capabilities, along with enhanced DeepSeek reasoning content features. Additional improvements include provider cancellation support and updated embedding model documentation.


### ✨ New Features & Enhancements

- **providers**: add Codex AI and fix DeepSeek reasoning_content `1c315e7f`
- **deepseek**: add reasoning content support `98497c54`
- **factory**: add local LLM provider support `9608f6d1`

### 🔧 Improvements & Optimizations

- **llm**: add cancellation support to providers `6cd2183d`
- **llm**: rename codex to cli provider `23054c48`

### 📚 Documentation & Examples

- **embedding**: document model dimensions and specs `39dd6a15`

### 📊 Release Summary

**Total commits**: 6 across 3 categories

✨ **3** new features - *Enhanced functionality*
🔧 **2** improvements - *Better performance & code quality*
📚 **1** documentation update - *Better developer experience*

## [0.5.1] - 2026-01-22

### 📋 Release Summary

This release enhances multi-provider support with improved thinking block parsing capabilities across providers (60010513, 8e66dc2c), and standardizes response field naming for a more consistent API experience (1c9be4e0).


### ✨ New Features & Enhancements

- **anthropic**: add thinking block parsing support `60010513`

### 🔧 Improvements & Optimizations

- **llm**: rename response_id fields to id `1c9be4e0`

### 🐛 Bug Fixes & Stability

- **zai**: handle thinking parsing for zai provider `8e66dc2c`

### 📊 Release Summary

**Total commits**: 3 across 3 categories

✨ **1** new feature - *Enhanced functionality*
🔧 **1** improvement - *Better performance & code quality*
🐛 **1** bug fix - *Improved stability*

## [0.5.0] - 2026-01-21

### 📋 Release Summary

This release enhances cost tracking with cache token pricing support for OpenAI and improves ZAI provider reliability through better model matching and documentation fixes. General improvements include updated OAuth documentation and cross-provider consistency enhancements.


### ✨ New Features & Enhancements

- **openai**: add cache token pricing for cost calculation `35723c98`

### 🔧 Improvements & Optimizations

- **providers**: add response_id across providers `9a754138`

### 🐛 Bug Fixes & Stability

- **llm/providers/zai**: format URL in documentation comment `95e226bf`
- **zai**: case-insensitive model matching `9c2b0053`

### 📚 Documentation & Examples

- update OAuth and provider support documentation `444b7b17`

### 📊 Release Summary

**Total commits**: 5 across 4 categories

✨ **1** new feature - *Enhanced functionality*
🔧 **1** improvement - *Better performance & code quality*
🐛 **2** bug fixes - *Improved stability*
📚 **1** documentation update - *Better developer experience*

## [0.4.2] - 2026-01-17

### 📋 Release Summary

Several bug fixes improve multi-provider functionality, including case-insensitive model name matching, fixed tool call argument handling for Zai, and structured output support for Minimax (bd85bc7c, 42da256b, 7723a7c9).


### 🐛 Bug Fixes & Stability

- **providers**: add case-insensitive model name matching `bd85bc7c`
- **zai**: fix argument handling for tool calls `42da256b`
- **minimax**: enable structured output support `7723a7c9`

### 📊 Release Summary

**Total commits**: 3 across 1 categories

🐛 **3** bug fixes - *Improved stability*

## [0.4.1] - 2026-01-13

### 📋 Release Summary

This release improves temperature and top_p parameter accuracy for consistent model inference (a7a9bac3) and updates reqwest to 0.13.1 for enhanced security and performance (c015ac94).


### 🐛 Bug Fixes & Stability

- **zai**: fix temperature and top_p precision `a7a9bac3`

### 🔄 Other Changes

- **deps**: update reqwest to 0.13.1 `c015ac94`

### 📊 Release Summary

**Total commits**: 2 across 2 categories

🐛 **1** bug fix - *Improved stability*
🔄 **1** other change - *Maintenance & tooling*

## [0.4.0] - 2026-01-08

### 📋 Release Summary

This release adds support for Z.ai and MiniMax providers, enhances reasoning token tracking, and introduces configurable API URLs for improved flexibility. Several optimizations improve model performance and stability, including fixes for Z.ai endpoint updates and enhanced thinking extraction for o-series models.


### ✨ New Features & Enhancements

- **openrouter**: support configurable API URL `eceec284`
- **zai**: add configurable API URL support `7ec4b35d`
- **llm**: add reasoning token tracking for providers `588ca7b1`
- **llm**: add thinking extraction for o-series models `cb49d0fd`
- **llm**: add Z.ai provider support `5e4d5899`
- **minimax**: add MiniMax provider support `fe749cef`

### 🐛 Bug Fixes & Stability

- **zai**: update api url endpoint `50f7b28e`

### 📊 Release Summary

**Total commits**: 7 across 2 categories

✨ **6** new features - *Enhanced functionality*
🐛 **1** bug fix - *Improved stability*

## [0.3.0] - 2026-01-07

### 📋 Release Summary

This release introduces enhanced model support with improved authentication and updated pricing for better cost tracking (8b3f2d93, 4904e94e, b08c6eb3). Several bug fixes and optimizations improve system stability, model compatibility, and test reliability (bd22c3fd, 013e5421, 0bb87c1f).


### ✨ New Features & Enhancements

- **openrouter**: add model catalog and mappings `8b3f2d93`
- **llm/auth**: prefer OAuth over API keys `4904e94e`

### 🐛 Bug Fixes & Stability

- **anthropic**: disable temp for haiku/opus `bd22c3fd`
- **pricing**: update model pricing & context `b08c6eb3`

### 🔄 Other Changes

- add serial test annotations to provider tests `013e5421`
- **openai**: update model list and pricing `0bb87c1f`

### 📊 Release Summary

**Total commits**: 6 across 3 categories

✨ **2** new features - *Enhanced functionality*
🐛 **2** bug fixes - *Improved stability*
🔄 **2** other changes - *Maintenance & tooling*

## [0.2.0] - 2025-11-29

### 📋 Release Summary

This release enhances LLM interoperability and reasoning traceability by converting model calls to a generic tool format and preserving Gemini thought signatures (7646e363, 620667c4). Core and documentation updates improve multi-provider workflows, making integration and debugging smoother for users.


### ✨ New Features & Enhancements

- **llm**: add conversion to GenericToolCall `7646e363`
- **llm**: preserve Gemini thought signatures `620667c4`

### 📊 Release Summary

**Total commits**: 2 across 1 categories

✨ **2** new features - *Enhanced functionality*

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-11-22

### 📋 Release Summary

This release adds multi-tool support and introduces new AI models including Gemini 3, GPT-5.1, and Claude Sonnet 4.5 with updated pricing details (baac12cd, 205c2e76, f917b001). It also expands embedding capabilities with a HuggingFace provider for local models and enhances output handling with structured JSON validation (40b4d50f, d15982fa). Additional improvements include unified pricing updates, API rate limit headers, and enriched provider integrations, alongside several bug fixes that enhance pricing accuracy, caching, and usage tracking for a more reliable user experience (bfc1cca8, 3a2ec8a8, e740c244).


### ✨ New Features & Enhancements

- **docs**: add tool calling example with multi-tool support `baac12cd`
- **llm**: add Gemini 3 and GPT-5.1 models with pricing and context tokens `205c2e76`
- **llm**: add pricing entry for gpt-5-codex model `a3bf7892`
- **llm**: add support for claude-sonnet-4-5 model pricing and temp check `f917b001`
- **embedding**: add HuggingFace provider with local model support `40b4d50f`
- **core**: add structured output support with JSON and schema validation `d15982fa`
- **deepseek**: unify pricing and update provider integration `6346a18e`
- **providers**: add rate limit headers to API responses `7efeb3b1`
- **openrouter**: set app title and referer via environment variables `79f54ffa`
- **octolib**: add initial multi-provider AI library `d6611301`

### 🔧 Improvements & Optimizations

- **embedding**: remove legacy provider parsing fallback `1b81d3fc`
- **embedding**: simplify API and remove config struct `99f3f4e7`
- **amazon**: update Bedrock provider and model support `43d68713`
- **llm**: reorganize modules under llm namespace `af7f02d0`
- **tool_calls**: unify tool call format and handling `1fbb94b3`
- **providers**: format openrouter.rs with cargo fmt `8f2de3e1`
- **core**: restructure modules and unify provider strategies `0679c3ff`

### 🐛 Bug Fixes & Stability

- resolve clippy warnings and test issues `ac9e9d53`
- **llm**: update Anthropic, DeepSeek, and Google pricing models `bfc1cca8`
- **cache**: correct Anthropic and OpenAI cache cost logic `525924c8`
- **deepseek**: avoid double consume of response for logging `3a2ec8a8`
- **openrouter**: add missing parameters and usage tracking `e740c244`
- **openai**: correct tool call handling in message conversion `7b37834d`
- **anthropic**: exclude opus-4-1 from temperature and top_p support `4aa7e701`

### 📚 Documentation & Examples

- add comprehensive Octolib development instructions `c7412c62`

### 🔄 Other Changes

- disable ONNX tests on Windows due to failures `6186ec3b`
- **ci**: run tests only without default features `b0ccaab8`
- **ci**: add GitHub release workflow `b480aa12`
- **deps**: update candle libs to 0.9.2-alpha `5c84ab47`
- **deps**: update and consolidate dependency versions `0473a070`
- **license**: add Apache 2.0 license file `99319e53`

### 📊 Release Summary

**Total commits**: 31 across 5 categories

✨ **10** new features - *Enhanced functionality*
🔧 **7** improvements - *Better performance & code quality*
🐛 **7** bug fixes - *Improved stability*
📚 **1** documentation update - *Better developer experience*
🔄 **6** other changes - *Maintenance & tooling*
