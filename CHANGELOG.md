# Changelog

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
