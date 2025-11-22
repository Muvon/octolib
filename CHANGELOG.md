# Changelog

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
