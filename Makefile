# octolib development tasks
.PHONY: help coverage

help: ## List available tasks
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

# Coverage report (requires cargo-llvm-cov + llvm-tools-preview).
# Mirrors CI: lib tests with all provider features on; test-only files
# are excluded so percentages describe product code.
coverage: ## Generate test coverage report (cargo-llvm-cov)
	@echo "Generating coverage report..."
	cargo llvm-cov --summary-only --lib --features fastembed,huggingface --ignore-filename-regex '_tests\.rs$$'
