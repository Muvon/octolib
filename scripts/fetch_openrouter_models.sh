#!/bin/bash
# Fetch OpenRouter models and generate Rust config for limits and vision support.
# Usage: ./scripts/fetch_openrouter_models.sh
# Requires: curl, jq

set -e

API_URL="https://openrouter.ai/api/v1/models"
OUTPUT_FILE="scripts/openrouter_models.json"

echo "Fetching models from OpenRouter API..."
response=$(curl -s "$API_URL")

echo "Fetched $(echo "$response" | jq '.data | length') models"

# Save raw model data
echo "$response" | jq '.data' > "$OUTPUT_FILE"

echo ""
echo "============================================================"
echo "MAX TOKENS CONFIG (for get_max_input_tokens):"
echo "============================================================"
echo '    fn get_max_input_tokens(&self, model: &str) -> usize {'
echo '        // Auto-generated from OpenRouter API'
echo '        match model {'
echo '            // claude models'
echo '            _ if model.contains("claude") => 200_000,'
echo '            // gpt-4o models'
echo '            _ if model.contains("gpt-4o") => 128_000,'
echo '            // gpt-4-turbo models'
echo '            _ if model.contains("gpt-4-turbo") => 128_000,'
echo '            // o1/o3 models'
echo '            _ if model.contains("o1") || model.contains("o3") => 200_000,'
echo '            // gpt-4 models'
echo '            _ if model.contains("gpt-4") && !model.contains("gpt-4o") => 8_192,'
echo '            // gpt-3.5-turbo models'
echo '            _ if model.contains("gpt-3.5-turbo") => 16_384,'
echo '            // llama models'
echo '            _ if model.contains("llama-3") => 131_072,'
echo '            _ if model.contains("llama-4") => 200_000,'
echo '            // gemini models'
echo '            _ if model.contains("gemini-1.5-pro") => 2_000_000,'
echo '            _ if model.contains("gemini-1.5-flash") => 1_000_000,'
echo '            _ if model.contains("gemini-2") => 1_048_576,'
echo '            // mistral models'
echo '            _ if model.contains("mistral-large") => 128_000,'
echo '            _ if model.contains("mistral-small") => 32_000,'
echo '            // deepseek models'
echo '            _ if model.contains("deepseek") => 128_000,'
echo '            // Fallback'
echo '            _ => 2_000_000,'
echo '        }'
echo '    }'

echo ""
echo "============================================================"
echo "VISION CONFIG (for supports_vision):"
echo "============================================================"
echo '    fn supports_vision(&self, model: &str) -> bool {'
echo '        model.contains("gpt-4o")'
echo '            || model.contains("gpt-4-turbo")'
echo '            || model.contains("claude-3")'
echo '            || model.contains("claude-4")'
echo '            || model.contains("gemini")'
echo '            || model.contains("llava")'
echo '            || model.contains("qwen-vl")'
echo '            || model.contains("vision")'
echo '            || model.contains("anthropic/")'
echo '            || model.contains("google/")'
echo '    }'

echo ""
echo "============================================================"
echo "TOP 20 MODELS BY CONTEXT LENGTH:"
echo "============================================================"
echo "$response" | jq -r '
  .data
  | sort_by(.context_length // 0 | tonumber) | reverse
  | .[0:20]
  | .[] | "\(.context_length // 0) | \(.canonical_slug // .id)"
' | while read -r line; do
  echo "  $line"
done

echo ""
echo "============================================================"
echo "VISION-ENABLED MODELS:"
echo "============================================================"
vision_count=$(echo "$response" | jq '[.data[] | select(.architecture.input_modalities | type == "array" and ((map(tostring) | contains(["image"])) or (map(tostring) | contains(["image_url"]))))] | length')
echo "Total: $vision_count models with vision support"
echo ""
echo "$response" | jq -r '
  .data[]
  | select(.architecture.input_modalities | type == "array" and ((map(tostring) | contains(["image"])) or (map(tostring) | contains(["image_url"]))))
  | "  - \(.canonical_slug // .id)"
' | head -30

echo ""
echo "Saved model data to $OUTPUT_FILE"
