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

use super::{ModelPricing, TokenUsage};

#[test]
fn reasoning_is_not_billed_twice() {
    // A z.ai-shaped response: prompt 1000 (200 cached), completion 500 of
    // which 300 was thinking; the provider's own total is 1500.
    let (output_tokens, reasoning_tokens) = TokenUsage::split_output(500, 300);
    let usage = TokenUsage {
        input_tokens: 800,
        cache_read_tokens: 200,
        cache_write_tokens: 0,
        output_tokens,
        reasoning_tokens,
        total_tokens: 1500,
        cost: None,
        request_time_ms: None,
    };
    assert_eq!(usage.output_tokens, 200);
    // The parts a consumer sums must reproduce the provider's own total.
    assert_eq!(
        usage.input_tokens
            + usage.cache_read_tokens
            + usage.cache_write_tokens
            + usage.output_tokens
            + usage.reasoning_tokens,
        usage.total_tokens
    );
}

#[test]
fn cost_bills_reasoning_at_the_output_rate() {
    // A real DashScope qwen3.6-flash response: prompt 22, completion 502 of
    // which 485 was thinking. Alibaba charges all 502 at the output rate,
    // so pricing must run on the billable counter, not the 17 left visible.
    let (output_tokens, reasoning_tokens) = TokenUsage::split_output(502, 485);
    let usage = TokenUsage {
        input_tokens: 22,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        output_tokens,
        reasoning_tokens,
        total_tokens: 524,
        cost: None,
        request_time_ms: None,
    };
    assert_eq!(usage.output_tokens, 17);
    assert_eq!(usage.billable_output_tokens(), 502);

    let pricing = ModelPricing::new(0.25, 1.50, 0.25, 0.025);
    let billed = pricing.calculate_cost(
        usage.input_tokens,
        usage.cache_write_tokens,
        usage.cache_read_tokens,
        usage.billable_output_tokens(),
    );
    let expected = 22.0 / 1_000_000.0 * 0.25 + 502.0 / 1_000_000.0 * 1.50;
    assert!((billed - expected).abs() < 1e-12);

    // Pricing the visible remainder instead is the regression this guards:
    // a thinking-heavy call would cost a fraction of what the provider charged.
    let visible_only = pricing.calculate_cost(
        usage.input_tokens,
        usage.cache_write_tokens,
        usage.cache_read_tokens,
        usage.output_tokens,
    );
    assert!(billed > visible_only * 10.0);
}

#[test]
fn estimated_reasoning_over_completion_clamps_both_sides() {
    // Estimated reasoning can exceed the completion count it was cut from.
    // Clamping only the output would leave the parts summing above the
    // provider's total again.
    assert_eq!(TokenUsage::split_output(120, 400), (0, 120));
}
