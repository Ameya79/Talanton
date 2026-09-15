# Talanton (τάλαντον)

> **The universal meter for what your AI calls actually cost.**

*Talanton* is a lightweight, zero-database Python library and CLI that accurately counts tokens across **OpenAI**, **Anthropic**, and **HuggingFace** models, translates tokens into real dollar costs, compares models side-by-side, and forecasts API spend at scale.

---

## Key Features

- **Universal Token Counting**: Exact counting via `tiktoken` (OpenAI), official `count_tokens` endpoint (Anthropic), and local `tokenizers` (HuggingFace / open-weight models).
- **Exact vs. Estimated Transparency**: Every single count and cost output explicitly carries an `exact: True/False` flag — never blurring provider-verified counts with heuristic approximations.
- **Chat Template Overhead Accounting**: Accurately accounts for provider-specific system/user message wrapping and reply primer tokens rather than just raw string length.
- **Side-by-Side Model Comparison**: Ranks candidate models by total cost (cheapest first) to answer *"Which model should I actually use for this prompt?"*
- **Spend Forecasting & Scale Analysis**: Projects monthly compounded spend over time horizons (3, 6, 12 months) and quantifies the exact switching savings across models.
- **Graceful Degradation**: Never crashes when offline or when an API key is missing — degrades safely to standard heuristic estimation with clear warnings.

---

## Installation

```bash
# Core package
pip install .

# Or with provider extras
pip install ".[all]"
```

---

## Python API Usage

```python
from talanton import (
    count_tokens,
    calculate_cost,
    compare_models,
    forecast,
    compare_at_scale,
)

prompt = "Analyze Q3 quarterly revenue and summarize operational highlights."

# 1. Count Tokens
count = count_tokens(prompt, model="gpt-4o")
print(count)
# {'tokens': 12, 'exact': True, 'provider': 'openai', 'model': 'gpt-4o'}

# Chat format with message overhead accounted for
chat_messages = [
    {"role": "system", "content": "You are a senior equity research analyst."},
    {"role": "user", "content": prompt}
]
print(count_tokens(chat_messages, model="gpt-4o"))

# 2. Calculate Cost (Input + Expected Output)
cost = calculate_cost(prompt, model="claude-3-5-sonnet-latest", expected_output_tokens=300)
print(cost)
# {
#   'input_tokens': 12, 'output_tokens': 300,
#   'input_cost': 0.000036, 'output_cost': 0.0045,
#   'total_cost': 0.004536, 'exact': True, 'model': 'claude-3-5-sonnet-latest'
# }

# 3. Compare Models (Ranked cheapest first)
candidates = ["gpt-4o", "gpt-4o-mini", "claude-3-5-haiku-latest"]
ranked = compare_models(prompt, models=candidates, expected_output_tokens=300)
for r in ranked:
    print(f"{r['model']}: ${r['total_cost']:.6f} (exact={r['exact']})")

# 4. Volume Spend Forecast (compounding monthly projection)
# 500 calls/day, 800 input tokens, 300 output tokens, 6 months, 15% monthly growth
projections = forecast(
    model="gpt-4o",
    calls_per_day=500,
    avg_input_tokens=800,
    avg_output_tokens=300,
    months=6,
    growth_rate=0.15
)
for p in projections:
    print(f"Month {p['month']}: {p['calls_per_day']:,.0f} calls/day -> ${p['cost']:,.2f}")

# 5. Compare at Scale (Switch analysis & savings)
scale_eval = compare_at_scale(
    models=["gpt-4o", "gpt-4o-mini"],
    calls_per_day=500,
    avg_input_tokens=800,
    avg_output_tokens=300,
    months=6,
    growth_rate=0.15
)
print("Cheapest model:", scale_eval["cheapest"])
print("Potential savings by switching:", f"${scale_eval['savings_by_switching']:,.2f}")
```

---

## CLI Usage

```bash
# Count tokens
talanton count "Analyze quarterly filings" --model gpt-4o

# Calculate cost
talanton cost "Analyze quarterly filings" --model claude-sonnet-4.5 --output-tokens 200

# Compare models side-by-side
talanton compare "Analyze quarterly filings" --models gpt-4o,claude-sonnet-4.5,gpt-4o-mini --output-tokens 200

# Forecast spend over 6 months with 15% growth
talanton forecast --model gpt-4o --calls-per-day 500 --avg-input 800 --avg-output 300 --months 6 --growth 0.15

# Compare spend at scale and calculate migration savings
talanton compare-at-scale --models gpt-4o,gpt-4o-mini --calls-per-day 500 --avg-input 800 --avg-output 300
```

---

## Testing

```bash
python -m pytest tests/ -v
```
