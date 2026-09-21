<p align="center">
  <a href="https://github.com/Ameya79/Talanton">
    <img src="https://raw.githubusercontent.com/Ameya79/Talanton/main/assets/talanton-logo.png" alt="Talanton — The Universal Meter for What Your AI Calls Actually Cost" width="300">
  </a>
</p>

<h1 align="center">Talanton (τάλαντον)</h1>

<p align="center">
  <strong>The universal meter and automatic spending limit for your AI calls.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/talanton/"><img src="https://img.shields.io/pypi/v/talanton.svg?color=0019ff&labelColor=070e24&logo=pypi&logoColor=white" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/talanton/"><img src="https://img.shields.io/pypi/pyversions/talanton.svg?color=0019ff&labelColor=070e24" alt="Python Versions"></a>
  <a href="https://github.com/Ameya79/Talanton/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-38bdf8.svg?labelColor=070e24" alt="License: MIT"></a>
  <a href="https://github.com/Ameya79/Talanton/actions"><img src="https://img.shields.io/badge/tests-75%2F75%20passing-10b981.svg?labelColor=070e24" alt="Tests Status"></a>
  <a href="https://github.com/Ameya79/Talanton"><img src="https://img.shields.io/badge/latency-0.08ms%20P50-38bdf8.svg?labelColor=070e24" alt="Latency P50"></a>
  <a href="https://github.com/Ameya79/Talanton"><img src="https://img.shields.io/badge/telemetry-100%25%20local--first-0019ff.svg?labelColor=070e24" alt="100% Local-First"></a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quickstart-in-30-seconds">Quickstart</a> •
  <a href="#budget-guardrails--circuit-breakers">Budget Guardrails</a> •
  <a href="#sdk-wrappers--auto-tracking">SDK Integrations</a> •
  <a href="#cli-usage">CLI Reference</a> •
  <a href="#benchmarks--slas">Performance SLAs</a> •
  <a href="#supported-models">Supported Models</a>
</p>

---

## What is Talanton?

**Talanton** (*τάλαντον* — ancient Greek for the balance scale used to weigh silver and gold) is an open-source, local-first Python library and CLI that acts like a **credit card spending limit for your AI**.

Traditional AI features fly blind until cloud invoices arrive. A single stuck coding agent or while-loop can retry 100 times on a syntax typo, silently ballooning conversation history and burning **$150+ in minutes**. Cloud observability proxies add 50ms–150ms of network latency and exfiltrate your private prompts to third-party clouds.

**Talanton solves this in 0.08 milliseconds directly on your CPU:**
1. **Pre-Flight Metrology**: Accurately counts tokens and computes exact dollar costs across **OpenAI**, **Anthropic**, and **HuggingFace** models *before* sending API requests.
2. **Hard Budget Guardrails**: Terminates runaway agent loops and unexpected spikes before a single dollar is wasted.
3. **Live In-Process Observability**: Auto-tracks every call with sub-millisecond SQLite WAL persistence (25,000+ events/sec). **Zero data leaves your machine.**

---

## Key Features

- **Exact vs. Estimated Transparency**: Every single count and cost output explicitly carries an `exact: True/False` flag — never blurring provider-verified counts with rough heuristic guesses.
- **Chat Template Overhead Accounting**: Accurately measures provider-specific system/user message wrapping and reply primer tokens (+7 to +9 tokens per turn) rather than naive raw string concatenation.
- **Sub-Millisecond Budget Guardrails**: Configurable soft warnings and hard spending caps (`ALLOW`, `WARN`, `BLOCK`) that raise `BudgetExceededError` in `0.08ms` to kill loops instantly.
- **Drop-In SDK Wrappers**: 1-line auto-tracking wrappers for `openai.OpenAI()`, `anthropic.Anthropic()`, LangChain, and LiteLLM.
- **Side-by-Side Model Comparison**: Ranks candidate models by total cost (cheapest first) to answer *"Which model should I actually use for this prompt?"*
- **Volume Spend Forecasting**: Projects monthly compounding spend over time horizons (3, 6, 12 months) and quantifies the exact switching savings across models.
- **100% Local-First & Zero Cloud Lock-in**: Powered by local SQLite WAL mode (`~/.talanton/telemetry.db`). Works completely offline with zero cloud data exfiltration.

---

## Installation

```bash
# Core package (token counting, cost calculation, live tracking, guardrails)
pip install talanton

# With OpenAI tiktoken support
pip install "talanton[openai]"

# With Anthropic API counting support
pip install "talanton[anthropic]"

# With local HuggingFace open-weight tokenizers
pip install "talanton[huggingface]"

# All providers + rich terminal CLI
pip install "talanton[all]"
```

---

## Quickstart in 30 Seconds

### 1. Pre-Flight Cost Metrology (Before Calling Any API)

```python
from talanton import count_tokens, calculate_cost, compare_models

prompt = "Analyze Q3 quarterly revenue and summarize operational highlights."

# 1. Count Tokens (exact via tiktoken for OpenAI)
count = count_tokens(prompt, model="gpt-4o")
print(count)
# {'tokens': 12, 'exact': True, 'provider': 'openai', 'model': 'gpt-4o'}

# Chat format accounts for message tags & reply primers automatically
chat = [
    {"role": "system", "content": "You are a financial analyst."},
    {"role": "user", "content": prompt}
]
print(count_tokens(chat, model="gpt-4o"))
# {'tokens': 24, 'exact': True, 'provider': 'openai', 'model': 'gpt-4o'}

# 2. Calculate Exact Cost in USD
cost = calculate_cost(prompt, model="claude-sonnet-4.5", expected_output_tokens=300)
print(f"Total: ${cost['total_cost']:.6f} (exact={cost['exact']})")
# Total: $0.004536 (exact=True)

# 3. Compare Models Side-by-Side (Ranked Cheapest First)
ranked = compare_models(prompt, models=["gpt-4o", "gpt-4o-mini", "claude-3-5-haiku-latest"], expected_output_tokens=300)
for r in ranked:
    print(f"{r['model']:<25} ${r['total_cost']:.6f}")
```

---

## Budget Guardrails & Circuit Breakers

Stop runaway agent loops, accidental prompt bloat, and runaway cron jobs before they drain your credit card:

```python
from talanton import TalantonTracker, BudgetGuardrail, BudgetExceededError

tracker = TalantonTracker()

# Set spending limit: $5.00 warning, $10.00 hard execution stop per day
guardrail = BudgetGuardrail(
    tracker=tracker,
    soft_limit=5.00,
    hard_limit=10.00,
    period="day",          # 'day', 'week', or 'month'
    team="backend-agent"   # optional multi-tenant team isolation
)

# Check before invoking an LLM call:
result = guardrail.check(estimated_cost=0.05)
if result.is_blocked:
    raise Exception(f"Spend limit reached! Current: ${result.current_spend:.2f}")

# Or let Talanton automatically raise BudgetExceededError:
from talanton.guardrails import raise_exception
guardrail = BudgetGuardrail(tracker=tracker, hard_limit=10.0, on_hard_limit=raise_exception)
```

---

## SDK Wrappers & Auto-Tracking

### Drop-in OpenAI Wrapper (`TalantonOpenAI`)

Wrap your existing `OpenAI()` client with one line. Every completion is automatically metered, cost-calculated, and checked against your guardrail:

```python
import openai
from talanton import TalantonTracker, BudgetGuardrail
from talanton.integrations import TalantonOpenAI

tracker = TalantonTracker()
guard = BudgetGuardrail(tracker=tracker, hard_limit=25.0, period="day")

# Drop-in replacement:
client = TalantonOpenAI(openai.OpenAI(), tracker=tracker, guardrail=guard)

# Use exactly like the standard OpenAI client:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello world!"}]
)

# Telemetry is recorded in-process to SQLite in 0.08ms!
```

### Drop-in Anthropic Wrapper (`TalantonAnthropic`)

```python
import anthropic
from talanton import TalantonTracker
from talanton.integrations import TalantonAnthropic

tracker = TalantonTracker()
client = TalantonAnthropic(anthropic.Anthropic(), tracker=tracker)

response = client.messages.create(
    model="claude-sonnet-4.5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Explain quantum computing simply."}]
)
```

### LangChain & LiteLLM Integrations

- **LangChain Tracer**: `from talanton.integrations import TalantonCallbackHandler`
- **LiteLLM Gateway Logger**: `from talanton.integrations import TalantonLiteLLMCallback`

---

## Volume Spend Forecasting & Scale Switch Analysis

Answer *"What will this feature cost in 6 months if it takes off?"* and *"Is migrating models worth the engineering effort?"*:

```python
from talanton import forecast, compare_at_scale

# Forecast 6 months at 15% monthly compounding growth:
projections = forecast(
    model="gpt-4o",
    calls_per_day=1000,
    avg_input_tokens=800,
    avg_output_tokens=300,
    months=6,
    growth_rate=0.15
)
for p in projections:
    print(f"Month {p['month']}: {p['calls_per_day']:,.0f} calls/day -> ${p['cost']:,.2f}")

# Quantify exact switching savings between models:
analysis = compare_at_scale(
    models=["gpt-4o", "gpt-4o-mini", "claude-3-5-haiku-latest"],
    calls_per_day=1000,
    avg_input_tokens=800,
    avg_output_tokens=300,
    months=6,
    growth_rate=0.15
)
print(f"Cheapest: {analysis['cheapest']}")
print(f"Net savings by switching: ${analysis['savings_by_switching']:,.2f}")
```

---

## CLI Usage

Talanton features a rich terminal interface for quick checks and CI/CD pipelines:

```bash
# 1. Count tokens (supports stdin piping)
talanton count "Analyze quarterly filings" --model gpt-4o

# 2. Calculate prompt and expected completion cost
talanton cost "Analyze quarterly filings" --model claude-sonnet-4.5 --output-tokens 200

# 3. Compare candidate models side-by-side
talanton compare "Analyze quarterly filings" --models gpt-4o,claude-sonnet-4.5,gpt-4o-mini --output-tokens 200

# 4. Forecast spend over 6 months with 15% growth
talanton forecast --model gpt-4o --calls-per-day 500 --avg-input 800 --avg-output 300 --months 6 --growth 0.15

# 5. Compare spend at scale and compute migration ROI
talanton compare-at-scale --models gpt-4o,gpt-4o-mini --calls-per-day 500 --avg-input 800 --avg-output 300

# 6. View live spend summaries
talanton track summary --period day
talanton track summary --period month --team backend-team

# 7. Inspect recent calls & export to CSV/JSON
talanton track calls --limit 20
talanton track export --format csv --output spend_report.csv

# 8. Check budget status
talanton budget status --period day
```

---

## Benchmarks & SLAs

Measured under multi-threaded concurrency using Python 3.12 with SQLite WAL mode on local NVMe:

| Metric / SLA Target | Talanton (Local-First) | Cloud Proxies (Langfuse/Helicone) | Status |
| :--- | :--- | :--- | :--- |
| **Guardrail Pre-Check Latency (P50)** | **`0.08 ms` (80 µs)** | `42 ms – 85 ms` (Network hop) | **1000x faster** |
| **Guardrail Pre-Check Latency (P99)** | **`1.94 ms`** | `120 ms – 350 ms` | **PASSED (< 5ms)** |
| **Batch Ingestion Throughput** | **`24,586 calls/sec`** | `200 – 500 req/sec` | **PASSED (> 5,000/s)** |
| **Concurrent Multi-Thread Writes** | **0 lock errors (25 threads)**| Frequent HTTP timeouts | **PASSED** |
| **Analytical Query Time (10k rows)** | **`0.85 ms`** | `250 ms – 1,200 ms` | **PASSED** |
| **Storage Density Footprint** | **`~208 bytes / call`** | Remote hosted database | **PASSED** |
| **Customer Data Privacy** | **100% In-Process / Local** | Exfiltrated across internet | **Zero Data Exfiltration** |

---

## Supported Models

Talanton maintains an offline pricing and tokenizer registry for **60+ leading models**:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`, `o1-mini`, `o3-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`, `text-embedding-3-small/large`
- **Anthropic**: `claude-sonnet-4.5`, `claude-3-7-sonnet-latest`, `claude-3-5-sonnet-latest`, `claude-3-5-haiku-latest`, `claude-3-opus-latest`, `claude-2.1`
- **Open-Weight (HuggingFace)**: `meta-llama/Llama-3.3-70B-Instruct`, `meta-llama/Llama-3.1-8B/70B`, `meta-llama/Llama-3.2-1B/3B`, `mistralai/Mistral-7B-v0.1`

Pricing is transparently stored in `talanton/pricing/pricing_table.json` in USD per 1,000,000 tokens.

---

## Running Tests

Talanton includes a complete test suite covering token counting accuracy, chat template accounting, compounding mathematics, thread concurrency, and integrations:

```bash
# Run all 75 unit, integration, and scalability tests:
python -m pytest tests/ -v

# Run the performance and latency SLA benchmark:
python benchmarks/scalability_benchmark.py
```

---

## License

Distributed under the permissive **MIT License**. See [LICENSE](LICENSE) for details.

Authored and maintained by **Ameya Kulkarni** ([acclaptop47@gmail.com](mailto:acclaptop47@gmail.com)).
GitHub: [github.com/Ameya79/Talanton](https://github.com/Ameya79/Talanton)
