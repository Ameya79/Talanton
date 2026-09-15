"""Talanton — The universal meter for what your AI calls actually cost.

Public API:
    - count_tokens(text, model)
    - calculate_cost(text, model, expected_output_tokens=0)
    - compare_models(text, models, expected_output_tokens=0)
    - forecast(model, calls_per_day, avg_input_tokens, avg_output_tokens, months=6, growth_rate=0.0)
    - compare_at_scale(models, calls_per_day, avg_input_tokens, avg_output_tokens, months=6, growth_rate=0.0)
    - detect_provider(model)
    - get_price(model)
"""

from talanton.cost import calculate_cost, compare_models
from talanton.counting import count_tokens
from talanton.forecast import compare_at_scale, forecast
from talanton.pricing import PricingNotFoundError, get_price, list_priced_models
from talanton.providers import (
    detect_provider,
    get_model_info,
    is_supported_model,
    list_supported_models,
)

__version__ = "0.1.0"

__all__ = [
    "count_tokens",
    "calculate_cost",
    "compare_models",
    "forecast",
    "compare_at_scale",
    "detect_provider",
    "get_model_info",
    "is_supported_model",
    "list_supported_models",
    "get_price",
    "list_priced_models",
    "PricingNotFoundError",
]
