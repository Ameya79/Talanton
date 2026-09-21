"""Talanton Integrations — SDK wrappers for automatic LLM call tracking.

Provides drop-in wrappers for OpenAI, Anthropic, LangChain, and LiteLLM
that auto-track every LLM call to the Talanton telemetry store.

Available integrations:
    - TalantonOpenAI: Wraps openai.OpenAI for auto-tracking
    - TalantonAnthropic: Wraps anthropic.Anthropic for auto-tracking
    - TalantonCallbackHandler: LangChain BaseCallbackHandler
    - TalantonLiteLLMCallback: LiteLLM CustomLogger
"""

from __future__ import annotations

__all__ = [
    "TalantonOpenAI",
    "TalantonAnthropic",
    "TalantonCallbackHandler",
    "TalantonLiteLLMCallback",
]


def __getattr__(name: str):
    """Lazy imports to avoid requiring all integration dependencies."""
    if name == "TalantonOpenAI":
        from talanton.integrations.openai_wrapper import TalantonOpenAI
        return TalantonOpenAI
    elif name == "TalantonAnthropic":
        from talanton.integrations.anthropic_wrapper import TalantonAnthropic
        return TalantonAnthropic
    elif name == "TalantonCallbackHandler":
        from talanton.integrations.langchain_callback import TalantonCallbackHandler
        return TalantonCallbackHandler
    elif name == "TalantonLiteLLMCallback":
        from talanton.integrations.litellm_callback import TalantonLiteLLMCallback
        return TalantonLiteLLMCallback
    raise AttributeError(f"module 'talanton.integrations' has no attribute {name!r}")
