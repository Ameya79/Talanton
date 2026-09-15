"""Tests for token counting across OpenAI and fallback heuristics."""

import pytest
import tiktoken
from talanton.counting import count_tokens, count_fallback
from talanton.counting.openai import count_openai


def test_openai_raw_string_counting():
    """Verify raw text counting matches tiktoken exact encoding length."""
    prompt = "Hello, world! Welcome to Talanton token counting."
    result = count_tokens(prompt, "gpt-4o")

    enc = tiktoken.encoding_for_model("gpt-4o")
    expected = len(enc.encode(prompt))

    assert result["tokens"] == expected
    assert result["exact"] is True
    assert result["provider"] == "openai"
    assert result["model"] == "gpt-4o"


def test_openai_chat_format_accounting():
    """Verify chat formatting matches OpenAI cookbook overhead."""
    # OpenAI Cookbook formula:
    # 3 tokens per message (<|im_start|>{role}\n{content}<|im_end|>)
    # + 3 tokens reply primer
    # total overhead for 2 messages without name = 3*2 + 3 = 9 tokens
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    result = count_tokens(messages, "gpt-4o")

    enc = tiktoken.encoding_for_model("gpt-4o")
    content_tokens = len(enc.encode("system")) + len(enc.encode("You are a helpful assistant.")) + \
                     len(enc.encode("user")) + len(enc.encode("Hello!"))
    expected_total = content_tokens + (3 * 2) + 3

    assert result["tokens"] == expected_total
    assert result["exact"] is True
    assert result["provider"] == "openai"


def test_openai_chat_with_name_field():
    """Verify that a name field adds 1 token per OpenAI cookbook specification."""
    messages = [
        {"role": "user", "name": "Alice", "content": "Hello!"}
    ]
    result = count_tokens(messages, "gpt-4o")

    enc = tiktoken.encoding_for_model("gpt-4o")
    content_tokens = len(enc.encode("user")) + len(enc.encode("Alice")) + len(enc.encode("Hello!"))
    # 3 per msg + 1 for name + 3 assistant reply primer = 7 overhead tokens
    expected_total = content_tokens + 3 + 1 + 3

    assert result["tokens"] == expected_total


def test_fallback_heuristic_returns_estimated():
    """Verify fallback estimator returns exact=False and approximate token count."""
    text = "A quick brown fox jumps over the lazy dog."  # 42 chars -> ~10-11 tokens
    result = count_fallback(text, "unknown-model")

    assert result["exact"] is False
    assert result["tokens"] == round(len(text) / 4)
    assert result["provider"] == "unknown"
    assert result["model"] == "unknown-model"


def test_count_tokens_unknown_model_graceful():
    """Verify unknown model runs through fallback without crashing."""
    prompt = "Test text for unknown model"
    result = count_tokens(prompt, "nonexistent-model-xyz")

    assert result["exact"] is False
    assert result["provider"] == "unknown"
    assert result["tokens"] > 0
    assert result["model"] == "nonexistent-model-xyz"


def test_anthropic_fallback_without_api_key(monkeypatch):
    """Verify Anthropic counting degrades to fallback without ANTHROPIC_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Clear cache to avoid hits from previous runs
    from talanton.counting.anthropic import _LOCAL_CACHE
    import talanton.counting.anthropic as ant_mod
    ant_mod._LOCAL_CACHE = {}

    result = count_tokens("Testing Claude sonnet without api key", "claude-sonnet-4.5")
    assert result["exact"] is False
    assert result["provider"] == "anthropic"
    assert result["tokens"] > 0
    assert result["model"] == "claude-sonnet-4.5"


def test_anthropic_uses_cache(monkeypatch):
    """Verify that cached counts return exact=True without making network calls."""
    import talanton.counting.anthropic as ant_mod
    prompt = "Cache test prompt for Claude"
    model = "claude-sonnet-4.5"
    key = ant_mod._compute_cache_key(prompt, model)
    ant_mod._LOCAL_CACHE = {key: 42}

    # Ensure no API key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = count_tokens(prompt, model)
    assert result["tokens"] == 42
    assert result["exact"] is True
    assert result["provider"] == "anthropic"


def test_huggingface_with_cached_mock_tokenizer():
    """Verify HuggingFace counting works with an in-memory cached tokenizer."""
    import talanton.counting.huggingface as hf_mod

    class MockEncoding:
        ids = [101, 102, 103, 104, 105]

    class MockTokenizer:
        def encode(self, text):
            return MockEncoding()

    model = "meta-llama/Meta-Llama-3-8B"
    hf_mod._TOKENIZER_CACHE[model] = MockTokenizer()

    res = count_tokens("Hello world open source", model)
    assert res["tokens"] == 5
    assert res["exact"] is True
    assert res["provider"] == "huggingface"


def test_huggingface_fallback_on_network_error(monkeypatch):
    """Verify HuggingFace gracefully falls back when loading fails."""
    import talanton.counting.huggingface as hf_mod
    hf_mod._TOKENIZER_CACHE.clear()

    # Model that cannot be loaded
    res = count_tokens("Testing HF failure fallback", "meta-llama/Nonexistent-Llama-Model")
    # Even if unknown or failing, it should return exact=False without raising
    assert res["tokens"] > 0
    assert res["exact"] is False


