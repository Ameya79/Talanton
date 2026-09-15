"""Tests for talanton.providers model detection."""

import pytest
from talanton.providers import (
    detect_provider,
    get_model_info,
    is_supported_model,
    list_supported_models,
)


def test_routes_gpt_4o_to_openai():
    """Verify gpt-4o routes to 'openai'."""
    assert detect_provider("gpt-4o") == "openai"


def test_routes_claude_sonnet_4_5_to_anthropic():
    """Verify claude-sonnet-4.5 routes to 'anthropic'."""
    assert detect_provider("claude-sonnet-4.5") == "anthropic"


def test_routes_huggingface_model():
    """Verify HuggingFace models route to 'huggingface'."""
    assert detect_provider("meta-llama/Meta-Llama-3-8B-Instruct") == "huggingface"


def test_unknown_model_returns_unknown_without_raising():
    """Verify arbitrary unknown models return 'unknown' and never raise."""
    unknown_inputs = [
        "some-random-unknown-model-xyz",
        "custom-finetuned-llama",
        "gpt-5-super",
        "my-model",
        "",
        "   ",
        None,
        12345,
    ]
    for model_name in unknown_inputs:
        provider = detect_provider(model_name)
        assert provider == "unknown", f"Expected 'unknown' for {model_name!r}, got {provider!r}"


def test_case_insensitivity_and_whitespace():
    """Verify model lookup handles casing and whitespace cleanly."""
    assert detect_provider("  gpt-4o  ") == "openai"
    assert detect_provider("GPT-4O") == "openai"
    assert detect_provider("Claude-Sonnet-4.5") == "anthropic"


def test_get_model_info():
    """Verify get_model_info retrieves metadata dictionary."""
    info = get_model_info("gpt-4o")
    assert info is not None
    assert info["provider"] == "openai"
    assert info["tokenizer"] == "o200k_base"

    claude_info = get_model_info("claude-sonnet-4.5")
    assert claude_info is not None
    assert claude_info["provider"] == "anthropic"

    assert get_model_info("unknown-model-foo") is None
    assert get_model_info(None) is None


def test_list_and_is_supported_models():
    """Verify list_supported_models and is_supported_model helpers."""
    all_models = list_supported_models()
    assert "gpt-4o" in all_models
    assert "claude-sonnet-4.5" in all_models

    openai_models = list_supported_models("openai")
    assert "gpt-4o" in openai_models
    assert "claude-sonnet-4.5" not in openai_models

    anthropic_models = list_supported_models("anthropic")
    assert "claude-sonnet-4.5" in anthropic_models
    assert "gpt-4o" not in anthropic_models

    assert is_supported_model("gpt-4o") is True
    assert is_supported_model("claude-sonnet-4.5") is True
    assert is_supported_model("totally-fake-model") is False
