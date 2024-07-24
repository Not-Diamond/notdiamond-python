import sys

import pytest

from notdiamond import LLMConfig
from notdiamond.exceptions import UnsupportedLLMProvider

sys.path.append("../")


def test_supported_llm_provider():
    openai = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    assert openai.provider == "openai"
    assert openai.model == "gpt-3.5-turbo"

    openai = LLMConfig(provider="anthropic", model="claude-2.1")
    assert openai.provider == "anthropic"
    assert openai.model == "claude-2.1"

    openai = LLMConfig(provider="google", model="gemini-pro")
    assert openai.provider == "google"
    assert openai.model == "gemini-pro"


def test_unsupported_model():
    with pytest.raises(UnsupportedLLMProvider):
        LLMConfig(provider="openai", model="gpt-71")


def test_prepare_for_request():
    llm_provider = LLMConfig(provider="openai", model="gpt-3.5-turbo")
    request = llm_provider.prepare_for_request()
    assert request == {"provider": "openai", "model": "gpt-3.5-turbo"}


def test_existing_openrouter_model():
    llm_provider = LLMConfig(provider="mistral", model="mistral-large-latest")
    assert llm_provider.openrouter_model == "mistralai/mistral-large"

    llm_provider = LLMConfig(
        provider="anthropic", model="claude-3-opus-20240229"
    )
    assert llm_provider.openrouter_model == "anthropic/claude-3-opus"


def test_not_existing_openrouter_model():
    llm_provider = LLMConfig(provider="openai", model="gpt-4-0125-preview")
    result = llm_provider.openrouter_model
    assert result is None
