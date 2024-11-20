import pytest

from notdiamond.exceptions import UnsupportedEmbeddingProvider
from notdiamond.llms.config import EmbeddingConfig


def test_supported_embedding_provider():
    config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
    assert config.provider == "openai"
    assert config.model == "text-embedding-3-small"

    config = EmbeddingConfig(provider="mistral", model="mistral-embed")
    assert config.provider == "mistral"
    assert config.model == "mistral-embed"

    config = EmbeddingConfig(provider="cohere", model="embed-english-v3.0")
    assert config.provider == "cohere"
    assert config.model == "embed-english-v3.0"


def test_unsupported_embedding_model():
    with pytest.raises(UnsupportedEmbeddingProvider):
        EmbeddingConfig(provider="openai", model="gpt-71")


def test_config_from_string():
    config = EmbeddingConfig.from_string("openai/text-embedding-3-small")
    assert config.provider == "openai"
    assert config.model == "text-embedding-3-small"
