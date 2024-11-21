import nltk
import pandas as pd
import pytest
from llama_index.core import SimpleDirectoryReader
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM

from notdiamond.llms.config import EmbeddingConfig, LLMConfig
from notdiamond.toolkit.rag.document_loaders import DirectoryLoader
from notdiamond.toolkit.rag.llms import get_embedding, get_llm
from notdiamond.toolkit.rag.testset import TestDataGenerator


@pytest.fixture
def test_data_langchain_docs():
    loader = DirectoryLoader("tests/static/", glob="airbnb_tos.md")
    docs = loader.load()
    return docs


@pytest.fixture
def test_data_llamaindex_docs():
    loader = SimpleDirectoryReader(
        input_files=[
            "tests/static/airbnb_tos.md",
        ]
    )
    docs = loader.load_data()
    return docs


@pytest.fixture
def generator_llm():
    return get_llm("openai/gpt-4o-mini")


@pytest.fixture
def generator_embedding():
    return get_embedding("openai/text-embedding-3-small")


def test_get_llm_by_config():
    llm_config = LLMConfig(provider="openai", model="gpt-4o-mini")
    llm = get_llm(llm_config)
    assert isinstance(llm, BaseRagasLLM)


def test_get_embedding_by_config():
    embed_config = EmbeddingConfig(
        provider="openai", model="text-embedding-3-small"
    )
    embed = get_embedding(embed_config)
    assert isinstance(embed, BaseRagasEmbeddings)


def test_dataset_generator_langchain_docs(
    test_data_langchain_docs, generator_llm, generator_embedding
):
    nltk.download("all")
    generator = TestDataGenerator(
        llm=generator_llm, embedding_model=generator_embedding
    )
    dataset = generator.generate_from_docs(
        test_data_langchain_docs, testset_size=2
    )
    assert len(dataset) == 2
    assert isinstance(dataset, pd.DataFrame)


def test_dataset_generator_llamaindex_docs(
    test_data_llamaindex_docs, generator_llm, generator_embedding
):
    generator = TestDataGenerator(
        llm=generator_llm, embedding_model=generator_embedding
    )
    dataset = generator.generate_from_docs(
        test_data_llamaindex_docs, testset_size=2
    )
    assert len(dataset) == 2
    assert isinstance(dataset, pd.DataFrame)
