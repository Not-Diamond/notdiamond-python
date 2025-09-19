import pandas as pd
import pytest
from llama_index.core import SimpleDirectoryReader

from notdiamond.llms.config import EmbeddingConfig, LLMConfig
from notdiamond.toolkit.rag.evaluation_dataset import (
    RAGEvaluationDataset,
    RAGSample,
)


def format_prompt(user_input: str, retrieved_contexts: list[str]) -> str:
    context = "\n".join(retrieved_contexts)
    prompt = f"""
    Use the following context to answer the question.

    Context: {context}

    Question: {user_input}
    """
    return prompt


@pytest.fixture
def user_input():
    return "What's the capital of France?"


@pytest.fixture
def retrieved_contexts():
    return ["Paris is the capital and most populous city of France."]


@pytest.fixture
def response():
    return "The capital of France is Paris."


@pytest.fixture
def reference():
    return "Paris"


@pytest.fixture
def gpt_4o():
    return LLMConfig.from_string("openai/gpt-4o")


@pytest.fixture
def sonnet_3_5():
    return LLMConfig.from_string("anthropic/claude-3-7-sonnet-latest")


@pytest.fixture
def openai_embedding():
    return EmbeddingConfig.from_string("openai/text-embedding-3-large")


@pytest.fixture
def dataset(gpt_4o, user_input, retrieved_contexts, response, reference):
    samples = [
        RAGSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference,
            generation_prompt=format_prompt(user_input, retrieved_contexts),
            generator_llm=str(gpt_4o),
        ),
        RAGSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference,
            generation_prompt=format_prompt(user_input, retrieved_contexts),
            generator_llm=str(gpt_4o),
        ),
    ]
    return RAGEvaluationDataset(samples)


@pytest.fixture
def pandas_dataset(
    gpt_4o, user_input, retrieved_contexts, response, reference
):
    data = {
        "user_input": [user_input, user_input],
        "retrieved_contexts": [retrieved_contexts, retrieved_contexts],
        "response": [response, response],
        "reference": [reference, reference],
        "generation_prompt": [
            format_prompt(user_input, retrieved_contexts),
            format_prompt(user_input, retrieved_contexts),
        ],
        "generator_llm": [str(gpt_4o), str(gpt_4o)],
    }
    df = pd.DataFrame(data=data)
    return df


@pytest.fixture
def llamaindex_documents():
    loader = SimpleDirectoryReader(
        input_files=[
            "tests/static/airbnb_tos.md",
        ]
    )
    docs = loader.load_data()
    return docs


@pytest.fixture
def test_queries():
    return ["summarize airbnb's ToS", "What are the cancellation policies?"]
