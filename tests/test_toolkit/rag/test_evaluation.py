#!/usr/bin/env python3
import pandas as pd
import pytest

from notdiamond.llms.config import EmbeddingConfig, LLMConfig
from notdiamond.toolkit.rag.evaluation import evaluate
from notdiamond.toolkit.rag.evaluation_dataset import (
    RAGEvaluationDataset,
    RAGSample,
)
from notdiamond.toolkit.rag.llms import get_embedding, get_llm
from notdiamond.toolkit.rag.metrics import Faithfulness, SemanticSimilarity


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
    return LLMConfig.from_string("anthropic/claude-3-5-sonnet-20241022")


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


def test_dataset_from_pandas(pandas_dataset):
    dataset = RAGEvaluationDataset.from_pandas(pandas_dataset)
    assert isinstance(dataset, RAGEvaluationDataset)


def test_dataset_format(dataset):
    assert len(dataset) == 2
    assert isinstance(dataset[0], RAGSample)


def test_evaluate(dataset, gpt_4o, sonnet_3_5, openai_embedding):
    evaluator_llm = get_llm(gpt_4o)
    evaluator_embedding = get_embedding(openai_embedding)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embedding),
    ]

    generator_llms = [gpt_4o, sonnet_3_5]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        generator_llms=generator_llms,
    )

    expected_results_keys = [str(gpt_4o), str(sonnet_3_5)]
    assert all([key in expected_results_keys for key in results.keys()])

    gpt_4o_result = results[str(gpt_4o)]
    expected_results_columns = [
        "user_input",
        "generation_prompt",
        "retrieved_contexts",
        "response",
        "reference",
        "faithfulness",
        "semantic_similarity",
    ]
    assert all(
        [
            col in expected_results_columns
            for col in list(gpt_4o_result.columns)
        ]
    )
