#!/usr/bin/env python3
import pytest

from notdiamond.toolkit.rag.evaluation import evaluate
from notdiamond.toolkit.rag.evaluation_dataset import (
    RAGEvaluationDataset,
    RAGSample,
)
from notdiamond.toolkit.rag.llms import get_embedding, get_llm
from notdiamond.toolkit.rag.metrics import Faithfulness, SemanticSimilarity


def test_dataset_from_pandas(pandas_dataset):
    dataset = RAGEvaluationDataset.from_pandas(pandas_dataset)
    assert isinstance(dataset, RAGEvaluationDataset)


def test_dataset_format(dataset):
    assert len(dataset) == 2
    assert isinstance(dataset[0], RAGSample)


@pytest.mark.vcr
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
