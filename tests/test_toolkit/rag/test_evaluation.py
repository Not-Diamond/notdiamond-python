#!/usr/bin/env python3
import pytest

# from notdiamond.toolkit.rag.evaluation_dataset import RAGEvaluationDataset, RAGSample
# from notdiamond.toolkit.rag.evaluation import evaluate
# from notdiamond.toolkit.rag.metrics import Faithfulness, SemanticSimilarity
from notdiamond.llms.config import LLMConfig


@pytest.fixture
def gpt_4o():
    return LLMConfig.from_string("openai/gpt-4o")


@pytest.fixture
def sonnet_3_5():
    return LLMConfig.from_string("anthropic/claude-3-5-sonnet-20241022")


@pytest.fixture
def dataset(gpt_4o):
    pass
    # samples = []


def test_dataset_format():
    # test column names
    # test length
    pass


def test_evaluate():
    # test workflow
    # test results dict keys
    # test results columns
    pass
