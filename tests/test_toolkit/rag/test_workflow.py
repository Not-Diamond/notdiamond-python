from typing import Annotated, Any, List

import pytest

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow, IntValueRange


class TestNDRagWorkflow(BaseNDRagWorkflow):
    parameter_specs = {
        "chunk_size": (Annotated[int, IntValueRange(1000, 2500, 500)], 1000)
    }

    def rag_workflow(self, documents: Any, test_queries: List[str]):
        pass


def test_set_param_values(llamaindex_documents, test_queries):
    workflow = TestNDRagWorkflow(
        llamaindex_documents, test_queries, objective_maximize=True
    )
    workflow._set_param_values({"chunk_size": 1500})
    assert workflow.chunk_size == 1500

    with pytest.raises(ValueError):
        workflow._set_param_values({"nonexistent_param": 1500})

    with pytest.raises(ValueError):
        workflow._set_param_values({"chunk_size": "not an int"})

    # value too low for range
    with pytest.raises(ValueError):
        workflow._set_param_values({"chunk_size": 100})


def test_set_bad_param_values(test_queries, llamaindex_documents):
    class BadNDRagWorkflow(BaseNDRagWorkflow):
        parameter_specs = {"chunk_size": (int, 1000)}

        def rag_workflow(self, documents: Any, test_queries: List[str]):
            pass

    # should fail bc we need ranges
    with pytest.raises(ValueError):
        BadNDRagWorkflow(
            llamaindex_documents, test_queries, objective_maximize=True
        )

    class BadNDRagWorkflow2(BaseNDRagWorkflow):
        parameter_specs = {"chunk_size": (int, "foo")}

    # should fail bc we need matching types
    with pytest.raises(ValueError):
        BadNDRagWorkflow2(
            llamaindex_documents, test_queries, objective_maximize=True
        )
