from typing import Annotated

import pytest

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow, IntValueRange


class TestNDRagWorkflow(BaseNDRagWorkflow):
    parameter_specs = {
        "chunk_size": (Annotated[int, IntValueRange(1000, 2500, 500)], 1000)
    }


def test_set_param_values(dataset, documents):
    workflow = TestNDRagWorkflow(dataset, documents)
    workflow._set_param_values({"chunk_size": 1500})
    assert workflow.chunk_size == 1500

    with pytest.raises(ValueError):
        workflow._set_param_values({"nonexistent_param": 1500})

    with pytest.raises(ValueError):
        workflow._set_param_values({"chunk_size": "not an int"})


def test_set_bad_param_values(dataset, documents):
    class BadNDRagWorkflow(BaseNDRagWorkflow):
        parameter_specs = {"chunk_size": (int, 1000)}

    # should fail bc we need ranges
    with pytest.raises(ValueError):
        BadNDRagWorkflow(dataset, documents)

    class BadNDRagWorkflow2(BaseNDRagWorkflow):
        parameter_specs = {"chunk_size": (int, "foo")}

    # should fail bc we need matching types
    with pytest.raises(ValueError):
        BadNDRagWorkflow2(dataset, documents)
