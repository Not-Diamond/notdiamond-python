from typing import Annotated

import pytest

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow, IntValueRange


class TestNDRagWorkflow(BaseNDRagWorkflow):
    parameter_specs = {
        "chunk_size": (Annotated[int, IntValueRange(1000, 2500, 500)], 1000)
    }


def test_set_param_values(dataset):
    workflow = TestNDRagWorkflow(dataset)
    workflow._set_param_values({"chunk_size": 1500})
    assert workflow.chunk_size == 1500

    with pytest.raises(ValueError):
        workflow._set_param_values({"nonexistent_param": 1500})
