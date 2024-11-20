from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Type, Union

import optuna

from notdiamond.toolkit.rag.evaluation_dataset import RAGEvaluationDataset


@dataclass
class IntValueRange:
    """
    A range of int values for an auto-evaluated RAG pipeline. Useful for, eg. RAG context chunk size.
    """

    lo: int
    hi: int
    step: int


@dataclass
class FloatValueRange:
    """
    A range of float values for an auto-evaluated RAG pipeline. Useful for, eg. LLM temperature.
    """

    lo: float
    hi: float
    step: float


@dataclass
class CategoricalValueOptions:
    """
    A list of categorical values for an auto-evaluated RAG pipeline. Useful for, eg. embedding algorithms.
    """

    values: List[str]


class BaseNDRagWorkflow:
    """
    A base interface for a RAG workflow to be auto-evaluated by Not Diamond.

    Subclasses should define parameter_specs to type parameters they need to optimize,
    by using type annotations with the above dataclasses. For example:

    class ExampleNDRagWorkflow(BaseNDRagWorkflow):
        parameter_specs = {
            "chunk_size": (Annotated[int, IntValueRange(1000, 2500, 500)], 1000),
            "chunk_overlap": (Annotated[int, IntValueRange(50, 200, 25)], 100),
            "top_k": (Annotated[int, IntValueRange(1, 20, 1)], 5),
            "algo": (
                Annotated[
                    str,
                    CategoricalValueOptions(
                        [
                            "BM25",
                            "openai_small",
                            "openai_large",
                            "cohere_eng",
                            "cohere_multi",
                        ]
                    ),
                ],
                "BM25",
            ),
            "temperature": (Annotated[float, FloatValueRange(0.0, 1.0, 0.1)], 0.9),
        }
    """

    parameter_specs: ClassVar[Dict[str, tuple[Type, Any]]] = {}

    def __init__(self, evaluation_dataset: RAGEvaluationDataset, **kwargs):
        if not self.parameter_specs:
            raise NotImplementedError(
                f"Class {self.__class__.__name__} must define parameter_specs"
            )

        self._param_types = {}
        for param_name, (
            param_type,
            default_value,
        ) in self.parameter_specs.items():
            value = kwargs.get(param_name, default_value)
            setattr(self, param_name, value)
            self._param_types[param_name] = param_type

        self.evaluation_dataset = evaluation_dataset

    def get_parameter_type(self, param_name: str) -> Type:
        return self._param_types.get(param_name)

    def get_parameter_constraint(self, param_name: str) -> Any:
        param_type = self._param_types.get(param_name)
        if hasattr(param_type, "__metadata__"):
            return param_type.__metadata__[0]
        return None

    def rag_workflow(self):
        """
        Users can define their RAG workflow components here by attaching them to `self`. This method will initiate those
        components at init-time, and they will be available in other methods.
        """
        raise NotImplementedError()

    def objective(self):
        raise NotImplementedError()

    def _outer_objective(self, trial: optuna.Trial):
        for param_name in self.parameter_specs.keys():
            param_type = self.get_parameter_type(param_name)
            if param_type == IntValueRange:
                param_value = trial.suggest_int(
                    param_name,
                    param_type.lo,
                    param_type.hi,
                    step=param_type.step,
                )
            elif param_type == FloatValueRange:
                param_value = trial.suggest_float(
                    param_name,
                    param_type.lo,
                    param_type.hi,
                    step=param_type.step,
                )
            elif param_type == CategoricalValueOptions:
                param_value = trial.suggest_categorical(
                    param_name, param_type.values
                )
            setattr(self, param_name, param_value)

        return self.objective()

    def _get_default_param_values(self):
        return {
            param_name: default_value
            for param_name, (_, default_value) in self.parameter_specs.items()
        }

    def _set_param_values(
        self, param_values: Dict[str, Union[int, float, str]]
    ):
        for param_name in self.parameter_specs.keys():
            param_value = param_values.get(param_name)
            if param_value is None:
                raise ValueError(
                    f"Best value for {param_name} not found. This should not happen."
                )
            setattr(self, param_name, param_value)

    def _reset_param_values(self):
        for param_name in self.parameter_specs.keys():
            setattr(
                self, param_name, self._get_default_param_values()[param_name]
            )

    def job_name(self):
        raise NotImplementedError()

    def get_retrieved_context(self, query: str) -> List[str]:
        raise NotImplementedError()

    def get_response(self, query: str) -> str:
        raise NotImplementedError()
