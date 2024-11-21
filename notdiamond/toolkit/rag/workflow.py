from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Type, Union, get_args

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

    def __contains__(self, value: int) -> bool:
        return (
            self.lo <= value <= self.hi and (value - self.lo) % self.step == 0
        )


@dataclass
class FloatValueRange:
    """
    A range of float values for an auto-evaluated RAG pipeline. Useful for, eg. LLM temperature.
    """

    lo: float
    hi: float
    step: float

    def __contains__(self, value: float) -> bool:
        return (
            self.lo <= value <= self.hi and (value - self.lo) % self.step == 0
        )


@dataclass
class CategoricalValueOptions:
    """
    A list of categorical values for an auto-evaluated RAG pipeline. Useful for, eg. embedding algorithms.
    """

    values: List[str]

    def __contains__(self, value: str) -> bool:
        return value in self.values


_ALLOWED_TYPES = [IntValueRange, FloatValueRange, CategoricalValueOptions]


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

    def __init__(
        self,
        evaluation_dataset: RAGEvaluationDataset,
        documents: Any,
        objective_maximize: bool,
        **kwargs,
    ):
        """
        Args:
            evaluation_dataset: A dataset of RAG evaluation samples.
            documents: The documents to use for RAG.
            objective_maximize: Whether to maximize or minimize the objective defined below.
        """
        if not self.parameter_specs:
            raise NotImplementedError(
                f"Class {self.__class__.__name__} must define parameter_specs"
            )

        self._param_ranges = {}
        self._base_param_types = {}
        for param_name, (
            param_type,
            default_value,
        ) in self.parameter_specs.items():
            type_args = get_args(param_type)
            base_type, range_type = type_args
            if range_type is None:
                raise ValueError(
                    f"Expected parameter type in {_ALLOWED_TYPES} but received {param_type}"
                )
            self._param_ranges[param_name] = range_type
            self._base_param_types[param_name] = base_type
            if not isinstance(default_value, base_type):
                raise ValueError(
                    f"Expected default value type {base_type} but received {type(default_value)}"
                )
            setattr(self, param_name, default_value)

        self.evaluation_dataset = evaluation_dataset
        self.documents = documents
        self.objective_maximize = objective_maximize
        self.rag_workflow(documents)

    def get_parameter_type(self, param_name: str) -> Type:
        param_type = self._base_param_types.get(param_name)
        if param_type is None:
            raise ValueError(
                f"Parameter {param_name} not found in parameter_specs"
            )
        return param_type

    def get_parameter_range(self, param_name: str) -> Type:
        param_range = self._param_ranges.get(param_name)
        if param_range is None:
            raise ValueError(
                f"Parameter {param_name} not found in parameter_specs"
            )
        return param_range

    def rag_workflow(self, documents: Any):
        """
        Define RAG workflow components here by setting instance attrs on `self`. Those attributes will be set
        at init-time and available when retrieving context or generating responses.
        """
        raise NotImplementedError()

    def objective(self):
        """
        Define the objective function for your RAG workflow. The workflow's hyperparameters will be optimized
        according to values of this objective.
        """
        raise NotImplementedError()

    def _outer_objective(self, trial: optuna.Trial):
        for param_name in self.parameter_specs.keys():
            param_range = self.get_parameter_range(param_name)
            if isinstance(param_range, IntValueRange):
                param_value = trial.suggest_int(
                    param_name,
                    param_range.lo,
                    param_range.hi,
                    step=param_range.step,
                )
            elif isinstance(param_range, FloatValueRange):
                param_value = trial.suggest_float(
                    param_name,
                    param_range.lo,
                    param_range.hi,
                    step=param_range.step,
                )
            elif isinstance(param_range, CategoricalValueOptions):
                param_value = trial.suggest_categorical(
                    param_name, param_range.values
                )
            else:
                raise ValueError(
                    f"Expected parameter type in {_ALLOWED_TYPES} but received unknown parameter type: {param_range}"
                )
            setattr(self, param_name, param_value)

        result = self.objective()
        self._reset_param_values()
        return result

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
            param_type = self.get_parameter_type(param_name)
            param_range = self.get_parameter_range(param_name)
            if param_value is None:
                raise ValueError(
                    f"Best value for {param_name} not found. This should not happen."
                )
            elif not isinstance(param_value, param_type):
                raise ValueError(
                    f"Expected parameter type {param_type} but received {type(param_value)}"
                )
            elif param_value not in param_range:
                raise ValueError(
                    f"Parameter value {param_value} not in range {param_range}"
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
