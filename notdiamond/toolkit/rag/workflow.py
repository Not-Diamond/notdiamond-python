from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Type


@dataclass
class IntValueRange:
    lo: int
    hi: int
    step: int


@dataclass
class FloatValueRange:
    lo: float
    hi: float
    step: float


@dataclass
class CategoricalValueOptions:
    values: List[str]


class BaseNDRagWorkflow:
    parameter_specs: ClassVar[Dict[str, tuple[Type, Any]]] = {}

    def __init__(self, **kwargs):
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

    def get_parameter_type(self, param_name: str) -> Type:
        return self._param_types.get(param_name)

    def get_parameter_constraint(self, param_name: str) -> Any:
        param_type = self._param_types.get(param_name)
        if hasattr(param_type, "__metadata__"):
            return param_type.__metadata__[0]
        return None

    def rag_workflow(self):
        raise NotImplementedError()

    def objective(self):
        # todo [t7 + a9] should we move this out of hyperopt.py / optuna.py
        raise NotImplementedError()

    def job_name(self):
        raise NotImplementedError()

    def get_retrieved_context(self, query: str) -> List[str]:
        raise NotImplementedError()

    def get_response(self, query: str) -> str:
        raise NotImplementedError()
