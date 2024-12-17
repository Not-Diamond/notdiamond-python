from notdiamond.llms.client import NotDiamond  # noqa: F401
from notdiamond.llms.config import LLMConfig  # noqa: F401
from notdiamond.metrics.metric import Metric  # noqa: F401

__all__ = []
try:
    import importlib.util

    spec = importlib.util.find_spec("openai")
    if spec is not None:
        from ._init import init  # noqa: F401

        __all__ = ["init"]
    else:
        __all__ = []
except ImportError:
    pass
