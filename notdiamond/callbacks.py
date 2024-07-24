import logging
from typing import Any

from notdiamond._utils import _module_check
from notdiamond.llms.config import LLMConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _nd_callback_handler_factory():
    try:
        BaseCallbackHandler = _module_check(
            "langchain_core.callbacks.base", "BaseCallbackHandler"
        )
    except (ModuleNotFoundError, ImportError) as ierr:
        log_msg = (
            "Attempted to instantiate NDLLMBaseCallbackHandler but langchain is not installed. ",
            f"Model callbacks not available. {ierr}",
        )
        LOGGER.warning(log_msg)
        return

    class _NDLLMBaseCallbackHandler(BaseCallbackHandler):
        """
        Base callback handler for NotDiamond LLMs.
        Accepts all of the langchain_core callbacks and adds new ones.
        """

        def on_model_select(
            self, model_provider: LLMConfig, model_name: str
        ) -> Any:
            """
            Called when a model is selected.
            """

        def on_latency_tracking(
            self,
            session_id: str,
            model_provider: LLMConfig,
            tokens_per_second: float,
        ):
            """
            Called when latency tracking is enabled.
            """

        def on_api_error(self, error_message: str):
            """
            Called when an NotDiamond API error occurs.
            """

    return _NDLLMBaseCallbackHandler


NDLLMBaseCallbackHandler = _nd_callback_handler_factory()
