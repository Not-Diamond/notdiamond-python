import pytest

from notdiamond.llms.client import _NDClientTarget, _ndllm_factory
from notdiamond.llms.providers import NDLLMProviders


@pytest.mark.vcr
@pytest.mark.parametrize(
    "nd_llm_cls",
    [
        (_ndllm_factory(_NDClientTarget.ROUTER)),
        (_ndllm_factory(_NDClientTarget.INVOKER)),
    ],
)
def test_model_select_callback(llm_base_callback_handler, nd_llm_cls):
    """
    This tests that the on_model_select method is called when a model is selected.
    """

    nd_llm = nd_llm_cls(
        llm_configs=[
            NDLLMProviders.GPT_4_TURBO_PREVIEW,
            NDLLMProviders.GPT_3_5_TURBO,
        ],
        callbacks=[llm_base_callback_handler],
    )
    nd_llm.model_select(
        messages=[{"role": "user", "content": "Hello, what's your name?"}]
    )
    assert llm_base_callback_handler.on_model_select_called


@pytest.mark.vcr
def test_latency_tracking_callback(llm_base_callback_handler, nd_invoker_cls):
    """
    This tests that latency tracking is enabled and the on_latency_tracking method is called.
    """
    nd_llm = nd_invoker_cls(
        llm_configs=[
            NDLLMProviders.GPT_4_TURBO_PREVIEW,
            NDLLMProviders.GPT_3_5_TURBO,
        ],
        callbacks=[llm_base_callback_handler],
        latency_tracking=True,
    )
    nd_llm.invoke(
        messages=[{"role": "user", "content": "Hello, what's your name?"}]
    )
    assert llm_base_callback_handler.on_latency_tracking_called


@pytest.mark.vcr
def test_llm_start_callback(llm_base_callback_handler, nd_invoker_cls):
    """
    This tests that callback handler is passed along correctly to
    the langchain LLM class, by checking if the on_llm_start method is called.
    """
    nd_llm = nd_invoker_cls(
        llm_configs=[
            NDLLMProviders.GPT_4_TURBO_PREVIEW,
            NDLLMProviders.GPT_3_5_TURBO,
        ],
        callbacks=[llm_base_callback_handler],
    )
    nd_llm.invoke(messages=[{"role": "user", "content": "Tell me a joke."}])
    assert llm_base_callback_handler.on_llm_start_called
