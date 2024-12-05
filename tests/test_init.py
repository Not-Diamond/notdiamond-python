import pytest
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from notdiamond import init


@pytest.fixture
def models():
    return ["openai/gpt-4o-mini", "anthropic/claude-3-5-haiku-20241022"]


@pytest.fixture
def max_retries():
    return 3


@pytest.fixture
def timeout():
    return 60.0


@pytest.fixture
def model_messages(models):
    return {
        model: [{"role": "user", "content": "Hello, how are you?"}]
        for model in models
    }


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.mark.parametrize(
    "client",
    [
        OpenAI(),
        Anthropic(),
        AzureOpenAI(),
        AsyncAzureOpenAI(),
        AsyncOpenAI(),
        AsyncAnthropic(),
        # [a9] need to set up bedrock to test these
        # AnthropicBedrock(),
        # AsyncAnthropicBedrock(),
    ],
)
def test_init_call(
    client, models, max_retries, timeout, model_messages, api_key
):
    result = init(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert result


def test_init_call_list(models, max_retries, timeout, model_messages, api_key):
    result = init(
        client=[OpenAI(), Anthropic()],
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert result
