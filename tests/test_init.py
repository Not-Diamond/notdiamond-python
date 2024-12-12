import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from notdiamond import init


@pytest.fixture
def models():
    return [
        "openai/gpt-4o-mini",
        "azure/gpt-4o-mini",
        "openai/gpt-4o",
        "azure/gpt-4o",
    ]


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
    ("client", "model"),
    [
        (OpenAI(), "openai/gpt-4o-mini"),
        (AzureOpenAI(), "azure/gpt-4o-mini"),
    ],
)
def test_init_call(
    client, model, max_retries, timeout, model_messages, api_key
):
    result = init(
        client=client,
        models=model,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert result


def test_init_call_list(models, max_retries, timeout, model_messages, api_key):
    result = init(
        client=[OpenAI(), AzureOpenAI()],
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert result

    async_result = init(
        client=[AsyncOpenAI(), AsyncAzureOpenAI()],
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert async_result


def test_init_call_single_client_multi_models(
    max_retries, timeout, model_messages, api_key
):
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    result = init(
        client=OpenAI(),
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert result
