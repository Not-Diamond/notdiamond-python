from unittest.mock import patch

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


@pytest.mark.vcr
def test_init_multi_provider_multi_model_azure_error(model_messages, api_key):
    openai_client = OpenAI()
    azure_client = AzureOpenAI(api_key="broken-api-key")

    manager = init(
        client=[openai_client, azure_client],
        models=["openai/gpt-4o-mini", "azure/gpt-4o-mini"],
        max_retries=1,
        timeout=60.0,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert manager

    azure_wrapper = manager.get_wrapper("azure/gpt-4o-mini")
    openai_wrapper = manager.get_wrapper("openai/gpt-4o-mini")

    with patch.object(
        azure_wrapper,
        "_default_create",
        wraps=azure_wrapper._default_create,
    ) as mock_azure, patch.object(
        openai_wrapper,
        "_default_create",
        wraps=openai_wrapper._default_create,
    ) as mock_openai:
        azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        # 2 calls to OpenAI - one after Azure failure as fallback, second when invoked directly
        assert mock_azure.call_count == 1
        assert mock_openai.call_count == 2


@pytest.mark.vcr
def test_init_multi_provider_multi_model(model_messages, api_key):
    openai_client = OpenAI()
    azure_client = AzureOpenAI()

    manager = init(
        client=[openai_client, azure_client],
        models=["openai/gpt-4o-mini", "azure/gpt-4o-mini"],
        max_retries=1,
        timeout=60.0,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert manager

    azure_wrapper = manager.get_wrapper("azure/gpt-4o-mini")
    openai_wrapper = manager.get_wrapper("openai/gpt-4o-mini")

    with patch.object(
        azure_wrapper,
        "_default_create",
        wraps=azure_wrapper._default_create,
    ) as mock_azure, patch.object(
        openai_wrapper,
        "_default_create",
        wraps=openai_wrapper._default_create,
    ) as mock_openai:
        azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        assert mock_azure.call_count == 1
        assert mock_openai.call_count == 1


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.timeout(20)
async def test_async_init_multi_provider_multi_model_azure_error(
    model_messages, api_key
):
    openai_client = AsyncOpenAI()
    azure_client = AsyncAzureOpenAI(api_key="broken-api-key")

    manager = init(
        client=[openai_client, azure_client],
        models=["openai/gpt-4o-mini", "azure/gpt-4o-mini"],
        max_retries=1,
        timeout=60.0,
        model_messages=model_messages,
        api_key=api_key,
        async_mode=True,
    )

    assert manager

    azure_wrapper = manager.get_wrapper("azure/gpt-4o-mini")
    openai_wrapper = manager.get_wrapper("openai/gpt-4o-mini")

    with patch.object(
        azure_wrapper,
        "_default_create",
        wraps=azure_wrapper._default_create,
    ) as mock_azure, patch.object(
        openai_wrapper,
        "_default_create",
        wraps=openai_wrapper._default_create,
    ) as mock_openai:
        await azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        # 2 calls to OpenAI - one after Azure failure as fallback, second when invoked directly
        assert mock_azure.call_count == 1
        assert mock_openai.call_count == 2


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.timeout(20)
async def test_async_init_multi_provider_multi_model(model_messages, api_key):
    openai_client = AsyncOpenAI()
    azure_client = AsyncAzureOpenAI()

    manager = init(
        client=[openai_client, azure_client],
        models=["openai/gpt-4o-mini", "azure/gpt-4o-mini"],
        max_retries=1,
        timeout=60.0,
        model_messages=model_messages,
        api_key=api_key,
        async_mode=True,
    )

    assert manager

    azure_wrapper = manager.get_wrapper("azure/gpt-4o-mini")
    openai_wrapper = manager.get_wrapper("openai/gpt-4o-mini")

    with patch.object(
        azure_wrapper,
        "_default_create",
        wraps=azure_wrapper._default_create,
    ) as mock_azure, patch.object(
        openai_wrapper,
        "_default_create",
        wraps=openai_wrapper._default_create,
    ) as mock_openai:
        await azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        assert mock_azure.call_count == 1
        assert mock_openai.call_count == 1


@pytest.mark.vcr
def test_init_multi_provider_multi_model_multi_config(api_key):
    openai_client = OpenAI()
    azure_client = AzureOpenAI(api_key="broken-api-key")

    max_retries = {
        "openai/gpt-4o-mini": 1,
        "azure/gpt-4o-mini": 3,
    }
    timeout = {
        "azure/gpt-4o-mini": 5.0,
        "openai/gpt-4o-mini": 20.0,
    }
    model_messages = {
        "azure/gpt-4o-mini": [
            {"role": "user", "content": "Hello, do you live on Azure?"}
        ],
        "openai/gpt-4o-mini": [
            {"role": "user", "content": "Hello, do you live on OpenAI?"}
        ],
    }
    default_messages = [{"role": "user", "content": "Hello, how are you?"}]

    manager = init(
        client=[openai_client, azure_client],
        models=["openai/gpt-4o-mini", "azure/gpt-4o-mini"],
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )

    assert manager

    azure_wrapper = manager.get_wrapper("azure/gpt-4o-mini")
    openai_wrapper = manager.get_wrapper("openai/gpt-4o-mini")

    with patch.object(
        azure_wrapper,
        "_default_create",
        wraps=azure_wrapper._default_create,
    ) as mock_azure, patch.object(
        openai_wrapper,
        "_default_create",
        wraps=openai_wrapper._default_create,
    ) as mock_openai:
        azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        # 2 calls to OpenAI - one after Azure failure as fallback, second when invoked directly
        assert mock_azure.call_count == max_retries["azure/gpt-4o-mini"]
        mock_azure.assert_called_with(
            model="gpt-4o-mini",
            messages=model_messages["azure/gpt-4o-mini"] + default_messages,
            timeout=timeout["azure/gpt-4o-mini"],
        )
        assert mock_openai.call_count == 2
        mock_openai.assert_called_with(
            model="gpt-4o-mini",
            messages=model_messages["openai/gpt-4o-mini"] + default_messages,
            timeout=timeout["openai/gpt-4o-mini"],
        )
