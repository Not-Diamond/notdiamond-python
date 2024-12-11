from unittest.mock import patch

import pytest
from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    AzureOpenAI,
    OpenAI,
)

from notdiamond import init
from notdiamond.toolkit.retry import (
    AsyncRetryWrapper,
    RetryManager,
    RetryWrapper,
)


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
    return 1


@pytest.fixture
def timeout():
    return 60.0


@pytest.fixture
def model_messages(models):
    return {
        m.split("/")[-1]: [{"role": "user", "content": "Hello, how are you?"}]
        for m in models
    }


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def broken_client():
    client = OpenAI(api_key="broken-api-key")
    with patch.object(
        client.chat.completions, "create", wraps=client.chat.completions.create
    ) as mock_create:
        yield mock_create


@pytest.fixture
def broken_async_client():
    client = AsyncOpenAI(api_key="broken-api-key")
    with patch.object(client.chat.completions, "create") as mock_create:
        client.chat.completions.create = mock_create
    return client


@pytest.mark.parametrize(
    ("client", "models"),
    [
        pytest.param(
            OpenAI(), ["openai/gpt-4o-mini", "openai/gpt-4o"], id="openai"
        ),
        pytest.param(
            AzureOpenAI(),
            ["azure/gpt-4o-mini", "azure/gpt-4o"],
            id="azure-openai",
        ),
    ],
)
@pytest.mark.vcr
def test_retry_wrapper_create(
    client, models, max_retries, timeout, model_messages, api_key
):
    wrapper = RetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=model_messages["gpt-4o-mini"],
    )
    assert result


@pytest.mark.parametrize(
    ("client", "models"),
    [
        pytest.param(
            AsyncOpenAI(),
            ["openai/gpt-4o-mini", "openai/gpt-4o"],
            id="async-openai",
        ),
        pytest.param(
            AsyncAzureOpenAI(),
            ["azure/gpt-4o-mini", "azure/gpt-4o"],
            id="async-azure-openai",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_retry_wrapper_async_create(
    client, models, max_retries, timeout, model_messages, api_key
):
    wrapper = AsyncRetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=model_messages["gpt-4o-mini"],
    )
    assert result


@pytest.mark.parametrize(
    ("client", "models"),
    [
        pytest.param(
            OpenAI(), ["openai/gpt-4o-mini", "openai/gpt-4o"], id="openai"
        ),
        pytest.param(
            AzureOpenAI(),
            ["azure/gpt-4o-mini", "azure/gpt-4o"],
            id="azure-openai",
        ),
    ],
)
@pytest.mark.vcr
def test_retry_wrapper_stream(
    client, models, max_retries, timeout, model_messages, api_key
):
    wrapper = RetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=model_messages["gpt-4o-mini"],
        stream=True,
    )
    assert result


@pytest.mark.parametrize(
    ("client", "models"),
    [
        pytest.param(
            AsyncOpenAI(),
            ["openai/gpt-4o-mini", "openai/gpt-4o"],
            id="async-openai",
        ),
        pytest.param(
            AsyncAzureOpenAI(),
            ["azure/gpt-4o-mini", "azure/gpt-4o"],
            id="async-azure-openai",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_retry_wrapper_async_stream(
    client, models, max_retries, timeout, model_messages, api_key
):
    wrapper = AsyncRetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    result = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=model_messages["gpt-4o-mini"],
        stream=True,
    )
    assert result


def test_retries(timeout, model_messages, api_key, broken_client):
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    wrapper = RetryWrapper(
        client=broken_client,
        models=models,
        max_retries=1,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    with pytest.raises(AuthenticationError):
        broken_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )

    print(broken_client.chat.completions.create)
    assert broken_client.chat.completions.create.mock_calls == [
        pytest.call(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
            timeout=timeout,
        ),
        pytest.call(
            model="gpt-4o", messages=model_messages["gpt-4o"], timeout=timeout
        ),
    ]


@pytest.mark.asyncio
async def test_retries_async(
    timeout, model_messages, api_key, broken_async_client
):
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    wrapper = AsyncRetryWrapper(
        client=broken_async_client,
        models=models,
        max_retries=1,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert wrapper

    manager = RetryManager(models, [wrapper])
    assert manager

    with pytest.raises(AuthenticationError):
        await broken_async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )

    broken_async_client.chat.completions.create.assert_has_calls(
        [
            pytest.call(
                model="gpt-4o-mini",
                messages=model_messages["gpt-4o-mini"],
                timeout=timeout,
            ),
            pytest.call(
                model="gpt-4o",
                messages=model_messages["gpt-4o"],
                timeout=timeout,
            ),
        ]
    )


def test_multi_model_multi_provider(
    models, max_retries, timeout, model_messages, api_key, broken_client
):
    clients = [
        OpenAI(),
        AzureOpenAI(),
        broken_client,
    ]
    manager = init(
        client=clients,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert manager

    client = clients[0]
    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=model_messages["gpt-4o-mini"],
        stream=True,
    )
    assert result
