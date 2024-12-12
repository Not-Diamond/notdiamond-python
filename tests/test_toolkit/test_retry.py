from unittest.mock import AsyncMock, call, patch

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
    return 20.0


@pytest.fixture
def model_messages(models):
    return {
        m.split("/")[-1]: [{"role": "user", "content": "Hello, how are you?"}]
        for m in models
    }


@pytest.fixture
def api_key():
    return "test-api-key"


class BrokenAsyncClientException(Exception):
    pass


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


def test_retries(timeout, model_messages, api_key):
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    patched_client = OpenAI(api_key="broken-api-key")

    with patch.object(
        patched_client.chat.completions,
        "create",
        wraps=patched_client.chat.completions.create,
    ) as mock_create:
        patched_client.chat.completions.create = mock_create
        wrapper = RetryWrapper(
            client=patched_client,
            models=models,
            max_retries=1,
            timeout=timeout,
            model_messages=model_messages,
            api_key=api_key,
        )
        _ = RetryManager(models, [wrapper])

        with pytest.raises(AuthenticationError):
            patched_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=model_messages["gpt-4o-mini"],
            )

        assert (
            mock_create.call_count == 2
        ), f"Expected 2 calls, got {mock_create.call_count}: {mock_create.call_args_list}"
        mock_create.assert_has_calls(
            [
                call(
                    model="gpt-4o-mini",
                    messages=model_messages["gpt-4o-mini"],
                    timeout=timeout,
                ),
                call(
                    model="gpt-4o",
                    messages=model_messages["gpt-4o"],
                    timeout=timeout,
                ),
            ],
            any_order=True,
        )


@pytest.mark.asyncio
async def test_retries_async(timeout, model_messages, api_key):
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    patched_client = AsyncOpenAI(api_key="broken-api-key")

    async def async_error(*args, **kwargs):
        raise BrokenAsyncClientException()

    with patch.object(
        patched_client.chat.completions,
        "create",
        wraps=async_error,
        new_callable=AsyncMock,
    ) as mock_create:
        patched_client.chat.completions.create = mock_create
        wrapper = AsyncRetryWrapper(
            client=patched_client,
            models=models,
            max_retries=1,
            timeout=timeout,
            model_messages=model_messages,
            api_key=api_key,
        )
        _ = RetryManager(models, [wrapper])

        with pytest.raises(BrokenAsyncClientException):
            await patched_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=model_messages["gpt-4o-mini"],
            )

        assert mock_create.call_count == 2
        mock_create.assert_has_calls(
            [
                call(
                    model="gpt-4o-mini",
                    messages=model_messages["gpt-4o-mini"],
                    timeout=timeout,
                ),
                call(
                    model="gpt-4o",
                    messages=model_messages["gpt-4o"],
                    timeout=timeout,
                ),
            ],
            any_order=True,
        )


def test_multi_model_multi_provider(
    max_retries, timeout, model_messages, api_key
):
    oai_client = OpenAI()
    azure_client = AzureOpenAI()
    clients = [
        oai_client,
        azure_client,
    ]
    models = [
        "openai/gpt-4o-mini",
        "azure/gpt-4o-mini",
        "openai/gpt-4o",
        "azure/gpt-4o",
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

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_openai, patch.object(
        azure_client.chat.completions,
        "create",
        wraps=azure_client.chat.completions.create,
    ) as mock_azure:
        mock_openai.return_value = "mocked_openai_response"
        mock_azure.return_value = "mocked_azure_response"

        client = oai_client
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        assert result == "mocked_openai_response"

        mock_openai.assert_called_once_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        mock_azure.assert_not_called()

        azure_result = azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        assert azure_result == "mocked_azure_response"
        mock_azure.assert_called_once_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )

        azure_gpt4o_result = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        assert azure_gpt4o_result == "mocked_azure_response"
        mock_azure.assert_called_with(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        assert not any(
            call(model="gpt-4o", messages=model_messages["gpt-4o"]) == c
            for c in mock_openai.call_args_list
        )


def test_multi_model_multi_provider_load_balance(
    models, max_retries, timeout, model_messages, api_key
):
    """
    Configure this test to ensure load balancing is working. In this setup, we should
    only invoke `gpt-4o-mini` via the OpenAI client, no matter which client or model
    is invoked by the user.
    """
    oai_client = OpenAI()
    azure_client = AzureOpenAI()
    clients = [
        oai_client,
        azure_client,
    ]
    models = {
        "openai/gpt-4o-mini": 1.0,
        "azure/gpt-4o-mini": 0.0,
        "openai/gpt-4o": 0.0,
        "azure/gpt-4o": 0.0,
    }
    manager = init(
        client=clients,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert manager

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_openai, patch.object(
        azure_client.chat.completions,
        "create",
        wraps=azure_client.chat.completions.create,
    ) as mock_azure:
        azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        mock_azure.assert_not_called()
        mock_openai.assert_called_once_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )

        azure_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        mock_azure.assert_not_called()
        mock_openai.call_count == 2
        mock_openai.reset_mock()

        # Even if the user calls `gpt-4o` from any client here, we should still invoke `gpt-4o-mini`
        azure_client.chat.completions.create(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        oai_client.chat.completions.create(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        mock_azure.assert_not_called()
        mock_openai.call_count == 2
        mock_openai.assert_called_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        assert not any(
            call(model="gpt-4o", messages=model_messages["gpt-4o"]) == c
            for c in mock_openai.call_args_list
        )


@pytest.mark.skip
def test_multi_model_multi_provider_azure_error(model_messages, api_key):
    """
    Configure this test so that AzureOpenAI fails on authentication. The RetryManager should ensure that
    we fall back to the OpenAI client.
    """
    oai_client = OpenAI()
    azure_client = AzureOpenAI(api_key="broken-api-key")
    clients = [
        oai_client,
        azure_client,
    ]
    models = [
        "azure/gpt-4o",
        "azure/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
    ]

    manager = init(
        client=clients,
        models=models,
        max_retries=1,
        timeout=10.0,
        model_messages=model_messages,
        api_key=api_key,
    )
    assert manager

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_openai, patch.object(
        azure_client.chat.completions,
        "create",
        wraps=azure_client.chat.completions.create,
    ) as mock_azure:
        mock_openai.return_value = "mocked_openai_response"

        client = azure_client
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        assert (
            result == "mocked_openai_response"
        ), f"Expected fallback invocation of OpenAI client, got {result}"

        mock_azure.assert_called_once_with(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        mock_azure.assert_called_once_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        mock_openai.assert_called_once_with(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )


def test_multi_model_timeout_config(model_messages, api_key):
    oai_client = OpenAI()
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    timeout = {"openai/gpt-4o-mini": 10.0, "openai/gpt-4o": 5.0}

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_create:
        wrapper = RetryWrapper(
            client=oai_client,
            models=models,
            max_retries=1,
            timeout=timeout,
            model_messages=model_messages,
            api_key=api_key,
        )
        _ = RetryManager(models, [wrapper])
        result = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
        assert result
        mock_create.assert_called_with(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
            timeout=10.0,
        )
        gpt4o_result = oai_client.chat.completions.create(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
        )
        assert gpt4o_result
        mock_create.assert_called_with(
            model="gpt-4o",
            messages=model_messages["gpt-4o"],
            timeout=5.0,
        )


def test_multi_model_max_retries_config(model_messages, api_key):
    oai_client = OpenAI(api_key="broken-api-key")
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]
    max_retries = {
        "openai/gpt-4o-mini": 1,
        "openai/gpt-4o": 2,
    }

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_create:
        wrapper = RetryWrapper(
            client=oai_client,
            models=models,
            max_retries=max_retries,
            timeout=20.0,
            model_messages=model_messages,
            api_key=api_key,
        )
        _ = RetryManager(models, [wrapper])
        with pytest.raises(AuthenticationError):
            oai_client.chat.completions.create(
                model="gpt-4o-mini",
            )
        assert mock_create.call_count == 3


def test_multi_model_model_messages_config(timeout, api_key):
    oai_client = OpenAI()
    models = ["openai/gpt-4o-mini", "openai/gpt-4o"]

    gpt4o_message = [{"role": "user", "content": "Is this 4o or 4o-mini?"}]
    gpt4o_mini_message = [{"role": "user", "content": "Hello, how are you?"}]
    model_messages = {
        "openai/gpt-4o-mini": gpt4o_mini_message,
        "openai/gpt-4o": gpt4o_message,
    }

    with patch.object(
        oai_client.chat.completions,
        "create",
        wraps=oai_client.chat.completions.create,
    ) as mock_create:
        wrapper = RetryWrapper(
            client=oai_client,
            models=models,
            max_retries=1,
            timeout=timeout,
            model_messages=model_messages,
            api_key=api_key,
        )
        _ = RetryManager(models, [wrapper])
        result = oai_client.chat.completions.create(
            model="gpt-4o-mini",
        )
        assert result
        mock_create.assert_called_with(
            model="gpt-4o-mini", messages=gpt4o_mini_message, timeout=20.0
        )
        gpt4o_result = oai_client.chat.completions.create(
            model="gpt-4o",
        )
        assert gpt4o_result
        mock_create.assert_called_with(
            model="gpt-4o", messages=gpt4o_message, timeout=20.0
        )
