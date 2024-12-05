import pytest
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit.retry import AsyncRetryWrapper, RetryWrapper


@pytest.fixture
def models():
    return [
        LLMConfig.from_string("openai/gpt-4o-mini"),
        LLMConfig.from_string("anthropic/claude-3-5-haiku-20241022"),
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
        model.model: [{"role": "user", "content": "Hello, how are you?"}]
        for model in models
    }


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(OpenAI(), id="openai"),
        pytest.param(Anthropic(), id="anthropic"),
        # pytest.param(AzureOpenAI(), id='azure-openai'),
    ],
)
@pytest.mark.vcr
def test_retry_wrapper(
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

    if isinstance(client, Anthropic):
        result = wrapper.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=model_messages["claude-3-5-haiku-20241022"],
            max_tokens=1024,
        )
    else:
        result = wrapper.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
    assert result


@pytest.mark.parametrize(
    "client",
    [
        pytest.param(AsyncOpenAI(), id="async-openai"),
        pytest.param(AsyncAnthropic(), id="async-anthropic"),
        # pytest.param(AsyncAzureOpenAI(), id='async-azure-openai'),
    ],
)
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_retry_wrapper_async(
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

    if isinstance(client, AsyncAnthropic):
        result = await wrapper.messages.create(
            model="claude-3-5-haiku-20241022",
            messages=model_messages["claude-3-5-haiku-20241022"],
            max_tokens=1024,
        )
    else:
        result = await wrapper.chat.completions.create(
            model="gpt-4o-mini",
            messages=model_messages["gpt-4o-mini"],
        )
    assert result
