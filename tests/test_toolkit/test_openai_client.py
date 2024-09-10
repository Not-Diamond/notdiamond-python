"""
Tests for our OpenAI client wrapper. Most of these ensure that the client supports
capabilities defined in the API reference:

https://platform.openai.com/docs/api-reference/introduction
"""

import pytest

from notdiamond import LLMConfig
from notdiamond.toolkit.openai import AsyncOpenAI, OpenAI


def test_openai_init():
    client = OpenAI()
    assert client is not None

    client2 = OpenAI(
        base_url="https://api.openai.com/v1",
    )
    assert client2 is not None

    client3 = OpenAI(
        base_url="https://api.openai.com/v1",
        organization="nd-oai-organization",
        project="nd-oai-project",
    )
    assert client3 is not None


def test_openai_create():
    nd_llm_configs = [
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-mini-2024-07-18",
    ]
    client = OpenAI()

    for model_kwarg in [
        nd_llm_configs,
        ",".join(nd_llm_configs),
        [LLMConfig.from_string(m) for m in nd_llm_configs],
    ]:
        response = client.create(
            model=model_kwarg,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, world! Does this route to gpt-3.5-turbo?",
                }
            ],
        )
        assert response is not None, f"Failed with model_kwarg={model_kwarg}"
        assert (
            len(response.choices[0].message.content) > 0
        ), f"Failed with model_kwarg={model_kwarg}"
        assert response.model in [
            provider.split("/")[-1] for provider in nd_llm_configs
        ], f"Failed with model_kwarg={model_kwarg}"


def test_openai_create_default_models():
    all_oai_model_response = OpenAI().create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Do you handle default models?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


def test_openai_chat_completions_create():
    all_oai_model_response = OpenAI().chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this chat.completions.create call work?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


def test_openai_chat_completions_create_stream():
    model_response_stream = OpenAI().chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this chat.completions.create call work?",
            }
        ],
        stream=True,
    )

    any_nonempty_chunk = False
    for chunk in model_response_stream:
        if chunk.choices[0].delta.content is not None:
            any_nonempty_chunk = True
            break
    assert any_nonempty_chunk


@pytest.mark.asyncio
async def test_async_openai_init():
    client = AsyncOpenAI()
    assert client is not None

    client2 = AsyncOpenAI(
        base_url="https://api.openai.com/v1",
    )
    assert client2 is not None

    client3 = AsyncOpenAI(
        base_url="https://api.openai.com/v1",
        organization="nd-oai-organization",
        project="nd-oai-project",
    )
    assert client3 is not None


@pytest.mark.asyncio
async def test_async_openai_create():
    nd_llm_configs = [
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-mini-2024-07-18",
    ]
    client = AsyncOpenAI()

    for model_kwarg in [
        nd_llm_configs,
        ",".join(nd_llm_configs),
        [LLMConfig.from_string(m) for m in nd_llm_configs],
    ]:
        response = await client.create(
            model=model_kwarg,
            messages=[
                {
                    "role": "user",
                    "content": "Hello, world! Does this route to gpt-3.5-turbo?",
                }
            ],
        )
        assert response is not None, f"Failed with model_kwarg={model_kwarg}"
        assert (
            len(response.choices[0].message.content) > 0
        ), f"Failed with model_kwarg={model_kwarg}"
        assert response.model in [
            provider.split("/")[-1] for provider in nd_llm_configs
        ], f"Failed with model_kwarg={model_kwarg}"


@pytest.mark.asyncio
async def test_async_openai_create_default_models():
    async_client = AsyncOpenAI()
    all_oai_model_response = await async_client.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Do you handle default models asynchronously?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_async_openai_chat_completions_create():
    async_client = AsyncOpenAI()
    all_oai_model_response = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this async chat.completions.create call work?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


@pytest.mark.asyncio
async def test_async_openai_chat_completions_create_stream():
    async_client = AsyncOpenAI()
    model_response_stream = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this async chat.completions.create call work with streaming?",
            }
        ],
        model=["openai/gpt-4o-mini", "openai/gpt-4o"],
        stream=True,
    )

    any_nonempty_chunk = False
    async for chunk in model_response_stream:
        if chunk.choices[0].delta.content is not None:
            any_nonempty_chunk = True
            break
    assert any_nonempty_chunk
