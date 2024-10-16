import os

import pytest

from notdiamond.settings import GOOGLE_API_KEY, PPLX_API_KEY, TOGETHER_API_KEY
from notdiamond.toolkit.litellm import acompletion, completion

os.environ["TOGETHERAI_API_KEY"] = TOGETHER_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
os.environ["PERPLEXITYAI_API_KEY"] = PPLX_API_KEY


# nd providers and models
# commenting various tests out to speed things up
ND_MODEL_LIST = [
    # {"provider": "openai", "model": "gpt-3.5-turbo"},
    # {"provider": "openai", "model": "gpt-3.5-turbo-0125"},
    # {"provider": "openai", "model": "gpt-4"},
    # {"provider": "openai", "model": "gpt-4-0613"},
    {"provider": "openai", "model": "gpt-4o"},
    # {"provider": "openai", "model": "gpt-4o-2024-05-13"},
    # {"provider": "openai", "model": "gpt-4-turbo"},
    # {"provider": "openai", "model": "gpt-4-turbo-2024-04-09"},
    # {"provider": "openai", "model": "gpt-4-turbo-preview"},
    # {"provider": "openai", "model": "gpt-4-0125-preview"},
    # {"provider": "openai", "model": "gpt-4-1106-preview"},
    {"provider": "openai", "model": "gpt-4o-mini"},
    # {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    # {"provider": "anthropic", "model": "claude-2.1"},
    # {"provider": "anthropic", "model": "claude-3-opus-20240229"},
    # {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620"},
    # {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    {"provider": "mistral", "model": "mistral-large-latest"},
    # {"provider": "mistral", "model": "mistral-medium-latest"},
    # {"provider": "mistral", "model": "mistral-small-latest"},
    # {"provider": "mistral", "model": "codestral-latest"},
    # {"provider": "mistral", "model": "open-mistral-7b"},
    # {"provider": "mistral", "model": "open-mixtral-8x7b"},
    # {"provider": "mistral", "model": "open-mixtral-8x22b"},
    # {"provider": "mistral", "model": "mistral-large-2407"},
    # {"provider": "mistral", "model": "mistral-large-2402"},
    {"provider": "perplexity", "model": "llama-3.1-sonar-large-128k-online"},
    # {"provider": "cohere", "model": "command-r"},
    {"provider": "cohere", "model": "command-r-plus"},
    # {"provider": "google", "model": "gemini-pro"},
    # {"provider": "google", "model": "gemini-1.5-pro-latest"},
    # {"provider": "google", "model": "gemini-1.5-flash-latest"},
    # {"provider": "google", "model": "gemini-1.0-pro-latest"},
    #
    # {"provider": "replicate", "model": "mistral-7b-instruct-v0.2"}, removed due to replicate side error
    # {"provider": "replicate", "model": "mixtral-8x7b-instruct-v0.1"}, removed due to replicate side error
    # {"provider": "replicate", "model": "meta-llama-3-70b-instruct"}, removed due to replicate side error
    # {"provider": "replicate", "model": "meta-llama-3.1-405b-instruct"}, removed due to replicate side error
    # {"provider": "replicate", "model": "meta-llama-3-8b-instruct"},
    #
    # {"provider": "togetherai", "model": "Mistral-7B-Instruct-v0.2"},
    # {"provider": "togetherai", "model": "Mixtral-8x7B-Instruct-v0.1"},
    # {"provider": "togetherai", "model": "Mixtral-8x22B-Instruct-v0.1"},
    # {"provider": "togetherai", "model": "Llama-3-70b-chat-hf"},
    # {"provider": "togetherai", "model": "Llama-3-8b-chat-hf"},
    # {"provider": "togetherai", "model": "Qwen2-72B-Instruct"},
    # {"provider": "togetherai", "model": "Meta-Llama-3.1-8B-Instruct-Turbo"},
    {"provider": "togetherai", "model": "Meta-Llama-3.1-70B-Instruct-Turbo"},
    # {"provider": "togetherai", "model": "Meta-Llama-3.1-405B-Instruct-Turbo"},
]

ND_TOOLS_MODEL_LIST = [
    # {"provider": "openai", "model": "gpt-3.5-turbo"},
    # {"provider": "openai", "model": "gpt-3.5-turbo-0125"},
    # {"provider": "openai", "model": "gpt-4"},
    # {"provider": "openai", "model": "gpt-4-0613"},
    {"provider": "openai", "model": "gpt-4o"},
    # {"provider": "openai", "model": "gpt-4o-2024-05-13"},
    # {"provider": "openai", "model": "gpt-4-turbo"},
    # {"provider": "openai", "model": "gpt-4-turbo-2024-04-09"},
    # {"provider": "openai", "model": "gpt-4-turbo-preview"},
    # {"provider": "openai", "model": "gpt-4-0125-preview"},
    # {"provider": "openai", "model": "gpt-4-1106-preview"},
    {"provider": "openai", "model": "gpt-4o-mini"},
    # {"provider": "openai", "model": "gpt-4o-mini-2024-07-18"},
    # {"provider": "anthropic", "model": "claude-3-opus-20240229"},
    # {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620"},
    # {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    {"provider": "mistral", "model": "mistral-large-latest"},
    # {"provider": "mistral", "model": "mistral-small-latest"},
    # {"provider": "cohere", "model": "command-r"},
    {"provider": "cohere", "model": "command-r-plus"},
    # {"provider": "google", "model": "gemini-pro"},
    # {"provider": "google", "model": "gemini-1.5-pro-latest"},
    # {"provider": "google", "model": "gemini-1.5-flash-latest"},
    # {"provider": "google", "model": "gemini-1.0-pro-latest"},
]


@pytest.mark.vcr
def test_completion_notdiamond():
    try:
        messages = [
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        for model in ND_MODEL_LIST:
            print(f"Testing {model}")
            _ = completion(
                model="notdiamond/notdiamond",
                messages=messages,
                llm_providers=[model],
            )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.vcr
def test_completion_notdiamond_stream():
    try:
        messages = [
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        for model in ND_MODEL_LIST:
            _ = completion(
                model="notdiamond/notdiamond",
                messages=messages,
                llm_providers=[model],
                stream=True,
            )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.vcr
def test_completion_notdiamond_tool_calling():
    try:
        messages = [
            {
                "role": "user",
                "content": "what is 2 + 5?",
            },
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Adds a and b.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ]
        for model in ND_TOOLS_MODEL_LIST:
            _ = completion(
                model="notdiamond/notdiamond",
                messages=messages,
                llm_providers=[model],
                tools=tools,
            )
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.vcr(allow_playback_repeats=True)
def test_async_completion_notdiamond():
    import asyncio

    async def test_get_response(model):
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            _ = await acompletion(
                model="notdiamond/notdiamond",
                messages=messages,
                llm_providers=[model],
                num_retries=3,
                timeout=10,
            )
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    async def run_concurrent_tests():
        _ = await asyncio.gather(
            *[test_get_response(model) for model in ND_MODEL_LIST]
        )

    asyncio.run(run_concurrent_tests())


@pytest.mark.vcr(allow_playback_repeats=True)
def test_async_completion_notdiamond_stream():
    import asyncio

    async def test_get_response(model):
        user_message = "Hello, how are you?"
        messages = [{"content": user_message, "role": "user"}]
        try:
            _ = await acompletion(
                model="notdiamond/notdiamond",
                messages=messages,
                llm_providers=[model],
                num_retries=3,
                timeout=10,
                stream=True,
            )
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")

    async def run_concurrent_tests():
        _ = await asyncio.gather(
            *[test_get_response(model) for model in ND_MODEL_LIST]
        )

    asyncio.run(run_concurrent_tests())
