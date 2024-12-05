from __future__ import annotations

from typing import Dict, List, Union

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit.openai import OpenAIRetryWrapper


def init(
    client: OpenAI | AsyncOpenAI | Anthropic | AsyncAnthropic,
    models: Union[Dict[str | LLMConfig, float], List[str | LLMConfig]],
    max_retries: int | Dict[str | LLMConfig, int],
    timeout: float | Dict[str | LLMConfig, float],
    model_messages: Dict[str | LLMConfig, List[Dict[str, str]]],
    api_key: str | None = None,
    fallback: List[str | LLMConfig] = [],
):
    """
    Usage:

    ```python
        openai_client = OpenAI(...)
        notdiamond.init(
            openai_client,
            models={"gpt-3.5-turbo": 0.9, "claude-3-5-sonnet-20240620": 0.1},
            max_retries=3,
            timeout=10.0,
            api_key="sk-...",
            fallback=["gpt-3.5-turbo", "claude-3-5-sonnet-20240620", "gpt-4o"],
        )
        response = openai_client.chat.completions.create(
            model="notdiamond",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    ```
    """
    # openai_client = gc.get_objects(type_filter=OpenAI)
    # if not openai_client:
    #     openai_client = gc.get_objects(type_filter=AsyncOpenAI)

    # if not openai_client:
    #     raise ValueError(
    #         "No OpenAI or AsyncOpenAI client found. Is this correct?"
    #     )

    client_wrapper = OpenAIRetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        fallback=fallback,
        model_messages=model_messages,
    )
    return client_wrapper
