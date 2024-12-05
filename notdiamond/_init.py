from __future__ import annotations

import os
from typing import Dict, List, Union, overload

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit.retry import RetryWrapper

ClientType = Union[OpenAI, AsyncOpenAI, Anthropic, AsyncAnthropic]
LLMType = Union[str, LLMConfig]


@overload
def init(
    client: List[ClientType],
    models: Union[Dict[LLMType, float], List[LLMType]],
    max_retries: int | Dict[LLMType, int],
    timeout: float | Dict[LLMType, float],
    model_messages: Dict[LLMType, List[Dict[str, str]]],
    api_key: str | None = None,
    fallback: List[LLMType] = [],
) -> RetryWrapper:
    ...


def init(
    client: ClientType,
    models: Union[Dict[LLMType, float], List[LLMType]],
    max_retries: int | Dict[LLMType, int],
    timeout: float | Dict[LLMType, float],
    model_messages: Dict[LLMType, List[Dict[str, str]]],
    api_key: str | None = None,
    fallback: List[LLMType] = [],
) -> RetryWrapper:
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
    api_key = api_key or os.getenv("NOTDIAMOND_API_KEY")

    client_wrapper = RetryWrapper(
        client=client,
        models=models,
        max_retries=max_retries,
        timeout=timeout,
        fallback=fallback,
        model_messages=model_messages,
    )
    return client_wrapper
