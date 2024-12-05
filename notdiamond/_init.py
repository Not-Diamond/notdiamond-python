from __future__ import annotations

from typing import Any, Dict, List, Union

from openai import AsyncOpenAI, OpenAI

from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit.openai import OpenAIRetryWrapper


def init(
    # todo [a9]: update annotation for correct client types
    client: OpenAI | AsyncOpenAI | Any,
    models: Union[Dict[str | LLMConfig, float], List[str | LLMConfig]],
    # todo [a9]: accept dict of model -> max_retries
    max_retries: int,
    # todo [a9]: accept dict of model -> timeout
    timeout: float,
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
        client, models, max_retries, timeout, fallback
    )
    return client_wrapper
