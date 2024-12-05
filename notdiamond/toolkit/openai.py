"""
Tools for working directly with OpenAI's various models.
"""
import logging
from typing import List, Union

from notdiamond import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.settings import NOTDIAMOND_API_KEY, OPENAI_API_KEY
from notdiamond.toolkit.retry import BaseRetryWrapper

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

_ND_PARAMS = [
    "llm_configs",
    "default",
    "max_model_depth",
    "latency_tracking",
    "hash_content",
    "tradeoff",
    "preference_id",
    "tools",
    "callbacks",
    "nd_api_url",
    "nd_api_key",
    "user_agent",
]
_SHARED_PARAMS = ["timeout", "max_retries"]


class _OpenAIBase:
    """
    Base class which wraps both an openai client and Not Diamond retry / fallback logic.
    """

    def __init__(self, oai_client_cls, *args, **kwargs):
        nd_kwargs = {
            k: v for k, v in kwargs.items() if k in _ND_PARAMS + _SHARED_PARAMS
        }

        # TODO [a9] remove llm_configs as valid constructor arg for ND client
        self._nd_client = NotDiamond(
            api_key=nd_kwargs.get("nd_api_key", NOTDIAMOND_API_KEY),
            llm_configs=["openai/gpt-3.5-turbo"],
            *args,
            **nd_kwargs,
        )

        # Create a OpenAI client with a dummy model - will ignore this during routing
        oai_kwargs = {k: v for k, v in kwargs.items() if k not in _ND_PARAMS}
        self._oai_client = oai_client_cls(
            *args, api_key=OPENAI_API_KEY, **oai_kwargs
        )

    def __getattr__(self, name):
        return getattr(self._oai_client, name)

    def __call__(self, *args, **kwargs):
        return self._oai_client(*args, **kwargs)

    def __dir__(self):
        return dir(self._oai_client)

    @property
    def chat(self):
        class ChatCompletions:
            def __init__(self, parent):
                self.parent = parent

            @property
            def completions(self):
                return self

            def create(self, *args, **kwargs):
                return self.parent.create(*args, **kwargs)

        return ChatCompletions(self)

    def _create_prep(self, model: Union[str, List], **kwargs):
        model = kwargs.get("model", model)

        if model is None:
            LOGGER.info(
                "No LLM configs provided. Not Diamond will route to all OpenAI models."
            )
            llm_configs = [
                str(p) for p in NDLLMProviders if p.provider == "openai"
            ]
        elif isinstance(model, str):
            llm_configs = model.split(",")
        elif isinstance(model, list):
            llm_configs = self._nd_client._parse_llm_configs_data(model)

        if "messages" not in kwargs:
            raise ValueError("'messages' argument is required")

        return llm_configs


class OpenAI(_OpenAIBase):
    """
    Encapsulating class for an openai.OpenAI client. This supports the same methods as
    the openai package, while also supporting routed prompts with calls to `completion`.
    """

    def __init__(self, *args, **kwargs):
        from openai import OpenAI as OpenAIClient

        super().__init__(OpenAIClient, *args, **kwargs)

    def create(self, *args, model: Union[str, List] = None, **kwargs):
        """
        Perform chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
        llm_configs = self._create_prep(model, **kwargs)
        session_id, best_llm = self._nd_client.model_select(
            *args, model=llm_configs, **kwargs
        )
        response = self._oai_client.chat.completions.create(
            *args, model=str(best_llm.model), **kwargs
        )
        LOGGER.info(f"Routed prompt to {best_llm} for session ID {session_id}")
        return response


class AsyncOpenAI(_OpenAIBase):
    """
    Encapsulating class for an openai.OpenAI client. This supports the same methods as
    the openai package, while also supporting routed prompts with calls to `completion`.
    """

    def __init__(self, *args, **kwargs):
        from openai import AsyncOpenAI as OpenAIClient

        super().__init__(OpenAIClient, *args, **kwargs)

    async def create(self, *args, model: Union[str, List] = None, **kwargs):
        """
        Perform async chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
        llm_configs = self._create_prep(model, **kwargs)
        session_id, best_llm = await self._nd_client.amodel_select(
            *args, model=llm_configs, **kwargs
        )
        response = await self._oai_client.chat.completions.create(
            *args, model=str(best_llm.model), **kwargs
        )
        LOGGER.debug(
            f"Routed prompt to {best_llm} for session ID {session_id}"
        )
        return response


class OpenAIRetryWrapper(BaseRetryWrapper, _OpenAIBase):
    @property
    def chat(self):
        class ChatCompletions:
            def __init__(self, parent):
                self.parent = parent

            @property
            def completions(self):
                return self

            def create(self, *args, **kwargs):
                return self._retry_decorator(
                    self.parent.create(*args, **kwargs)
                )

        return ChatCompletions(self)


class AsyncOpenAIRetryWrapper(BaseRetryWrapper, _OpenAIBase):
    @property
    def chat(self):
        class ChatCompletions:
            def __init__(self, parent):
                self.parent = parent

            @property
            def completions(self):
                return self

            async def create(self, *args, **kwargs):
                return await self._retry_decorator(
                    self.parent.create(*args, **kwargs)
                )

        return ChatCompletions(self)
