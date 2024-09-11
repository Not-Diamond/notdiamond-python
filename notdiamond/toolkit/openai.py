"""
Tools for working directly with OpenAI's various models.
"""
import logging
from typing import List, Union

from notdiamond import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.settings import NOTDIAMOND_API_KEY, OPENAI_API_KEY

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class OpenAI:
    """
    Encapsulating class for an openai.OpenAI client. This supports the same methods as
    the openai package, while also supporting routed prompts with calls to `completion`.
    """

    def __init__(self, *args, **kwargs):
        from openai import OpenAI as OpenAIClient

        nd_params = [
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
            "user_agent",
        ]

        nd_kwargs = {k: v for k, v in kwargs.items() if k in nd_params}

        # TODO [a9] remove llm_configs as valid constructor arg for ND client
        self._nd_client = NotDiamond(
            api_key=NOTDIAMOND_API_KEY,
            llm_configs=["openai/gpt-3.5-turbo"],
            *args,
            **nd_kwargs,
        )

        # Create a OpenAI client with a dummy model - will ignore this during routing
        oai_kwargs = {k: v for k, v in kwargs.items() if k not in nd_params}
        self._oai_client = OpenAIClient(
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

    def create(self, *args, model: Union[str, List] = None, **kwargs):
        """
        Perform chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
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

        session_id, best_llm = self._nd_client.model_select(
            *args, model=llm_configs, **kwargs
        )
        LOGGER.info(f"Routed prompt to {best_llm} for session ID {session_id}")
        return self._oai_client.chat.completions.create(
            *args, model=str(best_llm.model), **kwargs
        )


class AsyncOpenAI:
    """
    Encapsulating class for an openai.OpenAI client. This supports the same methods as
    the openai package, while also supporting routed prompts with calls to `completion`.
    """

    def __init__(self, *args, **kwargs):
        from openai import AsyncOpenAI as AsyncOpenAIClient

        nd_params = [
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
            "user_agent",
        ]

        nd_kwargs = {k: v for k, v in kwargs.items() if k in nd_params}

        # TODO [a9] remove llm_configs as valid constructor arg for ND client
        self._nd_client = NotDiamond(
            api_key=NOTDIAMOND_API_KEY,
            llm_configs=["openai/gpt-3.5-turbo"],
            *args,
            **nd_kwargs,
        )

        # Create a OpenAI client with a dummy model - will ignore this during routing
        oai_kwargs = {k: v for k, v in kwargs.items() if k not in nd_params}
        self._oai_client = AsyncOpenAIClient(
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

    async def create(self, *args, model: Union[str, List] = None, **kwargs):
        """
        Perform async chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
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

        session_id, best_llm = await self._nd_client.amodel_select(
            *args, model=llm_configs, **kwargs
        )
        LOGGER.debug(
            f"Routed prompt to {best_llm} for session ID {session_id}"
        )
        return await self._oai_client.chat.completions.create(
            *args, model=str(best_llm.model), **kwargs
        )
