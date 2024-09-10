"""
Tools for working directly with OpenAI's various models.
"""
import logging

from notdiamond import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.settings import NOTDIAMOND_API_KEY

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

        if "llm_configs" not in nd_kwargs:
            LOGGER.info(
                "No LLM configs provided. Not Diamond will route to all OpenAI models."
            )
            nd_kwargs["llm_configs"] = [
                str(p) for p in NDLLMProviders if p.provider == "openai"
            ]
        self._nd_client = NotDiamond(
            api_key=NOTDIAMOND_API_KEY, *args, **nd_kwargs
        )

        # Create a OpenAI client with a dummy model - will ignore this during routing
        oai_kwargs = {k: v for k, v in kwargs.items() if k not in nd_params}
        self._oai_client = OpenAIClient(*args, **oai_kwargs)

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

    def create(self, *args, **kwargs):
        """
        Perform chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
        print(f"create called with {args} and {kwargs}")
        if "model" in kwargs:
            LOGGER.warning(
                f"'model' argument {kwargs['model']} will override routing requests; ignoring"
            )
            kwargs.pop("model")

        if "messages" not in kwargs:
            raise ValueError("'messages' argument is required")

        session_id, best_llm = self._nd_client.model_select(*args, **kwargs)
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

        self._oai_client = AsyncOpenAIClient(*args, **kwargs)
        self._nd_client = NotDiamond(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._oai_client, name)

    def __call__(self, *args, **kwargs):
        return self._oai_client(*args, **kwargs)

    def __dir__(self):
        return dir(self._oai_client)

    def chat(self, *args, **kwargs):
        return self._oai_client.chat(*args, **kwargs)

    def completions(self, *args, **kwargs):
        """
        Wrapper around the OpenAI completions API. Patch in this class's create method
        to ensure routing.
        """
        completions = self._oai_client.chat.completions(*args, **kwargs)
        setattr(completions, "create", self.create)
        return completions

    async def create(self, *args, **kwargs):
        """
        Perform async chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
        if "model" in kwargs:
            LOGGER.warning(
                f"'model' argument {kwargs['model']} will override routing requests; ignoring"
            )
            kwargs.pop("model")

        if "messages" not in kwargs:
            raise ValueError("'messages' argument is required")

        session_id, best_llm = await self._nd_client.amodel_select(
            *args, **kwargs
        )
        LOGGER.info(f"Routed prompt to {best_llm} for session ID {session_id}")
        return await self._oai_client.chat.completions.create(
            *args, model=str(best_llm.model), **kwargs
        )
