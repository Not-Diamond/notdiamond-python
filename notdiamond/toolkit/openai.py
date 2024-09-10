"""
Tools for working directly with OpenAI's various models.
"""
import logging

from notdiamond import NotDiamond

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class OpenAI:
    """
    Encapsulating class for an openai.OpenAI client. This supports the same methods as
    the openai package, while also supporting routed prompts with calls to `completion`.
    """

    def __init__(self, *args, **kwargs):
        from openai import OpenAI as OpenAIClient

        self._oai_client = OpenAIClient(*args, **kwargs)
        self._nd_client = NotDiamond(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._oai_client, name)

    def __call__(self, *args, **kwargs):
        return self._oai_client(*args, **kwargs)

    def __dir__(self):
        return dir(self._oai_client)

    def create(self, *args, **kwargs):
        """
        Perform chat completion using OpenAI's API, after routing the prompt to a
        specific LLM via Not Diamond.
        """
        if "model" in kwargs:
            LOGGER.warning(
                f"'model' argument {kwargs['model']} will override routing requests; ignoring"
            )
            kwargs.pop("model")

        session_id, best_llm = self._nd_client.model_select(*args, **kwargs)
        LOGGER.info(f"Routed prompt to {best_llm} for session ID {session_id}")
        return self._oai_client.chat.completions.create(
            *args, model=best_llm, **kwargs
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

        session_id, best_llm = self._nd_client.amodel_select(*args, **kwargs)
        LOGGER.info(f"Routed prompt to {best_llm} for session ID {session_id}")
        return self._oai_client.chat.completions.create(
            *args, model=best_llm, **kwargs
        )
