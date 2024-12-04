"""
Tools for working directly with OpenAI's various models.
"""
import asyncio
import logging
import random
import time
from functools import wraps
from typing import Dict, List, Union

from notdiamond import NotDiamond
from notdiamond.llms.config import LLMConfig
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.settings import NOTDIAMOND_API_KEY, OPENAI_API_KEY

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


class _CumulativeModelSelectionWeights:
    def __init__(self, models: Dict[str | LLMConfig, float]):
        sorted_models = sorted(models.items(), key=lambda x: x[1])
        norm_sum = sum(weight for _, weight in sorted_models)
        self._cumulative_weights = []
        self._model_ordering = []
        previous_weight = 0.0
        for model, weight in sorted_models:
            weight_bin = (previous_weight, previous_weight + weight / norm_sum)
            self._cumulative_weights.append(weight_bin)
            self._model_ordering.append(model)
            previous_weight = weight_bin[1]

    def __getitem__(self, weight: float) -> LLMConfig:
        for bin_idx, weight_bin in enumerate(self._cumulative_weights):
            if (
                weight_bin[0] <= weight < weight_bin[1]
                or weight == weight_bin[1] == 1.0
            ):
                return self._model_ordering[bin_idx]
        raise KeyError(
            f"No model found for weight {weight} (model weights: {list(zip(self._model_ordering, self._cumulative_weights))})"
        )


class OpenAIRetryWrapper(_OpenAIBase):
    """
    Wrapper class for OpenAI clients which adds retry and fallback logic.
    """

    def __init__(
        self,
        openai_client: OpenAI,
        models: Union[Dict[str | LLMConfig, float], List[str | LLMConfig]],
        max_retries: int,
        timeout: float,
        fallback: List[str | LLMConfig],
        backoff: float = 2.0,
    ):
        self._openai_client = openai_client
        self._models = models
        self._max_retries = max_retries
        self._timeout = timeout
        self._backoff = backoff
        self._fallback = fallback or []

        if isinstance(self._models, dict):
            self._model_weights = _CumulativeModelSelectionWeights(
                self._models
            )
        else:
            self._model_weights = _CumulativeModelSelectionWeights(
                {m: 1.0 for m in self._models}
            )

    def _retry_decorator(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_delay = self._timeout
            last_exception = None
            target_model = kwargs.get("model", None)
            if target_model == "notdiamond":
                target_model = self._model_weights[random.random()]

            for attempt in range(self._max_retries):
                try:
                    return await func(*args, **kwargs)
                except self._retry_exceptions as e:
                    last_exception = e
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self._max_retries - 1:  # Last attempt
                        if self._fallback:
                            LOGGER.info("Attempting fallback models")
                            kwargs["model"] = self._fallback
                            try:
                                return await func(*args, **kwargs)
                            except Exception as fallback_e:
                                LOGGER.error(
                                    f"Fallback attempt failed: {str(fallback_e)}"
                                )
                                raise last_exception
                        raise

                    time.sleep(current_delay)
                    current_delay *= self._backoff

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_delay = self._timeout
            last_exception = None
            target_model = kwargs.get("model", None)
            if target_model == "notdiamond":
                target_model = self._model_weights[random.random()]

            for attempt in range(self._max_retries):
                try:
                    return func(*args, **kwargs)
                except self._retry_exceptions as e:
                    last_exception = e
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self._max_retries - 1:  # Last attempt
                        if self._fallback:
                            LOGGER.info("Attempting fallback models")
                            kwargs["model"] = self._fallback
                            try:
                                return func(*args, **kwargs)
                            except Exception as fallback_e:
                                LOGGER.error(
                                    f"Fallback attempt failed: {str(fallback_e)}"
                                )
                                raise last_exception
                        raise

                    time.sleep(current_delay)
                    current_delay *= self._backoff

        # Return appropriate wrapper based on the original function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def __getattr__(self, name):
        attr = getattr(self._openai_client, name)
        if callable(attr):
            return self._retry_decorator(attr)
        return attr
