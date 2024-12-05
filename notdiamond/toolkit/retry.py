import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI

from notdiamond.llms.config import LLMConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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


class BaseRetryWrapper:
    """
    Wrapper class for OpenAI clients which adds retry and fallback logic.
    """

    def __init__(
        self,
        client: Any,
        models: Union[Dict[str | LLMConfig, float], List[str | LLMConfig]],
        max_retries: int | Dict[str | LLMConfig, int],
        timeout: float | Dict[str | LLMConfig, float],
        fallback: List[str | LLMConfig],
        backoff: float = 2.0,
    ):
        self._client = client
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

    def _get_target_model(self, kwargs: Dict[str, Any]) -> str | LLMConfig:
        target_model = kwargs.get("model", None)
        if target_model == "notdiamond":
            return self._model_weights[random.random()]
        return target_model

    @property
    def timeout(self, target_model: Optional[str | LLMConfig] = None) -> float:
        if isinstance(self._timeout, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if timeout is a dict"
                )
            return self._timeout[target_model]
        return self._timeout

    @property
    def max_retries(
        self, target_model: Optional[str | LLMConfig] = None
    ) -> int:
        if isinstance(self._max_retries, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if max_retries is a dict"
                )
            return self._max_retries[target_model]
        return self._max_retries

    def _retry_decorator(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_model = self._get_target_model(kwargs)
            kwargs["model"] = target_model
            kwargs["timeout"] = self.timeout(target_model)

            last_exception = None
            attempt = 0
            while attempt < self.max_retries(target_model):
                try:
                    return await func(*args, **kwargs)
                except self._retry_exceptions as e:
                    last_exception = e
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.max_retries(target_model) - 1:
                        if self._fallback:
                            kwargs["model"] = self._fallback.pop(0)
                            kwargs["timeout"] = self.timeout(kwargs["model"])
                            LOGGER.info(
                                f"Attempting fallback model {kwargs['model']}"
                            )
                            attempt = 0
                            continue

                        raise last_exception

                    attempt += 1
                    await asyncio.sleep(self._backoff**attempt)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            target_model = self._get_target_model(kwargs)
            kwargs["model"] = target_model
            kwargs["timeout"] = self.timeout(target_model)

            last_exception = None
            attempt = 0
            while attempt < self.max_retries(target_model):
                try:
                    return func(*args, **kwargs)
                except self._retry_exceptions as e:
                    last_exception = e
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.max_retries(target_model) - 1:
                        if self._fallback:
                            kwargs["model"] = self._fallback.pop(0)
                            kwargs["timeout"] = self.timeout(kwargs["model"])
                            LOGGER.info(
                                f"Attempting fallback model {kwargs['model']}"
                            )
                            attempt = 0
                            continue

                        raise last_exception

                    attempt += 1
                    time.sleep(self._backoff**attempt)

        # Return appropriate wrapper based on the original function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def __getattr__(self, name):
        return getattr(self._client, name)


class RetryWrapper(BaseRetryWrapper):
    @property
    def chat(self):
        if not isinstance(self._client, OpenAI):
            return getattr(self._client, "chat")

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

    @property
    def messages(self):
        if not isinstance(self._client, Anthropic):
            return getattr(self._client, "messages")

        class Messages:
            def __init__(self, parent):
                self.parent = parent

            def create(self, *args, **kwargs):
                return self._retry_decorator(
                    self.parent.create(*args, **kwargs)
                )

        return Messages(self)


class AsyncRetryWrapper(BaseRetryWrapper):
    @property
    def chat(self):
        if not isinstance(self._client, AsyncOpenAI):
            return getattr(self._client, "chat")

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

    @property
    def messages(self):
        if not isinstance(self._client, AsyncAnthropic):
            return getattr(self._client, "messages")

        class Messages:
            def __init__(self, parent):
                self.parent = parent

            async def create(self, *args, **kwargs):
                return await self._retry_decorator(
                    self.parent.create(*args, **kwargs)
                )

        return Messages(self)
