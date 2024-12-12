import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AsyncAnthropic,
    AsyncAnthropicBedrock,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from notdiamond import NotDiamond
from notdiamond.llms.config import LLMConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class _CumulativeModelSelectionWeights:
    def __init__(self, models: Dict[str, float]):
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

    def __getitem__(self, value: float) -> LLMConfig:
        for bin_idx, weight_bin in enumerate(self._cumulative_weights):
            if (
                weight_bin[0] <= value < weight_bin[1]
                or value == weight_bin[1] == 1.0
            ):
                return self._model_ordering[bin_idx]
        raise KeyError(
            f"No model found for draw {value} (model weights: {list(zip(self._model_ordering, self._cumulative_weights))})"
        )


class _BaseRetryWrapper:
    """
    Wrapper class for OpenAI clients which adds retry and fallback logic.
    """

    def __init__(
        self,
        client: Any,
        models: Union[Dict[str, float], List[str]],
        max_retries: int | Dict[str, int] = 1,
        timeout: float | Dict[str, float] = 60.0,
        model_messages: Dict[str, List[Dict[str, Any]]] = {},
        api_key: str | None = None,
        backoff: float = 2.0,
    ):
        self._client = client

        # strip provider from model name
        if isinstance(models, list):
            # models is a list - preserve order
            self._models = [_m.split("/")[-1] for _m in models]
        else:
            # models is a load-balanced dict - order models by desc. weight
            self._models = sorted(
                [_m.split("/")[-1] for (_m, _w) in list(models.items())],
                key=lambda x: x[1],
                reverse=True,
            )

        self._max_retries = max_retries
        self._timeout = timeout
        self._model_messages = model_messages
        self._backoff = backoff

        self._api_key = api_key
        self._nd_client = None
        if self._api_key:
            self._nd_client = NotDiamond(api_key=self._api_key)

        # track manager to assist with model selection and load balancing when overriding
        # create or stream calls
        self.manager: Optional[RetryManager] = None

    def get_timeout(self, target_model: Optional[str] = None) -> float:
        out = self._timeout
        if isinstance(self._timeout, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if timeout is a dict"
                )
            out = self._timeout.get(target_model)
        return out

    def get_max_retries(self, target_model: Optional[str] = None) -> int:
        out = self._max_retries
        if isinstance(self._max_retries, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if max_retries is a dict"
                )
            out = self._max_retries.get(target_model)
        return out

    def _update_model_kwargs(
        self, kwargs: Dict[str, Any], target_model: str
    ) -> Dict[str, Any]:
        kwargs["model"] = target_model
        kwargs["timeout"] = self.get_timeout(target_model)
        kwargs["messages"] = self._model_messages[target_model]
        return kwargs

    def _get_fallback_model(self, current_model: str) -> str:
        """
        After failing to invoke current_model, use the user's model fallback list to choose the
        next invocation model.
        """
        model_idx = self._models.index(current_model)
        if model_idx == len(self._models) - 1:
            return
        return self._models[model_idx + 1]

    def _retry_decorator(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_model = kwargs["model"]
            kwargs = self._update_model_kwargs(kwargs, target_model)

            attempt = 0
            while attempt < self.get_max_retries(target_model):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        # get a fallback or raise exception if none found
                        target_model = self._get_fallback_model(target_model)
                        if not target_model:
                            raise e
                        kwargs = self._update_model_kwargs(
                            kwargs, target_model
                        )
                        LOGGER.info(
                            f"Attempting fallback model {kwargs['model']}"
                        )
                        attempt = 0
                        continue

                attempt += 1
                await asyncio.sleep(self._backoff**attempt)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            target_model = kwargs["model"]
            kwargs = self._update_model_kwargs(kwargs, target_model)

            attempt = 0
            while attempt < self.get_max_retries(target_model):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        target_model = self._get_fallback_model(target_model)
                        if not target_model:
                            raise e
                        kwargs = self._update_model_kwargs(
                            kwargs, target_model
                        )
                        LOGGER.info(
                            f"Attempting fallback model {kwargs['model']}"
                        )
                        attempt = 0
                        continue

                attempt += 1
                time.sleep(self._backoff**attempt)

        # Return appropriate wrapper based on the original function
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    def __getattr__(self, name):
        return getattr(self._client, name)

    def _log_to_nd(self, *args, **kwargs):
        if not self._api_key:
            LOGGER.warning(
                "No API key provided, skipping logging to Not Diamond."
            )

        LOGGER.info("Logging inference metadata to Not Diamond.")

        # todo [a9] implement logging endpoint


class RetryWrapper(_BaseRetryWrapper):
    @property
    def chat(self):
        if not isinstance(self._client, OpenAI):
            return getattr(self._client, "chat")

        self._default_create = self._client.chat.completions.create

        class ChatCompletions:
            def __init__(self, parent: RetryWrapper, manager: RetryManager):
                self.parent = parent
                self.manager = manager

            @property
            def completions(self):
                return self

            def create(self, *args, **kwargs):
                target_model = self.manager.sample_or_get_model(
                    random.random()
                )
                target_wrapper = self.manager.get_wrapper(target_model)
                wrapped = target_wrapper._retry_decorator(
                    self.parent._default_create
                )
                return wrapped(*args, **kwargs)

        return ChatCompletions(self, self.manager)


class AsyncRetryWrapper(_BaseRetryWrapper):
    @property
    def chat(self):
        if not isinstance(self._client, AsyncOpenAI):
            return getattr(self._client, "chat")

        self._default_create = self._client.chat.completions.create

        class AsyncCompletions:
            def __init__(
                self, parent: AsyncRetryWrapper, manager: RetryManager
            ):
                self.parent = parent
                self.manager = manager

            @property
            def completions(self):
                return self

            async def create(self, *args, **kwargs):
                target_model = self.manager.sample_or_get_model(
                    random.random()
                )
                target_wrapper = self.manager.get_wrapper(target_model)
                wrapped = target_wrapper._retry_decorator(
                    self.parent._default_create
                )
                result = await wrapped(*args, **kwargs)
                return result

        return AsyncCompletions(self, self.manager)


class RetryManager:
    """
    RetryManager handles model selection. If the user provides a load balancing dict,
    it will use that to select a model for each invocation. Otherwise, it will use the
    first model in the list.
    """

    def __init__(
        self,
        models: Union[Dict[str, float], List[str]],
        wrapped_clients: List[RetryWrapper],
    ):
        self._wrappers = wrapped_clients
        self._models = models

        self._model_weights = None
        if isinstance(self._models, dict):
            self._model_weights = _CumulativeModelSelectionWeights(
                self._models
            )

        for wrapper in self._wrappers:
            wrapper.manager = self
            wrapper._client.chat = wrapper.chat

        self._model_to_wrapper: Dict[str, RetryWrapper] = {}
        for model in self._models:
            self._model_to_wrapper[model] = self._get_model_wrapper(model)

    def _get_model_wrapper(self, model: str) -> RetryWrapper:
        target_wrapper = None
        try:
            if "bedrock" in model:
                instance_check = (AnthropicBedrock, AsyncAnthropicBedrock)
            elif "azure" in model:
                instance_check = (AzureOpenAI, AsyncAzureOpenAI)
            elif "anthropic" in model:
                instance_check = (Anthropic, AsyncAnthropic)
            elif "openai" in model:
                instance_check = (OpenAI, AsyncOpenAI)
            else:
                raise ValueError(
                    f"No wrapper found for model {model}. It may not currently be supported."
                )

            target_wrapper = [
                wrapper
                for wrapper in self._wrappers
                if isinstance(wrapper._client, instance_check)
            ][0]
        except IndexError:
            raise ValueError(
                f"No wrapped client found for model {model} among {[w._client for w in self._wrappers]}."
            )

        return target_wrapper

    def sample_or_get_model(self, prob: float) -> str:
        if self._model_weights:
            return self._model_weights[prob]
        return self._models[0]

    def _select_target_model(self, kwargs: Dict[str, Any]) -> str:
        target_model = kwargs.get("model", None)
        if target_model == "notdiamond":
            return self._model_weights[random.random()]
        return target_model

    def get_wrapper(self, model: str) -> Any:
        return self._model_to_wrapper[model]
