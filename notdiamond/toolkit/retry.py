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


class _RetryWrapperException(Exception):
    failed_models: List[str]
    failed_exception: Exception

    def __init__(self, failed_models: List[str], failed_exception: Exception):
        self.failed_models = failed_models
        self.failed_exception = failed_exception
        super().__init__(
            f"Failed to invoke {failed_models}: {failed_exception}"
        )


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
            models = [m for m in models if self._model_client_match(m)]
            # models is a list - preserve order
            self._models = [_m.split("/")[-1] for _m in models]
        else:
            # models is a load-balanced dict - order models by desc. weight
            models = {
                m: w for m, w in models.items() if self._model_client_match(m)
            }
            self._models = sorted(
                [_m.split("/")[-1] for (_m, _w) in list(models.items())],
                key=lambda x: x[1],
                reverse=True,
            )

        self._max_retries = (
            {m.split("/")[-1]: t for m, t in max_retries.items()}
            if isinstance(max_retries, dict)
            else max_retries
        )
        self._timeout = (
            {m.split("/")[-1]: t for m, t in timeout.items()}
            if isinstance(timeout, dict)
            else timeout
        )
        self._model_messages = (
            {m.split("/")[-1]: msgs for m, msgs in model_messages.items()}
            if model_messages
            else {}
        )
        self._backoff = backoff

        self._api_key = api_key
        self._nd_client = None
        if self._api_key:
            self._nd_client = NotDiamond(api_key=self._api_key)

        # track manager to assist with model selection and load balancing when overriding
        # create or stream calls
        self.manager: Optional[RetryManager] = None

    def get_provider(self) -> str:
        """
        Platform child clients should always precede the model-building clients (eg. Azure before OpenAI)
        since the former usually inherit from the latter clients.
        """
        if isinstance(self._client, (AsyncAzureOpenAI, AzureOpenAI)):
            return "azure"
        elif isinstance(self._client, (AsyncOpenAI, OpenAI)):
            return "openai"
        elif isinstance(
            self._client, (AnthropicBedrock, AsyncAnthropicBedrock)
        ):
            return "bedrock"
        elif isinstance(self._client, (Anthropic, AsyncAnthropic)):
            return "anthropic"

    def _model_client_match(self, target_model: str) -> bool:
        if isinstance(self._client, (AsyncOpenAI, OpenAI)):
            return target_model.split("/")[0] == "openai"
        elif isinstance(self._client, (AsyncAzureOpenAI, AzureOpenAI)):
            return target_model.split("/")[0] == "azure"
        else:
            raise ValueError(
                f"No client match found for model {target_model}. Client type: {type(self._client)}"
            )

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

    def _get_fallback_model(
        self, current_model: str, failed_models: List[str] = None
    ) -> str:
        """
        After failing to invoke current_model, use the user's model fallback list to choose the
        next invocation model.
        """
        models = self._models
        if failed_models:
            models = [m for m in models if m not in failed_models]

        if not models:
            return
        return models[0]

    def _retry_decorator(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_model = kwargs["model"]
            kwargs = self._update_model_kwargs(kwargs, target_model)

            attempt = 0
            failed_models = []
            while attempt < self.get_max_retries(target_model):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        previous_model = target_model
                        failed_models.append(previous_model)
                        target_model = self._get_fallback_model(
                            previous_model, failed_models
                        )
                        if not target_model:
                            raise _RetryWrapperException(failed_models, e)
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
            failed_models = []
            while attempt < self.get_max_retries(target_model):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        previous_model = target_model
                        failed_models.append(previous_model)
                        target_model = self._get_fallback_model(
                            previous_model, failed_models
                        )
                        if not target_model:
                            raise _RetryWrapperException(failed_models, e)
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

    def _update_failed_models(
        self,
        target_model: str,
        failed_models: List[str],
        all_failed_models: List[str],
    ) -> List[str]:
        target_provider = target_model.split("/")[0]
        all_failed_models.extend(
            ["/".join([target_provider, m]) for m in failed_models]
        )

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
                target_model = kwargs["model"]
                target_model = f"{self.parent.get_provider()}/{target_model}"

                all_failed_models = []
                _create_fn = self.parent._default_create
                while True:
                    target_wrapper = self.manager.get_wrapper(target_model)
                    # when this loop changes a client, this will override create with the _original_
                    # client's method - needs to be new client
                    wrapped = target_wrapper._retry_decorator(_create_fn)
                    kwargs["model"] = target_model.split("/")[-1]

                    try:
                        return wrapped(*args, **kwargs)
                    except _RetryWrapperException as e:
                        LOGGER.exception(e)
                        self.parent._update_failed_models(
                            target_model, e.failed_models, all_failed_models
                        )
                        target_model = self.manager.get_next_model(
                            all_failed_models
                        )
                        if not target_model:
                            raise e.failed_exception
                        target_wrapper = self.manager.get_wrapper(target_model)

                        # update state on the target wrapper's ChatCompletions by accessing `chat` property
                        _ = target_wrapper._client.chat
                        _create_fn = target_wrapper._default_create

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
                target_model = kwargs["model"]
                target_model = f"{self.parent.get_provider()}/{target_model}"

                all_failed_models = []
                _create_fn = self.parent._default_create
                while True:
                    target_wrapper = self.manager.get_wrapper(target_model)
                    wrapped = target_wrapper._retry_decorator(_create_fn)
                    kwargs["model"] = target_model.split("/")[-1]

                    try:
                        return await wrapped(*args, **kwargs)
                    except _RetryWrapperException as e:
                        LOGGER.exception(e)
                        self.parent._update_failed_models(
                            target_model, e.failed_models, all_failed_models
                        )
                        target_model = self.manager.get_next_model(
                            all_failed_models
                        )
                        if not target_model:
                            raise e.failed_exception
                        target_wrapper = self.manager.get_wrapper(target_model)

                        # update state on the target wrapper's ChatCompletions by accessing `chat` property
                        _ = target_wrapper._client.chat
                        _create_fn = target_wrapper._default_create

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

    def get_next_model(self, failed_models: List[str]) -> str:
        if self._model_weights:
            new_weights = {
                m: w for m, w in self._models.items() if m not in failed_models
            }
            return new_weights[random.random()]
        remaining_models = [m for m in self._models if m not in failed_models]
        if not remaining_models:
            return
        return remaining_models[0]

    def _select_target_model(self, kwargs: Dict[str, Any]) -> str:
        target_model = kwargs.get("model", None)
        if target_model == "notdiamond":
            return self._model_weights[random.random()]
        return target_model

    def get_wrapper(self, model: str) -> Any:
        return self._model_to_wrapper[model]
