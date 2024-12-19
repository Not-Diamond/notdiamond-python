"""
Private module for retry and fallback logic. Added to notdiamond.toolkit to avoid
introducing library dependencies on OpenAI into the core SDK.
"""
import asyncio
import logging
import random
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

from notdiamond import NotDiamond
from notdiamond.llms.config import LLMConfig

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

ClientType = Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]
ModelType = Union[Dict[str, float], List[str]]
OpenAIMessagesType = List[Dict[str, str]]


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
    """
    Helper class for load balancing between models. Overloads `__getitem__` to
    return a model based on a random float.
    """

    def __init__(self, models: Dict[str, float]):
        """
        Args:
            models: Dict[str, float]
                A dictionary mapping model names to their selection weights. The weights will be normalized
                to sum to 1.0 and used for weighted random selection of models.
        """
        sorted_models = sorted(models.items(), key=lambda x: x[1])
        norm_sum = sum(weight for _, weight in sorted_models)
        if norm_sum == 0.0:
            raise ValueError(
                "All model weights are zero. Do not need to load balance."
            )
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
    Wrapper class for an individual OpenAI client which adds retry and fallback logic.
    """

    def __init__(
        self,
        client: ClientType,
        models: ModelType,
        max_retries: Union[int, Dict[str, int]] = 1,
        timeout: Union[float, Dict[str, float]] = 60.0,
        model_messages: OpenAIMessagesType = {},
        api_key: Union[str, None] = None,
        backoff: Union[float, Dict[str, float]] = 2.0,
    ):
        """
        Args:
            client: OpenAI, AsyncOpenAI, AzureOpenAI, or AsyncAzureOpenAI
                The client to wrap.
            models: Dict[str, float] | List[str]
                The models to invoke.
            max_retries: int | Dict[str, int]
                The maximum number of retries. Configured globally or per-model.
            timeout: float | Dict[str, float]
                The timeout for each model. Configured globally or per-model.
            model_messages: OpenAIMessagesType
                The messages to send to each model. Prepended to any messages passed to the `create` method.
            api_key: str | None
                Not Diamond API key to use for logging. Currently unused.
            backoff: float | Dict[str, float]
                The backoff factor for the retry logic. Configured globally or per-model.
        """
        self._client = client

        # track the default create method - we'll need to reference it when overriding create
        # also allows us to spy during tests later
        self._default_create = self._client.chat.completions.create

        # strip provider from model name
        if isinstance(models, list):
            models = [m for m in models if self._model_client_match(m)]
            # models is a list - preserve order
            self._models = [_m.split("/")[-1] for _m in models]
        else:
            # models is a load-balanced dict - order models by desc. weight
            self._models = {
                m.split("/")[-1]: w
                for m, w in models.items()
                if self._model_client_match(m)
            }

        self._max_retries = (
            {
                m.split("/")[-1]: t
                for m, t in max_retries.items()
                if self._model_client_match(m)
            }
            if isinstance(max_retries, dict)
            else max_retries
        )
        self._timeout = (
            {
                m.split("/")[-1]: t
                for m, t in timeout.items()
                if self._model_client_match(m)
            }
            if isinstance(timeout, dict)
            else timeout
        )
        self._model_messages = (
            {
                m.split("/")[-1]: msgs
                for m, msgs in model_messages.items()
                if self._model_client_match(m)
            }
            if model_messages
            else {}
        )
        self._backoff = (
            {
                m.split("/")[-1]: b
                for m, b in backoff.items()
                if self._model_client_match(m)
            }
            if isinstance(backoff, dict)
            else backoff
        )

        self._api_key = api_key
        self._nd_client = None
        if self._api_key:
            self._nd_client = NotDiamond(api_key=self._api_key)

        # track RetryManager to assist with model selection and load balancing when overriding
        # create or stream calls
        self.manager: Optional[RetryManager] = None

    def get_provider(self) -> str:
        """
        Get the provider name for this client - "azure" or "openai".
        """
        # platform child clients should precede the model-building clients (eg. Azure before OpenAI)
        # since the former usually inherit from the latter.
        if isinstance(self._client, (AsyncAzureOpenAI, AzureOpenAI)):
            return "azure"
        elif isinstance(self._client, (AsyncOpenAI, OpenAI)):
            return "openai"

    def _model_client_match(self, target_model: str) -> bool:
        if isinstance(self._client, (AsyncAzureOpenAI, AzureOpenAI)):
            return target_model.split("/")[0] == "azure"
        elif isinstance(self._client, (AsyncOpenAI, OpenAI)):
            return target_model.split("/")[0] == "openai"
        else:
            raise ValueError(
                f"No client match found for model {target_model}. Client type: {type(self._client)}"
            )

    def get_timeout(self, target_model: Optional[str] = None) -> float:
        """
        Get the configured timeout (if per-model, for the target model).
        """
        out = self._timeout
        if isinstance(self._timeout, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if timeout is a dict"
                )
            out = self._timeout.get(target_model)
        return out

    def get_max_retries(self, target_model: Optional[str] = None) -> int:
        """
        Get the configured max retries (if per-model, for the target model).
        """
        out = self._max_retries
        if isinstance(self._max_retries, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if max_retries is a dict"
                )
            out = self._max_retries.get(target_model)
        return out

    def get_backoff(self, target_model: Optional[str] = None) -> float:
        """
        Get the configured backoff (if per-model, for the target model).
        """
        out = self._backoff
        if isinstance(self._backoff, dict):
            if not target_model:
                raise ValueError(
                    "target_model must be provided if backoff is a dict"
                )
            out = self._backoff.get(target_model)
        return out

    def _update_model_kwargs(
        self,
        kwargs: Dict[str, Any],
        target_model: str,
        user_messages: Optional[List[Dict[str, Any]]] = [],
    ) -> Dict[str, Any]:
        kwargs["model"] = target_model
        kwargs["timeout"] = self.get_timeout(target_model)
        kwargs["messages"] = (
            self._model_messages[target_model] + user_messages
            if self._model_messages.get(target_model)
            else user_messages
        )
        return kwargs

    def _get_fallback_model(self, failed_models: List[str] = []) -> str:
        """
        After failing to invoke current_model, use the user's model fallback list to choose the
        next invocation model.
        """
        # todo [a9] remove this so RetryManager handles selection
        models = [m for m in self._models if m not in failed_models]

        if not models:
            return
        return models[0]

    def _retry_decorator(self, func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_model = kwargs["model"]
            base_messages = kwargs.get("messages", [])
            kwargs = self._update_model_kwargs(
                kwargs, target_model, user_messages=base_messages
            )

            attempt = 0
            while attempt < self.get_max_retries(target_model):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        # throw exception - will invoke next model outside of wrapper
                        raise _RetryWrapperException([target_model], e)

                attempt += 1
                await asyncio.sleep(self.get_backoff(target_model) ** attempt)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            target_model = kwargs["model"]
            base_messages = kwargs.get("messages", [])
            kwargs = self._update_model_kwargs(
                kwargs, target_model, user_messages=base_messages
            )

            attempt = 0
            while attempt < self.get_max_retries(target_model):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    LOGGER.warning(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.get_max_retries(target_model) - 1:
                        raise _RetryWrapperException([target_model], e)

                attempt += 1
                time.sleep(self.get_backoff(target_model) ** attempt)

        if isinstance(self, AsyncRetryWrapper):
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
        raise NotImplementedError("Not Diamond logging not implemented.")


class _BaseCompletions:
    def __init__(self, parent: "_BaseRetryWrapper", manager: "RetryManager"):
        self.parent = parent
        self.manager = manager

    @property
    def completions(self):
        return self

    def _handle_retry_exception(
        self,
        e: _RetryWrapperException,
        target_model: str,
        all_failed_models: list,
    ) -> tuple:
        """Common exception handling logic"""
        LOGGER.exception(e)
        self.parent._update_failed_models(
            target_model, e.failed_models, all_failed_models
        )
        target_model = self.manager.get_next_model(all_failed_models)
        if not target_model:
            raise e.failed_exception
        target_wrapper = self.manager.get_wrapper(target_model)
        return target_model, target_wrapper, target_wrapper._default_create


class RetryWrapper(_BaseRetryWrapper):
    """
    Wrapper for OpenAI clients which adds retry and fallback logic. This method
    patches the `chat` property of the client to return a wrapped ChatCompletions
    object.

    If you need to patch the `create` method (eg. during unit testing), you can target
    the _default_create method of this class.
    """

    class ChatCompletions(_BaseCompletions):
        def create(self, *args, **kwargs):
            target_model = kwargs["model"]
            target_model = f"{self.parent.get_provider()}/{target_model}"
            target_wrapper = self.manager.get_wrapper(target_model)

            all_failed_models = []
            _create_fn = self.parent._default_create
            while True:
                wrapped = target_wrapper._retry_decorator(_create_fn)
                kwargs["model"] = target_model.split("/")[-1]

                try:
                    return wrapped(*args, **kwargs)
                except _RetryWrapperException as e:
                    (
                        target_model,
                        target_wrapper,
                        _create_fn,
                    ) = self._handle_retry_exception(
                        e, target_model, all_failed_models
                    )

    @property
    def chat(self):
        if not isinstance(self._client, OpenAI):
            return getattr(self._client, "chat")

        return self.ChatCompletions(self, self.manager)


class AsyncRetryWrapper(_BaseRetryWrapper):
    """
    Async wrapper for OpenAI clients which adds retry and fallback logic. This method
    patches the `chat` property of the client to return a wrapped AsyncCompletions
    object.

    If you need to patch the `create` method (eg. during unit testing), you can target
    the _default_create method of this class.
    """

    class AsyncCompletions(_BaseCompletions):
        async def create(self, *args, **kwargs):
            target_model = kwargs["model"]
            target_model = f"{self.parent.get_provider()}/{target_model}"
            target_wrapper = self.manager.get_wrapper(target_model)

            all_failed_models = []
            _create_fn = self.parent._default_create

            while True:
                wrapped = target_wrapper._retry_decorator(_create_fn)
                kwargs["model"] = target_model.split("/")[-1]

                try:
                    return await wrapped(*args, **kwargs)
                except _RetryWrapperException as e:
                    (
                        target_model,
                        target_wrapper,
                        _create_fn,
                    ) = self._handle_retry_exception(
                        e, target_model, all_failed_models
                    )

    @property
    def chat(self):
        if not isinstance(self._client, AsyncOpenAI):
            return getattr(self._client, "chat")

        return self.AsyncCompletions(self, self.manager)


class RetryManager:
    """
    RetryManager handles model selection and load balancing when using `notdiamond.init`.
    """

    def __init__(
        self,
        models: Union[Dict[str, float], List[str]],
        wrapped_clients: List[Union[RetryWrapper, AsyncRetryWrapper]],
    ):
        """
        Args:
            models: Dict[str, float] | List[str]
                The models to invoke.
            wrapped_clients: List[RetryWrapper | AsyncRetryWrapper]
                Clients wrapped by this manager.
        """
        self._wrappers = wrapped_clients
        self._models = models

        self._model_weights = (
            _CumulativeModelSelectionWeights(self._models)
            if isinstance(self._models, dict)
            else None
        )

        # Attach the manager to each wrapper and patch chat behavior
        # Without this hack we cannot wrap the create method successfully
        for wrapper in self._wrappers:
            wrapper.manager = self
            wrapper._client.chat = wrapper.chat

        self._model_to_wrapper = {
            model: self._get_model_wrapper(model) for model in self._models
        }

    def _get_model_wrapper(
        self, model: str
    ) -> Union[RetryWrapper, AsyncRetryWrapper]:
        target_wrapper = None
        try:
            if "azure" in model:
                instance_check = (AzureOpenAI, AsyncAzureOpenAI)
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

    def get_next_model(self, failed_models: List[str]) -> str:
        """
        Select the next model to invoke. Omit all models that have previously failed.
        """
        if not self._model_weights:
            remaining_models = [
                m for m in self._models if m not in failed_models
            ]
            if not remaining_models:
                return
            return remaining_models[0]

        try:
            new_weights = _CumulativeModelSelectionWeights(
                {
                    m: w
                    for m, w in self._models.items()
                    if m not in failed_models
                }
            )
        except ValueError:
            # model weights are all zero - do not need to load balance
            return
        return new_weights[random.random()]

    def _select_target_model(self, kwargs: Dict[str, Any]) -> str:
        target_model = kwargs.get("model", None)
        if target_model == "notdiamond":
            return self._model_weights[random.random()]
        return target_model

    def get_wrapper(
        self, model: str
    ) -> Union[RetryWrapper, AsyncRetryWrapper]:
        try:
            return self._model_to_wrapper[model]
        except KeyError:
            raise ValueError(f"No wrapper found for model {model}.")
