"""NotDiamond client class"""

import inspect
import logging
import time
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from litellm import token_counter

# Details: https://python.langchain.com/v0.1/docs/guides/development/pydantic_compatibility/
from pydantic.v1 import BaseModel
from pydantic_partial import create_partial_model

from notdiamond import settings
from notdiamond._utils import _module_check
from notdiamond.exceptions import (
    ApiError,
    CreateUnavailableError,
    MissingLLMConfigs,
)
from notdiamond.llms.config import LLMConfig
from notdiamond.llms.request import amodel_select, model_select, report_latency
from notdiamond.metrics.metric import Metric
from notdiamond.prompts import inject_system_prompt, _curly_escape
from notdiamond.types import NDApiKeyValidator

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class _NDClientTarget(Enum):
    ROUTER = "router"
    INVOKER = "invoker"


def _ndllm_factory(import_target: _NDClientTarget = None):
    _invoke_error_msg_tmpl = (
        "{fn_name} is not available. `notdiamond` can generate LLM responses after "
        "installing additional dependencies via `pip install notdiamond[create]`."
    )

    _default_llm_config_invalid_warning = "The default LLMConfig set is invalid. Defaulting to {provider}/{model}"

    _no_default_llm_config_warning = (
        "No default LLMConfig set. Defaulting to {provider}/{model}"
    )

    class _NDRouterClient(BaseModel):
        api_key: str
        llm_configs: Optional[List[Union[LLMConfig ,str]]]
        default: Union[LLMConfig, int, str]
        max_model_depth: Optional[int]
        latency_tracking: bool
        hash_content: bool
        tradeoff: Optional[str]
        preference_id: Optional[str]
        tools: Optional[Sequence[Union[Dict[str, Any], Callable]]]
        callbacks: Optional[List]

        class Config:
            arbitrary_types_allowed = True

        def __init__(
            self,
            llm_configs: Optional[List[Union[LLMConfig , str]]] = None,
            api_key: Optional[str] = None,
            default: Union[LLMConfig, int, str] = 0,
            max_model_depth: Optional[int] = None,
            latency_tracking: bool = True,
            hash_content: bool = False,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            callbacks: Optional[List] = None,
            tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = None,
            **kwargs,
        ):
            if api_key is None:
                api_key = settings.NOTDIAMOND_API_KEY
            NDApiKeyValidator(api_key=api_key)

            if llm_configs is not None:
                llm_configs = self._parse_llm_configs_data(llm_configs)

                if max_model_depth is None:
                    max_model_depth = len(llm_configs)

                if max_model_depth > len(llm_configs):
                    LOGGER.warning(
                        "WARNING: max_model_depth cannot be bigger than the number of LLMs."
                    )
                    max_model_depth = len(llm_configs)

            if tradeoff is not None:
                if tradeoff not in ["cost", "latency"]:
                    raise ValueError(
                        "Invalid tradeoff. Accepted values: cost, latency."
                    )

            super().__init__(
                api_key=api_key,
                llm_configs=llm_configs,
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
                tools=tools,
                callbacks=callbacks,
                **kwargs,
            )

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def amodel_select(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            model: Optional[List[LLMConfig]] = None,
            default: Optional[Union[LLMConfig, int, str]] = None,
            max_model_depth: Optional[int] = None,
            latency_tracking: Optional[bool] = None,
            hash_content: Optional[bool] = None,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            metric: Metric = Metric("accuracy"),
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, Optional[LLMConfig]]:
            """
            This function calls the NotDiamond backend to fetch the most suitable model for the given prompt,
            and leaves the execution of the LLM call to the developer.
            The function is async, so it's suitable for async codebases.

            Parameters:
                messages (List[Dict[str, str]]): List of messages, OpenAI style.
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                model (Optional[List[LLMConfig]]): List of models to choose from.
                default (Optional[Union[LLMConfig, int, str]]): Default LLM.
                max_model_depth (Optional[int]): If your top recommended model is down, specify up to which depth
                                                    of routing you're willing to go.
                latency_tracking (Optional[bool]): Latency tracking flag.
                hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
                tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                                    for the router to determine the best LLM for a given query.
                preference_id (Optional[str]): The ID of the router preference that was configured via the Dashboard.
                                                Defaults to None.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Returns:
                tuple[str, Optional[LLMConfig]]: returns the session_id and the chosen LLM
            """
            if input is None:
                input = {}

            if model is not None:
                llm_configs = self._parse_llm_configs_data(model)
                self.llm_configs = llm_configs

            self.validate_params(
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
            )

            best_llm, session_id = await amodel_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm
            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            return session_id, best_llm

        def model_select(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            model: Optional[List[LLMConfig]] = None,
            default: Optional[Union[LLMConfig, int, str]] = None,
            max_model_depth: Optional[int] = None,
            latency_tracking: Optional[bool] = None,
            hash_content: Optional[bool] = None,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            metric: Metric = Metric("accuracy"),
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, Optional[LLMConfig]]:
            """
            This function calls the NotDiamond backend to fetch the most suitable model for the given prompt,
            and leaves the execution of the LLM call to the developer.

            Parameters:
                messages (List[Dict[str, str]]): List of messages OpenAI style.
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                model (Optional[List[LLMConfig]]): List of models to choose from.
                default (Optional[Union[LLMConfig, int, str]]): Default LLM.
                max_model_depth (Optional[int]): If your top recommended model is down, specify up to which depth
                                                    of routing you're willing to go.
                latency_tracking (Optional[bool]): Latency tracking flag.
                hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
                tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                                    for the router to determine the best LLM for a given query.
                preference_id (Optional[str]): The ID of the router preference that was configured via the Dashboard.
                                                Defaults to None.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Returns:
                tuple[str, Optional[LLMConfig]]: returns the session_id and the chosen LLM
            """
            if input is None:
                input = {}

            if model is not None:
                llm_configs = self._parse_llm_configs_data(model)
                self.llm_configs = llm_configs

            self.validate_params(
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
            )

            best_llm, session_id = model_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm
            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            return session_id, best_llm

        @staticmethod
        def _parse_llm_configs_data(
            llm_configs: list,
        ) -> List[LLMConfig]:
            providers = []
            for llm_config in llm_configs:
                if isinstance(llm_config, LLMConfig):
                    providers.append(llm_config)
                    continue
                parsed_provider = LLMConfig.from_string(llm_config)
                providers.append(parsed_provider)
            return providers

        def validate_params(
            self,
            default: Optional[Union[LLMConfig, int, str]] = None,
            max_model_depth: Optional[int] = None,
            latency_tracking: Optional[bool] = None,
            hash_content: Optional[bool] = None,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
        ):
            self.default = default

            if max_model_depth is not None:
                self.max_model_depth = max_model_depth

            if self.llm_configs is None or len(self.llm_configs) == 0:
                raise MissingLLMConfigs(
                    "No LLM config speficied. Specify at least one."
                )

            if self.max_model_depth is None:
                self.max_model_depth = len(self.llm_configs)

            if self.max_model_depth == 0:
                raise ValueError("max_model_depth has to be bigger than 0.")

            if self.max_model_depth > len(self.llm_configs):
                LOGGER.warning(
                    "WARNING: max_model_depth cannot be bigger than the number of LLMs."
                )
                self.max_model_depth = len(self.llm_configs)

            if tradeoff is not None:
                if tradeoff not in ["cost", "latency"]:
                    raise ValueError(
                        "Invalid tradeoff. Accepted values: cost, latency."
                    )
            self.tradeoff = tradeoff

            if preference_id is not None:
                self.preference_id = preference_id

            if latency_tracking is not None:
                self.latency_tracking = latency_tracking

            if hash_content is not None:
                self.hash_content = hash_content

        def bind_tools(
            self, tools: Sequence[Union[Dict[str, Any], Callable]]
        ) -> "NotDiamond":
            """
            Bind tools to the LLM object. The tools will be passed to the LLM object when invoking it.
            Results in the tools being available in the LLM object.
            You can access the tool_calls in the result via `result.tool_calls`.
            """

            for provider in self.llm_configs:
                if provider.model not in settings.PROVIDERS[
                    provider.provider
                ].get("support_tools", []):
                    raise ApiError(
                        f"{provider.provider}/{provider.model} does not support function calling."
                    )
            self.tools = tools

            return self

        def call_callbacks(self, function_name: str, *args, **kwargs) -> None:
            """
            Call all callbacks with a specific function name.
            """

            if self.callbacks is None:
                return

            for callback in self.callbacks:
                if hasattr(callback, function_name):
                    getattr(callback, function_name)(*args, **kwargs)

        def create(*args, **kwargs):
            format_str = f"`{inspect.stack()[0].function}`"
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(fn_name=format_str)
            )

        async def acreate(*args, **kwargs):
            format_str = f"`{inspect.stack()[0].function}`"
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(fn_name=format_str)
            )

        def invoke(*args, **kwargs):
            format_str = f"`{inspect.stack()[0].function}`"
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(fn_name=format_str)
            )

        async def ainvoke(*args, **kwargs):
            format_str = f"`{inspect.stack()[0].function}`"
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(fn_name=format_str)
            )

        def stream(*args, **kwargs):
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(
                    fn_name=inspect.stack()[0].function
                )
            )

        async def astream(*args, **kwargs):
            raise CreateUnavailableError(
                _invoke_error_msg_tmpl.format(
                    fn_name=inspect.stack()[0].function
                )
            )

        @property
        def default_llm(self) -> LLMConfig:
            """
            Return the default LLM that's set on the NotDiamond client class.
            """
            if isinstance(self.default, int):
                if self.default < len(self.llm_configs):
                    return self.llm_configs[self.default]

            if isinstance(self.default, str):
                try:
                    default = LLMConfig.from_string(self.default)
                    if default in self.llm_configs:
                        return default
                except Exception as e:
                    LOGGER.debug(f"Error setting default llm: {e}")

            if isinstance(self.default, LLMConfig):
                return self.default

            default = self.llm_configs[0]
            if self.default is None:
                LOGGER.info(
                    _no_default_llm_config_warning.format(
                        provider=default.provider, model=default.model
                    )
                )
            else:
                LOGGER.info(
                    _default_llm_config_invalid_warning.format(
                        provider=default.provider, model=default.model
                    )
                )
            return default

    # Do not import from langchain_core directly, as it is now an optional SDK dependency
    try:
        LLM = _module_check("langchain_core.language_models.llms", "LLM")
        BaseMessageChunk = _module_check(
            "langchain_core.messages", "BaseMessageChunk"
        )
        JsonOutputParser = _module_check(
            "langchain_core.output_parsers", "JsonOutputParser"
        )
        ChatPromptTemplate = _module_check(
            "langchain_core.prompts", "ChatPromptTemplate"
        )
    except (ModuleNotFoundError, ImportError) as ierr:
        msg = _invoke_error_msg_tmpl.format(fn_name="NotDiamond creation")
        if import_target == _NDClientTarget.INVOKER:
            msg += " Create was requested, however - raising..."
            raise ImportError(msg) from ierr
        else:
            LOGGER.debug(msg)
            return _NDRouterClient

    class _NDInvokerClient(_NDRouterClient, LLM):
        """
        Implementation of NotDiamond class, the main class responsible for creating and invoking LLM prompts.
        The class inherits from Langchain's LLM class. Starting reference is from here:
        https://python.langchain.com/docs/modules/model_io/llms/custom_llm

        It's mandatory to have an API key set. If the api_key is not explicitly specified,
        it will check for NOTDIAMOND_API_KEY in the .env file.

        Raises:
            MissingLLMProviders: you must specify at least one LLM provider for the router to work
            ApiError: error raised when the NotDiamond API call fails.
                        Ensure to set a default LLM provider to not break the code.
        """

        api_key: str
        llm_configs: Optional[List[Union[LLMConfig , str]]]
        default: Union[LLMConfig, int, str]
        max_model_depth: Optional[int]
        latency_tracking: bool
        hash_content: bool
        tradeoff: Optional[str]
        preference_id: Optional[str]
        tools: Optional[Sequence[Union[Dict[str, Any], Callable]]]
        callbacks: Optional[List]

        def __init__(
            self,
            llm_configs: Optional[List[Union[LLMConfig , str]]] = None,
            api_key: Optional[str] = None,
            default: Union[LLMConfig, int, str] = 0,
            max_model_depth: Optional[int] = None,
            latency_tracking: bool = True,
            hash_content: bool = False,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = None,
            callbacks: Optional[List] = None,
            **kwargs,
        ) -> None:
            super().__init__(
                api_key=api_key,
                llm_configs=llm_configs,
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
                tools=tools,
                callbacks=callbacks,
                **kwargs,
            )

        def __repr__(self) -> str:
            class_name = self.__class__.__name__
            address = hex(id(self))  # Gets the memory address of the object
            return f"<{class_name} object at {address}>"

        @property
        def _llm_type(self) -> str:
            return "NotDiamond LLM"

        @staticmethod
        def _inject_model_instruction(messages, parser):
            format_instructions = parser.get_format_instructions()
            format_instructions = format_instructions.replace(
                "{", "{{"
            ).replace("}", "}}")
            messages[0]["content"] = (
                format_instructions + "\n" + messages[0]["content"]
            )
            return messages

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
        ) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            return "This function is deprecated for the latest LangChain version, use invoke instead"

        def create(
            self,
            messages: List[Dict[str, str]],
            model: Optional[List[LLMConfig]] = None,
            default: Optional[Union[LLMConfig, int, str]] = None,
            max_model_depth: Optional[int] = None,
            latency_tracking: Optional[bool] = None,
            hash_content: Optional[bool] = None,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, str, LLMConfig]:
            """
            Function call to invoke the LLM, with the same interface
            as the OpenAI Python library.

            Parameters:
                messages (Optional[List[Dict[str, str]], optional): Can be used instead of prompt_template to pass
                    the messages OpenAI style.
                model (Optional[List[LLMConfig]]): List of models to choose from.
                default (Optional[Union[LLMConfig, int, str]]): Default LLM.
                max_model_depth (Optional[int]): If your top recommended model is down, specify up to which depth
                                                    of routing you're willing to go.
                latency_tracking (Optional[bool]): Latency tracking flag.
                hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
                tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                                    for the router to determine the best LLM for a given query.
                preference_id (Optional[str]): The ID of the router preference that was configured via the Dashboard.
                                                Defaults to None.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a
                                                                dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Returns:
                tuple[Union[AIMessage, BaseModel], str, LLMConfig]:
                    result: response type defined by Langchain, contains the response from the LLM.
                    or object of the response_model
                    str: session_id returned by the NotDiamond API
                    LLMConfig: the best LLM selected by the router
            """

            if model is not None:
                llm_configs = self._parse_llm_configs_data(model)
                self.llm_configs = llm_configs

            self.validate_params(
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
            )

            return self.invoke(
                messages=messages,
                metric=metric,
                response_model=response_model,
                timeout=timeout,
                **kwargs,
            )

        async def acreate(
            self,
            messages: List[Dict[str, str]],
            model: Optional[List[LLMConfig]] = None,
            default: Optional[Union[LLMConfig, int, str]] = None,
            max_model_depth: Optional[int] = None,
            latency_tracking: Optional[bool] = None,
            hash_content: Optional[bool] = None,
            tradeoff: Optional[str] = None,
            preference_id: Optional[str] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, str, LLMConfig]:
            """
            Async function call to invoke the LLM, with the same interface
            as the OpenAI Python library.

            Parameters:
                messages (Optional[List[Dict[str, str]], optional): Can be used instead of prompt_template to pass
                    the messages OpenAI style.
                model (Optional[List[LLMConfig]]): List of models to choose from.
                default (Optional[Union[LLMConfig, int, str]]): Default LLM.
                max_model_depth (Optional[int]): If your top recommended model is down, specify up to which depth
                                                    of routing you're willing to go.
                latency_tracking (Optional[bool]): Latency tracking flag.
                hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
                tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                                    for the router to determine the best LLM for a given query.
                preference_id (Optional[str]): The ID of the router preference that was configured via the Dashboard.
                                                Defaults to None.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a
                                                                dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Returns:
                tuple[Union[AIMessage, BaseModel], str, LLMConfig]:
                    result: response type defined by Langchain, contains the response from the LLM.
                    or object of the response_model
                    str: session_id returned by the NotDiamond API
                    LLMConfig: the best LLM selected by the router
            """
            if model is not None and len(model) > 0:
                llm_configs = self._parse_llm_configs_data(model)
                self.llm_configs = llm_configs

            self.validate_params(
                default=default,
                max_model_depth=max_model_depth,
                latency_tracking=latency_tracking,
                hash_content=hash_content,
                tradeoff=tradeoff,
                preference_id=preference_id,
            )

            result = await self.ainvoke(
                messages=messages,
                metric=metric,
                response_model=response_model,
                timeout=timeout,
                **kwargs,
            )
            return result

        def invoke(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, str, LLMConfig]:
            """
            Function to invoke the LLM. Behind the scenes what happens:
            1. API call to NotDiamond backend to get the most suitable LLM for the given prompt
            2. Invoke the returned LLM client side
            3. Return the response

            Parameters:
                prompt_template (Optional(Union[ NDPromptTemplate, NDChatPromptTemplate, str, ])):
                    the prompt template defined by the user. It also supports Langchain prompt template types.
                messages (Optional[List[Dict[str, str]], optional): Can be used instead of prompt_template to pass
                    the messages OpenAI style.
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a
                                                                dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Returns:
                tuple[Union[AIMessage, BaseModel], str, LLMConfig]:
                    result: response type defined by Langchain, contains the response from the LLM.
                    or object of the response_model
                    str: session_id returned by the NotDiamond API
                    LLMConfig: the best LLM selected by the router
            """
            JsonOutputParser = _module_check(
                "langchain_core.output_parsers", "JsonOutputParser"
            )

            # If response_model is present, we will parse the response into the given model
            # doing this here so that if validation errors occur, we can raise them before making the API call
            response_model_parser = None
            if response_model is not None:
                self.verify_against_response_model()
                response_model_parser = JsonOutputParser(
                    pydantic_object=response_model
                )

            if input is None:
                input = {}

            best_llm, session_id = model_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            is_default = False
            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm
                is_default = True

            if best_llm.system_prompt is not None:
                messages = inject_system_prompt(
                    messages, best_llm.system_prompt
                )

            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            llm = self._llm_from_config(best_llm, callbacks=self.callbacks)

            if self.tools:
                llm = llm.bind_tools(self.tools)

            if response_model is not None:
                messages = _NDInvokerClient._inject_model_instruction(
                    messages, response_model_parser
                )
            chain_messages = [
                (msg["role"], _curly_escape(msg["content"]))
                for msg in messages
            ]
            prompt_template = ChatPromptTemplate.from_messages(chain_messages)
            chain = prompt_template | llm
            accepted_errors = _get_accepted_invoke_errors(best_llm.provider)

            try:
                if self.latency_tracking:
                    result = self._invoke_with_latency_tracking(
                        session_id=session_id,
                        chain=chain,
                        llm_config=best_llm,
                        is_default=is_default,
                        input=input,
                        **kwargs,
                    )
                else:
                    result = chain.invoke(input, **kwargs)
            except accepted_errors as e:
                if best_llm.provider == "google":
                    LOGGER.warning(
                        f"Submitted chat messages are violating Google requirements with error {e}. "
                        "If you see this message, `notdiamond` has returned a Google model as the best option, "
                        "but the LLM call will fail. If possible, `notdiamond` will fall back to a non-Google model."
                    )

                    non_google_llm = next(
                        (
                            llm_config
                            for llm_config in self.llm_configs
                            if llm_config.provider != "google"
                        ),
                        None,
                    )

                    if non_google_llm is not None:
                        best_llm = non_google_llm
                        llm = self._llm_from_config(
                            best_llm, callbacks=self.callbacks
                        )
                        if response_model is not None:
                            messages = (
                                _NDInvokerClient._inject_model_instruction(
                                    messages, response_model_parser
                                )
                            )
                        chain_messages = [
                            (msg["role"], _curly_escape(msg["content"]))
                            for msg in messages
                        ]
                        prompt_template = ChatPromptTemplate.from_messages(
                            chain_messages
                        )
                        chain = prompt_template | llm

                        if self.latency_tracking:
                            result = self._invoke_with_latency_tracking(
                                session_id=session_id,
                                chain=chain,
                                llm_config=best_llm,
                                is_default=is_default,
                                input=input,
                                **kwargs,
                            )
                        else:
                            result = chain.invoke(input, **kwargs)
                    else:
                        raise e
                else:
                    raise e

            if response_model is not None:
                parsed_dict = response_model_parser.parse(result.content)
                result = response_model.parse_obj(parsed_dict)

            return result, session_id, best_llm

        async def ainvoke(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> tuple[str, str, LLMConfig]:
            """
            Function to invoke the LLM. Behind the scenes what happens:
            1. API call to NotDiamond backend to get the most suitable LLM for the given prompt
            2. Invoke the returned LLM client side
            3. Return the response

            Parameters:
                messages (List[Dict[str, str]]): List of messages, OpenAI style
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Returns:
                tuple[Union[AIMessage, BaseModel], str, LLMConfig]:
                    result: response type defined by Langchain, contains the response from the LLM.
                    or object of the response_model
                    str: session_id returned by the NotDiamond API
                    LLMConfig: the best LLM selected by the router
            """
            JsonOutputParser = _module_check(
                "langchain_core.output_parsers", "JsonOutputParser"
            )

            response_model_parser = None
            if response_model is not None:
                self.verify_against_response_model()
                response_model_parser = JsonOutputParser(
                    pydantic_object=response_model
                )

            if input is None:
                input = {}

            best_llm, session_id = await amodel_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            is_default = False
            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm
                is_default = True

            if best_llm.system_prompt is not None:
                messages = inject_system_prompt(
                    messages, best_llm.system_prompt
                )

            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            llm = self._llm_from_config(best_llm, callbacks=self.callbacks)

            if self.tools:
                llm = llm.bind_tools(self.tools)

            if response_model is not None:
                messages = _NDInvokerClient._inject_model_instruction(
                    messages, response_model_parser
                )
            chain_messages = [
                (msg["role"], _curly_escape(msg["content"]))
                for msg in messages
            ]
            prompt_template = ChatPromptTemplate.from_messages(chain_messages)
            chain = prompt_template | llm
            accepted_errors = _get_accepted_invoke_errors(best_llm.provider)

            try:
                if self.latency_tracking:
                    result = await self._async_invoke_with_latency_tracking(
                        session_id=session_id,
                        chain=chain,
                        llm_config=best_llm,
                        is_default=is_default,
                        input=input,
                        **kwargs,
                    )
                else:
                    result = await chain.ainvoke(input, **kwargs)
            except accepted_errors as e:
                if best_llm.provider == "google":
                    LOGGER.warning(
                        f"Submitted chat messages are violating Google requirements with error {e}. "
                        "If you see this message, `notdiamond` has returned a Google model as the best option, "
                        "but the LLM call will fail. If possible, `notdiamond` will fall back to a non-Google model."
                    )

                    non_google_llm = next(
                        (
                            llm_config
                            for llm_config in self.llm_configs
                            if llm_config.provider != "google"
                        ),
                        None,
                    )

                    if non_google_llm is not None:
                        best_llm = non_google_llm
                        llm = self._llm_from_config(
                            best_llm, callbacks=self.callbacks
                        )
                        if response_model is not None:
                            messages = (
                                _NDInvokerClient._inject_model_instruction(
                                    messages, response_model_parser
                                )
                            )
                        chain_messages = [
                            (msg["role"], _curly_escape(msg["content"]))
                            for msg in messages
                        ]
                        prompt_template = ChatPromptTemplate.from_messages(
                            chain_messages
                        )
                        chain = prompt_template | llm

                        if self.latency_tracking:
                            result = (
                                await self._async_invoke_with_latency_tracking(
                                    session_id=session_id,
                                    chain=chain,
                                    llm_config=best_llm,
                                    is_default=is_default,
                                    input=input,
                                    **kwargs,
                                )
                            )
                        else:
                            result = await chain.ainvoke(input, **kwargs)
                    else:
                        raise e
                else:
                    raise e

            if response_model is not None:
                parsed_dict = response_model_parser.parse(result.content)
                result = response_model.parse_obj(parsed_dict)

            return result, session_id, best_llm

        def stream(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> Iterator[Union[BaseMessageChunk, BaseModel]]:
            """
            This function calls the NotDiamond backend to fetch the most suitable model for the given prompt,
            and calls the LLM client side to stream the response.

            Parameters:
                messages (Optional[List[Dict[str, str]], optional): List of messages, OpenAI style
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a
                                                                dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Yields:
                Iterator[Union[BaseMessageChunk, BaseModel]]: returns the response in chunks.
                    If response_model is present, it will return the partial model object
            """
            JsonOutputParser = _module_check(
                "langchain_core.output_parsers", "JsonOutputParser"
            )

            response_model_parser = None
            if response_model is not None:
                self.verify_against_response_model()
                response_model_parser = JsonOutputParser(
                    pydantic_object=response_model
                )

            if input is None:
                input = {}

            best_llm, session_id = model_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm

            if best_llm.system_prompt is not None:
                messages = inject_system_prompt(
                    messages, best_llm.system_prompt
                )

            if response_model is not None:
                messages = _NDInvokerClient._inject_model_instruction(
                    messages, response_model_parser
                )

            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            llm = self._llm_from_config(best_llm, callbacks=self.callbacks)
            if self.tools:
                llm = llm.bind_tools(self.tools)

            if response_model is not None:
                chain = llm | response_model_parser
            else:
                chain = llm

            for chunk in chain.stream(messages, **kwargs):
                if response_model is None:
                    yield chunk
                else:
                    partial_model = create_partial_model(response_model)
                    yield partial_model(**chunk)

        async def astream(
            self,
            messages: List[Dict[str, str]],
            input: Optional[Dict[str, Any]] = None,
            metric: Metric = Metric("accuracy"),
            response_model: Optional[Type[BaseModel]] = None,
            timeout: int = 5,
            **kwargs,
        ) -> AsyncIterator[Union[BaseMessageChunk, BaseModel]]:
            """
            This function calls the NotDiamond backend to fetch the most suitable model for the given prompt,
            and calls the LLM client side to stream the response. The function is async, so it's suitable for async codebases.

            Parameters:
                messages (Optional[List[Dict[str, str]], optional): List of messages, OpenAI style
                input (Optional[Dict[str, Any]], optional): If the prompt_template contains variables, use input to specify
                                                            the values for those variables. Defaults to None, assuming no
                                                            variables.
                metric (Metric, optional): Metric used by NotDiamond router to choose the best LLM.
                                                Defaults to Metric("accuracy").
                response_model (Optional[Type[BaseModel]], optional): If present, will use JsonOutputParser to parse the
                                                                response into the given model. In which case result will a dict.
                timeout (int): The number of seconds to wait before terminating the API call to Not Diamond backend.
                                Default to 5 seconds.
                **kwargs: Any other arguments that are supported by Langchain's invoke method, will be passed through.

            Raises:
                ApiError: when the NotDiamond API fails

            Yields:
                AsyncIterator[Union[BaseMessageChunk, BaseModel]]: returns the response in chunks.
                    If response_model is present, it will return the partial model object
            """
            response_model_parser = None
            if response_model is not None:
                self.verify_against_response_model()
                response_model_parser = JsonOutputParser(
                    pydantic_object=response_model
                )

            best_llm, session_id = await amodel_select(
                messages=messages,
                llm_configs=self.llm_configs,
                metric=metric,
                notdiamond_api_key=self.api_key,
                max_model_depth=self.max_model_depth,
                hash_content=self.hash_content,
                tradeoff=self.tradeoff,
                preference_id=self.preference_id,
                tools=self.tools,
                timeout=timeout,
            )

            if input is None:
                input = {}

            if not best_llm:
                LOGGER.warning(
                    f"ND API error. Falling back to default provider={self.default_llm.provider}/{self.default_llm.model}."
                )
                best_llm = self.default_llm

            if best_llm.system_prompt is not None:
                messages = inject_system_prompt(
                    messages, best_llm.system_prompt
                )
            if response_model is not None:
                messages = _NDInvokerClient._inject_model_instruction(
                    messages, response_model_parser
                )

            self.call_callbacks("on_model_select", best_llm, best_llm.model)

            llm = self._llm_from_config(best_llm, callbacks=self.callbacks)
            if self.tools:
                llm = llm.bind_tools(self.tools)

            if response_model is not None:
                chain = llm | response_model_parser
            else:
                chain = llm

            async for chunk in chain.astream(messages, **kwargs):
                if response_model is None:
                    yield chunk
                else:
                    partial_model = create_partial_model(response_model)
                    yield partial_model(**chunk)

        async def _async_invoke_with_latency_tracking(
            self,
            session_id: str,
            chain: Any,
            llm_config: LLMConfig,
            input: Optional[Dict[str, Any]] = {},
            is_default: bool = True,
            **kwargs,
        ):
            if session_id in ("NO-SESSION-ID", "") and not is_default:
                error_message = (
                    "ND session_id is not valid for latency tracking."
                    + "Please check the API response."
                )
                self.call_callbacks("on_api_error", error_message)
                raise ApiError(error_message)

            start_time = time.time()

            result = await chain.ainvoke(input, **kwargs)

            end_time = time.time()

            tokens_completed = token_counter(
                model=llm_config.model,
                messages=[{"role": "assistant", "content": result.content}],
            )
            tokens_per_second = tokens_completed / (end_time - start_time)

            report_latency(
                session_id=session_id,
                llm_config=llm_config,
                tokens_per_second=tokens_per_second,
                notdiamond_api_key=self.api_key,
            )
            self.call_callbacks(
                "on_latency_tracking",
                session_id,
                llm_config,
                tokens_per_second,
            )

            return result

        def _invoke_with_latency_tracking(
            self,
            session_id: str,
            chain: Any,
            llm_config: LLMConfig,
            input: Optional[Dict[str, Any]] = {},
            is_default: bool = True,
            **kwargs,
        ):
            LOGGER.debug(f"Latency tracking enabled, session_id={session_id}")
            if session_id in ("NO-SESSION-ID", "") and not is_default:
                error_message = (
                    "ND session_id is not valid for latency tracking."
                    + "Please check the API response."
                )
                self.call_callbacks("on_api_error", error_message)
                raise ApiError(error_message)

            start_time = time.time()
            result = chain.invoke(input, **kwargs)
            end_time = time.time()

            tokens_completed = token_counter(
                model=llm_config.model,
                messages=[{"role": "assistant", "content": result.content}],
            )
            tokens_per_second = tokens_completed / (end_time - start_time)

            report_latency(
                session_id=session_id,
                llm_config=llm_config,
                tokens_per_second=tokens_per_second,
                notdiamond_api_key=self.api_key,
            )
            self.call_callbacks(
                "on_latency_tracking",
                session_id,
                llm_config,
                tokens_per_second,
            )

            return result

        @staticmethod
        def _llm_from_config(
            provider: LLMConfig,
            callbacks: Optional[List],
        ) -> Any:
            default_kwargs = {"max_retries": 5, "timeout": 120}
            passed_kwargs = {**default_kwargs, **provider.kwargs}

            if provider.provider == "openai":
                ChatOpenAI = _module_check(
                    "langchain_openai.chat_models",
                    "ChatOpenAI",
                    provider.provider,
                )
                return ChatOpenAI(
                    openai_api_key=provider.api_key,
                    model_name=provider.model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "anthropic":
                ChatAnthropic = _module_check(
                    "langchain_anthropic", "ChatAnthropic", provider.provider
                )
                return ChatAnthropic(
                    anthropic_api_key=provider.api_key,
                    model=provider.model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "google":
                ChatGoogleGenerativeAI = _module_check(
                    "langchain_google_genai",
                    "ChatGoogleGenerativeAI",
                    provider.provider,
                )
                return ChatGoogleGenerativeAI(
                    google_api_key=provider.api_key,
                    model=provider.model,
                    convert_system_message_to_human=True,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "cohere":
                ChatCohere = _module_check(
                    "langchain_cohere.chat_models",
                    "ChatCohere",
                    provider.provider,
                )
                return ChatCohere(
                    cohere_api_key=provider.api_key,
                    model=provider.model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "mistral":
                ChatMistralAI = _module_check(
                    "langchain_mistralai.chat_models",
                    "ChatMistralAI",
                    provider.provider,
                )
                return ChatMistralAI(
                    mistral_api_key=provider.api_key,
                    model=provider.model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "togetherai":
                provider_settings = settings.PROVIDERS.get(
                    provider.provider, None
                )
                model_prefixes = provider_settings.get("model_prefix", None)
                model_prefix = model_prefixes.get(provider.model, None)
                del passed_kwargs["max_retries"]
                del passed_kwargs["timeout"]

                if model_prefix is not None:
                    model = f"{model_prefix}/{provider.model}"
                Together = _module_check(
                    "langchain_together", "Together", provider.provider
                )
                return Together(
                    together_api_key=provider.api_key,
                    model=model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "perplexity":
                del passed_kwargs["max_retries"]
                passed_kwargs["request_timeout"] = passed_kwargs["timeout"]
                del passed_kwargs["timeout"]
                ChatPerplexity = _module_check(
                    "langchain_community.chat_models",
                    "ChatPerplexity",
                    provider.provider,
                )
                return ChatPerplexity(
                    pplx_api_key=provider.api_key,
                    model=provider.model,
                    callbacks=callbacks,
                    **passed_kwargs,
                )
            if provider.provider == "replicate":
                provider_settings = settings.PROVIDERS.get(
                    provider.provider, None
                )
                model_prefixes = provider_settings.get("model_prefix", None)
                model_prefix = model_prefixes.get(provider.model, None)
                passed_kwargs["request_timeout"] = passed_kwargs["timeout"]
                del passed_kwargs["timeout"]

                if model_prefix is not None:
                    model = f"replicate/{model_prefix}/{provider.model}"
                ChatLiteLLM = _module_check(
                    "langchain_community.chat_models",
                    "ChatLiteLLM",
                    provider.provider,
                )
                return ChatLiteLLM(
                    model=model,
                    callbacks=callbacks,
                    replicate_api_key=provider.api_key,
                    **passed_kwargs,
                )
            raise ValueError(f"Unsupported provider: {provider.provider}")

        def verify_against_response_model(self) -> bool:
            """
            Verify that the LLMs support response modeling.
            """

            for provider in self.llm_configs:
                if provider.model not in settings.PROVIDERS[
                    provider.provider
                ].get("support_response_model", []):
                    raise ApiError(
                        f"{provider.provider}/{provider.model} does not support response modeling."
                    )

            return True

    if import_target is _NDClientTarget.ROUTER:
        return _NDRouterClient
    return _NDInvokerClient


_NDClient = _ndllm_factory()


class NotDiamond(_NDClient):
    api_key: str
    """
    API key required for making calls to NotDiamond.
    You can get an API key via our dashboard: https://app.notdiamond.ai
    If an API key is not set, it will check for NOTDIAMOND_API_KEY in .env file.
    """

    llm_configs: Optional[List[Union[LLMConfig , str]]]
    """The list of LLMs that are available to route between."""

    default: Union[LLMConfig, int, str]
    """
    Set a default LLM, so in case anything goes wrong in the flow,
    as for example NotDiamond API call fails, your code won't break and you have
    a fallback model. There are various ways to configure a default model:

    - Integer, specifying the index of the default provider from the llm_configs list
    - String, similar how you can specify llm_configs, of structure 'provider_name/model_name'
    - LLMConfig, just directly specify the object of the provider

    By default, we will set your first LLM in the list as the default.
    """

    max_model_depth: Optional[int]
    """
    If your top recommended model is down, specify up to which depth of routing you're willing to go.
    If max_model_depth is not set, it defaults to the length of the llm_configs list.
    If max_model_depth is set to 0, the init will fail.
    If the value is larger than the llm_configs list length, we reset the value to len(llm_configs).
    """

    latency_tracking: bool
    """
    Tracking and sending latency of LLM call to NotDiamond server as feedback, so we can improve our router.
    By default this is turned on, set it to False to turn off.
    """

    hash_content: bool
    """
    Hashing the content before being sent to the NotDiamond API.
    By default this is False.
    """

    tradeoff: Optional[str]
    """
    Define tradeoff between "cost" and "latency" for the router to determine the best LLM for a given query.
    If None is specified, then the router will not consider either cost or latency.

    The supported values: "cost", "latency"

    Defaults to None.
    """

    preference_id: Optional[str]
    """The ID of the router preference that was configured via the Dashboard. Defaults to None."""

    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]]
    """Bind tools to the LLM object. The tools will be passed to the LLM object when invoking it."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _get_accepted_invoke_errors(provider: str) -> Tuple:
    if provider == "google":
        ChatGoogleGenerativeAIError = _module_check(
            "langchain_google_genai.chat_models",
            "ChatGoogleGenerativeAIError",
            provider,
        )
        accepted_errors = (ChatGoogleGenerativeAIError, ValueError)
    else:
        accepted_errors = (ValueError,)
    return accepted_errors
