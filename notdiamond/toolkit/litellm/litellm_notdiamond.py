# flake8: noqa

# This file is a modified version of the original module provided by BerriAI.
# We have modified the file to add support for Not Diamond, and include the following
# license to comply with their license requirements:

# MIT License

# Copyright (c) 2023 Berri AI

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import types
from typing import Callable, Dict, List, Optional

import httpx
import litellm
import requests
from litellm._version import version
from litellm.utils import ModelResponse

# dict to map notdiamond providers and models to litellm providers and models
ND2LITELLM = {
    # openai
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
    "openai/gpt-4": "gpt-4",
    "openai/gpt-4-0613": "gpt-4-0613",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-2024-05-13": "gpt-4o-2024-05-13",
    "openai/gpt-4-turbo": "gpt-4-turbo",
    "openai/gpt-4-turbo-2024-04-09": "gpt-4-turbo-2024-04-09",
    "openai/gpt-4-turbo-preview": "gpt-4-turbo-preview",
    "openai/gpt-4-0125-preview": "gpt-4-0125-preview",
    "openai/gpt-4-1106-preview": "gpt-4-1106-preview",
    "openai/gpt-4-1106-preview": "gpt-4-1106-preview",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/gpt-4o-mini-2024-07-18": "gpt-4o-mini-2024-07-18",
    "openai/o1-preview-2024-09-12": "o1-preview-2024-09-12",
    "openai/o1-preview": "o1-preview",
    "openai/o1-mini-2024-09-12": "o1-mini-2024-09-12",
    "openai/o1-mini": "o1-mini",
    # anthropic
    "anthropic/claude-2.1": "claude-2.1",
    "anthropic/claude-3-opus-20240229": "claude-3-opus-20240229",
    "anthropic/claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "anthropic/claude-3-5-sonnet-20240620": "claude-3-5-sonnet-20240620",
    "anthropic/claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "anthropic/claude-3-haiku-20240307": "claude-3-haiku-20240307",
    "anthropic/claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    "anthropic/claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    # mistral
    "mistral/mistral-large-latest": "mistral/mistral-large-latest",
    "mistral/mistral-medium-latest": "mistral/mistral-medium-latest",
    "mistral/mistral-small-latest": "mistral/mistral-small-latest",
    "mistral/codestral-latest": "mistral/codestral-latest",
    "mistral/open-mistral-7b": "mistral/open-mistral-7b",
    "mistral/open-mixtral-8x7b": "mistral/open-mixtral-8x7b",
    "mistral/open-mixtral-8x22b": "mistral/open-mixtral-8x22b",
    "mistral/mistral-large-2407": "mistral/mistral-large-2407",
    "mistral/mistral-large-2402": "mistral/mistral-large-2402",
    # perplexity
    "perplexity/llama-3.1-sonar-large-128k-online": "perplexity/llama-3.1-sonar-large-128k-online",
    # cohere
    "cohere/command-r": "cohere_chat/command-r",
    "cohere/command-r-plus": "cohere_chat/command-r-plus",
    # google
    "google/gemini-pro": "gemini/gemini-pro",
    "google/gemini-1.5-pro-latest": "gemini/gemini-1.5-pro-latest",
    "google/gemini-1.5-flash-latest": "gemini/gemini-1.5-flash-latest",
    "google/gemini-1.0-pro-latest": "gemini/gemini-pro",
    # replicate
    "replicate/mistral-7b-instruct-v0.2": "replicate/mistralai/mistral-7b-instruct-v0.2",
    "replicate/mixtral-8x7b-instruct-v0.1": "replicate/mistralai/mixtral-8x7b-instruct-v0.1",
    "replicate/meta-llama-3-70b-instruct": "replicate/meta/meta-llama-3-70b-instruct",
    "replicate/meta-llama-3-8b-instruct": "replicate/meta/meta-llama-3-8b-instruct",
    "replicate/meta-llama-3.1-405b-instruct": "replicate/meta/meta-llama-3.1-405b-instruct",
    # togetherai
    "togetherai/Mistral-7B-Instruct-v0.2": "together_ai/mistralai/Mistral-7B-Instruct-v0.2",
    "togetherai/Mixtral-8x7B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "togetherai/Mixtral-8x22B-Instruct-v0.1": "together_ai/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "togetherai/Llama-3-70b-chat-hf": "together_ai/meta-llama/Llama-3-70b-chat-hf",
    "togetherai/Llama-3-8b-chat-hf": "together_ai/meta-llama/Llama-3-8b-chat-hf",
    "togetherai/Qwen2-72B-Instruct": "together_ai/Qwen/Qwen2-72B-Instruct",
    "togetherai/Meta-Llama-3.1-8B-Instruct-Turbo": "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "togetherai/Meta-Llama-3.1-70B-Instruct-Turbo": "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "togetherai/Meta-Llama-3.1-405B-Instruct-Turbo": "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
}


class NotDiamondError(Exception):
    def __init__(
        self,
        status_code,
        message,
        url="https://api.notdiamond.ai",
    ):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url=url)
        self.response = httpx.Response(
            status_code=status_code, request=self.request
        )
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class NotDiamondConfig:
    llm_providers: List[Dict[str, str]]
    tools: Optional[List[Dict[str, str]]] = None
    max_model_depth: int = 1
    # tradeoff params: "cost"/"latency"
    tradeoff: Optional[str] = None
    preference_id: Optional[str] = None
    hash_content: Optional[bool] = False

    def __init__(
        self,
        llm_providers: List[Dict[str, str]],
        tools: Optional[str] = None,
        max_model_depth: Optional[int] = 1,
        tradeoff: Optional[str] = None,
        preference_id: Optional[str] = None,
        hash_content: Optional[bool] = False,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
            or k == "llm_providers"
        }


def validate_environment(api_key):
    if api_key is None:
        raise ValueError(
            "Missing NOTDIAMOND_API_KEY in env - A call is being made to Not Diamond but no key is set either in the environment variables or via params"
        )
    headers = {
        "Authorization": "Bearer " + api_key,
        "accept": "application/json",
        "content-type": "application/json",
        "User-Agent": f"litellm/{version}",
    }
    return headers


def get_litellm_model(response: dict) -> str:
    nd_provider = response["providers"][0]["provider"]
    nd_model = response["providers"][0]["model"]
    nd_provider_model = f"{nd_provider}/{nd_model}"
    litellm_model = ND2LITELLM[nd_provider_model]
    return litellm_model


def update_litellm_params(litellm_params: dict):
    """
    Create a new litellm_params dict with non-default litellm_params from the original call, custom_llm_provider and api_base
    """
    new_litellm_params = dict()
    for k, v in litellm_params.items():
        # all litellm_params have defaults of None or False, except force_timeout
        if (k == "force_timeout" and v != 600) or v:
            new_litellm_params[k] = v
    if "custom_llm_provider" in new_litellm_params:
        del new_litellm_params["custom_llm_provider"]
    if "api_base" in new_litellm_params:
        del new_litellm_params["api_base"]
    if "api_key" in new_litellm_params:
        del new_litellm_params["api_key"]
    return new_litellm_params


def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    encoding,
    api_key,
    logging_obj,
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
):
    headers = validate_environment(api_key)
    completion_url = api_base

    ## Load Config
    config = NotDiamondConfig.get_config()
    for k, v in config.items():
        if k not in optional_params:
            optional_params[k] = v

    # separate ND optional params from litellm optional params
    nd_params = [
        "llm_providers",
        "tools",
        "max_model_depth",
        "tradeoff",
        "preference_id",
        "hash_content",
    ]
    selected_model_params = dict()
    for k, v in optional_params.items():
        if k not in nd_params:
            selected_model_params[k] = v
    if "tools" in optional_params:
        selected_model_params["tools"] = optional_params["tools"]
    # remove any optional params that are not in the ND params
    optional_params = {
        k: v for k, v in optional_params.items() if k in nd_params
    }

    data = {
        "messages": messages,
        **optional_params,
    }

    ## LOGGING
    logging_obj.pre_call(
        input=messages,
        api_key=api_key,
        additional_args={
            "complete_input_dict": data,
            "headers": headers,
            "api_base": completion_url,
        },
    )

    ## MODEL SELECTION CALL
    nd_response = requests.post(
        api_base,
        headers=headers,
        json=data,
    )
    print_verbose(f"Raw response from Not Diamond: {nd_response.text}")

    ## RESPONSE OBJECT
    if nd_response.status_code != 200:
        raise NotDiamondError(
            status_code=nd_response.status_code, message=nd_response.text
        )
    nd_response = nd_response.json()
    litellm_model = get_litellm_model(nd_response)

    ## COMPLETION CALL
    litellm_params = update_litellm_params(litellm_params)

    is_async_call = litellm_params.pop("acompletion", False)
    if is_async_call:
        return litellm.acompletion(
            model=litellm_model,
            messages=messages,
            **selected_model_params,
            **litellm_params,
        )
    else:
        return litellm.completion(
            model=litellm_model,
            messages=messages,
            **selected_model_params,
            **litellm_params,
        )
