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


import asyncio
import contextvars
import inspect
import os
import time
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Type, Union

import httpx
import litellm
import openai
import tiktoken
from litellm import (  # type: ignore
    client,
    exception_type,
    get_litellm_params,
    get_optional_params,
)
from litellm._logging import verbose_logger
from litellm.llms import (
    aleph_alpha,
    baseten,
    clarifai,
    cloudflare,
    maritalk,
    nlp_cloud,
    ollama,
    ollama_chat,
    oobabooga,
    openrouter,
    palm,
    petals,
    replicate,
    vllm,
)
from litellm.llms.AI21 import completion as ai21
from litellm.llms.AzureOpenAI.azure import _check_dynamic_azure_params
from litellm.llms.cohere import chat as cohere_chat
from litellm.llms.cohere import completion as cohere_completion  # type: ignore
from litellm.llms.custom_llm import CustomLLM, custom_chat_llm_router
from litellm.llms.prompt_templates.factory import (
    custom_prompt,
    function_call_prompt,
    map_system_message_pt,
    prompt_factory,
    stringify_json_tool_call_content,
)
from litellm.llms.vertex_ai_and_google_ai_studio import vertex_ai_non_gemini
from litellm.main import *
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import all_litellm_params
from litellm.utils import (
    CustomStreamWrapper,
    ModelResponse,
    TextCompletionResponse,
    completion_with_fallbacks,
    get_secret,
    supports_httpx_timeout,
)
from pydantic import BaseModel

from . import notdiamond_key, provider_list
from .litellm_notdiamond import completion as notdiamond_completion

encoding = tiktoken.get_encoding("cl100k_base")


def get_api_key(llm_provider: str, dynamic_api_key: Optional[str]):
    api_key = dynamic_api_key or litellm.api_key
    # openai
    if llm_provider == "openai" or llm_provider == "text-completion-openai":
        api_key = api_key or litellm.openai_key or get_secret("OPENAI_API_KEY")
    # anthropic
    elif llm_provider == "anthropic":
        api_key = (
            api_key or litellm.anthropic_key or get_secret("ANTHROPIC_API_KEY")
        )
    # ai21
    elif llm_provider == "ai21":
        api_key = api_key or litellm.ai21_key or get_secret("AI211_API_KEY")
    # aleph_alpha
    elif llm_provider == "aleph_alpha":
        api_key = (
            api_key
            or litellm.aleph_alpha_key
            or get_secret("ALEPH_ALPHA_API_KEY")
        )
    # baseten
    elif llm_provider == "baseten":
        api_key = (
            api_key or litellm.baseten_key or get_secret("BASETEN_API_KEY")
        )
    # cohere
    elif llm_provider == "cohere" or llm_provider == "cohere_chat":
        api_key = api_key or litellm.cohere_key or get_secret("COHERE_API_KEY")
    # huggingface
    elif llm_provider == "huggingface":
        api_key = (
            api_key
            or litellm.huggingface_key
            or get_secret("HUGGINGFACE_API_KEY")
        )
    # notdiamond
    elif llm_provider == "notdiamond":
        api_key = api_key or notdiamond_key or get_secret("NOTDIAMOND_API_KEY")
    # nlp_cloud
    elif llm_provider == "nlp_cloud":
        api_key = (
            api_key or litellm.nlp_cloud_key or get_secret("NLP_CLOUD_API_KEY")
        )
    # replicate
    elif llm_provider == "replicate":
        api_key = (
            api_key or litellm.replicate_key or get_secret("REPLICATE_API_KEY")
        )
    # together_ai
    elif llm_provider == "together_ai":
        api_key = (
            api_key
            or litellm.togetherai_api_key
            or get_secret("TOGETHERAI_API_KEY")
            or get_secret("TOGETHER_AI_TOKEN")
        )
    return api_key


def get_llm_provider(
    model: str,
    custom_llm_provider: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    litellm_params: Optional[LiteLLM_Params] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Returns the provider for a given model name - e.g. 'azure/chatgpt-v-2' -> 'azure'

    For router -> Can also give the whole litellm param dict -> this function will extract the relevant details

    Raises Error - if unable to map model to a provider
    """
    try:
        ## IF LITELLM PARAMS GIVEN ##
        if litellm_params is not None:
            assert (
                custom_llm_provider is None
                and api_base is None
                and api_key is None
            ), "Either pass in litellm_params or the custom_llm_provider/api_base/api_key. Otherwise, these values will be overriden."
            custom_llm_provider = litellm_params.custom_llm_provider
            api_base = litellm_params.api_base
            api_key = litellm_params.api_key

        dynamic_api_key = None
        # check if llm provider provided
        # AZURE AI-Studio Logic - Azure AI Studio supports AZURE/Cohere
        # If User passes azure/command-r-plus -> we should send it to cohere_chat/command-r-plus
        if model.split("/", 1)[0] == "azure":
            if _is_non_openai_azure_model(model):
                custom_llm_provider = "openai"
                return model, custom_llm_provider, dynamic_api_key, api_base

        if custom_llm_provider:
            return model, custom_llm_provider, dynamic_api_key, api_base

        if api_key and api_key.startswith("os.environ/"):
            dynamic_api_key = get_secret(api_key)
        # check if llm provider part of model name
        if (
            model.split("/", 1)[0] in provider_list
            and model.split("/", 1)[0] not in litellm.model_list
            and len(model.split("/"))
            > 1  # handle edge case where user passes in `litellm --model mistral` https://github.com/BerriAI/litellm/issues/1351
        ):
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if custom_llm_provider == "perplexity":
                # perplexity is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.perplexity.ai
                api_base = api_base or "https://api.perplexity.ai"
                dynamic_api_key = api_key or get_secret("PERPLEXITYAI_API_KEY")
            elif custom_llm_provider == "anyscale":
                # anyscale is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = api_base or "https://api.endpoints.anyscale.com/v1"
                dynamic_api_key = api_key or get_secret("ANYSCALE_API_KEY")
            elif custom_llm_provider == "deepinfra":
                # deepinfra is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = api_base or "https://api.deepinfra.com/v1/openai"
                dynamic_api_key = api_key or get_secret("DEEPINFRA_API_KEY")
            elif custom_llm_provider == "empower":
                api_base = api_base or "https://app.empower.dev/api/v1"
                dynamic_api_key = api_key or get_secret("EMPOWER_API_KEY")
            elif custom_llm_provider == "groq":
                # groq is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.groq.com/openai/v1
                api_base = api_base or "https://api.groq.com/openai/v1"
                dynamic_api_key = api_key or get_secret("GROQ_API_KEY")
            elif custom_llm_provider == "nvidia_nim":
                # nvidia_nim is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = api_base or "https://integrate.api.nvidia.com/v1"
                dynamic_api_key = api_key or get_secret("NVIDIA_NIM_API_KEY")
            elif custom_llm_provider == "volcengine":
                # volcengine is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.endpoints.anyscale.com/v1
                api_base = (
                    api_base or "https://ark.cn-beijing.volces.com/api/v3"
                )
                dynamic_api_key = api_key or get_secret("VOLCENGINE_API_KEY")
            elif custom_llm_provider == "codestral":
                # codestral is openai compatible, we just need to set this to custom_openai and have the api_base be https://codestral.mistral.ai/v1
                api_base = api_base or "https://codestral.mistral.ai/v1"
                dynamic_api_key = api_key or get_secret("CODESTRAL_API_KEY")
            elif custom_llm_provider == "deepseek":
                # deepseek is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.deepseek.com/v1
                api_base = api_base or "https://api.deepseek.com/v1"
                dynamic_api_key = api_key or get_secret("DEEPSEEK_API_KEY")
            elif custom_llm_provider == "fireworks_ai":
                # fireworks is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.fireworks.ai/inference/v1
                if not model.startswith("accounts/"):
                    model = f"accounts/fireworks/models/{model}"
                api_base = api_base or "https://api.fireworks.ai/inference/v1"
                dynamic_api_key = api_key or (
                    get_secret("FIREWORKS_API_KEY")
                    or get_secret("FIREWORKS_AI_API_KEY")
                    or get_secret("FIREWORKSAI_API_KEY")
                    or get_secret("FIREWORKS_AI_TOKEN")
                )
            elif custom_llm_provider == "azure_ai":
                api_base = api_base or get_secret("AZURE_AI_API_BASE")  # type: ignore
                dynamic_api_key = api_key or get_secret("AZURE_AI_API_KEY")
            elif custom_llm_provider == "mistral":
                # mistral is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.mistral.ai
                api_base = (
                    api_base
                    or get_secret(
                        "MISTRAL_AZURE_API_BASE"
                    )  # for Azure AI Mistral
                    or "https://api.mistral.ai/v1"
                )  # type: ignore

                # if api_base does not end with /v1 we add it
                if api_base is not None and not api_base.endswith(
                    "/v1"
                ):  # Mistral always needs a /v1 at the end
                    api_base = api_base + "/v1"
                dynamic_api_key = (
                    api_key
                    or get_secret(
                        "MISTRAL_AZURE_API_KEY"
                    )  # for Azure AI Mistral
                    or get_secret("MISTRAL_API_KEY")
                )
            elif custom_llm_provider == "voyage":
                # voyage is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.voyageai.com/v1
                api_base = "https://api.voyageai.com/v1"
                dynamic_api_key = api_key or get_secret("VOYAGE_API_KEY")
            elif custom_llm_provider == "together_ai":
                api_base = "https://api.together.xyz/v1"
                dynamic_api_key = api_key or (
                    get_secret("TOGETHER_API_KEY")
                    or get_secret("TOGETHER_AI_API_KEY")
                    or get_secret("TOGETHERAI_API_KEY")
                    or get_secret("TOGETHER_AI_TOKEN")
                )
            elif custom_llm_provider == "friendliai":
                api_base = (
                    api_base
                    or get_secret("FRIENDLI_API_BASE")
                    or "https://inference.friendli.ai/v1"
                )
                dynamic_api_key = (
                    api_key
                    or get_secret("FRIENDLIAI_API_KEY")
                    or get_secret("FRIENDLI_TOKEN")
                )
            elif custom_llm_provider == "notdiamond":
                api_base = "/".join(
                    [
                        get_secret(
                            "NOTDIAMOND_API_URL", "https://api.notdiamond.ai"
                        ),
                        "v2/optimizer/modelSelect",
                    ]
                )
                dynamic_api_key = get_secret("NOTDIAMOND_API_KEY") or None
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(
                        api_base
                    )
                )
            if dynamic_api_key is not None and not isinstance(
                dynamic_api_key, str
            ):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        elif model.split("/", 1)[0] in provider_list:
            custom_llm_provider = model.split("/", 1)[0]
            model = model.split("/", 1)[1]
            if api_base is not None and not isinstance(api_base, str):
                raise Exception(
                    "api base needs to be a string. api_base={}".format(
                        api_base
                    )
                )
            if dynamic_api_key is not None and not isinstance(
                dynamic_api_key, str
            ):
                raise Exception(
                    "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                        dynamic_api_key
                    )
                )
            return model, custom_llm_provider, dynamic_api_key, api_base
        # check if api base is a known openai compatible endpoint
        if api_base:
            for endpoint in litellm.openai_compatible_endpoints:
                if endpoint in api_base:
                    if endpoint == "api.perplexity.ai":
                        custom_llm_provider = "perplexity"
                        dynamic_api_key = get_secret("PERPLEXITYAI_API_KEY")
                    elif endpoint == "api.endpoints.anyscale.com/v1":
                        custom_llm_provider = "anyscale"
                        dynamic_api_key = get_secret("ANYSCALE_API_KEY")
                    elif endpoint == "api.deepinfra.com/v1/openai":
                        custom_llm_provider = "deepinfra"
                        dynamic_api_key = get_secret("DEEPINFRA_API_KEY")
                    elif endpoint == "api.mistral.ai/v1":
                        custom_llm_provider = "mistral"
                        dynamic_api_key = get_secret("MISTRAL_API_KEY")
                    elif endpoint == "api.groq.com/openai/v1":
                        custom_llm_provider = "groq"
                        dynamic_api_key = get_secret("GROQ_API_KEY")
                    elif endpoint == "https://integrate.api.nvidia.com/v1":
                        custom_llm_provider = "nvidia_nim"
                        dynamic_api_key = get_secret("NVIDIA_NIM_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "codestral"
                        dynamic_api_key = get_secret("CODESTRAL_API_KEY")
                    elif endpoint == "https://codestral.mistral.ai/v1":
                        custom_llm_provider = "text-completion-codestral"
                        dynamic_api_key = get_secret("CODESTRAL_API_KEY")
                    elif endpoint == "app.empower.dev/api/v1":
                        custom_llm_provider = "empower"
                        dynamic_api_key = get_secret("EMPOWER_API_KEY")
                    elif endpoint == "api.deepseek.com/v1":
                        custom_llm_provider = "deepseek"
                        dynamic_api_key = get_secret("DEEPSEEK_API_KEY")
                    elif endpoint == "inference.friendli.ai/v1":
                        custom_llm_provider = "friendliai"
                        dynamic_api_key = get_secret(
                            "FRIENDLIAI_API_KEY"
                        ) or get_secret("FRIENDLI_TOKEN")

                    if api_base is not None and not isinstance(api_base, str):
                        raise Exception(
                            "api base needs to be a string. api_base={}".format(
                                api_base
                            )
                        )
                    if dynamic_api_key is not None and not isinstance(
                        dynamic_api_key, str
                    ):
                        raise Exception(
                            "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                                dynamic_api_key
                            )
                        )
                    return model, custom_llm_provider, dynamic_api_key, api_base  # type: ignore

        # check if model in known model provider list  -> for huggingface models, raise exception as they don't have a fixed provider (can be togetherai, anyscale, baseten, runpod, et.)
        ## openai - chatcompletion + text completion
        if (
            model in litellm.open_ai_chat_completion_models
            or "ft:gpt-3.5-turbo" in model
            or "ft:gpt-4" in model  # catches ft:gpt-4-0613, ft:gpt-4o
            or model in litellm.openai_image_generation_models
        ):
            custom_llm_provider = "openai"
        elif model in litellm.open_ai_text_completion_models:
            custom_llm_provider = "text-completion-openai"
        ## anthropic
        elif model in litellm.anthropic_models:
            custom_llm_provider = "anthropic"
        ## cohere
        elif (
            model in litellm.cohere_models
            or model in litellm.cohere_embedding_models
        ):
            custom_llm_provider = "cohere"
        ## cohere chat models
        elif model in litellm.cohere_chat_models:
            custom_llm_provider = "cohere_chat"
        ## replicate
        elif model in litellm.replicate_models or (
            ":" in model and len(model) > 64
        ):
            model_parts = model.split(":")
            if (
                len(model_parts) > 1 and len(model_parts[1]) == 64
            ):  ## checks if model name has a 64 digit code - e.g. "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
                custom_llm_provider = "replicate"
            elif model in litellm.replicate_models:
                custom_llm_provider = "replicate"
        ## openrouter
        elif model in litellm.openrouter_models:
            custom_llm_provider = "openrouter"
        ## openrouter
        elif model in litellm.maritalk_models:
            custom_llm_provider = "maritalk"
        ## vertex - text + chat + language (gemini) models
        elif (
            model in litellm.vertex_chat_models
            or model in litellm.vertex_code_chat_models
            or model in litellm.vertex_text_models
            or model in litellm.vertex_code_text_models
            or model in litellm.vertex_language_models
            or model in litellm.vertex_embedding_models
            or model in litellm.vertex_vision_models
        ):
            custom_llm_provider = "vertex_ai"
        ## ai21
        elif model in litellm.ai21_models:
            custom_llm_provider = "ai21"
        ## aleph_alpha
        elif model in litellm.aleph_alpha_models:
            custom_llm_provider = "aleph_alpha"
        ## baseten
        elif model in litellm.baseten_models:
            custom_llm_provider = "baseten"
        ## nlp_cloud
        elif model in litellm.nlp_cloud_models:
            custom_llm_provider = "nlp_cloud"
        ## petals
        elif model in litellm.petals_models:
            custom_llm_provider = "petals"
        ## bedrock
        elif (
            model in litellm.bedrock_models
            or model in litellm.bedrock_embedding_models
        ):
            custom_llm_provider = "bedrock"
        elif model in litellm.watsonx_models:
            custom_llm_provider = "watsonx"
        # openai embeddings
        elif model in litellm.open_ai_embedding_models:
            custom_llm_provider = "openai"
        elif model in litellm.empower_models:
            custom_llm_provider = "empower"
        elif model == "*":
            custom_llm_provider = "openai"
        if custom_llm_provider is None or custom_llm_provider == "":
            if litellm.suppress_debug_info == False:
                print()  # noqa
                print(  # noqa
                    "\033[1;31mProvider List: https://docs.litellm.ai/docs/providers\033[0m"  # noqa
                )  # noqa
                print()  # noqa
            error_str = f"LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model={model}\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers"
            # maps to openai.NotFoundError, this is raised when openai does not recognize the llm
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=error_str,
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )
        if api_base is not None and not isinstance(api_base, str):
            raise Exception(
                "api base needs to be a string. api_base={}".format(api_base)
            )
        if dynamic_api_key is not None and not isinstance(
            dynamic_api_key, str
        ):
            raise Exception(
                "dynamic_api_key needs to be a string. dynamic_api_key={}".format(
                    dynamic_api_key
                )
            )
        return model, custom_llm_provider, dynamic_api_key, api_base
    except Exception as e:
        if isinstance(e, litellm.exceptions.BadRequestError):
            raise e
        else:
            error_str = f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}"
            raise litellm.exceptions.BadRequestError(  # type: ignore
                message=f"GetLLMProvider Exception - {str(e)}\n\noriginal model: {model}",
                model=model,
                response=httpx.Response(
                    status_code=400,
                    content=error_str,
                    request=httpx.Request(method="completion", url="https://github.com/BerriAI/litellm"),  # type: ignore
                ),
                llm_provider="",
            )


@client
async def acompletion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    deployment_id=None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    extra_headers: Optional[dict] = None,
    # Optional liteLLM function params
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Asynchronously executes a litellm.completion() call for any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)

    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only use this if stream is True.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        timeout (float, optional): The maximum execution time in seconds for the completion request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Notes:
        - This function is an asynchronous version of the `completion` function.
        - The `completion` function is called using `run_in_executor` to execute synchronously in the event loop.
        - If `stream` is True, the function returns an async generator that yields completion lines.
    """
    loop = asyncio.get_event_loop()
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    # Adjusted to use explicit arguments instead of *args and **kwargs
    completion_kwargs = {
        "model": model,
        "messages": messages,
        "functions": functions,
        "function_call": function_call,
        "timeout": timeout,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stream_options": stream_options,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias,
        "user": user,
        "response_format": response_format,
        "seed": seed,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "deployment_id": deployment_id,
        "base_url": base_url,
        "api_version": api_version,
        "api_key": api_key,
        "model_list": model_list,
        "extra_headers": extra_headers,
        "acompletion": True,  # assuming this is a required parameter
    }
    if custom_llm_provider is None:
        _, custom_llm_provider, _, _ = get_llm_provider(
            model=model, api_base=completion_kwargs.get("base_url", None)
        )
    try:
        # Use a partial function to pass your keyword arguments
        func = partial(completion, **completion_kwargs, **kwargs)

        # Add the context to the function
        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)

        if (
            custom_llm_provider == "openai"
            or custom_llm_provider == "azure"
            or custom_llm_provider == "azure_text"
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "openrouter"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "codestral"
            or custom_llm_provider == "text-completion-codestral"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "text-completion-openai"
            or custom_llm_provider == "huggingface"
            or custom_llm_provider == "ollama"
            or custom_llm_provider == "ollama_chat"
            or custom_llm_provider == "replicate"
            or custom_llm_provider == "vertex_ai"
            or custom_llm_provider == "vertex_ai_beta"
            or custom_llm_provider == "gemini"
            or custom_llm_provider == "sagemaker"
            or custom_llm_provider == "sagemaker_chat"
            or custom_llm_provider == "anthropic"
            or custom_llm_provider == "predibase"
            or custom_llm_provider == "bedrock"
            or custom_llm_provider == "databricks"
            or custom_llm_provider == "triton"
            or custom_llm_provider == "clarifai"
            or custom_llm_provider == "watsonx"
            or custom_llm_provider == "notdiamond"
            or custom_llm_provider in litellm.openai_compatible_providers
            or custom_llm_provider in litellm._custom_providers
        ):  # currently implemented aiohttp calls for just azure, openai, hf, ollama, vertex ai soon all.
            init_response = await loop.run_in_executor(None, func_with_context)
            if isinstance(init_response, dict) or isinstance(
                init_response, ModelResponse
            ):  ## CACHING SCENARIO
                if isinstance(init_response, dict):
                    response = ModelResponse(**init_response)
                response = init_response
            elif asyncio.iscoroutine(init_response):
                response = await init_response
            else:
                response = init_response  # type: ignore

            if (
                custom_llm_provider == "text-completion-openai"
                or custom_llm_provider == "text-completion-codestral"
            ) and isinstance(response, TextCompletionResponse):
                response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
                    response_object=response,
                    model_response_object=litellm.ModelResponse(),
                )
        else:
            # Call the synchronous function using run_in_executor
            response = await loop.run_in_executor(None, func_with_context)  # type: ignore
        if isinstance(response, CustomStreamWrapper):
            response.set_logging_event_loop(
                loop=loop
            )  # sets the logging event loop if the user does sync streaming (e.g. on proxy for sagemaker calls)
        return response
    except Exception as e:
        custom_llm_provider = custom_llm_provider or "openai"
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )


@client
def completion(
    model: str,
    # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
    messages: List = [],
    timeout: Optional[Union[float, str, httpx.Timeout]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    response_format: Optional[Union[dict, Type[BaseModel]]] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[Union[str, dict]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    parallel_tool_calls: Optional[bool] = None,
    deployment_id=None,
    extra_headers: Optional[dict] = None,
    # soon to be deprecated params by OpenAI
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Perform a completion() using any of litellm supported llms (example gpt-4, gpt-3.5-turbo, claude-2, command-nightly)
    Parameters:
        model (str): The name of the language model to use for text completion. see all supported LLMs: https://docs.litellm.ai/docs/providers/
        messages (List): A list of message objects representing the conversation context (default is an empty list).

        OPTIONAL PARAMS
        functions (List, optional): A list of functions to apply to the conversation messages (default is an empty list).
        function_call (str, optional): The name of the function to call within the conversation (default is an empty string).
        temperature (float, optional): The temperature parameter for controlling the randomness of the output (default is 1.0).
        top_p (float, optional): The top-p parameter for nucleus sampling (default is 1.0).
        n (int, optional): The number of completions to generate (default is 1).
        stream (bool, optional): If True, return a streaming response (default is False).
        stream_options (dict, optional): A dictionary containing options for the streaming response. Only set this when you set stream: true.
        stop(string/list, optional): - Up to 4 sequences where the LLM API will stop generating further tokens.
        max_tokens (integer, optional): The maximum number of tokens in the generated completion (default is infinity).
        presence_penalty (float, optional): It is used to penalize new tokens based on their existence in the text so far.
        frequency_penalty: It is used to penalize new tokens based on their frequency in the text so far.
        logit_bias (dict, optional): Used to modify the probability of specific tokens appearing in the completion.
        user (str, optional):  A unique identifier representing your end-user. This can help the LLM provider to monitor and detect abuse.
        logprobs (bool, optional): Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message
        top_logprobs (int, optional): An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.
        metadata (dict, optional): Pass in additional metadata to tag your completion calls - eg. prompt version, details, etc.
        api_base (str, optional): Base URL for the API (default is None).
        api_version (str, optional): API version (default is None).
        api_key (str, optional): API key (default is None).
        model_list (list, optional): List of api base, version, keys
        extra_headers (dict, optional): Additional headers to include in the request.

        LITELLM Specific Params
        mock_response (str, optional): If provided, return a mock completion response for testing or debugging purposes (default is None).
        custom_llm_provider (str, optional): Used for Non-OpenAI LLMs, Example usage for bedrock, set model="amazon.titan-tg1-large" and custom_llm_provider="bedrock"
        max_retries (int, optional): The number of retries to attempt (default is 0).
    Returns:
        ModelResponse: A response object containing the generated completion and associated metadata.

    Note:
        - This function is used to perform completions() using the specified language model.
        - It supports various optional parameters for customizing the completion behavior.
        - If 'mock_response' is provided, a mock completion response is returned for testing or debugging.
    """
    ######### unpacking kwargs #####################
    args = locals()
    api_base = kwargs.get("api_base", None)
    mock_response = kwargs.get("mock_response", None)
    mock_tool_calls = kwargs.get("mock_tool_calls", None)
    force_timeout = kwargs.get("force_timeout", 600)  ## deprecated
    logger_fn = kwargs.get("logger_fn", None)
    verbose = kwargs.get("verbose", False)
    custom_llm_provider = kwargs.get("custom_llm_provider", None)
    litellm_logging_obj = kwargs.get("litellm_logging_obj", None)
    id = kwargs.get("id", None)
    metadata = kwargs.get("metadata", None)
    model_info = kwargs.get("model_info", None)
    proxy_server_request = kwargs.get("proxy_server_request", None)
    fallbacks = kwargs.get("fallbacks", None)
    headers = kwargs.get("headers", None) or extra_headers
    num_retries = kwargs.get(
        "num_retries", None
    )  ## alt. param for 'max_retries'. Use this to pass retries w/ instructor.
    max_retries = kwargs.get("max_retries", None)
    cooldown_time = kwargs.get("cooldown_time", None)
    context_window_fallback_dict = kwargs.get(
        "context_window_fallback_dict", None
    )
    organization = kwargs.get("organization", None)
    ### CUSTOM MODEL COST ###
    input_cost_per_token = kwargs.get("input_cost_per_token", None)
    output_cost_per_token = kwargs.get("output_cost_per_token", None)
    input_cost_per_second = kwargs.get("input_cost_per_second", None)
    output_cost_per_second = kwargs.get("output_cost_per_second", None)
    ### CUSTOM PROMPT TEMPLATE ###
    initial_prompt_value = kwargs.get("initial_prompt_value", None)
    roles = kwargs.get("roles", None)
    final_prompt_value = kwargs.get("final_prompt_value", None)
    bos_token = kwargs.get("bos_token", None)
    eos_token = kwargs.get("eos_token", None)
    preset_cache_key = kwargs.get("preset_cache_key", None)
    hf_model_name = kwargs.get("hf_model_name", None)
    supports_system_message = kwargs.get("supports_system_message", None)
    ### TEXT COMPLETION CALLS ###
    text_completion = kwargs.get("text_completion", False)
    atext_completion = kwargs.get("atext_completion", False)
    ### ASYNC CALLS ###
    acompletion = kwargs.get("acompletion", False)
    client = kwargs.get("client", None)
    ### Admin Controls ###
    no_log = kwargs.get("no-log", False)
    ### COPY MESSAGES ### - related issue https://github.com/BerriAI/litellm/discussions/4489
    messages = deepcopy(messages)
    ######## end of unpacking kwargs ###########
    openai_params = [
        "functions",
        "function_call",
        "temperature",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stream_options",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "request_timeout",
        "api_base",
        "api_version",
        "api_key",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "max_retries",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "extra_headers",
    ]
    litellm_params = all_litellm_params  # use the external var., used in creating cache key as well.

    default_params = openai_params + litellm_params
    non_default_params = {
        k: v for k, v in kwargs.items() if k not in default_params
    }  # model-specific params - pass them straight to the model/provider

    try:
        if base_url is not None:
            api_base = base_url
        if num_retries is not None:
            max_retries = num_retries
        logging = litellm_logging_obj
        fallbacks = fallbacks or litellm.model_fallbacks
        if fallbacks is not None:
            return completion_with_fallbacks(**args)
        if model_list is not None:
            deployments = [
                m["litellm_params"]
                for m in model_list
                if m["model_name"] == model
            ]
            return batch_completion_models(deployments=deployments, **args)
        if litellm.model_alias_map and model in litellm.model_alias_map:
            model = litellm.model_alias_map[
                model
            ]  # update the model to the actual value if an alias has been passed in
        model_response = ModelResponse()
        setattr(model_response, "usage", litellm.Usage())
        if (
            kwargs.get("azure", False) == True
        ):  # don't remove flag check, to remain backwards compatible for repos like Codium
            custom_llm_provider = "azure"
        if deployment_id != None:  # azure llms
            model = deployment_id
            custom_llm_provider = "azure"
        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            api_base,
        ) = get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            api_key=api_key,
        )
        if model_response is not None and hasattr(
            model_response, "_hidden_params"
        ):
            model_response._hidden_params[
                "custom_llm_provider"
            ] = custom_llm_provider
            model_response._hidden_params["region_name"] = kwargs.get(
                "aws_region_name", None
            )  # support region-based pricing for bedrock

        ### TIMEOUT LOGIC ###
        timeout = timeout or kwargs.get("request_timeout", 600) or 600
        # set timeout for 10 minutes by default
        if isinstance(timeout, httpx.Timeout) and not supports_httpx_timeout(
            custom_llm_provider
        ):
            timeout = timeout.read or 600  # default 10 min timeout
        elif not isinstance(timeout, httpx.Timeout):
            timeout = float(timeout)  # type: ignore

        ### REGISTER CUSTOM MODEL PRICING -- IF GIVEN ###
        if (
            input_cost_per_token is not None
            and output_cost_per_token is not None
        ):
            litellm.register_model(
                {
                    f"{custom_llm_provider}/{model}": {
                        "input_cost_per_token": input_cost_per_token,
                        "output_cost_per_token": output_cost_per_token,
                        "litellm_provider": custom_llm_provider,
                    },
                    model: {
                        "input_cost_per_token": input_cost_per_token,
                        "output_cost_per_token": output_cost_per_token,
                        "litellm_provider": custom_llm_provider,
                    },
                }
            )
        elif (
            input_cost_per_second is not None
        ):  # time based pricing just needs cost in place
            output_cost_per_second = output_cost_per_second
            litellm.register_model(
                {
                    f"{custom_llm_provider}/{model}": {
                        "input_cost_per_second": input_cost_per_second,
                        "output_cost_per_second": output_cost_per_second,
                        "litellm_provider": custom_llm_provider,
                    },
                    model: {
                        "input_cost_per_second": input_cost_per_second,
                        "output_cost_per_second": output_cost_per_second,
                        "litellm_provider": custom_llm_provider,
                    },
                }
            )
        ### BUILD CUSTOM PROMPT TEMPLATE -- IF GIVEN ###
        custom_prompt_dict = {}  # type: ignore
        if (
            initial_prompt_value
            or roles
            or final_prompt_value
            or bos_token
            or eos_token
        ):
            custom_prompt_dict = {model: {}}
            if initial_prompt_value:
                custom_prompt_dict[model][
                    "initial_prompt_value"
                ] = initial_prompt_value
            if roles:
                custom_prompt_dict[model]["roles"] = roles
            if final_prompt_value:
                custom_prompt_dict[model][
                    "final_prompt_value"
                ] = final_prompt_value
            if bos_token:
                custom_prompt_dict[model]["bos_token"] = bos_token
            if eos_token:
                custom_prompt_dict[model]["eos_token"] = eos_token

        if (
            supports_system_message is not None
            and isinstance(supports_system_message, bool)
            and supports_system_message is False
        ):
            messages = map_system_message_pt(messages=messages)
        model_api_key = get_api_key(
            llm_provider=custom_llm_provider, dynamic_api_key=api_key
        )  # get the api key from the environment if required for the model

        if dynamic_api_key is not None:
            api_key = dynamic_api_key
        # check if user passed in any of the OpenAI optional params
        optional_params = get_optional_params(
            functions=functions,
            function_call=function_call,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            # params to identify the model
            model=model,
            custom_llm_provider=custom_llm_provider,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            max_retries=max_retries,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            extra_headers=extra_headers,
            api_version=api_version,
            parallel_tool_calls=parallel_tool_calls,
            **non_default_params,
        )

        if litellm.add_function_to_prompt and optional_params.get(
            "functions_unsupported_model", None
        ):  # if user opts to add it to prompt, when API doesn't support function calling
            functions_unsupported_model = optional_params.pop(
                "functions_unsupported_model"
            )
            messages = function_call_prompt(
                messages=messages, functions=functions_unsupported_model
            )

        # For logging - save the values of the litellm-specific params passed in
        litellm_params = get_litellm_params(
            acompletion=acompletion,
            api_key=api_key,
            force_timeout=force_timeout,
            logger_fn=logger_fn,
            verbose=verbose,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            litellm_call_id=kwargs.get("litellm_call_id", None),
            model_alias_map=litellm.model_alias_map,
            completion_call_id=id,
            metadata=metadata,
            model_info=model_info,
            proxy_server_request=proxy_server_request,
            preset_cache_key=preset_cache_key,
            no_log=no_log,
            input_cost_per_second=input_cost_per_second,
            input_cost_per_token=input_cost_per_token,
            output_cost_per_second=output_cost_per_second,
            output_cost_per_token=output_cost_per_token,
            cooldown_time=cooldown_time,
            text_completion=kwargs.get("text_completion"),
            azure_ad_token_provider=kwargs.get("azure_ad_token_provider"),
            user_continue_message=kwargs.get("user_continue_message"),
        )
        logging.update_environment_variables(
            model=model,
            user=user,
            optional_params=optional_params,
            litellm_params=litellm_params,
            custom_llm_provider=custom_llm_provider,
        )
        if mock_response or mock_tool_calls:
            return mock_completion(
                model,
                messages,
                stream=stream,
                n=n,
                mock_response=mock_response,
                mock_tool_calls=mock_tool_calls,
                logging=logging,
                acompletion=acompletion,
                mock_delay=kwargs.get("mock_delay", None),
                custom_llm_provider=custom_llm_provider,
            )

        if custom_llm_provider == "azure":
            # azure configs
            ## check dynamic params ##
            dynamic_params = False
            if client is not None and (
                isinstance(client, openai.AzureOpenAI)
                or isinstance(client, openai.AsyncAzureOpenAI)
            ):
                dynamic_params = _check_dynamic_azure_params(
                    azure_client_params={"api_version": api_version},
                    azure_client=client,
                )

            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = (
                api_base or litellm.api_base or get_secret("AZURE_API_BASE")
            )

            api_version = (
                api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
                or litellm.AZURE_DEFAULT_API_VERSION
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.get("extra_body", {}).pop(
                "azure_ad_token", None
            ) or get_secret("AZURE_AD_TOKEN")

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            response = azure_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                dynamic_params=dynamic_params,
                azure_ad_token=azure_ad_token,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,  # type: ignore
                client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
            )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )

        elif custom_llm_provider == "notdiamond":
            notdiamond_key = (
                api_key
                or notdiamond_key
                or get_secret("NOTDIAMOND_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("NOTDIAMOND_API_BASE")
                or "https://api.notdiamond.ai/v2/optimizer/modelSelect"
            )

            # since notdiamond.completion() internally calls other models' completion functions
            # streaming does not need to be handled separately
            response = notdiamond_completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=notdiamond_key,
                logging_obj=logging,
            )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )

            response = response

        elif custom_llm_provider == "azure_text":
            # azure configs
            api_type = get_secret("AZURE_API_TYPE") or "azure"

            api_base = (
                api_base or litellm.api_base or get_secret("AZURE_API_BASE")
            )

            api_version = (
                api_version
                or litellm.api_version
                or get_secret("AZURE_API_VERSION")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.azure_key
                or get_secret("AZURE_OPENAI_API_KEY")
                or get_secret("AZURE_API_KEY")
            )

            azure_ad_token = optional_params.get("extra_body", {}).pop(
                "azure_ad_token", None
            ) or get_secret("AZURE_AD_TOKEN")

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.AzureOpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > azure_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            response = azure_text_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                api_type=api_type,
                azure_ad_token=azure_ad_token,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,
                client=client,  # pass AsyncAzureOpenAI, AzureOpenAI client
            )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={
                        "headers": headers,
                        "api_version": api_version,
                        "api_base": api_base,
                    },
                )
        elif custom_llm_provider == "azure_ai":
            api_base = (
                api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or get_secret("AZURE_AI_API_BASE")
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or get_secret("AZURE_AI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## FOR COHERE
            if "command-r" in model:  # make sure tool call in messages are str
                messages = stringify_json_tool_call_content(messages=messages)

            ## COMPLETION CALL
            try:
                response = openai_chat_completions.completion(
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,
                    client=client,  # pass AsyncOpenAI, OpenAI client
                    organization=organization,
                    custom_llm_provider=custom_llm_provider,
                    drop_params=non_default_params.get("drop_params"),
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif (
            custom_llm_provider == "text-completion-openai"
            or "ft:babbage-002" in model
            or "ft:davinci-002"
            in model  # support for finetuned completion models
            or custom_llm_provider
            in litellm.openai_text_completion_compatible_providers
            and kwargs.get("text_completion") is True
        ):
            openai.api_type = "openai"

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )

            openai.api_version = None
            # set API KEY

            api_key = (
                api_key
                or litellm.api_key
                or litellm.openai_key
                or get_secret("OPENAI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.OpenAITextCompletionConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_text_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v
            if litellm.organization:
                openai.organization = litellm.organization

            if (
                len(messages) > 0
                and "content" in messages[0]
                and type(messages[0]["content"]) == list
            ):
                # text-davinci-003 can accept a string or array, if it's an array, assume the array is set in messages[0]['content']
                # https://platform.openai.com/docs/api-reference/completions/create
                prompt = messages[0]["content"]
            else:
                prompt = " ".join([message["content"] for message in messages])  # type: ignore

            ## COMPLETION CALL
            _response = openai_text_completions.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                client=client,  # pass AsyncOpenAI, OpenAI client
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,  # type: ignore
            )

            if (
                optional_params.get("stream", False) == False
                and acompletion == False
                and text_completion == False
            ):
                # convert to chat completion response
                _response = litellm.OpenAITextCompletionConfig().convert_to_chat_model_response_object(
                    response_object=_response,
                    model_response_object=model_response,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=_response,
                    additional_args={"headers": headers},
                )
            response = _response

        elif (
            model in litellm.open_ai_chat_completion_models
            or custom_llm_provider == "custom_openai"
            or custom_llm_provider == "deepinfra"
            or custom_llm_provider == "perplexity"
            or custom_llm_provider == "groq"
            or custom_llm_provider == "nvidia_nim"
            or custom_llm_provider == "cerebras"
            or custom_llm_provider == "ai21_chat"
            or custom_llm_provider == "volcengine"
            or custom_llm_provider == "codestral"
            or custom_llm_provider == "deepseek"
            or custom_llm_provider == "anyscale"
            or custom_llm_provider == "mistral"
            or custom_llm_provider == "openai"
            or custom_llm_provider == "together_ai"
            or custom_llm_provider in litellm.openai_compatible_providers
            or "ft:gpt-3.5-turbo" in model  # finetune gpt-3.5-turbo
        ):  # allow user to make an openai call with a custom base
            # note: if a user sets a custom base - we should ensure this works
            # allow for the setting of dynamic and stateful api-bases
            api_base = (
                api_base  # for deepinfra/perplexity/anyscale/groq/friendliai we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or get_secret("OPENAI_API_BASE")
                or "https://api.openai.com/v1"
            )
            openai.organization = (
                organization
                or litellm.organization
                or get_secret("OPENAI_ORGANIZATION")
                or None  # default - https://github.com/openai/openai-python/blob/284c1799070c723c6a553337134148a7ab088dd8/openai/util.py#L105
            )
            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for deepinfra/perplexity/anyscale/friendliai we check in get_llm_provider and pass in the api key from there
                or litellm.openai_key
                or get_secret("OPENAI_API_KEY")
            )

            headers = headers or litellm.headers

            ## LOAD CONFIG - if set
            config = litellm.OpenAIConfig.get_config()
            for k, v in config.items():
                if (
                    k not in optional_params
                ):  # completion(top_k=3) > openai_config(top_k=3) <- allows for dynamic variables to be passed in
                    optional_params[k] = v

            ## COMPLETION CALL
            try:
                response = openai_chat_completions.completion(
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,
                    client=client,  # pass AsyncOpenAI, OpenAI client
                    organization=organization,
                    custom_llm_provider=custom_llm_provider,
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif (
            "replicate" in model
            or custom_llm_provider == "replicate"
            or model in litellm.replicate_models
        ):
            # Setting the relevant API KEY for replicate, replicate defaults to using os.environ.get("REPLICATE_API_TOKEN")
            replicate_key = None
            replicate_key = (
                api_key
                or litellm.replicate_key
                or litellm.api_key
                or get_secret("REPLICATE_API_KEY")
                or get_secret("REPLICATE_API_TOKEN")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("REPLICATE_API_BASE")
                or "https://api.replicate.com/v1"
            )

            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )

            model_response = replicate.completion(  # type: ignore
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=replicate_key,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
                acompletion=acompletion,
            )

            if optional_params.get("stream", False) == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=replicate_key,
                    original_response=model_response,
                )

            response = model_response
        elif (
            "clarifai" in model
            or custom_llm_provider == "clarifai"
            or model in litellm.clarifai_models
        ):
            clarifai_key = None
            clarifai_key = (
                api_key
                or litellm.clarifai_key
                or litellm.api_key
                or get_secret("CLARIFAI_API_KEY")
                or get_secret("CLARIFAI_API_TOKEN")
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("CLARIFAI_API_BASE")
                or "https://api.clarifai.com/v2"
            )

            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            model_response = clarifai.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                acompletion=acompletion,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=clarifai_key,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=model_response,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=clarifai_key,
                    original_response=model_response,
                )
            response = model_response

        elif custom_llm_provider == "anthropic":
            api_key = (
                api_key
                or litellm.anthropic_key
                or litellm.api_key
                or os.environ.get("ANTHROPIC_API_KEY")
            )
            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )

            if (model == "claude-2") or (model == "claude-instant-1"):
                # call anthropic /completion, only use this route for claude-2, claude-instant-1
                api_base = (
                    api_base
                    or litellm.api_base
                    or get_secret("ANTHROPIC_API_BASE")
                    or get_secret("ANTHROPIC_BASE_URL")
                    or "https://api.anthropic.com/v1/complete"
                )

                if api_base is not None and not api_base.endswith(
                    "/v1/complete"
                ):
                    api_base += "/v1/complete"

                response = anthropic_text_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=api_key,
                    logging_obj=logging,
                    headers=headers,
                )
            else:
                # call /messages
                # default route for all anthropic models
                api_base = (
                    api_base
                    or litellm.api_base
                    or get_secret("ANTHROPIC_API_BASE")
                    or get_secret("ANTHROPIC_BASE_URL")
                    or "https://api.anthropic.com/v1/messages"
                )

                if api_base is not None and not api_base.endswith(
                    "/v1/messages"
                ):
                    api_base += "/v1/messages"

                response = anthropic_chat_completions.completion(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    acompletion=acompletion,
                    custom_prompt_dict=litellm.custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,  # for calculating input/output tokens
                    api_key=api_key,
                    logging_obj=logging,
                    headers=headers,
                    timeout=timeout,
                    client=client,
                )
            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )
            response = response
        elif custom_llm_provider == "nlp_cloud":
            nlp_cloud_key = (
                api_key
                or litellm.nlp_cloud_key
                or get_secret("NLP_CLOUD_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("NLP_CLOUD_API_BASE")
                or "https://api.nlpcloud.io/v1/gpu/"
            )

            response = nlp_cloud.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=nlp_cloud_key,
                logging_obj=logging,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    response,
                    model,
                    custom_llm_provider="nlp_cloud",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )

            response = response
        elif custom_llm_provider == "aleph_alpha":
            aleph_alpha_key = (
                api_key
                or litellm.aleph_alpha_key
                or get_secret("ALEPH_ALPHA_API_KEY")
                or get_secret("ALEPHALPHA_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("ALEPH_ALPHA_API_BASE")
                or "https://api.aleph-alpha.com/complete"
            )

            model_response = aleph_alpha.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                default_max_tokens_to_sample=litellm.max_tokens,
                api_key=aleph_alpha_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="aleph_alpha",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "cohere":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.ai/v1/generate"
            )

            headers = headers or litellm.headers or {}
            if headers is None:
                headers = {}

            if extra_headers is not None:
                headers.update(extra_headers)

            model_response = cohere_completion.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                headers=headers,
                api_key=cohere_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="cohere",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "cohere_chat":
            cohere_key = (
                api_key
                or litellm.cohere_key
                or get_secret("COHERE_API_KEY")
                or get_secret("CO_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("COHERE_API_BASE")
                or "https://api.cohere.ai/v1/chat"
            )

            headers = headers or litellm.headers or {}
            if headers is None:
                headers = {}

            if extra_headers is not None:
                headers.update(extra_headers)

            model_response = cohere_chat.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=cohere_key,
                logging_obj=logging,  # model call logging done inside the class as we make need to modify I/O to fit aleph alpha's requirements
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="cohere_chat",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "maritalk":
            maritalk_key = (
                api_key
                or litellm.maritalk_key
                or get_secret("MARITALK_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("MARITALK_API_BASE")
                or "https://chat.maritaca.ai/api/chat/inference"
            )

            model_response = maritalk.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=maritalk_key,
                logging_obj=logging,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="maritalk",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "huggingface":
            custom_llm_provider = "huggingface"
            huggingface_key = (
                api_key
                or litellm.huggingface_key
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_API_KEY")
                or litellm.api_key
            )
            hf_headers = headers or litellm.headers

            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            model_response = huggingface.completion(
                model=model,
                messages=messages,
                api_base=api_base,  # type: ignore
                headers=hf_headers,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=huggingface_key,
                acompletion=acompletion,
                logging_obj=logging,
                custom_prompt_dict=custom_prompt_dict,
                timeout=timeout,  # type: ignore
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
                and acompletion is False
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="huggingface",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "oobabooga":
            custom_llm_provider = "oobabooga"
            model_response = oobabooga.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                api_base=api_base,  # type: ignore
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=None,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="oobabooga",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "databricks":
            api_base = (
                api_base  # for databricks we check in get_llm_provider and pass in the api base from there
                or litellm.api_base
                or os.getenv("DATABRICKS_API_BASE")
            )

            # set API KEY
            api_key = (
                api_key
                or litellm.api_key  # for databricks we check in get_llm_provider and pass in the api key from there
                or litellm.databricks_key
                or get_secret("DATABRICKS_API_KEY")
            )

            headers = headers or litellm.headers

            ## COMPLETION CALL
            try:
                response = databricks_chat_completions.completion(
                    model=model,
                    messages=messages,
                    headers=headers,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    api_key=api_key,
                    api_base=api_base,
                    acompletion=acompletion,
                    logging_obj=logging,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,  # type: ignore
                    custom_prompt_dict=custom_prompt_dict,
                    client=client,  # pass AsyncOpenAI, OpenAI client
                    encoding=encoding,
                    custom_llm_provider="databricks",
                )
            except Exception as e:
                ## LOGGING - log the original exception returned
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=str(e),
                    additional_args={"headers": headers},
                )
                raise e

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                    additional_args={"headers": headers},
                )
        elif custom_llm_provider == "openrouter":
            api_base = (
                api_base or litellm.api_base or "https://openrouter.ai/api/v1"
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.openrouter_key
                or get_secret("OPENROUTER_API_KEY")
                or get_secret("OR_API_KEY")
            )

            openrouter_site_url = (
                get_secret("OR_SITE_URL") or "https://litellm.ai"
            )
            openrouter_app_name = get_secret("OR_APP_NAME") or "liteLLM"

            openrouter_headers = {
                "HTTP-Referer": openrouter_site_url,
                "X-Title": openrouter_app_name,
            }

            _headers = headers or litellm.headers
            if _headers:
                openrouter_headers.update(_headers)

            headers = openrouter_headers

            ## Load Config
            config = openrouter.OpenrouterConfig.get_config()
            for k, v in config.items():
                if k == "extra_body":
                    # we use openai 'extra_body' to pass openrouter specific params - transforms, route, models
                    if "extra_body" in optional_params:
                        optional_params[k].update(v)
                    else:
                        optional_params[k] = v
                elif k not in optional_params:
                    optional_params[k] = v

            data = {"model": model, "messages": messages, **optional_params}

            ## COMPLETION CALL
            response = openai_chat_completions.completion(
                model=model,
                messages=messages,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,  # type: ignore
                custom_llm_provider="openrouter",
            )
            ## LOGGING
            logging.post_call(
                input=messages,
                api_key=openai.api_key,
                original_response=response,
            )
        elif (
            custom_llm_provider == "together_ai"
            or ("togethercomputer" in model)
            or (model in litellm.together_ai_models)
        ):
            """
            Deprecated. We now do together ai calls via the openai client - https://docs.together.ai/docs/openai-api-compatibility
            """
            pass
        elif custom_llm_provider == "palm":
            palm_api_key = (
                api_key or get_secret("PALM_API_KEY") or litellm.api_key
            )

            # palm does not support streaming as yet :(
            model_response = palm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=palm_api_key,
                logging_obj=logging,
            )
            # fake palm streaming
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # fake streaming for palm
                resp_string = model_response["choices"][0]["message"][
                    "content"
                ]
                response = CustomStreamWrapper(
                    resp_string,
                    model,
                    custom_llm_provider="palm",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif (
            custom_llm_provider == "vertex_ai_beta"
            or custom_llm_provider == "gemini"
        ):
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
            )

            gemini_api_key = (
                api_key
                or get_secret("GEMINI_API_KEY")
                or get_secret(
                    "PALM_API_KEY"
                )  # older palm api key should also work
                or litellm.api_key
            )

            new_params = deepcopy(optional_params)
            response = vertex_chat_completion.completion(  # type: ignore
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=new_params,
                litellm_params=litellm_params,  # type: ignore
                logger_fn=logger_fn,
                encoding=encoding,
                vertex_location=vertex_ai_location,
                vertex_project=vertex_ai_project,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                logging_obj=logging,
                acompletion=acompletion,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                client=client,
                api_base=api_base,
                extra_headers=extra_headers,
            )

        elif custom_llm_provider == "vertex_ai":
            vertex_ai_project = (
                optional_params.pop("vertex_project", None)
                or optional_params.pop("vertex_ai_project", None)
                or litellm.vertex_project
                or get_secret("VERTEXAI_PROJECT")
            )
            vertex_ai_location = (
                optional_params.pop("vertex_location", None)
                or optional_params.pop("vertex_ai_location", None)
                or litellm.vertex_location
                or get_secret("VERTEXAI_LOCATION")
            )
            vertex_credentials = (
                optional_params.pop("vertex_credentials", None)
                or optional_params.pop("vertex_ai_credentials", None)
                or get_secret("VERTEXAI_CREDENTIALS")
            )

            new_params = deepcopy(optional_params)
            if "claude-3" in model:
                model_response = vertex_ai_anthropic.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    logging_obj=logging,
                    acompletion=acompletion,
                    headers=headers,
                    custom_prompt_dict=custom_prompt_dict,
                    timeout=timeout,
                    client=client,
                )
            elif (
                model.startswith("meta/")
                or model.startswith("mistral")
                or model.startswith("codestral")
                or model.startswith("jamba")
            ):
                model_response = (
                    vertex_partner_models_chat_completion.completion(
                        model=model,
                        messages=messages,
                        model_response=model_response,
                        print_verbose=print_verbose,
                        optional_params=new_params,
                        litellm_params=litellm_params,  # type: ignore
                        logger_fn=logger_fn,
                        encoding=encoding,
                        vertex_location=vertex_ai_location,
                        vertex_project=vertex_ai_project,
                        vertex_credentials=vertex_credentials,
                        logging_obj=logging,
                        acompletion=acompletion,
                        headers=headers,
                        custom_prompt_dict=custom_prompt_dict,
                        timeout=timeout,
                        client=client,
                    )
                )
            elif "gemini" in model:
                model_response = vertex_chat_completion.completion(  # type: ignore
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,  # type: ignore
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    gemini_api_key=None,
                    logging_obj=logging,
                    acompletion=acompletion,
                    timeout=timeout,
                    custom_llm_provider=custom_llm_provider,
                    client=client,
                    api_base=api_base,
                    extra_headers=extra_headers,
                )
            else:
                model_response = vertex_ai_non_gemini.completion(
                    model=model,
                    messages=messages,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=new_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    vertex_location=vertex_ai_location,
                    vertex_project=vertex_ai_project,
                    vertex_credentials=vertex_credentials,
                    logging_obj=logging,
                    acompletion=acompletion,
                )

                if (
                    "stream" in optional_params
                    and optional_params["stream"] is True
                    and acompletion is False
                ):
                    response = CustomStreamWrapper(
                        model_response,
                        model,
                        custom_llm_provider="vertex_ai",
                        logging_obj=logging,
                    )
                    return response
            response = model_response
        elif custom_llm_provider == "predibase":
            tenant_id = (
                optional_params.pop("tenant_id", None)
                or optional_params.pop("predibase_tenant_id", None)
                or litellm.predibase_tenant_id
                or get_secret("PREDIBASE_TENANT_ID")
            )

            api_base = (
                api_base
                or optional_params.pop("api_base", None)
                or optional_params.pop("base_url", None)
                or litellm.api_base
                or get_secret("PREDIBASE_API_BASE")
            )

            api_key = (
                api_key
                or litellm.api_key
                or litellm.predibase_key
                or get_secret("PREDIBASE_API_KEY")
            )

            _model_response = predibase_chat_completions.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                api_key=api_key,
                tenant_id=tenant_id,
                timeout=timeout,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] is True
                and acompletion is False
            ):
                return _model_response
            response = _model_response
        elif custom_llm_provider == "text-completion-codestral":
            api_base = (
                api_base
                or optional_params.pop("api_base", None)
                or optional_params.pop("base_url", None)
                or litellm.api_base
                or "https://codestral.mistral.ai/v1/fim/completions"
            )

            api_key = (
                api_key or litellm.api_key or get_secret("CODESTRAL_API_KEY")
            )

            text_completion_model_response = litellm.TextCompletionResponse(
                stream=stream
            )

            _model_response = codestral_text_completions.completion(  # type: ignore
                model=model,
                messages=messages,
                model_response=text_completion_model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                api_base=api_base,
                custom_prompt_dict=custom_prompt_dict,
                api_key=api_key,
                timeout=timeout,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] is True
                and acompletion is False
            ):
                return _model_response
            response = _model_response
        elif custom_llm_provider == "ai21":
            custom_llm_provider = "ai21"
            ai21_key = (
                api_key
                or litellm.ai21_key
                or os.environ.get("AI21_API_KEY")
                or litellm.api_key
            )

            api_base = (
                api_base
                or litellm.api_base
                or get_secret("AI21_API_BASE")
                or "https://api.ai21.com/studio/v1/"
            )

            model_response = ai21.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=ai21_key,
                logging_obj=logging,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="ai21",
                    logging_obj=logging,
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif (
            custom_llm_provider == "sagemaker"
            or custom_llm_provider == "sagemaker_chat"
        ):
            # boto3 reads keys from .env
            model_response = sagemaker_llm.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                custom_prompt_dict=custom_prompt_dict,
                hf_model_name=hf_model_name,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                acompletion=acompletion,
                use_messages_api=(
                    True if custom_llm_provider == "sagemaker_chat" else False
                ),
            )
            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=model_response,
                )

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "bedrock":
            # boto3 reads keys from .env
            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )

            if "aws_bedrock_client" in optional_params:
                verbose_logger.warning(
                    "'aws_bedrock_client' is a deprecated param. Please move to another auth method - https://docs.litellm.ai/docs/providers/bedrock#boto3---authentication."
                )
                # Extract credentials for legacy boto3 client and pass thru to httpx
                aws_bedrock_client = optional_params.pop("aws_bedrock_client")
                creds = (
                    aws_bedrock_client._get_credentials().get_frozen_credentials()
                )

                if creds.access_key:
                    optional_params["aws_access_key_id"] = creds.access_key
                if creds.secret_key:
                    optional_params["aws_secret_access_key"] = creds.secret_key
                if creds.token:
                    optional_params["aws_session_token"] = creds.token
                if (
                    "aws_region_name" not in optional_params
                    or optional_params["aws_region_name"] is None
                ):
                    optional_params[
                        "aws_region_name"
                    ] = aws_bedrock_client.meta.region_name

            if model in litellm.BEDROCK_CONVERSE_MODELS:
                response = bedrock_converse_chat_completion.completion(
                    model=model,
                    messages=messages,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,  # type: ignore
                    logger_fn=logger_fn,
                    encoding=encoding,
                    logging_obj=logging,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    acompletion=acompletion,
                    client=client,
                    api_base=api_base,
                )
            else:
                response = bedrock_chat_completion.completion(
                    model=model,
                    messages=messages,
                    custom_prompt_dict=custom_prompt_dict,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    encoding=encoding,
                    logging_obj=logging,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    acompletion=acompletion,
                    client=client,
                    api_base=api_base,
                )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=response,
                )

            ## RESPONSE OBJECT
            response = response
        elif custom_llm_provider == "watsonx":
            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            response = watsonxai.completion(
                model=model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,  # type: ignore
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
                timeout=timeout,  # type: ignore
                acompletion=acompletion,
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
                and not isinstance(response, CustomStreamWrapper)
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    iter(response),
                    model,
                    custom_llm_provider="watsonx",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False):
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=None,
                    original_response=response,
                )
            ## RESPONSE OBJECT
            response = response
        elif custom_llm_provider == "vllm":
            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            model_response = vllm.completion(
                model=model,
                messages=messages,
                custom_prompt_dict=custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )

            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):  ## [BETA]
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="vllm",
                    logging_obj=logging,
                )
                return response

            ## RESPONSE OBJECT
            response = model_response
        elif custom_llm_provider == "ollama":
            api_base = (
                litellm.api_base
                or api_base
                or get_secret("OLLAMA_API_BASE")
                or "http://localhost:11434"
            )
            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            if model in custom_prompt_dict:
                # check if the model has a registered custom prompt
                model_prompt_details = custom_prompt_dict[model]
                prompt = custom_prompt(
                    role_dict=model_prompt_details["roles"],
                    initial_prompt_value=model_prompt_details[
                        "initial_prompt_value"
                    ],
                    final_prompt_value=model_prompt_details[
                        "final_prompt_value"
                    ],
                    messages=messages,
                )
            else:
                prompt = prompt_factory(
                    model=model,
                    messages=messages,
                    custom_llm_provider=custom_llm_provider,
                )
                if isinstance(prompt, dict):
                    # for multimode models - ollama/llava prompt_factory returns a dict {
                    #     "prompt": prompt,
                    #     "images": images
                    # }
                    prompt, images = prompt["prompt"], prompt["images"]
                    optional_params["images"] = images

            ## LOGGING
            generator = ollama.get_ollama_response(
                api_base=api_base,
                model=model,
                prompt=prompt,
                optional_params=optional_params,
                logging_obj=logging,
                acompletion=acompletion,
                model_response=model_response,
                encoding=encoding,
            )
            if (
                acompletion is True
                or optional_params.get("stream", False) == True
            ):
                return generator

            response = generator
        elif custom_llm_provider == "ollama_chat":
            api_base = (
                litellm.api_base
                or api_base
                or get_secret("OLLAMA_API_BASE")
                or "http://localhost:11434"
            )

            api_key = (
                api_key
                or litellm.ollama_key
                or os.environ.get("OLLAMA_API_KEY")
                or litellm.api_key
            )
            ## LOGGING
            generator = ollama_chat.get_ollama_response(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                optional_params=optional_params,
                logging_obj=logging,
                acompletion=acompletion,
                model_response=model_response,
                encoding=encoding,
            )
            if (
                acompletion is True
                or optional_params.get("stream", False) is True
            ):
                return generator

            response = generator

        elif custom_llm_provider == "triton":
            api_base = litellm.api_base or api_base
            model_response = triton_chat_completions.completion(
                api_base=api_base,
                timeout=timeout,  # type: ignore
                model=model,
                messages=messages,
                model_response=model_response,
                optional_params=optional_params,
                logging_obj=logging,
                stream=stream,
                acompletion=acompletion,
            )

            ## RESPONSE OBJECT
            response = model_response
            return response

        elif custom_llm_provider == "cloudflare":
            api_key = (
                api_key
                or litellm.cloudflare_api_key
                or litellm.api_key
                or get_secret("CLOUDFLARE_API_KEY")
            )
            account_id = get_secret("CLOUDFLARE_ACCOUNT_ID")
            api_base = (
                api_base
                or litellm.api_base
                or get_secret("CLOUDFLARE_API_BASE")
                or f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
            )

            custom_prompt_dict = (
                custom_prompt_dict or litellm.custom_prompt_dict
            )
            response = cloudflare.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                custom_prompt_dict=litellm.custom_prompt_dict,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,  # for calculating input/output tokens
                api_key=api_key,
                logging_obj=logging,
            )
            if (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    response,
                    model,
                    custom_llm_provider="cloudflare",
                    logging_obj=logging,
                )

            if optional_params.get("stream", False) or acompletion == True:
                ## LOGGING
                logging.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=response,
                )
            response = response
        elif (
            custom_llm_provider == "baseten"
            or litellm.api_base == "https://app.baseten.co"
        ):
            custom_llm_provider = "baseten"
            baseten_key = (
                api_key
                or litellm.baseten_key
                or os.environ.get("BASETEN_API_KEY")
                or litellm.api_key
            )

            model_response = baseten.completion(
                model=model,
                messages=messages,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                api_key=baseten_key,
                logging_obj=logging,
            )
            if inspect.isgenerator(model_response) or (
                "stream" in optional_params
                and optional_params["stream"] == True
            ):
                # don't try to access stream object,
                response = CustomStreamWrapper(
                    model_response,
                    model,
                    custom_llm_provider="baseten",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "petals" or model in litellm.petals_models:
            api_base = api_base or litellm.api_base

            custom_llm_provider = "petals"
            stream = optional_params.pop("stream", False)
            model_response = petals.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                encoding=encoding,
                logging_obj=logging,
            )
            if stream == True:  ## [BETA]
                # Fake streaming for petals
                resp_string = model_response["choices"][0]["message"][
                    "content"
                ]
                response = CustomStreamWrapper(
                    resp_string,
                    model,
                    custom_llm_provider="petals",
                    logging_obj=logging,
                )
                return response
            response = model_response
        elif custom_llm_provider == "custom":
            import requests

            url = litellm.api_base or api_base or ""
            if url == None or url == "":
                raise ValueError(
                    "api_base not set. Set api_base or litellm.api_base for custom endpoints"
                )

            """
            assume input to custom LLM api bases follow this format:
            resp = requests.post(
                api_base,
                json={
                    'model': 'meta-llama/Llama-2-13b-hf', # model name
                    'params': {
                        'prompt': ["The capital of France is P"],
                        'max_tokens': 32,
                        'temperature': 0.7,
                        'top_p': 1.0,
                        'top_k': 40,
                    }
                }
            )

            """
            prompt = " ".join([message["content"] for message in messages])  # type: ignore
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "params": {
                        "prompt": [prompt],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": kwargs.get("top_k", 40),
                    },
                },
                verify=litellm.ssl_verify,
            )
            response_json = resp.json()
            """
            assume all responses from custom api_bases of this format:
            {
                'data': [
                    {
                        'prompt': 'The capital of France is P',
                        'output': ['The capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France is PARIS.\nThe capital of France'],
                        'params': {'temperature': 0.7, 'top_k': 40, 'top_p': 1}}],
                        'message': 'ok'
                    }
                ]
            }
            """
            string_response = response_json["data"][0]["output"][0]
            ## RESPONSE OBJECT
            model_response.choices[0].message.content = string_response  # type: ignore
            model_response.created = int(time.time())
            model_response.model = model
            response = model_response
        elif (
            custom_llm_provider in litellm._custom_providers
        ):  # Assume custom LLM provider
            # Get the Custom Handler
            custom_handler: Optional[CustomLLM] = None
            for item in litellm.custom_provider_map:
                if item["provider"] == custom_llm_provider:
                    custom_handler = item["custom_handler"]

            if custom_handler is None:
                raise ValueError(
                    f"Unable to map your input to a model. Check your input - {args}"
                )

            ## ROUTE LLM CALL ##
            handler_fn = custom_chat_llm_router(
                async_fn=acompletion, stream=stream, custom_llm=custom_handler
            )

            headers = headers or litellm.headers

            ## CALL FUNCTION
            response = handler_fn(
                model=model,
                messages=messages,
                headers=headers,
                model_response=model_response,
                print_verbose=print_verbose,
                api_key=api_key,
                api_base=api_base,
                acompletion=acompletion,
                logging_obj=logging,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,  # type: ignore
                custom_prompt_dict=custom_prompt_dict,
                client=client,  # pass AsyncOpenAI, OpenAI client
                encoding=encoding,
            )
            if stream is True:
                return CustomStreamWrapper(
                    completion_stream=response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    logging_obj=logging,
                )

        else:
            raise ValueError(
                f"Unable to map your input to a model. Check your input - {args}"
            )
        return response
    except Exception as e:
        ## Map to OpenAI Exception
        raise exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=args,
            extra_kwargs=kwargs,
        )
