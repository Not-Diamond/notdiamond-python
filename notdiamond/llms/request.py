import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import aiohttp
import requests

from notdiamond import settings
from notdiamond._utils import convert_tool_to_openai_function
from notdiamond.llms.config import LLMConfig
from notdiamond.metrics.metric import Metric
from notdiamond.types import ModelSelectRequestPayload

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def model_select_prepare(
    messages: List[Dict[str, str]],
    llm_configs: List[LLMConfig],
    metric: Metric,
    notdiamond_api_key: str,
    max_model_depth: int,
    hash_content: bool,
    tradeoff: Optional[str] = None,
    preference_id: Optional[str] = None,
    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = [],
):
    """
    This is the core method for the model_select endpoint.
    It returns the best fitting LLM to call and a session ID that can be used for feedback.

    Parameters:
        messages (List[Dict[str, str]]): list of messages to be used for the LLM call
        llm_configs (List[LLMConfig]): a list of available LLMs that the router can decide from
        metric (Metric): metric based off which the router makes the decision. As of now only 'accuracy' supported.
        notdiamond_api_key (str): API key generated via the NotDiamond dashboard.
        max_model_depth (int): if your top recommended model is down, specify up to which depth of routing you're willing to go.
        hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
        tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                            for the router to determine the best LLM for a given query.
        preference_id (Optional[str], optional): The ID of the router preference that was configured via the Dashboard.
                                                    Defaults to None.
        async_mode (bool, optional): whether to run the request in async mode. Defaults to False.

    Returns:
        tuple(url, payload, headers): returns data to be used for the API call of modelSelect
    """

    url = f"{settings.ND_BASE_URL}/v2/modelRouter/modelSelect"
    tools_dict = get_tools_in_openai_format(tools)

    payload: ModelSelectRequestPayload = {
        "messages": messages,
        "llm_providers": [
            llm_provider.prepare_for_request() for llm_provider in llm_configs
        ],
        "metric": metric.metric,
        "max_model_depth": max_model_depth,
        "hash_content": hash_content,
    }

    if tools_dict:
        payload["tools"] = tools_dict
    if tradeoff is not None:
        payload["tradeoff"] = tradeoff
    if preference_id is not None:
        payload["preference_id"] = preference_id

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {notdiamond_api_key}",
        "User-Agent": f"Python-SDK/{settings.VERSION}",
    }

    return url, payload, headers


def get_tools_in_openai_format(
    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]],
):
    """
    This function converts the tools list into the format that OpenAI expects.
    Does this by using langchains Model that automatically creates the dictionary on bind_tools

    Parameters:
        tools (Optional[Sequence[Union[Dict[str, Any], Callable]]]): list of tools to be converted

    Returns:
        dict: dictionary of tools in the format that OpenAI expects
    """
    if tools:
        return [
            {
                "type": "function",
                "function": convert_tool_to_openai_function(tool),
            }
            for tool in tools
        ]

    return None


def model_select_parse(response_code, response_json, llm_configs):
    if response_code == 200:
        providers = response_json["providers"]
        session_id = response_json["session_id"]

        top_provider = providers[0]

        best_llm = list(
            filter(
                lambda x: (x.model == top_provider["model"])
                & (x.provider == top_provider["provider"]),
                llm_configs,
            )
        )[0]
        return best_llm, session_id

    error_message = response_json["detail"]
    LOGGER.error(f"API error: {response_code}. {error_message}")
    return None, "NO-SESSION-ID"


def model_select(
    messages: List[Dict[str, str]],
    llm_configs: List[LLMConfig],
    metric: Metric,
    notdiamond_api_key: str,
    max_model_depth: int,
    hash_content: bool,
    tradeoff: Optional[str] = None,
    preference_id: Optional[str] = None,
    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = [],
    timeout: Optional[int] = 5,
):
    """
    This endpoint receives the prompt and routing settings, and makes a call to the NotDiamond API.
    It returns the best fitting LLM to call and a session ID that can be used for feedback.

    Parameters:
        messages (List[Dict[str, str]]): list of messages to be used for the LLM call
        llm_configs (List[LLMConfig]): a list of available LLMs that the router can decide from
        metric (Metric): metric based off which the router makes the decision. As of now only 'accuracy' supported.
        notdiamond_api_key (str): API key generated via the NotDiamond dashboard.
        max_model_depth (int): if your top recommended model is down, specify up to which depth of routing you're willing to go.
        hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
        tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                            for the router to determine the best LLM for a given query.
        preference_id (Optional[str], optional): The ID of the router preference that was configured via the Dashboard.
                                                    Defaults to None.
        timeout (int, optional): timeout for the request. Defaults to 5.

    Returns:
        tuple(LLMConfig, string): returns a tuple of the chosen LLMConfig to call and a session ID string.
                                        In case of an error the LLM defaults to None and the session ID defaults
                                        to 'NO-SESSION-ID'.
    """
    url, payload, headers = model_select_prepare(
        messages=messages,
        llm_configs=llm_configs,
        metric=metric,
        notdiamond_api_key=notdiamond_api_key,
        max_model_depth=max_model_depth,
        hash_content=hash_content,
        tradeoff=tradeoff,
        preference_id=preference_id,
        tools=tools,
    )

    try:
        response = requests.post(
            url, data=json.dumps(payload), headers=headers, timeout=timeout
        )
        response_code = response.status_code
        response_json = response.json()
    except Exception as e:
        LOGGER.error(f"API error: {e}", exc_info=True)
        return None, "NO-SESSION-ID"

    best_llm, session_id = model_select_parse(
        response_code, response_json, llm_configs
    )

    return best_llm, session_id


async def amodel_select(
    messages: List[Dict[str, str]],
    llm_configs: List[LLMConfig],
    metric: Metric,
    notdiamond_api_key: str,
    max_model_depth: int,
    hash_content: bool,
    tradeoff: Optional[str] = None,
    preference_id: Optional[str] = None,
    tools: Optional[Sequence[Union[Dict[str, Any], Callable]]] = [],
    timeout: Optional[int] = 5,
):
    """
    This endpoint receives the prompt and routing settings, and makes a call to the NotDiamond API.
    It returns the best fitting LLM to call and a session ID that can be used for feedback.

    Parameters:
        messages (List[Dict[str, str]]): list of messages to be used for the LLM call
        llm_configs (List[LLMConfig]): a list of available LLMs that the router can decide from
        metric (Metric): metric based off which the router makes the decision. As of now only 'accuracy' supported.
        notdiamond_api_key (str): API key generated via the NotDiamond dashboard.
        max_model_depth (int): if your top recommended model is down, specify up to which depth of routing you're willing to go.
        hash_content (Optional[bool]): Flag for hashing content before sending to NotDiamond API.
        tradeoff (Optional[str], optional): Define the "cost" or "latency" tradeoff
                                            for the router to determine the best LLM for a given query.
        preference_id (Optional[str], optional): The ID of the router preference that was configured via the Dashboard.
                                                    Defaults to None.
        timeout (int, optional): timeout for the request. Defaults to 5.

    Returns:
        tuple(LLMConfig, string): returns a tuple of the chosen LLMConfig to call and a session ID string.
                                        In case of an error the LLM defaults to None and the session ID defaults
                                        to 'NO-SESSION-ID'.
    """
    url, payload, headers = model_select_prepare(
        messages=messages,
        llm_configs=llm_configs,
        metric=metric,
        notdiamond_api_key=notdiamond_api_key,
        max_model_depth=max_model_depth,
        hash_content=hash_content,
        tradeoff=tradeoff,
        preference_id=preference_id,
        tools=tools,
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, data=json.dumps(payload), headers=headers, timeout=timeout
            ) as response:
                response_code = response.status
                response_json = await response.json()
    except Exception as e:
        LOGGER.error(f"API error: {e}", exc_info=True)
        return None, "NO-SESSION-ID"

    best_llm, session_id = model_select_parse(
        response_code, response_json, llm_configs
    )

    return best_llm, session_id


def report_latency(
    session_id: str,
    llm_config: LLMConfig,
    tokens_per_second: float,
    notdiamond_api_key: str,
):
    """
    This method makes an API call to the NotDiamond server to report the latency of an LLM call.
    It helps fine-tune our model router and ensure we offer recommendations that meet your latency expectation.

    This feature can be disabled on the NDLLM class level by setting `latency_tracking` to False.

    Parameters:
        session_id (str): the session ID that was returned from the `invoke` or `model_select` calls, so we know which
                            router call your latency report refers to.
        llm_provider (LLMConfig): specifying the LLM provider for which the latency is reported
        tokens_per_second (float): latency of the model call calculated based on time elapsed, input tokens, and output tokens
        notdiamond_api_key (str): NotDiamond API call used for authentication

    Returns:
        int: status code of the API call, 200 if it's success

    Raises:
        ApiError: if the API call to the NotDiamond backend fails, this error is raised
    """
    url = f"{settings.ND_BASE_URL}/v2/report/metrics/latency"

    payload = {
        "session_id": session_id,
        "provider": llm_config.prepare_for_request(),
        "feedback": {"tokens_per_second": tokens_per_second},
    }

    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {notdiamond_api_key}",
        "User-Agent": f"Python-SDK/{settings.VERSION}",
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
    except Exception as e:
        LOGGER.error(
            f"API error for report metrics latency: {e}", exc_info=True
        )
        return 500

    return response.status_code
