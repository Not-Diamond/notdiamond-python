import os
from typing import Dict, List, Union

from notdiamond.toolkit._retry import (
    AsyncRetryWrapper,
    ClientType,
    ModelType,
    OpenAIMessagesType,
    RetryManager,
    RetryWrapper,
)


def init(
    client: Union[ClientType, List[ClientType]],
    models: ModelType,
    max_retries: Union[int, Dict[str, int]],
    timeout: Union[float, Dict[str, float]],
    model_messages: Dict[str, OpenAIMessagesType],
    api_key: Union[str, None] = None,
    async_mode: bool = False,
) -> RetryManager:
    """Entrypoint for fallback and retry features without changing existing code.

    Add this to existing codebase without other modifications to enable the following capabilities:

    - Fallback to a different model if a model invocation fails.
    - If configured, fallback to a different *provider* if a model invocation fails
      (eg. azure/gpt-4o fails -> invoke openai/gpt-4o)
    - Load-balance between models and providers, if specified.
    - Pass timeout and retry configurations to each invoke, optionally configured per model.
    - Pass model-specific messages on each retry (prepended to the provided `messages` parameter)

    Parameters:
        client (Union[ClientType, List[ClientType]]): Clients to apply retry/fallback logic to.
        models (Union[Dict[str, float], List[str]]):
            Models to use of the format <provider>/<model>.
            Supports two formats:
                - List of models, eg. ["openai/gpt-4o", "azure/gpt-4o"]. Models will be prioritized as listed.
                - Dict of models to weights for load balancing, eg. {"openai/gpt-4o": 0.9, "azure/gpt-4o": 0.1}.
                  If a model invocation fails, the next model is selected by sampling using the *remaining* weights.
        max_retries (Union[int, Dict[str, int]]):
            Maximum number of retries. Can be configured globally or per model.
        timeout (Union[float, Dict[str, float]]):
            Timeout in seconds per model. Can be configured globally or per model.
        model_messages (Dict[str, OpenAIMessagesType]):
            Model-specific messages to prepend to `messages` on each invocation, formatted OpenAI-style. Can be
            configured using any role which is valid as an initial message (eg. "system" or "user", but not "assistant").
        api_key (Optional[str]):
            Not Diamond API key for authentication. Unused for now - will offer logging and metrics in the future.
        async_mode (bool):
            Whether to manage clients as async.

    Returns:
        RetryManager: Manager object that handles retries and fallbacks. Not required for usage.

    Model Fallback Prioritization
    -----------------------------

    - If models is a list, the fallback model is selected in order after removing the failed model.
      eg. If "openai/gpt-4o" fails for the list:
        - ["openai/gpt-4o", "azure/gpt-4o"], "azure/gpt-4o" will be tried next
        - ["openai/gpt-4o-mini", "openai/gpt-4o", "azure/gpt-4o"], "openai/gpt-4o-mini" will be tried next.
    - If models is a dict, the next model is selected by sampling using the *remaining* weights.
      eg. If "openai/gpt-4o" fails for the dict:
        - {"openai/gpt-4o": 0.9, "azure/gpt-4o": 0.1}, "azure/gpt-4o" will be invoked 100% of the time
        - {"openai/gpt-4o": 0.5, "azure/gpt-4o": 0.25, "openai/gpt-4o-mini": 0.25}, then "azure/gpt-4o" and
          "openai/gpt-4o-mini" can be invoked with 50% probability each.

    Usage
    -----

    Please refer to tests/test_init.py for more examples on how to use notdiamond.init.

    .. code-block:: python

        # ...existing workflow code, including client initialization...
        openai_client = OpenAI(...)
        azure_client = AzureOpenAI(...)

        # Add `notdiamond.init` to the workflow.
        notdiamond.init(
            [openai_client, azure_client],
            models={"openai/gpt-4o": 0.9, "azure/gpt-4o": 0.1},
            max_retries={"openai/gpt-4o": 3, "azure/gpt-4o": 1},
            timeout={"openai/gpt-4o": 10.0, "azure/gpt-4o": 5.0},
            model_messages={
                "openai/gpt-4o": [{"role": "user", "content": "Here is a prompt for OpenAI."}],
                "azure/gpt-4o": [{"role": "user", "content": "Here is a prompt for Azure."}],
            },
            api_key="sk-...",
        )

        # ...continue existing workflow code...
        response = openai_client.chat.completions.create(
            model="notdiamond",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    """
    api_key = api_key or os.getenv("NOTDIAMOND_API_KEY")

    if async_mode:
        wrapper_cls = AsyncRetryWrapper
    else:
        wrapper_cls = RetryWrapper

    for model in models:
        if len(model.split("/")) != 2:
            raise ValueError(
                f"Model {model} must be in the format <provider>/<model>."
            )

    if not isinstance(client, List):
        client_wrappers = [
            wrapper_cls(
                client=client,
                models=models,
                max_retries=max_retries,
                timeout=timeout,
                model_messages=model_messages,
                api_key=api_key,
            )
        ]
    else:
        client_wrappers = [
            wrapper_cls(
                client=cc,
                models=models,
                max_retries=max_retries,
                timeout=timeout,
                model_messages=model_messages,
                api_key=api_key,
            )
            for cc in client
        ]
    retry_manager = RetryManager(models, client_wrappers)

    return retry_manager
