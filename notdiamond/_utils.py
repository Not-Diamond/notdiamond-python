import importlib.util
import logging
import tiktoken
from typing import Dict, cast, List, Optional, Union

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _module_check(module_name: str, attr_name: str, provider_name: str = None):
    parent_module_name = module_name.split(".")[0]

    if provider_name:
        provider_title = provider_name.title()
        if provider_title == "Openai":
            provider_title = "OpenAI"
        err_msg = (
            f"`notdiamond` requires `{parent_module_name}` to generate responses from {provider_title} models. "
            "Please install via `pip install notdiamond[create]`."
        )
    else:
        err_msg = (
            f"`notdiamond` requires `{parent_module_name}` to generate responses. Please install via "
            "`pip install notdiamond[create]`."
        )

    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, attr_name)
    except (ModuleNotFoundError, AttributeError) as err:
        raise ModuleNotFoundError(err_msg) from err
    return cls


def convert_tool_to_openai_function(tool):
    convert_python_function_to_openai_function = None
    BaseTool = None
    format_tool_to_openai_function = None
    try:
        convert_python_function_to_openai_function = _module_check(
            "langchain_core.utils.function_calling",
            "convert_python_function_to_openai_function",
        )
        BaseTool = _module_check("langchain_core.tools", "BaseTool")
        format_tool_to_openai_function = _module_check(
            "langchain_core.utils.function_calling",
            "format_tool_to_openai_function",
        )
    except (ModuleNotFoundError, ImportError) as ierr:
        LOGGER.info(
            "Could not import function calling and tool helpers from langchain_core."
            f"{ierr}"
        )

    if isinstance(tool, dict) and all(
        k in tool for k in ("name", "description", "parameters")
    ):
        return tool
    # a JSON schema with title and description
    elif isinstance(tool, dict) and all(
        k in tool for k in ("title", "description", "properties")
    ):
        tool = tool.copy()
        return {
            "name": tool.pop("title"),
            "description": tool.pop("description"),
            "parameters": tool,
        }
    elif isinstance(tool, BaseTool):
        if not format_tool_to_openai_function:
            raise ValueError(
                "Provided tool is a langchain BaseTool, but langchain is not available to import. "
                "Please run pip install notdiamond[create] to install langchain."
            )
        return cast(Dict, format_tool_to_openai_function(tool))
    elif callable(tool):
        if not convert_python_function_to_openai_function:
            raise ValueError(
                "Provided tool is a Python function, but langchain is not available to import. "
                "Tool calling of functions is only available with langchain. Please run pip install notdiamond[create] "
                "to install langchain."
            )
        return cast(Dict, convert_python_function_to_openai_function(tool))

    else:
        raise ValueError(
            f"Unsupported function\n\n{tool}\n\nFunctions must be passed in"
            " as Dict, pydantic.BaseModel, or Callable. If they're a dict they must"
            " either be in OpenAI function format or valid JSON schema with top-level"
            " 'title' and 'description' keys."
        )


def _default_headers(
    notdiamond_api_key: str, user_agent: str
) -> Dict[str, str]:
    return {
        "content-type": "application/json",
        "Authorization": f"Bearer {notdiamond_api_key}",
        "User-Agent": user_agent,
    }


def token_counter(
    model: str = "",
    text: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> int:
    """
    Count the number of tokens in a given text or messages using a specified model.
    
    Args:
        model (str): The name of the model to use for tokenization.
        text (Optional[str]): The raw text string to count tokens for.
        messages (Optional[List[Dict[str, str]]]): List of messages with "role" and "content" keys.
        
    Returns:
        int: The number of tokens in the text or messages.
    """
    if text is None and messages is None:
        raise ValueError("Either text or messages must be provided")
    
    if text is None:
        text = ""
        for message in messages:
            if message.get("content", None) is not None:
                content = message.get("content")
                if isinstance(content, str):
                    text += content
    
    # Use tiktoken for OpenAI models
    try:
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            # Default to cl100k_base for other models
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))
    except Exception as e:
        LOGGER.warning(f"Error using tiktoken: {e}. Using simple character count approximation.")
        # Fallback: rough approximation (4 chars ≈ 1 token)
        return len(text) // 4
