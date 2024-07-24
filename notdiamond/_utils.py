import importlib.util
import logging
from typing import Dict, cast

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
