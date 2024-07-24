import requests
from langchain_core.tools import tool

from notdiamond import settings
from notdiamond.llms.client import NotDiamond


def test_function_calling():
    # Defining our tools
    @tool
    def add(a: int, b: int) -> int:
        "Adds a and b."
        return a + b

    @tool
    def multiply(a: int, b: int) -> int:
        "Multiplies a and b."
        return a * b

    # Creating an instance of NDLLM with specified providers
    nd_llm = NotDiamond(
        llm_configs=["openai/gpt-4", "anthropic/claude-3-opus-20240229"]
    )

    # Binding the add and multiply tools to the nd_llm instance
    nd_llm.bind_tools([add, multiply])

    # Creating a list of messages
    messages = [
        {
            "role": "system",
            "content": """You are a bot that helps with doing different mathematic calculations
        using provided functions. For every question that is asked, call the correct function.""",
        },
        {"role": "user", "content": "What is 3288273827373 * 523283927371?"},
    ]

    # Invoking the nd_llm instance with the messages
    result, session_id, provider = nd_llm.invoke(messages)

    # Looping through the tool calls in the result
    for tool_call in result.tool_calls:
        # Selecting the tool based on the name
        selected_tool = {"add": add, "multiply": multiply}[
            tool_call["name"].lower()
        ]
        # Invoking the selected tool with the arguments
        tool_output = selected_tool.invoke(tool_call["args"])

    print(
        "ND session ID:", session_id
    )  # A unique ID of the invoke. Useful for personalizing ND through feedback
    print("LLM called:", provider.model)  # The LLM routed to
    print("Selected tool:", result.tool_calls[0]["name"])  # The selected tool
    print(
        "Arguments:", result.tool_calls[0]["args"]
    )  # The corresponding arguments
    print(
        "Function output:", tool_output
    )  # The output of the selected function


def test_function_calling_via_rest_api():
    modelselect_url = (
        "https://not-diamond-server.onrender.com/v2/optimizer/hashModelSelect"
    )

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {settings.NOTDIAMOND_API_KEY}",
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Adds a and b.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiplies a and b.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]

    modelselect_payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a mathematician with tools at your disposal.",
            },
            {"role": "assistant", "content": "How can I help you today?"},
            {"role": "user", "content": "What is 234 + 82234?"},
        ],
        # The models you want to route between
        "llm_providers": [
            {"provider": "openai", "model": "gpt-4-1106-preview"},
            {"provider": "anthropic", "model": "claude-3-opus-20240229"},
        ],
        "tools": tools,
    }

    # Send a POST request to the API endpoint
    response = requests.post(
        modelselect_url, json=modelselect_payload, headers=headers
    )

    # Extract the session ID and the selected model
    assert (
        "session_id" in response.json()
    ), f"Expected 'session_id' in response but got {response.json()}"
    session_id = response.json()["session_id"]
    model_selected = response.json()["providers"][
        0
    ]  # The provider to call in the format {"provider": "openai", "model": "gpt-4"}

    print(session_id)
    print(model_selected)
