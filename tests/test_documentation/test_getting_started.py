import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.config import LLMConfig


@pytest.fixture
def start_prompt():
    return [
        {
            "role": "system",
            "content": "You are a world class software developer.",
        },
        {"role": "user", "content": "Write a merge sort in Python"},
    ]


@pytest.mark.vcr
def test_main_example():
    model_list = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-sonnet-4-0",
    ]

    client = NotDiamond()

    # After fuzzy hashing the inputs, the best LLM is determined by the ND API and the LLM is called client-side
    result, session_id, provider = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a world class programmer."},
            {"role": "user", "content": "Write a merge sort in Python"},
        ],
        model=model_list,
        tradeoff="cost",
    )

    print(
        "ND session ID: ", session_id
    )  # A unique ID of the model call. Important for personalizing ND to your use-case
    print("LLM called: ", provider.model)  # The LLM routed to
    print("LLM output: ", result.content)  # The LLM response


@pytest.mark.vcr
def test_pass_array_of_messages(start_prompt):
    # Define the available LLMs you'd like to route between
    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-sonnet-20241022",
    ]

    # Create the NDLLM object -> like a 'meta-LLM' combining all of the specified models
    nd_llm = NotDiamond(
        llm_configs=llm_configs,
        tradeoff="cost",
    )  # Define preferences

    # After fuzzy hashing the inputs, the best LLM is determined by the ND API and the LLM is called client-side
    result, session_id, provider = nd_llm.invoke(messages=start_prompt)

    print(
        "ND session ID: ", session_id
    )  # A unique ID of the invoke. Important for personalizing ND to your use-case
    print("LLM called: ", provider.model)  # The LLM routed to
    print("LLM output: ", result.content)  # The LLM response


@pytest.mark.vcr
def test_programatic_define_llm_configs(start_prompt):
    # Define the available LLMs you'd like to route between
    llm_configs = [
        LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.8,
            max_tokens=256,
        ),
        LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            temperature=0.8,
            max_tokens=256,
        ),
    ]

    # Create the NDLLM object -> like a 'meta-LLM' combining all of the specified models
    nd_llm = NotDiamond(
        llm_configs=llm_configs,
        tradeoff="cost",
    )  # Define preferences

    # After fuzzy hashing the inputs, the best LLM is determined by the ND API and the LLM is called client-side
    result, session_id, provider = nd_llm.invoke(start_prompt)

    print(
        "ND session ID: ", session_id
    )  # A unique ID of the invoke. Important for personalizing ND to your use-case
    print("LLM called: ", provider.model)  # The LLM routed to
    print("LLM output: ", result.content)  # The LLM response


@pytest.mark.vcr
def test_model_select(start_prompt):
    # Define the available LLMs you'd like to route between
    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-sonnet-20241022",
    ]

    # Create the NDLLM object -> like a 'meta-LLM' combining all of the specified models
    nd_llm = NotDiamond()

    session_id, provider = nd_llm.model_select(
        messages=start_prompt,
        model=llm_configs,
        tradeoff="cost",
    )

    print(
        "ND session ID: ", session_id
    )  # A unique ID of the model_select. Important for personalizing ND to your use-case
    print("LLM called: ", provider.model)  # The LLM routed to
