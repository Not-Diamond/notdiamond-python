import pytest
from openai import OpenAI

from notdiamond import settings
from notdiamond.llms.client import NotDiamond


@pytest.mark.vcr
def test_fallback_model():
    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-2.1",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-sonnet-4-0",
        "google/gemini-pro",
    ]

    # Setting a fallback model
    NotDiamond(
        llm_configs=llm_configs,
        default="openai/gpt-4-turbo-preview",  # The model from llm_configs you want to fallback to
    )


@pytest.mark.vcr
def test_set_max_model_depth():
    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-2.1",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-5-sonnet-20241022",
        "google/gemini-pro",
    ]

    # Setting recommendation depth
    NotDiamond(
        llm_configs=llm_configs,
        max_model_depth=3,  # The maximum depth in the recommendation ranking to consider
    )


@pytest.mark.vcr
def test_custom_logic():
    # Define the string that will be routed to the best LLM
    prompt = "You are a world class software developer. Write a merge sort in Python."

    # Define the available LLMs you'd like to route between
    llm_configs = [
        "openai/gpt-3.5-turbo",
    ]

    # Create the NDLLM object -> like a 'meta-LLM' combining all of the specified models
    nd_llm = NotDiamond(
        llm_configs=llm_configs,
        tradeoff="cost",
    )  # Define preferences

    session_id, provider = nd_llm.model_select(messages=prompt)

    print(session_id)
    print(provider)

    client = OpenAI(api_key=settings.OPENAI_API_KEY)

    max_retries = 3

    if provider.model == "gpt-3.5-turbo":
        for _ in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-3.5-turbo",
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(e)
                continue
