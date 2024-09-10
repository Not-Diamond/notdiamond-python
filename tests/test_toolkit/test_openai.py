"""
Tests for our OpenAI client wrapper. Most of these ensure that the client supports
capabilities defined in the API reference:

https://platform.openai.com/docs/api-reference/introduction
"""
from notdiamond.settings import OPENAI_API_KEY
from notdiamond.toolkit.openai import OpenAI


def test_openai_init():
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        llm_configs=["openai/gpt-4o", "openai/gpt-4o-mini"],
    )
    assert client is not None

    client2 = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        llm_configs=["openai/gpt-4o", "openai/gpt-4o-mini"],
    )
    assert client2 is not None

    client3 = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.openai.com/v1",
        llm_configs=["openai/gpt-4o", "openai/gpt-4o-mini"],
        organization="nd-oai-organization",
        project="nd-oai-project",
    )
    assert client3 is not None


def test_openai_create():
    nd_llm_configs = [
        "openai/gpt-4o-2024-05-13",
        "openai/gpt-4o-mini-2024-07-18",
    ]
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        llm_configs=nd_llm_configs,
    )

    response = client.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this route to gpt-3.5-turbo?",
            }
        ],
    )
    assert response is not None
    assert len(response.choices[0].message.content) > 0
    # Ensure provided model is ignored for LLM configs instead
    assert response.model in [
        provider.split("/")[-1] for provider in nd_llm_configs
    ]

    # Ensure no model will still route to the chosen LLM
    no_model_response = client.create(
        messages=[
            {"role": "user", "content": "Hello, world! What about no model?"}
        ],
    )
    assert no_model_response is not None
    assert len(no_model_response.choices[0].message.content) > 0
    # Ensure provided model is ignored for LLM configs instead
    assert no_model_response.model in [
        provider.split("/")[-1] for provider in nd_llm_configs
    ]


def test_openai_create_default_models():
    all_oai_model_response = OpenAI(api_key=OPENAI_API_KEY).create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Do you handle default models?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


def test_openai_chat_completions_create():
    all_oai_model_response = OpenAI(
        api_key=OPENAI_API_KEY
    ).chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this chat.completions.create call work?",
            }
        ],
    )
    assert all_oai_model_response is not None
    assert len(all_oai_model_response.choices[0].message.content) > 0


def test_openai_chat_completions_create_stream():
    model_response_stream = OpenAI(
        api_key=OPENAI_API_KEY
    ).chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello, world! Does this chat.completions.create call work?",
            }
        ],
        stream=True,
    )

    any_nonempty_chunk = False
    for chunk in model_response_stream:
        if chunk.choices[0].delta.content is not None:
            any_nonempty_chunk = True
            break
    assert any_nonempty_chunk
