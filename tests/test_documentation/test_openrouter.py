import json
import os

import pytest
import requests
from dotenv import load_dotenv

from notdiamond.llms.client import NotDiamond

load_dotenv(os.getcwd() + "/.env")


@pytest.mark.vcr
def test_openrouter_integration():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", default="")
    messages = [
        {
            "role": "system",
            "content": "You are a world class software developer.",
        },
        {"role": "user", "content": "Write a merge sort in Python"},
    ]

    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-opus-20240229",
    ]

    nd_llm = NotDiamond(llm_configs=llm_configs)
    session_id, provider = nd_llm.model_select(messages=messages)

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_api_key}",
        },
        data=json.dumps(
            {"model": provider.openrouter_model, "messages": messages}
        ),
    )

    print(response)
