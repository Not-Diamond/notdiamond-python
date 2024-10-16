import pytest

from notdiamond.llms.client import NotDiamond


@pytest.mark.vcr
def test_streaming():
    messages = [
        {
            "role": "system",
            "content": "You are a world class software developer.",
        },
        {"role": "user", "content": "Write merge sort in Python."},
    ]

    chat = NotDiamond(
        llm_configs=[
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-2.1",
            "google/gemini-pro",
        ]
    )

    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
