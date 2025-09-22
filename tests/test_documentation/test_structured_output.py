from json import JSONDecoder

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field

from notdiamond.llms.client import NotDiamond


@pytest.mark.vcr
def test_structured_output():
    class LanguageChoice(BaseModel):
        language: str = Field(
            description="The programming language of choice."
        )
        reason: str = Field(
            description="The reason to pick the programming language."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a world class software developer.",
        },
        {
            "role": "user",
            "content": "What language would you suggest for developing a web application?",
        },
    ]

    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-2.1",
    ]

    nd_llm = NotDiamond(llm_configs=llm_configs)

    def extract_json_objects(text, decoder=JSONDecoder()):
        """Find JSON objects in text, and yield the decoded JSON data

        Does not attempt to look for JSON arrays, text, or other JSON types outside
        of a parent JSON object.

        """
        pos = 0
        while True:
            match = text.find("{", pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                yield result
                pos = match + index
            except ValueError:
                pos = match + 1

    try:
        result, session_id, _ = nd_llm.invoke(
            messages=messages, response_model=LanguageChoice
        )
    except OutputParserException:
        result, session_id, _ = nd_llm.invoke(messages=messages)
        json_obj = list(extract_json_objects(result.content))
        assert len(json_obj) == 1
        json_response = json_obj[0]
        assert len(json_response) == 2
        assert "language" in json_response
        assert "reason" in json_response

    print(result)


@pytest.mark.vcr
def test_structured_output_streaming():
    class LanguageChoice(BaseModel):
        language: str = Field(
            description="The programming language of choice."
        )
        reason: str = Field(
            description="The reason to pick the programming language."
        )

    messages = [
        {
            "role": "system",
            "content": "You are a world class software developer.",
        },
        {
            "role": "user",
            "content": "What language would you suggest for developing a web application?",
        },
    ]

    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-2.1",
    ]

    nd_llm = NotDiamond(llm_configs=llm_configs)

    for chunk in nd_llm.stream(
        messages=messages, response_model=LanguageChoice
    ):
        print(chunk)
