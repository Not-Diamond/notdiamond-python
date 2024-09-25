import logging
import re
from typing import Dict, List

from notdiamond.llms.config import LLMConfig
from notdiamond.llms.providers import NDLLMProviders

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def inject_system_prompt(
    messages: List[Dict[str, str]], system_prompt: str
) -> List[Dict[str, str]]:
    """
    Add a system prompt to an OpenAI-style message list. If a system prompt is already present, replace it.
    """
    new_messages = []
    found = False
    for msg in messages:
        # t7: replace the first system prompt with the new one
        if msg["role"] == "system" and not found:
            new_messages.append({"role": "system", "content": system_prompt})
            found = True
        else:
            new_messages.append(msg)
    if not found:
        new_messages.insert(0, {"role": "system", "content": system_prompt})
    return new_messages


def _curly_escape(text: str) -> str:
    """
    Escape curly braces in the text, but only for single occurrences of alphabetic characters.
    This function will not escape double curly braces or non-alphabetic characters.
    """
    return re.sub(r"(?<!{){([a-zA-Z])}(?!})", r"{{\1}}", text)


def _is_o1_model(llm: LLMConfig):
    if llm in (
        NDLLMProviders.O1_PREVIEW,
        NDLLMProviders.O1_PREVIEW_2024_09_12,
        NDLLMProviders.O1_MINI,
        NDLLMProviders.O1_MINI_2024_09_12,
    ):
        return True
    return False


def o1_system_prompt_translate(
    messages: List[Dict[str, str]], llm: LLMConfig
) -> List[Dict[str, str]]:
    if _is_o1_model(llm):
        translated_messages = []
        for msg in messages:
            if msg["role"] == "system":
                translated_messages.append(
                    {"role": "user", "content": msg["content"]}
                )
            else:
                translated_messages.append(msg)
        return translated_messages
    return messages
