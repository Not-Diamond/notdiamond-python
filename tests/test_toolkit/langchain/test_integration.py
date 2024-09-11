import os
import uuid
from importlib import import_module
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from notdiamond import LLMConfig, NotDiamond
from notdiamond.toolkit.langchain import NotDiamondRoutedRunnable

load_dotenv()


@pytest.fixture
def nd_routed_runnable() -> NotDiamondRoutedRunnable:
    api_key = os.getenv("NOTDIAMOND_API_KEY")
    llm_configs = [
        LLMConfig(provider="openai", model="gpt-4o-2024-08-06"),
        LLMConfig(provider="openai", model="gpt-4o-mini-2024-07-18"),
    ]
    nd_client = NotDiamond(
        api_key=api_key,
        llm_configs=llm_configs,
        default="openai/gpt-4o-mini-2024-07-18",
    )
    return NotDiamondRoutedRunnable(nd_client=nd_client)


def test_notdiamond_routed_runnable(
    nd_routed_runnable: NotDiamondRoutedRunnable,
) -> None:
    result = nd_routed_runnable.invoke("Hello, world!")
    assert result.response_metadata is not None
    assert "gpt" in result.response_metadata["model_name"]


def test_notdiamond_routed_runnable_chain(
    nd_routed_runnable: NotDiamondRoutedRunnable,
) -> None:
    def fn(x: str) -> str:
        return x + "!"

    chain = RunnableLambda(fn) | nd_routed_runnable
    result = chain.invoke(
        "Hello there! Not Diamond sent me to you. Which OpenAI model are you?"
    )
    assert result.response_metadata is not None
    assert "gpt" in result.response_metadata["model_name"]


@pytest.mark.parametrize(
    "target_model,patch_class",
    [
        ("openai/gpt-4o", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-2024-08-06", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-2024-05-13", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-turbo-2024-04-09", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-0125-preview", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-1106-preview", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4-0613", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-3.5-turbo-0125", "langchain_openai.ChatOpenAI"),
        ("openai/gpt-4o-mini-2024-07-18", "langchain_openai.ChatOpenAI"),
        (
            "anthropic/claude-3-5-sonnet-20240620",
            "langchain_anthropic.ChatAnthropic",
        ),
        (
            "anthropic/claude-3-opus-20240229",
            "langchain_anthropic.ChatAnthropic",
        ),
        (
            "anthropic/claude-3-sonnet-20240229",
            "langchain_anthropic.ChatAnthropic",
        ),
        (
            "anthropic/claude-3-haiku-20240307",
            "langchain_anthropic.ChatAnthropic",
        ),
        (
            "google/gemini-1.5-pro-latest",
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ),
        (
            "google/gemini-1.5-flash-latest",
            "langchain_google_genai.ChatGoogleGenerativeAI",
        ),
        ("mistral/open-mixtral-8x22b", "langchain_mistralai.ChatMistralAI"),
        ("mistral/codestral-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/open-mixtral-8x7b", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-large-2407", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-large-2402", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-medium-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/mistral-small-latest", "langchain_mistralai.ChatMistralAI"),
        ("mistral/open-mistral-7b", "langchain_mistralai.ChatMistralAI"),
        ("togetherai/Llama-3-70b-chat-hf", "langchain_together.ChatTogether"),
        ("togetherai/Llama-3-8b-chat-hf", "langchain_together.ChatTogether"),
        (
            "togetherai/Meta-Llama-3.1-8B-Instruct-Turbo",
            "langchain_together.ChatTogether",
        ),
        (
            "togetherai/Meta-Llama-3.1-70B-Instruct-Turbo",
            "langchain_together.ChatTogether",
        ),
        (
            "togetherai/Meta-Llama-3.1-405B-Instruct-Turbo",
            "langchain_together.ChatTogether",
        ),
        ("togetherai/Qwen2-72B-Instruct", "langchain_together.ChatTogether"),
        (
            "togetherai/Mixtral-8x22B-Instruct-v0.1",
            "langchain_together.ChatTogether",
        ),
        (
            "togetherai/Mixtral-8x7B-Instruct-v0.1",
            "langchain_together.ChatTogether",
        ),
        (
            "togetherai/Mistral-7B-Instruct-v0.2",
            "langchain_together.ChatTogether",
        ),
        ("cohere/command-r-plus", "langchain_cohere.ChatCohere"),
        ("cohere/command-r", "langchain_cohere.ChatCohere"),
    ],
)
def test_invokable(target_model: str, patch_class: str) -> None:
    nd_client = MagicMock(
        spec=NotDiamond,
        llm_configs=[target_model],
        api_key="",
        default=target_model,
    )
    nd_client.chat.completions.model_select = MagicMock(
        return_value=(uuid.uuid4(), target_model)
    )

    module_name, cls_name = patch_class.split(".")
    cls = getattr(import_module(module_name), cls_name)
    mock_client = MagicMock(spec=cls)

    with patch(patch_class, autospec=True) as mock_class:
        mock_class.return_value = mock_client
        runnable = NotDiamondRoutedRunnable(nd_client=nd_client)
        runnable.invoke("Test prompt")
        assert (
            mock_client.invoke.called  # type: ignore[attr-defined]
        ), f"{mock_client}"

    mock_client.reset_mock()

    with patch(patch_class, autospec=True) as mock_class:
        mock_class.return_value = mock_client
        runnable = NotDiamondRoutedRunnable(
            nd_api_key="sk-...", nd_llm_configs=[target_model]
        )
        runnable.invoke("Test prompt")
        assert (
            mock_client.invoke.called  # type: ignore[attr-defined]
        ), f"{mock_client}"


def test_init_perplexity() -> None:
    target_model = "perplexity/llama-3.1-sonar-large-128k-online"
    nd_client = MagicMock(
        spec=NotDiamond,
        llm_configs=[target_model],
        api_key="",
        default=target_model,
    )
    nd_client.chat.completions.model_select = MagicMock(
        return_value=(uuid.uuid4(), target_model)
    )

    with pytest.raises(ValueError):
        NotDiamondRoutedRunnable(nd_client=nd_client)
