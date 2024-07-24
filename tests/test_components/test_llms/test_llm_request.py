import pytest
from langchain_core.messages import AIMessage

from notdiamond import Metric, NotDiamond
from notdiamond.llms.providers import NDLLMProviders


def test_llm_invoke_with_latency_tracking_success():
    metric = Metric("accuracy")
    openai = NDLLMProviders.GPT_3_5_TURBO
    llm_configs = [openai]

    llm = NotDiamond(llm_configs=llm_configs, latency_tracking=True)

    llm_result, session_id, _ = llm.invoke(
        messages=[{"role": "user", "content": "Prompt: Tell me a joke."}],
        metric=metric,
    )

    assert session_id
    assert llm_result


@pytest.mark.asyncio
async def test_async_llm_invoke_with_latency_tracking_success():
    metric = Metric("accuracy")
    openai = NDLLMProviders.GPT_3_5_TURBO
    llm_configs = [openai]

    llm = NotDiamond(llm_configs=llm_configs, latency_tracking=True)

    llm_result, session_id, _ = await llm.ainvoke(
        messages=[{"role": "user", "content": "Prompt: Tell me a joke."}],
        metric=metric,
    )

    assert session_id
    assert llm_result


def test_llm_invoke_with_latency_tracking_nd_chat_prompt_success():
    metric = Metric("accuracy")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI bot. Your name is Bob.",
        },
        {"role": "user", "content": "Hello, how are you doing?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
        {"role": "user", "content": "What is your name?"},
    ]

    nd_llm = NotDiamond(
        llm_configs=["openai/gpt-3.5-turbo"], hash_content=True
    )
    result, session_id, _ = nd_llm.invoke(
        messages=messages,
        metric=metric,
    )

    print(session_id)
    assert len(session_id) == 36
    assert len(session_id.split("-")) == 5
    assert type(result) is AIMessage
    assert "Bob" in result.content
