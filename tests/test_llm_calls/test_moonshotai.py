import pytest
from helpers import astream_chunks, stream_chunks

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders


@pytest.mark.longrun
@pytest.mark.vcr
class Test_Moonshotai_LLMs:
    def test_kimi_k2_thinking_with_streaming(self, prompt):
        provider = NDLLMProviders.KIMI_K2_THINKING
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        stream_chunks(nd_llm.stream(prompt))

    @pytest.mark.asyncio
    async def test_kimi_k2_thinking_with_async_streaming(self, prompt):
        provider = NDLLMProviders.KIMI_K2_THINKING
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        await astream_chunks(nd_llm.astream(prompt))

    def test_kimi_k2_thinking_response_model(self, response_model):
        provider = NDLLMProviders.KIMI_K2_THINKING
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_kimi_k2_thinking_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.KIMI_K2_THINKING
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_kimi_k2_thinking_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.KIMI_K2_THINKING
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

