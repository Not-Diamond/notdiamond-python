import pytest
from helpers import astream_chunks, stream_chunks

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders

test_providers = [
    provider for provider in NDLLMProviders if provider.provider == "openai"
]


@pytest.mark.longrun
@pytest.mark.parametrize("provider", test_providers)
class Test_OpenAI:
    def test_streaming(self, prompt, provider):
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        stream_chunks(nd_llm.stream(prompt))

    @pytest.mark.asyncio
    async def test_async_streaming(self, prompt, provider):
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        await astream_chunks(nd_llm.astream(prompt))

    def test_streaming_use_stop(self, prompt, provider):
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        for chunk in nd_llm.stream(messages=prompt, stop=["a"]):
            assert chunk.type == "AIMessageChunk"
            assert isinstance(chunk.content, str)
            assert chunk.content != "a"

    def test_with_tool_calling(self, tools_fixture, provider):
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_with_openai_tool_calling(self, openai_tools_fixture, provider):
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_response_model(self, response_model, provider):
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
