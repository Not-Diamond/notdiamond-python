import pytest
from helpers import astream_chunks, stream_chunks

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders


@pytest.mark.longrun
@pytest.mark.vcr
class Test_Anthropic_LLMs:
    def test_claude_21_with_streaming(self, prompt):
        provider = NDLLMProviders.CLAUDE_2_1
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        stream_chunks(nd_llm.stream(prompt))

    @pytest.mark.asyncio
    async def test_claude_21_with_async_streaming(self, prompt):
        provider = NDLLMProviders.CLAUDE_2_1
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        await astream_chunks(nd_llm.astream(prompt))

    def test_claude_21_response_model(self, response_model):
        provider = NDLLMProviders.CLAUDE_2_1
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

    def test_claude_3_opus_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_OPUS_20240229
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

    def test_claude_3_opus_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_3_OPUS_20240229
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

    def test_claude_3_opus_response_model(self, response_model):
        provider = NDLLMProviders.CLAUDE_3_OPUS_20240229
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

    def test_claude_3_sonnet_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_SONNET_20240229
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

    def test_claude_3_sonnet_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_3_SONNET_20240229
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

    def test_claude_3_haiku_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_HAIKU_20240307
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

    def test_claude_3_5_sonnet_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_5_SONNET_20240620
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

    def test_claude_3_5_sonnet_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_3_5_SONNET_20240620
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

    def test_claude_3_5_sonnet_20241022_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_5_SONNET_20241022
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

    def test_claude_3_5_sonnet_20241022_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_3_5_SONNET_20241022
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

    def test_claude_3_5_haiku_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_5_HAIKU_20241022
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

    def test_claude_3_5_haiku_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_3_5_HAIKU_20241022
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

    def test_claude_3_7_sonnet_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_7_SONNET_LATEST
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

    def test_claude_3_7_sonnet_20250219_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_3_7_SONNET_20250219
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

    def test_claude_opus_4_1_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_OPUS_4_1
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

    def test_claude_opus_4_1_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_OPUS_4_1
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

    def test_claude_opus_4_1_20250805_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_OPUS_4_1_20250805
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

    def test_claude_opus_4_1_20250805_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_OPUS_4_1_20250805
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

    def test_claude_sonnet_4_5_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.CLAUDE_SONNET_4_5
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

    def test_claude_sonnet_4_5_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.CLAUDE_SONNET_4_5
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
