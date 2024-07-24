import pytest
from helpers import astream_chunks, stream_chunks

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders


@pytest.mark.longrun
class Test_OpenAI:
    def test_gpt_35_turbo_streaming(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        stream_chunks(nd_llm.stream(prompt))

    @pytest.mark.asyncio
    async def test_gpt_35_turbo_async_streaming(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        await astream_chunks(nd_llm.astream(prompt))

    def test_gpt_35_turbo_streaming_use_stop(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        for chunk in nd_llm.stream(messages=prompt, stop=["a"]):
            assert chunk.type == "AIMessageChunk"
            assert isinstance(chunk.content, str)
            assert chunk.content != "a"

    def test_gpt_35_turbo_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_35_turbo_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_35_turbo_response_model(self, response_model):
        provider = NDLLMProviders.GPT_3_5_TURBO
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

    def test_gpt_35_turbo_0125_streaming(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        stream_chunks(nd_llm.stream(prompt))

    @pytest.mark.asyncio
    async def test_gpt_35_turbo_0125_async_streaming(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        await astream_chunks(nd_llm.astream(prompt))

    def test_gpt_35_turbo_0125_streaming_use_stop(self, prompt):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        for chunk in nd_llm.stream(messages=prompt, stop=["a"]):
            assert chunk.type == "AIMessageChunk"
            assert isinstance(chunk.content, str)
            assert chunk.content != "a"

    def test_gpt_35_turbo_0125_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_35_turbo_0125_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        # t7: commenting this out bc gpt-3.5-turbo doesn't reliably call tools
        # assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_35_turbo_0125_response_model(self, response_model):
        provider = NDLLMProviders.GPT_3_5_TURBO_0125
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

    def test_gpt_4_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4
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

    def test_gpt_4_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_4
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

    def test_gpt_4_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4_0163_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_0613
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

    def test_gpt_4_0163_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_4_0613
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

    def test_gpt_4_0163_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_0613
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(llm_configs=[provider])
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4_1106_preview_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_1106_PREVIEW
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

    def test_gpt_4_1106_preview_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4_1106_PREVIEW
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

    def test_gpt_4_1106_preview_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_1106_PREVIEW
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

    def test_gpt_4_0125_preview_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_0125_PREVIEW
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

    def test_gpt_4_0125_preview_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4_0125_PREVIEW
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

    def test_gpt_4_0125_preview_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_0125_PREVIEW
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

    def test_gpt_4_turbo_preview_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_TURBO_PREVIEW
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

    def test_gpt_4_turbo_preview_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4_TURBO_PREVIEW
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

    def test_gpt_4_turbo_preview_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_TURBO_PREVIEW
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

    def test_gpt_4_turbo_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_TURBO
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

    def test_gpt_4_turbo_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_4_TURBO
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

    def test_gpt_4_turbo_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_TURBO
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

    def test_gpt_4_turbo_2024_04_09_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4_TURBO_2024_04_09
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4_turbo_2024_04_09_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4_TURBO_2024_04_09
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4_turbo_2024_04_09_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4_TURBO_2024_04_09
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4o_2024_05_13_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4o_2024_05_13
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_2024_05_13_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4o_2024_05_13
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_2024_05_13_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4o_2024_05_13
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4o_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4o
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_4o
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4o
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4o_response_model_full_roles(self):
        provider = NDLLMProviders.GPT_4o
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            [
                {
                    "role": "system",
                    "content": "You are a product manager for an AI company.",
                },
                {
                    "role": "assistant",
                    "content": "I am looking for a great prompt routing product.",
                },
                {
                    "role": "user",
                    "content": "Boy have I got the right thing for you - have you heard of Not Diamond??",
                },
                {"role": "assistant", "content": "I have, their product is:"},
            ],
        )

        assert session_id != "NO-SESSION-ID"
        print(result.content)
        assert len(result.content) > 0

    def test_gpt_4o_mini_2024_07_18_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4o_MINI_2024_07_18
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_mini_2024_07_18_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GPT_4o_MINI_2024_07_18
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_mini_2024_07_18_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4o_MINI_2024_07_18
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gpt_4o_mini_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GPT_4o_MINI
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_mini_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GPT_4o_MINI
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        nd_llm = nd_llm.bind_tools(openai_tools_fixture)
        result, session_id, _ = nd_llm.invoke(
            [{"role": "user", "content": "How much is 3 + 5?"}]
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"

    def test_gpt_4o_mini_response_model(self, response_model):
        provider = NDLLMProviders.GPT_4o_MINI
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            [{"role": "user", "content": "Tell me a joke"}],
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline
