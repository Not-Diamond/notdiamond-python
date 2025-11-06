import logging
from unittest.mock import Mock, patch

import pytest

from notdiamond import LLMConfig
from notdiamond._utils import _module_check
from notdiamond.exceptions import (
    ApiError,
    CreateUnavailableError,
    MissingLLMConfigs,
)
from notdiamond.llms.client import _NDClientTarget, _ndllm_factory
from notdiamond.llms.providers import NDLLMProviders

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

test_params = [
    (_NDClientTarget.ROUTER, _ndllm_factory(_NDClientTarget.ROUTER)),
]
SKIP_TOOL_TESTS = False
try:
    _module_check("langchain_core.tools", "tool")
    test_params.append(
        (_NDClientTarget.INVOKER, _ndllm_factory(_NDClientTarget.INVOKER))
    )
except (ModuleNotFoundError, ImportError) as ierr:
    print(
        f"Could not import function calling and tool helpers from langchain_core. {ierr}. Will only test _NDInvokerClient."
    )
    SKIP_TOOL_TESTS = True


pytestmark = pytest.mark.parametrize("ndtarget, NDLLM", test_params)


class Test_NDLLM:
    @pytest.mark.vcr
    def test_with_string_llm_configs(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]
        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True

    @pytest.mark.vcr
    def test_with_llm_provider_enums(self, ndtarget, NDLLM):
        llm_configs = [
            NDLLMProviders.GPT_3_5_TURBO,
            NDLLMProviders.GPT_4,
            NDLLMProviders.CLAUDE_HAIKU_4_5_20251001,
            NDLLMProviders.GEMINI_25_PRO,
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True

    @pytest.mark.vcr
    def test_with_no_default_llm_provider(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True
            assert nd_llm.default == 0

    @pytest.mark.vcr
    def test_default_llm_provider_set_by_index(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(default=2, llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True
            assert nd_llm.default_llm.provider == "anthropic"
            assert nd_llm.default_llm.model == "claude-haiku-4-5-20251001"

    @pytest.mark.vcr
    def test_default_llm_provider_set_by_string(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(default="openai/gpt-4", llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True
            assert nd_llm.default_llm.provider == "openai"
            assert nd_llm.default_llm.model == "gpt-4"

    @pytest.mark.vcr
    def test_invalid_default_llm_provider_set_by_string(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(default="openai/gpt-4-abc", llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True
            assert nd_llm.default_llm.provider == "openai"
            assert nd_llm.default_llm.model == "gpt-3.5-turbo"

    @pytest.mark.vcr
    def test_invalid_default_llm_provider_set_index(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(default=4, llm_configs=llm_configs)
            assert len(nd_llm.llm_configs) == 4
            assert nd_llm.llm_configs[0].provider == "openai"
            assert nd_llm.llm_configs[0].model == "gpt-3.5-turbo"
            assert nd_llm.latency_tracking is True
            assert nd_llm.default_llm.provider == "openai"
            assert nd_llm.default_llm.model == "gpt-3.5-turbo"

    @pytest.mark.vcr
    def test_no_max_model_depth(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs)
            assert nd_llm.max_model_depth == len(llm_configs)

    @pytest.mark.vcr
    def test_with_correct_max_model_depth(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs, max_model_depth=2)
            assert nd_llm.max_model_depth == 2

    @pytest.mark.vcr
    def test_with_max_model_depth_too_big(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs, max_model_depth=7)
            assert nd_llm.max_model_depth == len(llm_configs)

    @pytest.mark.vcr
    def test_model_select_with_strings(self, prompt, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)
        session_id, provider = nd_llm.model_select(messages=prompt)
        assert session_id != "NO-SESSION-ID"
        assert provider is not None

    @pytest.mark.skip("Expected to fail with session id")
    @pytest.mark.vcr
    def test_model_select_with_messages_and_model_and_params(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        nd_llm = NDLLM(hash_content=True)
        session_id, provider = nd_llm.model_select(
            messages=openai_style_messages,
            model=llm_configs,
            default=0,
            max_model_depth=2,
        )
        assert session_id != "NO-SESSION-ID"
        assert provider is not None

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_async_model_select_with_strings(
        self, prompt, ndtarget, NDLLM
    ):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)
        session_id, provider = await nd_llm.amodel_select(messages=prompt)
        assert session_id != "NO-SESSION-ID"
        assert provider is not None

    @pytest.mark.skipif(
        SKIP_TOOL_TESTS,
        reason="Skipping tool tests since langchain_core is not installed.",
    )
    @pytest.mark.vcr
    def test_ndllm_tool_calling(self, ndtarget, NDLLM):
        from langchain_core.tools import tool

        llm_configs = [
            "openai/gpt-4-turbo",
        ]
        messages = [{"role": "user", "content": "How much is 3 + 5?"}]

        @tool
        def add_fct(a: int, b: int) -> int:
            """Adds a and b."""
            return a + b

        tools = [add_fct]

        nd_llm = NDLLM(llm_configs=llm_configs)
        nd_llm = nd_llm.bind_tools(tools)

        assert len(nd_llm.tools) == 1

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                result, session_id, provider = nd_llm.invoke(messages=messages)
            return

        result, session_id, provider = nd_llm.invoke(messages=messages)

        assert session_id != "NO-SESSION-ID"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"
        assert result.tool_calls[0]["args"] == {"a": 3, "b": 5}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_TOOL_TESTS,
        reason="Skipping tool tests since langchain_core is not installed.",
    )
    @pytest.mark.vcr
    async def test_ndllm_async_tool_calling(self, ndtarget, NDLLM):
        from langchain_core.tools import tool

        llm_configs = [
            "openai/gpt-4-turbo",
        ]
        messages = [{"role": "user", "content": "How much is 3 + 5?"}]

        @tool
        def add_fct(a: int, b: int) -> int:
            """Adds a and b."""
            return a + b

        tools = [add_fct]

        nd_llm = NDLLM(llm_configs=llm_configs)
        nd_llm = nd_llm.bind_tools(tools)

        assert len(nd_llm.tools) == 1

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                result, session_id, provider = await nd_llm.ainvoke(
                    messages=messages
                )
            return

        result, session_id, provider = await nd_llm.ainvoke(messages=messages)

        assert session_id != "NO-SESSION-ID"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add_fct"
        assert result.tool_calls[0]["args"] == {"a": 3, "b": 5}

    @pytest.mark.skipif(
        SKIP_TOOL_TESTS,
        reason="Skipping tool tests since langchain_core is not installed.",
    )
    @pytest.mark.vcr
    def test_ndllm_tool_calling_stream(self, ndtarget, NDLLM):
        from langchain_core.tools import tool

        llm_configs = [
            "openai/gpt-4-turbo",
        ]
        messages = [{"role": "user", "content": "How much is 3 + 5?"}]

        @tool
        def add_fct(a: int, b: int) -> int:
            """Adds a and b."""
            return a + b

        tools = [add_fct]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)
        nd_llm = nd_llm.bind_tools(tools)

        assert len(nd_llm.tools) == 1

        first = True

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.stream(messages)
            return

        for chunk in nd_llm.stream(messages):
            if first:
                gathered = chunk
                first = False
            else:
                gathered = gathered + chunk

        assert len(gathered.tool_calls) == 1
        assert gathered.tool_calls[0]["name"] == "add_fct"
        assert gathered.tool_calls[0]["args"] == {"a": 3, "b": 5}

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        SKIP_TOOL_TESTS,
        reason="Skipping tool tests since langchain_core is not installed.",
    )
    @pytest.mark.vcr
    async def test_ndllm_tool_calling_astream(self, ndtarget, NDLLM):
        from langchain_core.tools import tool

        llm_configs = [
            "openai/gpt-4-turbo",
        ]
        messages = [{"role": "user", "content": "How much is 3 + 5?"}]

        @tool
        def add_fct(a: int, b: int) -> int:
            """Adds a and b."""
            return a + b

        tools = [add_fct]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)
        nd_llm = nd_llm.bind_tools(tools)

        assert len(nd_llm.tools) == 1

        first = True

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                chunks = await nd_llm.astream(messages)
                chunk = [c for c in chunks][0]
            return

        async for chunk in nd_llm.astream(messages):
            if first:
                gathered = chunk
                first = False
            else:
                gathered = gathered + chunk

        assert len(gathered.tool_calls) == 1
        assert gathered.tool_calls[0]["name"] == "add_fct"
        assert gathered.tool_calls[0]["args"] == {"a": 3, "b": 5}

    @pytest.mark.skipif(
        SKIP_TOOL_TESTS,
        reason="Skipping tool tests since langchain_core is not installed.",
    )
    @pytest.mark.vcr
    def test_ndllm_tool_calling_unsupported_model(self, ndtarget, NDLLM):
        from langchain_core.tools import tool

        # Note: This test is skipped as all current Anthropic models support tools
        # The deprecated claude-2.1 model that didn't support tools has been removed
        pytest.skip("All current Anthropic models support tools")

        llm_configs = [
            "anthropic/claude-opus-4-1-20250805",
        ]

        @tool
        def add_fct(a: int, b: int) -> int:
            """Adds a and b."""
            return a + b

        tools = [add_fct]

        nd_llm = NDLLM(llm_configs=llm_configs)

        # test that error is thrown when trying to bind tools to unsupported model
        with pytest.raises(ApiError):
            nd_llm = nd_llm.bind_tools(tools)

    @pytest.mark.vcr
    def test_ndllm_invoke_response_model(
        self, response_model, ndtarget, NDLLM
    ):
        llm_configs = ["openai/gpt-3.5-turbo"]
        messages = [{"role": "user", "content": "Tell me a joke"}]

        nd_llm = NDLLM(llm_configs=llm_configs)

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                result, _, _ = nd_llm.invoke(
                    messages=messages, response_model=response_model
                )
            return

        result, _, _ = nd_llm.invoke(
            messages=messages, response_model=response_model
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    @pytest.mark.vcr
    def test_ndllm_invoke_with_curly_braces(self, ndtarget, NDLLM):
        llm_configs = ["openai/gpt-3.5-turbo"]
        messages = [
            {
                "role": "user",
                "content": "Tell me a joke about LaTeX like \\mathbb{x}",
            }
        ]

        nd_llm = NDLLM(llm_configs=llm_configs)

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                result, _, _ = nd_llm.invoke(messages=messages)
            return

        result, _, _ = nd_llm.invoke(messages=messages)

        assert result.content, f"Expected content but got {result}"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_ndllm_ainvoke_response_model(
        self, response_model, ndtarget, NDLLM
    ):
        llm_configs = ["openai/gpt-3.5-turbo"]
        messages = [{"role": "user", "content": "Tell me a joke"}]

        nd_llm = NDLLM(llm_configs=llm_configs)

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                await nd_llm.ainvoke(
                    messages=messages, response_model=response_model
                )
            return

        result, _, _ = await nd_llm.ainvoke(
            messages=messages, response_model=response_model
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    @pytest.mark.vcr
    def test_ndllm_stream_response_model(
        self, response_model, ndtarget, NDLLM
    ):
        llm_configs = ["openai/gpt-4"]
        messages = [{"role": "user", "content": "Tell me a joke"}]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.stream(messages=messages, response_model=response_model)
            return

        for chunk in nd_llm.stream(
            messages=messages, response_model=response_model
        ):
            assert isinstance(chunk, response_model)
            last_chunk = chunk

        assert last_chunk.setup
        assert last_chunk.punchline

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_ndllm_astream_response_model(
        self, response_model, ndtarget, NDLLM
    ):
        llm_configs = ["openai/gpt-4"]
        messages = [{"role": "user", "content": "Tell me a joke"}]

        nd_llm = NDLLM(llm_configs=llm_configs, hash_content=True)

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                chunks = await nd_llm.astream(
                    messages=messages, response_model=response_model
                )
                [c for c in chunks][0]
            return

        async for chunk in nd_llm.astream(
            messages=messages, response_model=response_model
        ):
            assert isinstance(chunk, response_model)
            last_chunk = chunk

        assert last_chunk.setup
        assert last_chunk.punchline

    @pytest.mark.vcr
    def test_ndllm_openai_interface(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        nd_llm = NDLLM(hash_content=True)
        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.chat.completions.create(
                    messages=openai_style_messages, model=llm_configs
                )
            return

        result, session_id, provider = nd_llm.chat.completions.create(
            messages=openai_style_messages, model=llm_configs
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

        session_id, result = nd_llm.chat.completions.model_select(
            messages=openai_style_messages, model=llm_configs
        )
        assert session_id != "NO-SESSION-ID"
        assert str(result) in llm_configs

    # no cassettes here as async openai interface can lead to nested event loop errors
    @pytest.mark.asyncio
    async def test_ndllm_async_openai_interface(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]

        nd_llm = NDLLM(hash_content=True)
        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.chat.completions.create(
                    messages=openai_style_messages, model=llm_configs
                )
            return

        result, session_id, provider = nd_llm.chat.completions.create(
            messages=openai_style_messages, model=llm_configs
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    @pytest.mark.vcr
    def test_create_unavailable_error(self, ndtarget, NDLLM):
        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client = NDLLM()
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "Tell me a joke"}],
                    model=["openai/gpt-3.5-turbo"],
                    hash_content=True,
                )

    @pytest.mark.vcr
    def test_preference_id(self, ndtarget, NDLLM):
        llm_configs = [
            "openai/gpt-3.5-turbo",
            "openai/gpt-4",
            "anthropic/claude-haiku-4-5-20251001",
            "google/gemini-2.5-pro",
        ]
        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            nd_llm = NDLLM(llm_configs=llm_configs)
            preference_id = nd_llm.create_preference_id(name="test")
            assert preference_id is not None
            assert isinstance(preference_id, str)


@pytest.mark.vcr
class Test_OpenAI_style_input:
    def test_openai_style_input_invoke(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """Test open ai style input for invoke"""
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.invoke(messages=openai_style_messages)
            return

        result, session_id, _ = nd_llm.invoke(messages=openai_style_messages)

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_openai_style_input_ainvoke(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """Test open ai style input for async invoke"""
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                await nd_llm.ainvoke(messages=openai_style_messages)
            return

        result, session_id, _ = await nd_llm.ainvoke(
            messages=openai_style_messages
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_openai_style_input_stream(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """Test open ai style input for stream"""
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.stream(messages=openai_style_messages)
            return

        for chunk in nd_llm.stream(messages=openai_style_messages):
            assert chunk.type == "AIMessageChunk"
            assert isinstance(chunk.content, str)
            break

    @pytest.mark.asyncio
    async def test_openai_style_input_astream(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """Test open ai style input for async stream"""
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                result = await nd_llm.astream(messages=openai_style_messages)
                [c for c in result][0]
            return

        async for chunk in nd_llm.astream(messages=openai_style_messages):
            assert chunk.type == "AIMessageChunk"
            assert isinstance(chunk.content, str)
            break

    def test_openai_style_input_invoke_claude(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """
        Test open ai style input but with model that is not openai
        """
        provider = NDLLMProviders.CLAUDE_HAIKU_4_5_20251001
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                nd_llm.invoke(messages=openai_style_messages)
            return

        result, session_id, _ = nd_llm.invoke(messages=openai_style_messages)

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_openai_style_input_model_select(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        """Test open ai style input for invoke"""
        provider = NDLLMProviders.GPT_3_5_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NDLLM(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        session_id, best_llm = nd_llm.model_select(
            messages=openai_style_messages
        )

        assert session_id != "NO-SESSION-ID"
        assert best_llm is not None

    def test_llm_without_ndllm_configs(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    tradeoff="cost",
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=[
                "anthropic/claude-haiku-4-5-20251001",
                "openai/gpt-4-1106-preview",
            ],
            tradeoff="cost",
        )

        print("ND session ID: ", session_id)
        print("Result: ", result)

    def test_llm_with_empty_llm_configs(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    tradeoff="cost",
                )
        else:
            with pytest.raises(MissingLLMConfigs):
                result, session_id, provider = client.chat.completions.create(
                    messages=openai_style_messages,
                    tradeoff="cost",
                )

    def test_llm_with_tradeoff(self, openai_style_messages, ndtarget, NDLLM):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    tradeoff="cost",
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=[
                "anthropic/claude-haiku-4-5-20251001",
                "openai/gpt-4-1106-preview",
            ],
            tradeoff="cost",
        )

        print("ND session ID: ", session_id)
        print("Result: ", result)

    def test_llm_with_tradeoff_with_exception(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    tradeoff="speed",
                )
        else:
            with pytest.raises(ValueError):
                result, session_id, provider = client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    tradeoff="speed",
                )

    def test_create_with_default(self, openai_style_messages, ndtarget, NDLLM):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    default=1,
                    hash_content=True,
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=[
                "anthropic/claude-haiku-4-5-20251001",
                "openai/gpt-4-1106-preview",
            ],
            default=1,
            hash_content=True,
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0
        assert client.default == 1

    def test_create_with_max_model_depth(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    max_model_depth=2,
                    hash_content=True,
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=[
                "anthropic/claude-haiku-4-5-20251001",
                "openai/gpt-4-1106-preview",
            ],
            max_model_depth=2,
            hash_content=True,
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_create_with_latency_tracking(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=[
                        "anthropic/claude-haiku-4-5-20251001",
                        "openai/gpt-4-1106-preview",
                    ],
                    latency_tracking=False,
                    hash_content=True,
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=[
                "anthropic/claude-haiku-4-5-20251001",
                "openai/gpt-4-1106-preview",
            ],
            latency_tracking=False,
            hash_content=True,
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    @pytest.mark.skip("Expected to fail with session id")
    def test_create_with_preference_id(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=["openai/gpt-3.5-turbo", "anthropic/claude-haiku-4-5-20251001"],
                    preference_id="de2852f7-10f6-4dd3-8428-970022a986ca",
                    hash_content=True,
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=["openai/gpt-3.5-turbo", "anthropic/claude-haiku-4-5-20251001"],
            preference_id="de2852f7-10f6-4dd3-8428-970022a986ca",
            hash_content=True,
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_create_with_provider_system_prompts(
        self, openai_style_messages, ndtarget, NDLLM
    ):
        client = NDLLM()
        llm_configs = [
            LLMConfig(
                provider="anthropic",
                model="claude-3-haiku-20240307",
                system_prompt="You are a helpful assistant",
            ),
            LLMConfig(
                provider="openai",
                model="gpt-4-1106-preview",
                system_prompt="You are NOT a helpful assistant",
            ),
        ]

        if ndtarget == _NDClientTarget.ROUTER:
            with pytest.raises(CreateUnavailableError):
                client.chat.completions.create(
                    messages=openai_style_messages,
                    model=llm_configs,
                    default=1,
                    hash_content=True,
                )
            return

        result, session_id, provider = client.chat.completions.create(
            messages=openai_style_messages,
            model=llm_configs,
            default=1,
            hash_content=True,
        )
        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0
        assert client.default == 1
