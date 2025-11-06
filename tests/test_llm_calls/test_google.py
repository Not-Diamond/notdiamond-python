import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.metrics.metric import Metric


@pytest.mark.vcr
@pytest.mark.longrun
@pytest.mark.skip(reason="Skipping due to API issues")
class Test_Google_LLMs:
    def test_gemini_pro_with_prompt_template(self, prompt_template):
        provider = NDLLMProviders.GEMINI_25_PRO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_with_chat_prompt_template(self, chat_prompt_template):
        provider = NDLLMProviders.GEMINI_25_PRO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=chat_prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GEMINI_25_PRO
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

    def test_gemini_pro_with_openai_tool_calling(self, openai_tools_fixture):
        provider = NDLLMProviders.GEMINI_25_PRO
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

    def test_gemini_pro_response_model(self, response_model):
        provider = NDLLMProviders.GEMINI_25_PRO
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, _, _ = nd_llm.invoke(
            {"role": "user", "content": "Tell me a joke"},
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gemini_pro_1_latest_with_prompt_template(self, prompt_template):
        provider = NDLLMProviders.GEMINI_25_PRO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_15_with_prompt_template(self, prompt_template):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_15_with_chat_prompt_template(
        self, chat_prompt_template
    ):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=chat_prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_15_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
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

    def test_gemini_pro_15_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
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

    def test_gemini_pro_15_response_model(self, response_model):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            {"role": "user", "content": "Tell me a joke"},
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gemini_flash_15_with_prompt_template(self, prompt_template):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_flash_15_with_chat_prompt_template(
        self, chat_prompt_template
    ):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=chat_prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_flash_15_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
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

    def test_gemini_flash_15_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
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

    def test_gemini_flash_15_response_model(self, response_model):
        provider = NDLLMProviders.GEMINI_15_FLASH_LATEST
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            {"role": "user", "content": "Tell me a joke"},
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gemini_pro_15_exp_0801_with_prompt_template(
        self, prompt_template
    ):
        provider = NDLLMProviders.GEMINI_15_PRO_EXP_0801
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_15_exp_0801_with_chat_prompt_template(
        self, chat_prompt_template
    ):
        provider = NDLLMProviders.GEMINI_15_PRO_EXP_0801
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=chat_prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_pro_15_exp_0801_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GEMINI_15_PRO_EXP_0801
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

    def test_gemini_pro_15_exp_0801_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GEMINI_15_PRO_EXP_0801
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

    def test_gemini_pro_15_exp_0801_response_model(self, response_model):
        provider = NDLLMProviders.GEMINI_15_PRO_EXP_0801
        provider.kwargs = {"max_tokens": 200}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, _, _ = nd_llm.invoke(
            {"role": "user", "content": "Tell me a joke"},
            response_model=response_model,
        )

        assert isinstance(result, response_model)
        assert result.setup
        assert result.punchline

    def test_gemini_20_flash_with_prompt_template(self, prompt_template):
        provider = NDLLMProviders.GEMINI_20_FLASH
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_20_flash_with_chat_prompt_template(
        self, chat_prompt_template
    ):
        provider = NDLLMProviders.GEMINI_20_FLASH
        provider.kwargs = {"max_tokens": 10}

        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            prompt_template=chat_prompt_template,
            metric=Metric("accuracy"),
            input={"query": "Write a short novel."},
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_gemini_20_flash_with_tool_calling(self, tools_fixture):
        provider = NDLLMProviders.GEMINI_20_FLASH
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

    def test_gemini_20_flash_with_openai_tool_calling(
        self, openai_tools_fixture
    ):
        provider = NDLLMProviders.GEMINI_20_FLASH
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
