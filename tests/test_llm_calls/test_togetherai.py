import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.metrics.metric import Metric


@pytest.mark.vcr
@pytest.mark.longrun
class Test_TogetherAI_LLMs:
    def test_mistral_7b_instruct_v02(self):
        provider = NDLLMProviders.TOGETHER_MISTRAL_7B_INSTRUCT_V0_2
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_mixtral_8x7b_instruct_v01(self):
        provider = NDLLMProviders.TOGETHER_MIXTRAL_8X7B_INSTRUCT_V0_1
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    @pytest.mark.skip(
        "Currently failing with a 400 for 'max_new_tokens', an undocumented param"
    )
    def test_mixtral_8x22b_instruct_v01(self):
        provider = NDLLMProviders.TOGETHER_MIXTRAL_8X22B_INSTRUCT_V0_1
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "Write a short novel."}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    @pytest.mark.skip("Currently failing, need to debug test")
    def test_llama_3_70b_chat_hf(self):
        provider = NDLLMProviders.TOGETHER_LLAMA_3_70B_CHAT_HF
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_llama_3_8b_chat_hf(self):
        provider = NDLLMProviders.TOGETHER_LLAMA_3_8B_CHAT_HF
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_qwen2_72b(self):
        provider = NDLLMProviders.TOGETHER_QWEN2_72B_INSTRUCT
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_llama_3_1_8b_instruct_turbo(self):
        provider = NDLLMProviders.TOGETHER_LLAMA_3_1_8B_INSTRUCT_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_llama_3_1_70b_instruct_turbo(self):
        provider = NDLLMProviders.TOGETHER_LLAMA_3_1_70B_INSTRUCT_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0

    def test_llama_3_1_405b_instruct_turbo(self):
        provider = NDLLMProviders.TOGETHER_LLAMA_3_1_405B_INSTRUCT_TURBO
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider],
            latency_tracking=False,
            hash_content=False,
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result) > 0
