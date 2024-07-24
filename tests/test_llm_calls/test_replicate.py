import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.metrics.metric import Metric


@pytest.mark.longrun
class Test_Replicate_LLMs:
    def test_mistral_7b_instruct_v02(self):
        provider = NDLLMProviders.REPLICATE_MISTRAL_7B_INSTRUCT_V0_2
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_mixtral_8x7b_instruct_v01(self):
        provider = NDLLMProviders.REPLICATE_MIXTRAL_8X7B_INSTRUCT_V0_1
        provider.kwargs = {"max_tokens": 10}
        nd_llm = NotDiamond(
            llm_configs=[provider], latency_tracking=False, hash_content=True
        )
        result, session_id, _ = nd_llm.invoke(
            messages=[{"role": "user", "content": "How much is 3 + 5?"}],
            metric=Metric("accuracy"),
        )

        assert session_id != "NO-SESSION-ID"
        assert len(result.content) > 0

    def test_llama_3_70b_instruct(self):
        provider = NDLLMProviders.REPLICATE_META_LLAMA_3_70B_INSTRUCT
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
        assert len(result.content) > 0

    def test_llama_3_8b_instruct_with_prompt_template(self):
        provider = NDLLMProviders.REPLICATE_META_LLAMA_3_8B_INSTRUCT
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
        assert len(result.content) > 0

    def test_llama_3_1_405b_instruct(self):
        provider = NDLLMProviders.REPLICATE_META_LLAMA_3_1_405B_INSTRUCT
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
        assert len(result.content) > 0
