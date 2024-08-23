import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.metrics.metric import Metric


@pytest.mark.longrun
class Test_Perplexity_LLMs:
    def test_llama_3_1_sonar_large_128k_online(self):
        provider = NDLLMProviders.LLAMA_3_1_SONAR_LARGE_128K_ONLINE
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
