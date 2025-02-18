import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders
from notdiamond.metrics.metric import Metric


@pytest.mark.vcr
@pytest.mark.longrun
class Test_Perplexity_LLMs:
    def test_sonar(self):
        provider = NDLLMProviders.SONAR
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
