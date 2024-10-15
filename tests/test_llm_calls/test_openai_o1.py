import pytest

from notdiamond.llms.client import NotDiamond
from notdiamond.llms.providers import NDLLMProviders

test_providers = [
    provider
    for provider in NDLLMProviders
    if provider.provider == "openai" and provider.model[:2] == "o1"
]


@pytest.mark.vcr
@pytest.mark.longrun
@pytest.mark.parametrize("provider", test_providers)
def test_o1_with_system_prompt(provider):
    nd_llm = NotDiamond(
        llm_configs=[provider], latency_tracking=False, hash_content=True
    )
    result, session_id, _ = nd_llm.invoke(
        [
            {"role": "system", "content": "You are a funny AI"},
            {"role": "user", "content": "Tell me a joke"},
        ],
    )

    assert session_id != "NO-SESSION-ID"
    assert len(result.content) > 0
