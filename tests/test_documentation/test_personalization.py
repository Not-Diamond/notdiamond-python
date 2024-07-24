from notdiamond.llms.client import NotDiamond
from notdiamond.metrics.metric import Metric


def test_personalization():
    prompt = [
        {
            "role": "user",
            "content": "You are a world class software developer. Write a merge sort in Python.",
        }
    ]

    llm_configs = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4-turbo-preview",
        "anthropic/claude-2.1",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-opus-20240229",
        "google/gemini-pro",
    ]

    nd_llm = NotDiamond(
        llm_configs=llm_configs,
        preference_id="5c5af286-b715-4d8b-8cf9-151230ef96a3",
    )
    metric = Metric("accuracy")

    result, session_id, provider = nd_llm.invoke(prompt, metric=metric)

    # Application logic...
    # Let's assume the result from the LLM achieved the outcome you're looking for!

    score = metric.feedback(
        session_id=session_id, llm_config=provider, value=1
    )
    print(score)
