from typing import Any

from ragas import EvaluationDataset, SingleTurnSample

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow


def get_eval_dataset(
    test_queries: Any, workflow: BaseNDRagWorkflow, llm_prompt: str
):
    # [a9] let's handle this for users - it's boilerplate - they shouldn't
    # touch this method
    # todo [t7 + a9] - should this be a separate method / class?
    # should this move out of hyperopt logic?
    samples = []
    for query, reference in test_queries:
        retrieved_contexts = workflow.get_retrieved_context(query)
        response = workflow.get_response(query)

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference,
            llm_prompt=llm_prompt,
            generator_llm="openai/gpt-4o",
        )
        samples.append(sample)
    eval_dataset = EvaluationDataset(samples)
    return eval_dataset
