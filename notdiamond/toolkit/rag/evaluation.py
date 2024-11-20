import optuna
import pandas as pd
from ragas import EvaluationDataset, SingleTurnSample

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow


def get_eval_dataset(test_queries: pd.DataFrame, workflow: BaseNDRagWorkflow):
    """
    Create a dataset of SingleTurnSample objects to evaluate the performance of a RAG workflow.

    Args:
        test_queries: A pandas DataFrame with schema implied by the method below.
        workflow: BaseNDRagWorkflow subclass created by the user.

    Schema for test_queries can be found below
    https://docs.ragas.io/en/stable/references/evaluation_schema/#ragas.dataset_schema.SingleTurnSample
    """
    samples = []
    # for query, reference in test_queries:
    for _, row in test_queries.iterrows():
        query = row["query"]
        reference = row["reference"]
        generation_prompt = row.get(
            "generation_prompt"
        )  # avail w/ t7's completed impl
        generator_llm = row.get(
            "generator_llm"
        )  # avail w/ t7's completed impl

        retrieved_contexts = workflow.get_retrieved_context(query)
        response = workflow.get_response(query)

        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference,
            generation_prompt=generation_prompt,
            generator_llm=generator_llm,
        )
        samples.append(sample)
    eval_dataset = EvaluationDataset(samples)
    return eval_dataset


def parameter_optimizer(
    workflow: BaseNDRagWorkflow, n_trials: int, maximize: bool = True
):
    if maximize:
        direction = "maximize"
    else:
        direction = "minimize"
    study = optuna.create_study(
        study_name=workflow.job_name, direction=direction
    )
    study.optimize(workflow.objective, n_trials=n_trials)

    print(study.best_params)
    return {"best_params": study.best_params, "trials": study.trials}
