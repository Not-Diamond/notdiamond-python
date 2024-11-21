from typing import Dict, List, Optional, Sequence, Tuple, Union

import optuna
import pandas as pd
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from langchain_core.prompt_values import StringPromptValue
from ragas import EvaluationDataset, SingleTurnSample
from ragas._analytics import track_was_completed
from ragas.cost import TokenUsageParser
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.llms import BaseRagasLLM
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig
from tqdm import tqdm

from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit.rag.evaluation_dataset import (
    RAGEvaluationDataset,
    RAGSample,
)
from notdiamond.toolkit.rag.llms import get_llm
from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow


def get_eval_dataset(
    test_queries: pd.DataFrame,
    workflow: BaseNDRagWorkflow,
    generation_prompt: str = None,
    generator_llm: Union[LLMConfig, str] = None,
):
    """
    Create a dataset of RAGSample objects to evaluate the performance of a RAG workflow.

    Args:
        test_queries: A pandas DataFrame with schema implied by the method below.
        workflow: BaseNDRagWorkflow subclass created by the user.

    Schema for test_queries can be found below
    https://docs.ragas.io/en/stable/references/evaluation_schema/#ragas.dataset_schema.SingleTurnSample
    """
    samples = []
    for _, row in test_queries.iterrows():
        query = row["user_input"]
        reference = row["reference"]
        generation_prompt = generation_prompt or row.get("generation_prompt")
        generator_llm = generator_llm or row.get("generator_llm")

        retrieved_contexts = workflow.get_retrieved_context(query)
        response = workflow.get_response(query)

        sample = RAGSample(
            user_input=query,
            retrieved_contexts=retrieved_contexts,
            response=response,
            reference=reference,
            generation_prompt=generation_prompt,
            generator_llm=generator_llm,
        )
        samples.append(sample)
    eval_dataset = RAGEvaluationDataset(samples)
    return eval_dataset


def auto_optimize(
    workflow: BaseNDRagWorkflow, n_trials: int, maximize: bool = True
):
    direction = "maximize" if maximize else "minimize"
    study = optuna.create_study(
        study_name=workflow.job_name, direction=direction
    )
    study.optimize(workflow._outer_objective, n_trials=n_trials)
    workflow._set_param_values(study.best_params)
    return {"best_params": study.best_params, "trials": study.trials}


def _map_to_ragas_samples(
    dataset: RAGEvaluationDataset,
) -> Tuple[EvaluationDataset, pd.DataFrame]:
    ragas_samples = []
    extra_columns = {
        "generation_prompt": [],
    }
    for sample in dataset:
        ragas_sample = SingleTurnSample(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference_contexts=sample.reference_contexts,
            response=sample.response,
            multi_responses=sample.multi_responses,
            reference=sample.reference,
            rubrics=sample.rubrics,
        )
        ragas_samples.append(ragas_sample)
        extra_columns["generation_prompt"].append(sample.generation_prompt)

    extra_columns_df = pd.DataFrame.from_dict(extra_columns)
    return EvaluationDataset(ragas_samples), extra_columns_df


def _evaluate_dataset(
    generator_llm: LLMConfig,
    dataset: EvaluationDataset,
    metrics: Optional[Sequence[Metric]] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[
        Union[BaseRagasEmbeddings, LangchainEmbeddings]
    ] = None,
    callbacks: Callbacks = None,
    in_ci: bool = False,
    run_config: RunConfig = RunConfig(),
    token_usage_parser: Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    print(f"Evaluating generations from {str(generator_llm)}")

    result = ragas_evaluate(
        dataset,
        metrics,
        llm,
        embeddings,
        callbacks,
        in_ci,
        run_config,
        token_usage_parser,
        raise_exceptions,
        column_map,
        show_progress,
        batch_size,
    )
    return result.to_pandas()


def _generate_rag_eval_dataset(
    generator_llm: LLMConfig, dataset: RAGEvaluationDataset
) -> RAGEvaluationDataset:
    print(f"Generating responses from {str(generator_llm)}")

    llm = get_llm(generator_llm)
    temperature = generator_llm.kwargs.get("temperature", 0.7)

    eval_samples = []
    for sample in tqdm(dataset):
        response = llm.generate_text(
            StringPromptValue(text=sample.generation_prompt),
            temperature=temperature,
        )
        eval_sample = RAGSample(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference_contexts=sample.reference_contexts,
            response=response.generations[0][0].text,
            multi_responses=sample.multi_responses,
            reference=sample.reference,
            rubrics=sample.rubrics,
            generation_prompt=sample.generation_prompt,
            generator_llm=str(generator_llm),
        )
        eval_samples.append(eval_sample)
    return RAGEvaluationDataset(eval_samples)


@track_was_completed
def evaluate(
    dataset: RAGEvaluationDataset,
    metrics: Optional[Sequence[Metric]] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[
        Union[BaseRagasEmbeddings, LangchainEmbeddings]
    ] = None,
    callbacks: Callbacks = None,
    in_ci: bool = False,
    run_config: RunConfig = RunConfig(),
    token_usage_parser: Optional[TokenUsageParser] = None,
    raise_exceptions: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
    generator_llms: List[LLMConfig] = [],
) -> Dict[str, pd.DataFrame]:
    dataset_llm_str = dataset[0].generator_llm
    dataset_llm_config = LLMConfig.from_string(dataset_llm_str)

    if dataset_llm_config not in generator_llms:
        generator_llms.append(dataset_llm_config)

    ragas_dataset, extra_columns = _map_to_ragas_samples(dataset)

    dataset_results = _evaluate_dataset(
        dataset_llm_config,
        ragas_dataset,
        metrics,
        llm,
        embeddings,
        callbacks,
        in_ci,
        run_config,
        token_usage_parser,
        raise_exceptions,
        column_map,
        show_progress,
        batch_size,
    )

    evaluation_results = {
        str(dataset_llm_config): pd.concat(
            [dataset_results, extra_columns], axis=1
        )
    }

    for llm_config in generator_llms:
        if str(llm_config) in evaluation_results:
            continue

        llm_dataset = _generate_rag_eval_dataset(llm_config, dataset)
        ragas_dataset, extra_columns = _map_to_ragas_samples(llm_dataset)
        dataset_results = _evaluate_dataset(
            llm_config,
            ragas_dataset,
            metrics,
            llm,
            embeddings,
            callbacks,
            in_ci,
            run_config,
            token_usage_parser,
            raise_exceptions,
            column_map,
            show_progress,
            batch_size,
        )
        evaluation_results[str(llm_config)] = pd.concat(
            [dataset_results, extra_columns], axis=1
        )
    return evaluation_results
