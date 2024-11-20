from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

import optuna
import pandas as pd
from langchain_core.callbacks import Callbacks
from langchain_core.embeddings import Embeddings as LangchainEmbeddings
from langchain_core.language_models import BaseLanguageModel as LangchainLLM
from langchain_core.prompt_values import StringPromptValue
from ragas import EvaluationDataset, MultiTurnSample, SingleTurnSample
from ragas._analytics import track_was_completed
from ragas.cost import TokenUsageParser
from ragas.dataset_schema import RagasDataset
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.evaluation import evaluate as ragas_evaluate
from ragas.llms import BaseRagasLLM
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig
from tqdm import tqdm

from notdiamond.toolkit.rag.workflow import BaseNDRagWorkflow

from ...llms.config import LLMConfig
from .llms import get_llm


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


class RAGSample(SingleTurnSample):
    """
    Represents RAG evaluation samples.

    Attributes:
        user_input (Optional[str]): The input query from the user.
        retrieved_contexts (Optional[List[str]]): List of contexts retrieved for the query.
        reference_contexts (Optional[List[str]]): List of reference contexts for the query.
        response (Optional[str]): The generated response for the query.
        generation_prompt (str): The input prompt to the generator LLM.
        generator_llm (str): The LLM used to generate the response.
        multi_responses (Optional[List[str]]): List of multiple responses generated for the query.
        reference (Optional[str]): The reference answer for the query.
        rubric (Optional[Dict[str, str]]): Evaluation rubric for the sample.
    """

    generation_prompt: str
    generator_llm: str


@dataclass
class RAGEvaluationDataset(RagasDataset[RAGSample]):
    """
    Represents a dataset of RAG evaluation samples.

    Attributes:
        samples (List[BaseSample]): A list of evaluation samples.

    Methods:
        validate_samples(samples): Validates that all samples are of the same type.
        get_sample_type(): Returns the type of the samples in the dataset.
        to_hf_dataset(): Converts the dataset to a Hugging Face Dataset.
        to_pandas(): Converts the dataset to a pandas DataFrame.
        features(): Returns the features of the samples.
        from_list(mapping): Creates an EvaluationDataset from a list of dictionaries.
        from_dict(mapping): Creates an EvaluationDataset from a dictionary.
        to_csv(path): Converts the dataset to a CSV file.
        to_jsonl(path): Converts the dataset to a JSONL file.
        from_jsonl(path): Creates an EvaluationDataset from a JSONL file.
    """

    @overload
    def __getitem__(self, idx: int) -> RAGSample:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "RAGEvaluationDataset":
        ...

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[RAGSample, "RAGEvaluationDataset"]:
        if isinstance(idx, int):
            return self.samples[idx]
        elif isinstance(idx, slice):
            return type(self)(samples=self.samples[idx])
        else:
            raise TypeError("Index must be int or slice")

    def is_multi_turn(self) -> bool:
        return False

    def to_list(self) -> List[Dict]:
        rows = [sample.to_dict() for sample in self.samples]
        return rows

    @classmethod
    def from_list(cls, data: List[Dict]):
        samples = []
        if all(
            "user_input" in item and isinstance(data[0]["user_input"], list)
            for item in data
        ):
            samples.extend(MultiTurnSample(**sample) for sample in data)
        else:
            samples.extend(SingleTurnSample(**sample) for sample in data)
        return cls(samples=samples)

    def __repr__(self) -> str:
        return f"RAGEvaluationDataset(features={self.features()}, len={len(self.samples)})"


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
    llm: Optional[BaseRagasLLM | LangchainLLM] = None,
    embeddings: Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
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
        response = llm.generate(
            StringPromptValue(text=sample.generation_prompt),
            temperature=temperature,
        )
        eval_sample = RAGSample(
            user_input=sample.user_input,
            retrieved_contexts=sample.retrieved_contexts,
            reference_contexts=sample.reference_contexts,
            response=response.content,
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
    llm: Optional[BaseRagasLLM | LangchainLLM] = None,
    embeddings: Optional[BaseRagasEmbeddings | LangchainEmbeddings] = None,
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
