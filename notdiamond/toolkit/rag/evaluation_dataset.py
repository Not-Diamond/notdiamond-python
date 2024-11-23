from dataclasses import dataclass
from typing import Dict, List, Union, overload

from ragas import MultiTurnSample, SingleTurnSample
from ragas.dataset_schema import RagasDataset


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
