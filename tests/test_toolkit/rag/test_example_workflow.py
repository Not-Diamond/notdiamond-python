from typing import Annotated, Any, List

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI

from notdiamond.toolkit.rag.evaluation import auto_optimize, evaluate
from notdiamond.toolkit.rag.llms import get_embedding, get_llm
from notdiamond.toolkit.rag.metrics import (
    FactualCorrectness,
    Faithfulness,
    LLMContextRecall,
    SemanticSimilarity,
)
from notdiamond.toolkit.rag.workflow import (
    BaseNDRagWorkflow,
    CategoricalValueOptions,
    FloatValueRange,
    IntValueRange,
)


class ExampleNDRagWorkflow(BaseNDRagWorkflow):
    parameter_specs = {
        "chunk_size": (Annotated[int, IntValueRange(1000, 2500, 500)], 1000),
        "chunk_overlap": (Annotated[int, IntValueRange(50, 200, 25)], 100),
        "top_k": (Annotated[int, IntValueRange(1, 20, 1)], 5),
        "algo": (
            Annotated[
                str,
                CategoricalValueOptions(
                    [
                        "BM25",
                        "openai_small",
                        "openai_large",
                        "cohere_eng",
                        "cohere_multi",
                    ]
                ),
            ],
            "BM25",
        ),
        "temperature": (Annotated[float, FloatValueRange(0.0, 1.0, 0.1)], 0.9),
    }

    def job_name(self):
        return "my-awesome-workflow"

    def rag_workflow(self, documents: Any):
        self.index = VectorStoreIndex.from_documents(
            documents,
            transformations=[
                SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            ],
        )

        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )

        response_synthesizer = get_response_synthesizer(llm=OpenAI("gpt-4o"))
        self.query_engine = RetrieverQueryEngine(
            retriever=self.retriever, response_synthesizer=response_synthesizer
        )

        print(f"Dummy access for temperature: {self.temperature}")
        print(f"Dummy access for algo: {self.algo}")

    def get_retrieved_context(self, query: str) -> List[str]:
        return self.retriever.retrieve(query)

    def get_response(self, query: str) -> str:
        return self.query_engine.query(query)

    def objective(self):
        evaluator_llm = get_llm("openai/gpt-4o")
        evaluator_embeddings = get_embedding("openai/text-embedding-3-large")
        metrics = [
            LLMContextRecall(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            SemanticSimilarity(embeddings=evaluator_embeddings),
        ]
        results = evaluate(dataset=self.evaluation_dataset, metrics=metrics)
        return results["openai/gpt-4o"]["faithfulness"].mean()


def test_example_workflow(dataset, llamaindex_documents):
    example_workflow = ExampleNDRagWorkflow(
        dataset, llamaindex_documents, objective_maximize=True
    )
    results = auto_optimize(example_workflow, n_trials=1)
    assert results["best_params"] is not None


def test_workflow_attrs_init(dataset, llamaindex_documents):
    example_workflow = ExampleNDRagWorkflow(
        dataset, llamaindex_documents, objective_maximize=True
    )
    assert hasattr(example_workflow, "index")
