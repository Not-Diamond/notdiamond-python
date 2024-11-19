#!/usr/bin/env python3

from typing import List, Optional, Sequence, Union

import pandas as pd
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document as LCDocument
from llama_index.core.base.embeddings.base import (
    BaseEmbedding as LlamaIndexEmbedding,
)
from llama_index.core.base.llms.base import BaseLLM as LlamaIndexLLM
from llama_index.core.schema import Document as LlamaIndexDocument
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.transforms import Transforms


class TestDataGenerator(TestsetGenerator):
    def __init__(
        self,
        llm: BaseRagasLLM,
        embedding_model: BaseRagasEmbeddings,
        knowledge_graph: KnowledgeGraph = KnowledgeGraph(),
        persona_list: Optional[List[Persona]] = None,
    ):
        super().__init__(
            llm=llm,
            embedding_model=embedding_model,
            knowledge_graph=knowledge_graph,
            persona_list=persona_list,
        )

    def generate_from_docs(
        self,
        documents: Union[Sequence[LCDocument], Sequence[LlamaIndexDocument]],
        testset_size: int,
        transforms: Optional[Transforms] = None,
        transforms_llm: Optional[Union[BaseRagasLLM, LlamaIndexLLM]] = None,
        transforms_embedding_model: Optional[
            Union[BaseRagasEmbeddings, LlamaIndexEmbedding]
        ] = None,
        query_distribution: Optional[QueryDistribution] = None,
        run_config: Optional[RunConfig] = None,
        callbacks: Optional[Callbacks] = None,
        with_debugging_logs: bool = False,
        raise_exceptions: bool = True,
    ) -> pd.DataFrame:
        assert isinstance(
            documents, list
        ), "Documents must be a list of langchain or llama-index documents."

        if isinstance(documents[0], LCDocument):
            dataset = self.generate_with_langchain_docs(
                documents=documents,
                testset_size=testset_size,
                transforms=transforms,
                transforms_llm=transforms_llm,
                transforms_embedding_model=transforms_embedding_model,
                query_distribution=query_distribution,
                run_config=run_config,
                callbacks=callbacks,
                with_debugging_logs=with_debugging_logs,
                raise_exceptions=raise_exceptions,
            )
            return dataset.to_pandas()

        elif isinstance(documents[0], LlamaIndexDocument):
            dataset = self.generate_with_llamaindex_docs(
                documents=documents,
                testset_size=testset_size,
                transforms=transforms,
                transforms_llm=transforms_llm,
                transforms_embedding_model=transforms_embedding_model,
                query_distribution=query_distribution,
                run_config=run_config,
                callbacks=callbacks,
                with_debugging_logs=with_debugging_logs,
                raise_exceptions=raise_exceptions,
            )
            return dataset.to_pandas()

        raise Exception(f"Document type {type(documents[0])} not supported.")
