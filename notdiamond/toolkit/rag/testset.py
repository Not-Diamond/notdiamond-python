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
from ragas.embeddings import BaseRagasEmbeddings, LlamaIndexEmbeddingsWrapper
from ragas.llms import BaseRagasLLM, LlamaIndexLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import QueryDistribution
from ragas.testset.transforms import (
    Transforms,
    apply_transforms,
    default_transforms,
)


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

    def generate_with_llamaindex_docs(
        self,
        documents: Sequence[LlamaIndexDocument],
        testset_size: int,
        transforms: Optional[Transforms] = None,
        transforms_llm: Optional[LlamaIndexLLM] = None,
        transforms_embedding_model: Optional[LlamaIndexEmbedding] = None,
        query_distribution: Optional[QueryDistribution] = None,
        run_config: Optional[RunConfig] = None,
        callbacks: Optional[Callbacks] = None,
        with_debugging_logs=False,
        raise_exceptions: bool = True,
    ):
        """
        Generates an evaluation dataset based on given scenarios and parameters.
        """

        run_config = run_config or RunConfig()

        # force the user to provide an llm and embedding client to prevent use of default LLMs
        if not self.llm and not transforms_llm:
            raise ValueError(
                "An llm client was not provided."
                " Provide an LLM on TestsetGenerator instantiation or as an argument for transforms_llm parameter."
                " Alternatively you can provide your own transforms through the `transforms` parameter."
            )
        if not self.embedding_model and not transforms_embedding_model:
            raise ValueError(
                "An embedding client was not provided."
                " Provide an embedding through the transforms_embedding_model parameter."
                " Alternatively you can provide your own transforms through the `transforms` parameter."
            )

        if not transforms:
            # use TestsetGenerator's LLM and embedding model if no transforms_llm or transforms_embedding_model is provided
            if transforms_llm is None:
                llm_for_transforms = self.llm
            else:
                llm_for_transforms = LlamaIndexLLMWrapper(transforms_llm)
            if transforms_embedding_model is None:
                embedding_model_for_transforms = self.embedding_model
            else:
                embedding_model_for_transforms = LlamaIndexEmbeddingsWrapper(
                    transforms_embedding_model
                )

            # create the transforms
            transforms = default_transforms(
                documents=[
                    LCDocument(page_content=doc.text) for doc in documents
                ],
                llm=llm_for_transforms,
                embedding_model=embedding_model_for_transforms,
            )

        # convert the documents to Ragas nodes
        nodes = []
        for doc in documents:
            if doc.text is not None and doc.text.strip() != "":
                node = Node(
                    type=NodeType.DOCUMENT,
                    properties={
                        "page_content": doc.text,
                        "document_metadata": doc.metadata,
                    },
                )
                nodes.append(node)

        kg = KnowledgeGraph(nodes=nodes)

        # apply transforms and update the knowledge graph
        apply_transforms(kg, transforms, run_config)
        self.knowledge_graph = kg

        return self.generate(
            testset_size=testset_size,
            query_distribution=query_distribution,
            run_config=run_config,
            callbacks=callbacks,
            with_debugging_logs=with_debugging_logs,
            raise_exceptions=raise_exceptions,
        )
