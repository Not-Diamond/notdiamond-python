from typing import Union

from langchain_cohere import CohereEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import HuggingfaceEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from ...exceptions import UnsupportedEmbeddingProvider
from ...llms.client import NotDiamond
from ...llms.config import EmbeddingConfig, LLMConfig


def get_llm(llm_config_or_str: Union[LLMConfig, str]) -> LangchainLLMWrapper:
    """
    Build the LLM object compatible with evaluation metrics.

    Parameters:
        llm_config_or_str (Union[LLMConfig, str]): a LLMConfig object or a model string
            that specifies the LLM to construct.
    """
    if isinstance(llm_config_or_str, str):
        llm_config = LLMConfig.from_string(llm_config_or_str)
    else:
        llm_config = llm_config_or_str

    lc_llm = NotDiamond._llm_from_config(llm_config)
    return LangchainLLMWrapper(lc_llm)


def get_embedding(
    embedding_model_config_or_str: Union[EmbeddingConfig, str]
) -> Union[LangchainEmbeddingsWrapper, HuggingfaceEmbeddings]:
    """
    Build the embedding model object compatible with evaluation metrics.

    Parameters:
        embedding_model_config_or_str (Union[EmbeddingConfig, str]): an EmbeddingConfig object
            or an embedding model string that specifies the embedding model to construct.
    """
    if isinstance(embedding_model_config_or_str, str):
        embedding_config = EmbeddingConfig.from_string(
            embedding_model_config_or_str
        )
    else:
        embedding_config = embedding_model_config_or_str

    if embedding_config.provider == "openai":
        lc_embedding = OpenAIEmbeddings(
            model=embedding_config.model, **embedding_config.kwargs
        )

    elif embedding_config.provider == "cohere":
        lc_embedding = CohereEmbeddings(
            model=embedding_config.model, **embedding_config.kwargs
        )

    elif embedding_config.provider == "mistral":
        lc_embedding = MistralAIEmbeddings(
            model=embedding_config.model, **embedding_config.kwargs
        )

    elif embedding_config.provider == "huggingface":
        return HuggingfaceEmbeddings(model_name=embedding_config.model)

    else:
        raise UnsupportedEmbeddingProvider(
            f"Embedding model {str(embedding_config)} not supported."
        )
    return LangchainEmbeddingsWrapper(lc_embedding)
