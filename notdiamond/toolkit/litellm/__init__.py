from enum import Enum
from typing import List, Optional, Union

from litellm.__init__ import *  # noqa

from .litellm_notdiamond import NotDiamondConfig  # noqa

notdiamond_key: Optional[str] = None


class LlmProviders(str, Enum):
    OPENAI = "openai"
    CUSTOM_OPENAI = "custom_openai"
    TEXT_COMPLETION_OPENAI = "text-completion-openai"
    COHERE = "cohere"
    COHERE_CHAT = "cohere_chat"
    CLARIFAI = "clarifai"
    ANTHROPIC = "anthropic"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    TOGETHER_AI = "together_ai"
    OPENROUTER = "openrouter"
    VERTEX_AI = "vertex_ai"
    VERTEX_AI_BETA = "vertex_ai_beta"
    PALM = "palm"
    GEMINI = "gemini"
    AI21 = "ai21"
    BASETEN = "baseten"
    AZURE = "azure"
    AZURE_TEXT = "azure_text"
    AZURE_AI = "azure_ai"
    SAGEMAKER = "sagemaker"
    SAGEMAKER_CHAT = "sagemaker_chat"
    BEDROCK = "bedrock"
    VLLM = "vllm"
    NLP_CLOUD = "nlp_cloud"
    PETALS = "petals"
    OOBABOOGA = "oobabooga"
    OLLAMA = "ollama"
    OLLAMA_CHAT = "ollama_chat"
    DEEPINFRA = "deepinfra"
    PERPLEXITY = "perplexity"
    ANYSCALE = "anyscale"
    MISTRAL = "mistral"
    GROQ = "groq"
    NVIDIA_NIM = "nvidia_nim"
    CEREBRAS = "cerebras"
    AI21_CHAT = "ai21_chat"
    VOLCENGINE = "volcengine"
    CODESTRAL = "codestral"
    TEXT_COMPLETION_CODESTRAL = "text-completion-codestral"
    DEEPSEEK = "deepseek"
    MARITALK = "maritalk"
    VOYAGE = "voyage"
    CLOUDFLARE = "cloudflare"
    XINFERENCE = "xinference"
    FIREWORKS_AI = "fireworks_ai"
    FRIENDLIAI = "friendliai"
    WATSONX = "watsonx"
    TRITON = "triton"
    PREDIBASE = "predibase"
    DATABRICKS = "databricks"
    EMPOWER = "empower"
    GITHUB = "github"
    CUSTOM = "custom"
    NOTDIAMOND = "notdiamond"


provider_list: List[Union[LlmProviders, str]] = list(LlmProviders)

from .main import *  # noqa
