import os
from importlib import metadata

from dotenv import load_dotenv

load_dotenv(os.getcwd() + "/.env")

VERSION = metadata.version("notdiamond")

NOTDIAMOND_API_KEY = os.getenv("NOTDIAMOND_API_KEY", default="")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", default="")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", default="")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", default="")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", default="")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", default="")
PPLX_API_KEY = os.getenv("PPLX_API_KEY", default="")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", default="")


NOTDIAMOND_API_URL = os.getenv(
    "NOTDIAMOND_API_URL", "https://api.notdiamond.ai"
)

PROVIDERS = {
    "openai": {
        "models": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-mini-2024-09-12",
            "chatgpt-4o-latest",
        ],
        "api_key": OPENAI_API_KEY,
        "support_tools": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
        ],
        "support_response_model": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "o1-preview",
            "o1-preview-2024-09-12",
            "o1-mini",
            "o1-mini-2024-09-12",
            "chatgpt-4o-latest",
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
        ],
        "openrouter_identifier": {
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            "gpt-4": "openai/gpt-4",
            "gpt-4-turbo": "openai/gpt-4-turbo",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06": "openai/gpt-4o-2024-08-06",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview": "openai/gpt-4-turbo-preview",
            "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
            "o1-preview": "openai/o1-preview",
            "o1-preview-2024-09-12": "openai/o1-preview-2024-09-12",
            "o1-mini": "openai/o1-mini",
            "o1-mini-2024-09-12": "openai/o1-mini-2024-09-12",
            "chatgpt-4o-latest": "openai/chatgpt-4o-latest",
            "gpt-4.5-preview": "openai/gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27": "openai/gpt-4.5-preview-2025-02-27",
        },
        "price": {
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-3.5-turbo-0125": {"input": 0.5, "output": 1.5},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-0613": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-4-turbo-2024-04-09": {"input": 10.0, "output": 30.0},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
            "gpt-4o-2024-08-06": {"input": 2.5, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
            "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-4-0125-preview": {"input": 10.0, "output": 30.0},
            "gpt-4-1106-preview": {"input": 10.0, "output": 30.0},
            "o1-preview": {"input": 15.0, "output": 60.0},
            "o1-preview-2024-09-12": {"input": 15.0, "output": 60.0},
            "o1-mini": {"input": 3.0, "output": 12.0},
            "o1-mini-2024-09-12": {"input": 3.0, "output": 12.0},
            "chatgpt-4o-latest": {"input": 5.0, "output": 15.0},
            "gpt-4.5-preview": {"input": 75.0, "output": 150.0},
            "gpt-4.5-preview-2025-02-27": {"input": 75.0, "output": 150.0},
        },
    },
    "anthropic": {
        "models": [
            "claude-2.1",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-7-sonnet-latest",
            "claude-3-7-sonnet-20250219",
        ],
        "api_key": ANTHROPIC_API_KEY,
        "support_tools": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-latest",
            "claude-3-7-sonnet-latest",
            "claude-3-7-sonnet-20250219",
        ],
        "support_response_model": [
            "claude-2.1",
            "claude-3-opus-20240229",
        ],
        "openrouter_identifier": {
            "claude-2.1": "anthropic/claude-2.1",
            "claude-3-opus-20240229": "anthropic/claude-3-opus",
            "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
            "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
            "claude-3-5-sonnet-20240620": "anthropic/claude-3.5-sonnet-20240620",
            "claude-3-5-sonnet-latest": "anthropic/claude-3.5-sonnet",
            "claude-3-5-haiku-20241022": "anthropic/claude-3.5-haiku",
            "claude-3-7-sonnet-latest": "anthropic/claude-3.7-sonnet",
            "claude-3-7-sonnet-20250219": "anthropic/claude-3.7-sonnet",
        },
        "price": {
            "claude-2.1": {"input": 8.0, "output": 24.0},
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
            "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
            "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
            "claude-3-5-sonnet-latest": {"input": 3.0, "output": 15.0},
            "claude-3-7-sonnet-latest": {"input": 3.0, "output": 15.0},
            "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
        },
    },
    "google": {
        "models": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-exp-0801",
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
        ],
        "api_key": GOOGLE_API_KEY,
        "support_tools": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-exp-0801",
            "gemini-1.5-flash-latest",
        ],
        "support_response_model": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-exp-0801",
            "gemini-1.5-flash-latest",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
        ],
        "openrouter_identifier": {
            "gemini-pro": "google/gemini-pro",
            "gemini-1.0-pro-latest": "google/gemini-pro",
            "gemini-1.5-pro-latest": "google/gemini-pro-1.5",  #
            "gemini-1.5-pro-exp-0801": "google/gemini-pro-1.5-exp",  #
            "gemini-1.5-flash-latest": "google/gemini-flash-1.5",  #
            "gemini-2.0-flash": "google/gemini-2.0-flash",  #
            "gemini-2.0-flash-001": "google/gemini-2.0-flash",  #
        },
        "price": {
            "gemini-pro": {"input": 0.5, "output": 1.5},
            "gemini-1.0-pro-latest": {"input": 0.5, "output": 1.5},
            "gemini-1.5-pro-latest": {"input": 1.75, "output": 10.5},
            "gemini-1.5-pro-exp-0801": {"input": 1.75, "output": 10.5},
            "gemini-1.5-flash-latest": {"input": 0.35, "output": 1.05},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
        },
    },
    "cohere": {
        "models": ["command-r", "command-r-plus"],
        "api_key": COHERE_API_KEY,
        "support_tools": ["command-r", "command-r-plus"],
        "support_response_model": ["command-r", "command-r-plus"],
        "openrouter_identifier": {
            "command-r": "cohere/command-r",
            "command-r-plus": "cohere/command-r-plus",
        },
        "price": {
            "command-r": {"input": 0.5, "output": 1.5},
            "command-r-plus": {"input": 3.0, "output": 15.0},
        },
    },
    "mistral": {
        "models": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "open-mistral-nemo",
            "codestral-latest",
        ],
        "api_key": MISTRAL_API_KEY,
        "support_tools": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-small-latest",
            "open-mixtral-8x22b",
            "open-mistral-nemo",
        ],
        "support_response_model": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "open-mistral-nemo",
            "codestral-latest",
        ],
        "openrouter_identifier": {
            "mistral-large-latest": "mistralai/mistral-large",
            "mistral-large-2407": "mistralai/mistral-large",
            "mistral-medium-latest": "mistralai/mistral-medium",
            "mistral-small-latest": "mistralai/mistral-small",
            "open-mistral-7b": "mistralai/mistral-7b-instruct",
            "open-mixtral-8x7b": "mistralai/mixtral-8x7b",
            "open-mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",
            "open-mistral-nemo": "mistralai/mistral-nemo",
        },
        "price": {
            "mistral-large-latest": {"input": 2.0, "output": 6.0},
            "mistral-large-2402": {"input": 4.0, "output": 12.0},
            "mistral-large-2407": {"input": 2.0, "output": 6.0},
            "mistral-medium-latest": {"input": 2.7, "output": 8.1},
            "mistral-small-latest": {"input": 1.0, "output": 3.0},
            "open-mistral-7b": {"input": 0.25, "output": 0.25},
            "open-mixtral-8x7b": {"input": 0.7, "output": 0.7},
            "open-mixtral-8x22b": {"input": 2.0, "output": 6.0},
            "open-mistral-nemo": {"input": 0.15, "output": 0.15},
            "codestral-latest": {"input": 1.0, "output": 3.0},
        },
    },
    "togetherai": {
        "models": [
            "Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B-Instruct-v0.1",
            "Mixtral-8x22B-Instruct-v0.1",
            "Llama-3-70b-chat-hf",
            "Llama-3-8b-chat-hf",
            "Qwen2-72B-Instruct",
            "Meta-Llama-3.1-8B-Instruct-Turbo",
            "Meta-Llama-3.1-70B-Instruct-Turbo",
            "Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek-R1",
        ],
        "api_key": TOGETHER_API_KEY,
        "model_prefix": {
            "Mistral-7B-Instruct-v0.2": "mistralai",
            "Mixtral-8x7B-Instruct-v0.1": "mistralai",
            "Mixtral-8x22B-Instruct-v0.1": "mistralai",
            "Llama-3-70b-chat-hf": "meta-llama",
            "Llama-3-8b-chat-hf": "meta-llama",
            "Qwen2-72B-Instruct": "Qwen",
            "Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama",
            "Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama",
            "Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama",
            "DeepSeek-R1": "deepseek-ai",
        },
        "openrouter_identifier": {
            "Llama-3-70b-chat-hf": "meta-llama/llama-3-70b-instruct",
            "Llama-3-8b-chat-hf": "meta-llama/llama-3-8b-instruct",
            "Qwen2-72B-Instruct": "qwen/qwen-2-72b-instruct",
            "Mistral-7B-Instruct-v0.2": "mistralai/mistral-7b-instruct",
            "Mixtral-8x7B-Instruct-v0.1": "mistralai/mixtral-8x7b",
            "Mixtral-8x22B-Instruct-v0.1": "mistralai/mixtral-8x22b-instruct",
            "Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/llama-3.1-8b-instruct",
            "Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/llama-3.1-70b-instruct",
            "Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/llama-3.1-405b-instruct",
            "DeepSeek-R1": "deepseek/deepseek-r1",
        },
        "price": {
            "Mistral-7B-Instruct-v0.2": {"input": 0.2, "output": 0.2},
            "Mixtral-8x7B-Instruct-v0.1": {"input": 0.6, "output": 0.6},
            "Mixtral-8x22B-Instruct-v0.1": {"input": 0.6, "output": 0.6},
            "Llama-3-70b-chat-hf": {"input": 0.7, "output": 0.7},
            "Llama-3-8b-chat-hf": {"input": 0.2, "output": 0.2},
            "Qwen2-72B-Instruct": {"input": 0.9, "output": 0.9},
            "Meta-Llama-3.1-8B-Instruct-Turbo": {
                "input": 0.18,
                "output": 0.18,
            },
            "Meta-Llama-3.1-70B-Instruct-Turbo": {
                "input": 0.88,
                "output": 0.88,
            },
            "Meta-Llama-3.1-405B-Instruct-Turbo": {
                "input": 5.0,
                "output": 15.0,
            },
            "DeepSeek-R1": {
                "input": 7.0,
                "output": 7.0,
            },
        },
    },
    "perplexity": {
        "models": [
            "sonar",
        ],
        "api_key": PPLX_API_KEY,
        "openrouter_identifier": {
            "sonar": "perplexity/sonar",
        },
        "price": {
            "sonar": {"input": 1.0, "output": 1.0},
        },
    },
    "replicate": {
        "models": [
            "mistral-7b-instruct-v0.2",
            "mixtral-8x7b-instruct-v0.1",
            "meta-llama-3-70b-instruct",
            "meta-llama-3-8b-instruct",
            "meta-llama-3.1-405b-instruct",
        ],
        "api_key": REPLICATE_API_KEY,
        "model_prefix": {
            "mistral-7b-instruct-v0.2": "mistralai",
            "mixtral-8x7b-instruct-v0.1": "mistralai",
            "meta-llama-3-70b-instruct": "meta",
            "meta-llama-3-8b-instruct": "meta",
            "meta-llama-3.1-405b-instruct": "meta",
        },
        "openrouter_identifier": {
            "mistral-7b-instruct-v0.2": "mistralai/mistral-7b-instruct",
            "mixtral-8x7b-instruct-v0.1": "mistralai/mixtral-8x7b",
            "meta-llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
            "meta-llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
            "meta-llama-3.1-405b-instruct": "meta-llama/llama-3.1-405b-instruct",
        },
        "price": {
            "mistral-7b-instruct-v0.2": {"input": 0.2, "output": 0.2},
            "mixtral-8x7b-instruct-v0.1": {"input": 0.6, "output": 0.6},
            "meta-llama-3-70b-instruct": {"input": 0.65, "output": 2.75},
            "meta-llama-3-8b-instruct": {"input": 0.05, "output": 0.25},
            "meta-llama-3.1-405b-instruct": {"input": 9.5, "output": 9.5},
        },
    },
}


EMBEDDING_PROVIDERS = {
    "openai": {
        "models": [
            "text-embedding-3-large",
            "text-embedding-3-small",
            "text-embedding-ada-002",
        ],
        "api_key": OPENAI_API_KEY,
    },
    "cohere": {
        "models": [
            "embed-english-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-v3.0",
            "embed-multilingual-light-v3.0",
        ],
        "api_key": COHERE_API_KEY,
    },
    "mistral": {
        "models": [
            "mistral-embed",
        ],
        "api_key": MISTRAL_API_KEY,
    },
}

DEFAULT_USER_AGENT = f"Python-SDK/{VERSION}"
